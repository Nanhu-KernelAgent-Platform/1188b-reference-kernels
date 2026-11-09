"""Reference case for

musa_asm_tf32tf32tf32tf32ssgemm_tn_tce_256_128x256B128_epilogue_persis_stage4_btmenc_tf32_rne

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: RouterGatingLinearFunctionBackward.

This launch is the router wgrad. Layout NT in Transformer Engine is
swapped to tn when muDNN is called with (grad_output, inp). The dgrad
sibling is
musa_asm_tf32tf32tf32tf32ssgemm_nn_tce_768_384x256B128_epilogue_persis_stage2_tf32_rne.

  grad_weight[320, 8192] = grad_output[16384, 320].T @ inp[16384, 8192]

  inp is the saved bf16 activation, cast to fp32. grad_output is already
  fp32, the dtype of the router logits. Both operands are then rounded
  to tf32 with round-to-nearest-even (the tf32_rne suffix) before the MMA.
  Accumulation and the stored result are fp32. No bias, alpha=1, beta=0.
  The autograd function casts this fp32 result to the weight dtype (bf16)
  after the kernel returns.

Shape evidence, same router as the forward NT kernel:
  weight is [num_experts, hidden] = [320, 8192], and so is grad_weight.
  16384 = seq 131072 / CP 8.
  64 launches, one per backward; the forward GEMM runs 128 times because
  full recompute runs the forward twice.
  Duration 1459.849..1550.889 us.

Launch recorded on MTT S5000 (60 SMs):
  grid (60, 1, 1), block (256, 1, 1), dynamic shared memory 196608 bytes.
  The 256 in the kernel name is the block size. Tile 128x256, K-block 128.
  epilogue_persis_stage4_btmenc is the persistent fp32 store.

On MUSA this is the muDNN matmul selected by
  te_general_gemm(inp.to(fp32), grad_output, out_dtype=fp32, layout="NT", grad=True)
"""

from __future__ import annotations

import torch


M = 16384
N = 320  # num_experts, rows of grad_weight
K = 8192  # hidden, cols of grad_weight

_TF32_LOW = 0x1FFF
_TF32_TIE = 0x1000
_TF32_MASK = 0xFFFFE000


def round_tf32_rne(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to tf32, ties to even. Matches the kernel's tf32_rne inputs."""
    bits = x.float().contiguous().view(torch.int32)
    low = bits & _TF32_LOW
    lsb = (bits >> 13) & 1
    add = torch.where((low == _TF32_TIE) & (lsb == 0), torch.zeros_like(bits), torch.full_like(bits, _TF32_TIE))
    return ((bits + add) & _TF32_MASK).view(torch.float32)


def reference_tf32(grad_output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """TN wgrad golden.

    grad_output: [M, N] fp32
    inp:         [M, K] fp32 (the bf16 activation cast to fp32)
    return:      [N, K] fp32
    """
    return round_tf32_rne(grad_output).transpose(0, 1).matmul(round_tf32_rne(inp))


def _check_tf32_ties() -> None:
    stay = torch.tensor([0x3F801000], dtype=torch.int32).view(torch.float32)
    up = torch.tensor([0x3F803000], dtype=torch.int32).view(torch.float32)
    got_stay = round_tf32_rne(stay).view(torch.int32).item()
    got_up = round_tf32_rne(up).view(torch.int32).item()
    if got_stay != 0x3F800000 or got_up != 0x3F804000:
        raise SystemExit(f"tf32 RNE tie check failed: {got_stay:#x} {got_up:#x}")


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    grad_output = torch.randn(M, N, generator=g, dtype=torch.float32)
    inp = torch.randn(M, K, generator=g, dtype=torch.bfloat16).float()
    return grad_output.to(device), inp.to(device)


def run_case(device: str = "cpu", seed: int = 0) -> None:
    grad_output, inp = make_inputs(seed=seed, device=device)
    cols = torch.tensor([0, N // 2, N - 1], device=device)
    got = reference_tf32(grad_output[:, cols], inp)
    g = round_tf32_rne(grad_output[:, cols])
    x = round_tf32_rne(inp)
    manual = torch.empty_like(got)
    for i in range(cols.numel()):
        manual[i] = (g[:, i].unsqueeze(1) * x).sum(dim=0)

    diff = (got - manual).abs()
    max_abs = diff.max().item()
    print(f"device={device} M={M} N={N} K={K}")
    print(f"out {tuple(got.shape)} {got.dtype} (experts {cols.tolist()} of {N})")
    print(f"max_abs={max_abs:.6e}")
    if got.shape != (cols.numel(), K) or max_abs > 1e-2:
        raise SystemExit("tf32 wgrad disagrees with the explicit sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_tf32_ties()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
