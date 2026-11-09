"""Reference case for

musa_asm_tf32tf32tf32tf32ssgemm_nn_tce_768_384x256B128_epilogue_persis_stage2_tf32_rne

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: RouterGatingLinearFunctionBackward.

This launch is the router dgrad (layout NN), not the wgrad.
The wgrad is the sibling kernel
musa_asm_tf32tf32tf32tf32ssgemm_tn_tce_256_128x256B128_epilogue_persis_stage4_btmenc_tf32_rne.

  grad_input[M, 8192] = grad_output[M, 320] @ weight[320, 8192]

  Both inputs are fp32 and the tensor-core kernel rounds them to tf32
  with round-to-nearest-even (the tf32_rne suffix) before the MMA.
  Accumulation and the stored result are fp32. No bias, alpha=1, beta=0.
  The autograd function casts this fp32 result to the activation dtype
  (bf16) after the kernel returns.

Shape evidence, same router as the forward NT kernel:
  weight is [num_experts, hidden] = [320, 8192].
  grad_output matches the forward logits, [16384, 320].
  16384 = seq 131072 / CP 8. 8192 / 256 = 32, so the 256-wide N tile
  covers hidden exactly. 64 launches, one per backward; forward was 128
  because full recompute runs the forward twice.
  Duration 729..817 us.

Launch recorded on MTT S5000 (60 SMs):
  grid (60, 1, 1), block (768, 1, 1), dynamic shared memory 163840 bytes.
  Tile encoded in the kernel name: 384x256, K-block 128.
  epilogue_persis_stage2 is the persistent fp32 store.

On MUSA this is the muDNN matmul selected by
  te_general_gemm(weight.to(fp32), grad_output, out_dtype=fp32, layout="NN", grad=True)
"""

from __future__ import annotations

import torch


M = 16384
K = 320  # num_experts
N = 8192  # hidden_size

_TF32_LOW = 0x1FFF
_TF32_TIE = 0x1000
_TF32_MASK = 0xFFFFE000


def round_tf32_rne(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to tf32, ties to even. Matches the kernel's tf32_rne inputs."""
    bits = x.float().contiguous().view(torch.int32)
    low = bits & _TF32_LOW
    lsb = (bits >> 13) & 1
    # Adding the tie bit rounds; a tie whose kept LSB is already even must not.
    add = torch.where((low == _TF32_TIE) & (lsb == 0), torch.zeros_like(bits), torch.full_like(bits, _TF32_TIE))
    return ((bits + add) & _TF32_MASK).view(torch.float32)


def reference_tf32(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """NN dgrad golden.

    grad_output: [M, K] fp32
    weight:      [K, N] fp32 (the bf16 parameter cast to fp32)
    return:      [M, N] fp32
    """
    return round_tf32_rne(grad_output).matmul(round_tf32_rne(weight))


def _check_tf32_ties() -> None:
    # Halfway between two tf32 values of 1.0, kept LSB even: stay at 1.0.
    stay = torch.tensor([0x3F801000], dtype=torch.int32).view(torch.float32)
    # Halfway, kept LSB odd: round away to the next even tf32 value.
    up = torch.tensor([0x3F803000], dtype=torch.int32).view(torch.float32)
    got_stay = round_tf32_rne(stay).view(torch.int32).item()
    got_up = round_tf32_rne(up).view(torch.int32).item()
    if got_stay != 0x3F800000 or got_up != 0x3F804000:
        raise SystemExit(f"tf32 RNE tie check failed: {got_stay:#x} {got_up:#x}")


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    grad_output = torch.randn(M, K, generator=g, dtype=torch.float32)
    # Parameter is bf16; backward casts it to fp32 before this GEMM.
    weight = torch.randn(K, N, generator=g, dtype=torch.bfloat16).float()
    return grad_output.to(device), weight.to(device)


def run_case(device: str = "cpu", seed: int = 0) -> None:
    grad_output, weight = make_inputs(seed=seed, device=device)
    rows = torch.tensor([0, M // 2, M - 1], device=device)
    got = reference_tf32(grad_output[rows], weight)
    a = round_tf32_rne(grad_output[rows])
    b = round_tf32_rne(weight)
    manual = (a.unsqueeze(-1) * b.unsqueeze(0)).sum(dim=1)

    diff = (got - manual).abs()
    max_abs = diff.max().item()
    print(f"device={device} M={M} K={K} N={N}")
    print(f"out {tuple(got.shape)} {got.dtype} (rows {rows.tolist()} of {M})")
    print(f"max_abs={max_abs:.6e}")
    if max_abs > 1e-3:
        raise SystemExit("tf32 GEMM disagrees with the explicit sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_tf32_ties()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
