"""Reference case for

musa_asm_bf16bf16ssgemm_nt_tce_768_320x256B128_squad_level_epilogue

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: RouterGatingLinearFunction (Megatron router gate forward).

  logits[M, 320] = inp[M, 8192] @ weight[320, 8192].T

  inp and weight are bf16. logits are fp32 (moe_router_dtype=fp32).
  Layout is TN: weight is transposed, inp is not. No bias
  (--disable-bias-linear, router bias is not enabled).
  alpha=1, beta=0. squad_level_epilogue stores fp32.

Shape evidence from the same trace and the 1188B script:
  num_experts = 320
    make_row_id_map grid.x = 320, router weight is [num_experts, hidden].
    The 320 in the kernel tile matches this N.
  hidden = 8192
    moe_unpermute grid.x = ceil(hidden / 1024) = 8.
  M = 16384
    local tokens after CP=8 on seq 131072. deep_ep permute grid is (16384, 1, 1).
    16384 / 256 = 64, so the 256-wide tile covers M exactly.
  128 launches, duration 402..413 us, grid always (60, 1, 1).
  One router GEMM per layer call; full activation recompute doubles the forward count.

Launch recorded on MTT S5000 (60 SMs):
  grid (60, 1, 1), block (768, 1, 1), dynamic shared memory 196608 bytes.
  Tile encoded in the kernel name: 320x256, K-block 128.

On MUSA this is the muDNN matmul selected by
  te_general_gemm(weight, inp, out_dtype=torch.float32, layout="TN")
Inputs stay bf16; the output dtype argument is fp32.
"""

from __future__ import annotations

import torch


M = 16384
K = 8192  # hidden_size
N = 320  # num_experts


def reference_fp32(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """TN GEMM golden: logits = inp @ weight.T, accumulated in fp32.

    inp:    [M, K] bf16
    weight: [N, K] bf16
    return: [M, N] fp32
    """
    return inp.float().matmul(weight.float().t())


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    inp = torch.randn(M, K, generator=g, dtype=torch.bfloat16)
    weight = torch.randn(N, K, generator=g, dtype=torch.bfloat16)
    return inp.to(device), weight.to(device)


def run_case(device: str = "cpu", seed: int = 0) -> None:
    inp, weight = make_inputs(seed=seed, device=device)
    # Check the full shape on a few rows. The explicit sum is independent of torch.mm.
    rows = torch.tensor([0, M // 2, M - 1], device=device)
    got = reference_fp32(inp[rows], weight)
    manual = (inp[rows].float().unsqueeze(-1) * weight.float().t().unsqueeze(0)).sum(dim=1)

    diff = (got - manual).abs()
    max_abs = diff.max().item()
    print(f"device={device} M={M} K={K} N={N}")
    print(f"out {tuple(got.shape)} {got.dtype} (rows {rows.tolist()} of {M})")
    print(f"max_abs={max_abs:.6e}")
    if max_abs > 1e-3:
        raise SystemExit("fp32 GEMM disagrees with the explicit sum")
    print("reference check passed")


if __name__ == "__main__":
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
