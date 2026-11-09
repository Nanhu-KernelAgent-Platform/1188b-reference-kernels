"""Reference case for

musa::dnn::LayerNormGammaBetaBwdKernelSplit<
    __mt_bfloat16, float, __mt_bfloat16, __mt_bfloat16, float, 16, 16, 8,
    (musa::dnn::NormType)1>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: aten::_fused_rmsnorm_backward. NormType 1 is RMSNorm, so this
launch only writes dgamma. The second output pointer is the LayerNorm
dbeta slot and is unused: RMSNorm has no beta.

  rstd[m]    = rsqrt(mean_i(x[m, i]^2) + eps)     # forward, eps = 1e-6
  dgamma[i]  = sum_m dy[m, i] * x[m, i] * rstd[m] # this kernel

dy and x are bf16, rstd is fp32, the sum is fp32, and dgamma is stored
bf16 (same dtype as the RMSNorm weight). The mean pointer in the
signature is not part of the RMSNorm formula.

Rows are the local tokens. CP=8, seq 131072, so M = 131072 / 8 = 16384.
The same specialization is launched for three normalized widths:

  N = 8192  hidden,   two norms per layer   grid (64, 16, 1)  128 times  402..423 us
  N = 1536  q_lora,   one norm per layer    grid (12, 16, 1)   64 times   91..98 us
  N = 512   kv_lora,  one norm per layer    grid (4, 16, 1)    64 times   41..50 us

Launch recorded on MTT S5000 for the hidden norm:
  grid (64, 16, 1), block (16, 16, 1), shared memory 1040 bytes.
  block (16, 16) and the template 8 match the tile: 16 threads times
  vec 8 cover 128 columns, so grid.x = N / 128. grid.y = 16 splits the
  row reduction; the partial sums are combined into one dgamma[N].
"""

from __future__ import annotations

import torch


M = 16384
N = 8192
ROW_CHUNK = 2048


def reference_rmsnorm_dgamma(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """dgamma[i] = sum_m dy[m, i] * x[m, i] * rstd[m].

    Returns the fp32 sum and the bf16 value stored by the kernel.
    """
    if dy.shape != x.shape or rstd.shape != (dy.shape[0],):
        raise ValueError(f"shapes dy {tuple(dy.shape)} x {tuple(x.shape)} rstd {tuple(rstd.shape)}")
    acc = torch.zeros(dy.shape[1], dtype=torch.float32, device=dy.device)
    scale = rstd.float()
    for start in range(0, dy.shape[0], ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        acc += (dy[rows].float() * x[rows].float() * scale[rows, None]).sum(dim=0)
    return acc, acc.to(torch.bfloat16)


def make_inputs(m: int = M, n: int = N, seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    dy = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    x = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    # rstd = rsqrt(mean(x^2) + eps), kept as the forward kernel stored it.
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
    return dy.to(device), x.to(device), rstd.to(device)


def _check_ones() -> None:
    dy = torch.ones(4, 8, dtype=torch.bfloat16)
    x = torch.ones_like(dy)
    rstd = torch.full((4,), 0.5, dtype=torch.float32)
    acc, stored = reference_rmsnorm_dgamma(dy, x, rstd)
    # 4 rows * 1 * 1 * 0.5 = 2, exact in bf16.
    if not torch.equal(acc, torch.full((8,), 2.0)) or not torch.equal(stored, acc.to(torch.bfloat16)):
        raise SystemExit("dgamma of a constant row is not 2")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    dy, x, rstd = make_inputs(seed=seed, device=device)
    acc, stored = reference_rmsnorm_dgamma(dy, x, rstd)
    max_abs = 0.0
    for col in (0, N // 2, N - 1):
        manual = (dy[:, col].float() * x[:, col].float() * rstd.float()).sum()
        max_abs = max(max_abs, (acc[col] - manual).abs().item())
        if stored[col] != acc[col].to(torch.bfloat16):
            raise SystemExit(f"bf16 store disagrees with the fp32 sum at column {col}")
    print(
        f"device={device} dy={tuple(dy.shape)} {dy.dtype} "
        f"dgamma={tuple(stored.shape)} {stored.dtype}"
    )
    print(f"max_abs={max_abs:.6e}")
    if stored.shape != (N,) or stored.dtype != torch.bfloat16 or max_abs > 1e-2:
        raise SystemExit("dgamma disagrees with the explicit column sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_ones()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
