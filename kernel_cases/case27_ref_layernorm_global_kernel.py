"""Reference case for

musa::dnn::LayerNormGlobalKernelVlen<
    __mt_bfloat16, float, __mt_bfloat16, float, 128, 4, 8,
    (musa::dnn::NormType)1, false>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent stack: CheckpointFunction / _OperationFuserAutogradFunction
(TE fused RMSNorm forward via at::musa::FusedRMSNormForwardOut).
NormType 1 is RMSNorm. The trailing false is zero_centered_gamma=false,
so gamma is applied as stored (not 1 + gamma). This run does not enable
--apply-layernorm-1p.

  rstd[m] = rsqrt(mean_i(x[m, i]^2) + eps)
  y[m, i] = x[m, i] * rstd[m] * gamma[i]

x and gamma are bf16, the mean/rsqrtrt accumulate in fp32, rstd is stored
fp32, and y is stored bf16. eps = 1e-6 (--norm-epsilon). RMSNorm has no
mean centering and no beta; the mean and beta pointers in the signature
are unused.

Rows are the local tokens. CP=8, seq 131072, so M = 131072 / 8 = 16384.
This <128, 4, 8> specialization is the hidden norm, N = 8192. The matching
backward dx kernel is LayerNormBwdGlobalKernelVlen with the same tile.

Launch recorded on MTT S5000:
  grid (4096, 1, 1), block (128, 4, 1), shared memory 4112 bytes.
  Template 128, 4, 8 is the block: 128 threads, 4 rows, vec 8.
  4096 * 4 = 16384 tokens. 8192 / 8 = 1024 elements per row, covered by
  8 passes of the 128 threads.
  256 launches (attention and MLP RMSNorm; full recompute runs the
  forward twice).
  Duration 425.842..480.083 us (median ~445 us).
"""

from __future__ import annotations

import torch


M = 16384
N = 8192
ROW_CHUNK = 1024
EPS = 1e-6


def reference_rmsnorm_fwd(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm forward. Returns (fp32 y, bf16 y, fp32 rstd)."""
    if x.ndim != 2 or gamma.shape != (x.shape[1],):
        raise ValueError(f"shapes x {tuple(x.shape)} gamma {tuple(gamma.shape)}")
    y_acc = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    weight = gamma.float()
    for start in range(0, x.shape[0], ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        xf = x[rows].float()
        inv = torch.rsqrt(xf.square().mean(dim=-1) + eps)
        rstd[rows] = inv
        y_acc[rows] = xf * inv[:, None] * weight
    return y_acc, y_acc.to(x.dtype), rstd


def make_inputs(m: int = M, n: int = N, seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    gamma = torch.randn(n, generator=g, dtype=torch.bfloat16)
    return x.to(device), gamma.to(device)


def _check_closed_form() -> None:
    # Constant row x=1, gamma=2, eps=0 -> rstd=1, y=2.
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    gamma = torch.full((4,), 2, dtype=torch.bfloat16)
    acc, stored, rstd = reference_rmsnorm_fwd(x, gamma, eps=0.0)
    if not torch.equal(rstd, torch.ones(2)) or not torch.equal(acc, torch.full((2, 4), 2.0)):
        raise SystemExit(f"constant-row RMSNorm mismatch: rstd={rstd} y={acc}")
    if not torch.equal(stored, acc.to(torch.bfloat16)):
        raise SystemExit("bf16 store disagrees with the fp32 result")

    # x=[3,4], ||x||^2/2 = 12.5, rstd = 1/sqrt(12.5), gamma=[1,2]
    x = torch.tensor([[3, 4]], dtype=torch.bfloat16)
    gamma = torch.tensor([1, 2], dtype=torch.bfloat16)
    acc, stored, rstd = reference_rmsnorm_fwd(x, gamma, eps=0.0)
    expect_rstd = torch.rsqrt(torch.tensor([12.5]))
    expect_y = x.float() * expect_rstd[:, None] * gamma.float()
    if (rstd - expect_rstd).abs().max().item() > 1e-6:
        raise SystemExit(f"rstd mismatch: {rstd} vs {expect_rstd}")
    if (acc - expect_y).abs().max().item() > 1e-5:
        raise SystemExit(f"y mismatch: {acc} vs {expect_y}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    x, gamma = make_inputs(seed=seed, device=device)
    acc, stored, rstd = reference_rmsnorm_fwd(x, gamma)
    max_abs = 0.0
    max_rstd = 0.0
    weight = gamma.float()
    for row in (0, M // 2, M - 1):
        xf = x[row].float()
        inv = torch.rsqrt(xf.square().mean() + EPS)
        manual = xf * inv * weight
        max_abs = max(max_abs, (acc[row] - manual).abs().max().item())
        max_rstd = max(max_rstd, abs(rstd[row].item() - inv.item()))
        if not torch.equal(stored[row], acc[row].to(torch.bfloat16)):
            raise SystemExit(f"bf16 store disagrees with the fp32 y at row {row}")
    print(
        f"device={device} x={tuple(x.shape)} {x.dtype} "
        f"y={tuple(stored.shape)} {stored.dtype} rstd={tuple(rstd.shape)} {rstd.dtype}"
    )
    print(f"max_abs={max_abs:.6e} max_rstd={max_rstd:.6e}")
    if max_abs > 1e-4 or max_rstd > 1e-6:
        raise SystemExit("RMSNorm forward disagrees with the explicit formula")
    print("reference check passed")


if __name__ == "__main__":
    _check_closed_form()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
