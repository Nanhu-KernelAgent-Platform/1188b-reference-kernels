"""Reference case for

musa::dnn::LayerNormBwdGlobalKernelVlen<
    __mt_bfloat16, float, __mt_bfloat16, float, 128, 4, 8,
    (musa::dnn::NormType)1>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: aten::_fused_rmsnorm_backward. NormType 1 is RMSNorm. This
kernel writes dx. dgamma is the separate LayerNormGammaBetaBwdKernelSplit.

  x_hat = x * rstd
  wdy   = dy * gamma
  dx    = rstd * (wdy - x_hat * mean(wdy * x_hat))

mean is over the normalized hidden axis. x, dy and gamma are bf16, rstd
is the fp32 value saved by the forward
(rstd = rsqrt(mean(x^2) + eps), eps = 1e-6). The sum is fp32 and dx is
stored bf16. RMSNorm has no mean, so the mean pointer in the signature
is not used. gamma is applied as stored; this run does not use the
1+gamma centering.

Rows are the local tokens. CP=8, seq 131072, so M = 131072 / 8 = 16384.
This <128, 4, 8> specialization is the hidden norm, N = 8192. The same
backward for q_lora 1536 and kv_lora 512 uses the <16, 32, 8> kernel.

Launch recorded on MTT S5000:
  grid (4096, 1, 1), block (128, 4, 1), shared memory 2064 bytes.
  Template 128, 4, 8 is the block: 128 threads, 4 rows, vec 8.
  4096 * 4 = 16384 tokens. 8192 / 8 = 1024 elements per row, 8 passes
  of the 128 threads.
  128 launches (attention and MLP RMSNorm, once per layer call).
  The matching forward kernel runs 256 times because full recompute
  runs the forward twice.
  Duration 730.084..826.325 us.
"""

from __future__ import annotations

import torch


M = 16384
N = 8192
ROW_CHUNK = 1024
EPS = 1e-6


def reference_rmsnorm_dx(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """dx = rstd * (dy * gamma - x_hat * mean(dy * gamma * x_hat)).

    Returns the fp32 result and the bf16 value stored by the kernel.
    """
    if dy.shape != x.shape or rstd.shape != (dy.shape[0],) or gamma.shape != (dy.shape[1],):
        raise ValueError(
            f"shapes dy {tuple(dy.shape)} x {tuple(x.shape)} "
            f"rstd {tuple(rstd.shape)} gamma {tuple(gamma.shape)}"
        )
    acc = torch.empty(dy.shape, dtype=torch.float32, device=dy.device)
    weight = gamma.float()
    scale = rstd.float()
    for start in range(0, dy.shape[0], ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        x_hat = x[rows].float() * scale[rows, None]
        wdy = dy[rows].float() * weight
        center = (x_hat * wdy).mean(dim=-1, keepdim=True)
        acc[rows] = scale[rows, None] * (wdy - x_hat * center)
    return acc, acc.to(dy.dtype)


def make_inputs(m: int = M, n: int = N, seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    dy = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    gamma = torch.randn(n, generator=g, dtype=torch.bfloat16)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + EPS)
    return dy.to(device), x.to(device), rstd.to(device), gamma.to(device)


def _check_closed_form() -> None:
    ones = torch.ones(2, 4, dtype=torch.bfloat16)
    acc, stored = reference_rmsnorm_dx(ones, ones, torch.ones(2), torch.ones(4, dtype=torch.bfloat16))
    if not torch.equal(acc, torch.zeros_like(acc)) or not torch.equal(stored, torch.zeros_like(stored)):
        raise SystemExit("dx of a constant RMSNorm row is not 0")

    x = torch.tensor([[3, 4], [3, 4]], dtype=torch.bfloat16)
    dy = torch.ones(2, 2, dtype=torch.bfloat16)
    gamma = torch.tensor([1, 2], dtype=torch.bfloat16)
    rstd = torch.tensor([0.5, 0.5])
    acc, stored = reference_rmsnorm_dx(dy, x, rstd, gamma)
    expect = torch.tensor([[-1.5625, -1.75], [-1.5625, -1.75]])
    if not torch.equal(acc, expect) or not torch.equal(stored, expect.to(torch.bfloat16)):
        raise SystemExit(f"closed-form dx mismatch: {acc}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    dy, x, rstd, gamma = make_inputs(seed=seed, device=device)
    acc, stored = reference_rmsnorm_dx(dy, x, rstd, gamma)
    max_abs = 0.0
    weight = gamma.float()
    for row in (0, M // 2, M - 1):
        x_hat = x[row].float() * rstd[row]
        wdy = dy[row].float() * weight
        manual = rstd[row] * (wdy - x_hat * (x_hat * wdy).mean())
        max_abs = max(max_abs, (acc[row] - manual).abs().max().item())
        if not torch.equal(stored[row], acc[row].to(torch.bfloat16)):
            raise SystemExit(f"bf16 store disagrees with the fp32 dx at row {row}")
    print(
        f"device={device} dy={tuple(dy.shape)} {dy.dtype} "
        f"dx={tuple(stored.shape)} {stored.dtype}"
    )
    print(f"max_abs={max_abs:.6e}")
    if stored.shape != (M, N) or stored.dtype != torch.bfloat16 or max_abs > 1e-4:
        raise SystemExit("dx disagrees with the explicit row formula")
    print("reference check passed")


if __name__ == "__main__":
    _check_closed_form()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
