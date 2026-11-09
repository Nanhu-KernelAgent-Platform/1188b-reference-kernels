"""Reference case for

musa_asm_bf16bf16bf16bf16gemm_nn_tce_768_256x384B128_squad_level_epilogue

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: _GroupedLinearBackward (TransformerEngine grouped GEMM, layout "NN").

This launch is one expert of the MoE fc1 dgrad, not fc2 and not wgrad.

  dgrad[M, 8192] = grad_output[M, 5120] @ weight[5120, 8192]

  dtype bf16, both inputs row-major and not transposed, alpha=1, beta=0, no bias.
  The "squad_level_epilogue" is the bf16 store epilogue (no GELU, no accumulate).

Shape evidence from the same trace:
  hidden = 8192
    moe_unpermute grid.x = ceil(hidden / 1024) = 8, and the 1188B script uses 8192.
  moe fc1 out = 5120, fc1 intermediate = 2560 (SwiGLU)
    triton cat grid / silu grid = 2, and both are numel/128,
    so the two tensors are [tokens, 5120] and [tokens, 2560].
  40 local experts (320 experts / EP 8). This kernel appears only on the
  GroupedLinear whose wgrad is the 2x-FLOP fc1
  (ssgemm_tn_tce_512_256x256), 116 times, one expert per launch.
  Token count of that expert is about 288
  (trace estimate 269..336, median 287). M=288 is the representative.

Launch recorded on MTT S5000 (60 SMs):
  grid (60, 1, 1), block (768, 1, 1), dynamic shared memory 196608 bytes.
  Tile encoded in the kernel name: 256x384, K-block 128.

On MUSA, torch.matmul(bf16, bf16) goes through muDNN and is the launch to
compare against. If the picked kernel name is different, sweep M in 256..384;
N and K stay 8192 and 5120.
"""

from __future__ import annotations

import torch


# Representative expert token count. See module docstring for the band.
M = 288
K = 5120  # fc1 out_features, SwiGLU 2 * moe_ffn_hidden_size
N = 8192  # hidden_size, dgrad columns


def reference_fp32(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """NN GEMM golden: dgrad = grad_output @ weight, accumulated in fp32."""
    return grad_output.float().matmul(weight.float())


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    grad_output = torch.randn(M, K, generator=g, dtype=torch.bfloat16)
    weight = torch.randn(K, N, generator=g, dtype=torch.bfloat16)
    return grad_output.to(device), weight.to(device)


def run_case(device: str = "cpu", seed: int = 0) -> None:
    grad_output, weight = make_inputs(seed=seed, device=device)
    golden = reference_fp32(grad_output, weight)

    # bf16 NN GEMM. On MUSA this is the muDNN matmul that selects the asm kernel.
    got = torch.matmul(grad_output, weight)

    diff = (got.float() - golden).abs()
    max_abs = diff.max().item()
    # bf16 matmul over K=5120. Allow a few ulps at the output magnitude.
    denom = golden.abs().clamp(min=1.0)
    max_rel = (diff / denom).max().item()
    print(f"device={device} M={M} K={K} N={N}")
    print(f"out {tuple(got.shape)} {got.dtype}")
    print(f"max_abs={max_abs:.6f} max_rel={max_rel:.6f}")
    if max_rel > 2e-2 and max_abs > 2e-1:
        raise SystemExit("bf16 GEMM disagrees with the fp32 reference")
    print("reference check passed")


if __name__ == "__main__":
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
