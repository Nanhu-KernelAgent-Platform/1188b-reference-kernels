"""Reference case for

musa_asm_bf16bf16bf16bf16gemm_nt_tce_768_256x384B128_squad_level_epilogue

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: _GroupedLinear (TransformerEngine grouped GEMM, layout "TN").

This launch is one expert of the MoE fc1 forward. It is the forward
counterpart of musa_asm_bf16bf16bf16bf16gemm_nn_tce_768_256x384B128_squad_level_epilogue,
which is the same expert's dgrad. 232 launches = 116 experts x 2, because
full recompute runs the forward again during backward.

  out[M, 5120] = inp[M, 8192] @ weight[5120, 8192].T

  dtype bf16, layout TN: weight is transposed, inp is not.
  alpha=1, beta=0, no bias, no GELU. squad_level_epilogue stores bf16.
  muDNN swaps the two arguments, so the assembled kernel is named nt.

Shape evidence is the same fc1 expert as the NN dgrad reference:
  hidden = 8192, SwiGLU fc1 out = 2 * 2560 = 5120.
  M=288 is the representative token count (trace band 269..336, median 287).
  If the picked kernel name differs, sweep M in 256..384. K and N stay.

Launch recorded on MTT S5000 (60 SMs):
  grid (60, 1, 1), block (768, 1, 1), dynamic shared memory 196608 bytes.
  Tile encoded in the kernel name: 256x384, K-block 128.
  Duration 230..277 us.
"""

from __future__ import annotations

import torch


# Representative expert token count. See module docstring for the band.
M = 288
K = 8192  # hidden_size
N = 5120  # fc1 out_features, SwiGLU 2 * moe_ffn_hidden_size


def reference_fp32(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """TN GEMM golden: out = inp @ weight.T, accumulated in fp32.

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
    golden = reference_fp32(inp, weight)

    # bf16 TN GEMM. On MUSA this is the muDNN matmul that selects the asm kernel.
    got = torch.matmul(inp, weight.t())

    diff = (got.float() - golden).abs()
    max_abs = diff.max().item()
    # bf16 matmul over K=8192. Allow a few ulps at the output magnitude.
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
