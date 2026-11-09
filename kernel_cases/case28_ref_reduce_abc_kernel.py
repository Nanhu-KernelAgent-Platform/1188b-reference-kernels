"""Reference case for

musa::dnn::reduce_abc::ReduceABCKernel<
    __mt_bfloat16, float, __mt_bfloat16,
    musa::dnn::ReduceType<float, (musa::dnn::ReduceImplType)0>,
    musa::dnn::ReduceOp<__mt_bfloat16, float, __mt_bfloat16,
                        musa::dnn::ReduceType<float, (musa::dnn::ReduceImplType)0>,
                        (musa::dnn::ReduceMode)1>,
    (musa::dnn::ReduceMode)1, 32, 8, musa::dnn::FastDivmodU32>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: _LayerNormLinearBackward (100% of the 128 launches).

This is the muDNN column reduction used as the bias-gradient path of a
non-FP8 GEMM. Transformer Engine wires it as

  rdc.SetMode(REDUCE_MODE::ADD)   # mudnn_gemm.cpp no_fp8_grad_bias

so (ReduceMode)1 is ADD. The kernel reduces the token axis of a bf16
matrix and stores a bf16 vector:

  out[j] = sum_i inp[i, j]     # fp32 accumulate, bf16 store

The mangled signature has three mutable bf16 pointers and one const bf16
input. The extra two destinations are reduction workspaces; the
mathematical result is a single length-N vector. Template 32 / 8 match
the block tile: block.y = 32 rows, vec 8. block.x = 16 threads cover
16 * 8 = 128 columns per block, so grid.x = N / 128.

Two problem sizes from MLA LayerNormLinear (bias=False in the module
config, but this reduce still sits between the wgrad GEMM and the
RMSNorm backward of q_up / kv_up):

  grid (192, 1, 1)  N = 192 * 128 = 24576 = 128 heads * q_head_dim 192
      q_up  grad_output [M, 24576], duration 540.883..551.603 us
  grid (256, 1, 1)  N = 256 * 128 = 32768 = 128 heads * (qk 128 + v 128)
      kv_up grad_output [M, 32768], duration 1112.926..1148.767 us

M = 16384 = 131072 / CP 8, the local token count.

This reference uses the kv_up shape (larger N). Launch for that size:
  grid (256, 1, 1), block (16, 32, 1), shared memory 17420 bytes.
  64 launches (one kv_up backward per layer). The q_up twin adds another
  64, for 128 total.
"""

from __future__ import annotations

import torch


M = 16384
N = 32768  # kv_up out_features; q_up uses 24576
ROW_CHUNK = 1024


def reference_reduce_abc(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Column sum: out[j] = sum_i inp[i, j].

    Returns the fp32 accumulator and the bf16 stored vector.
    """
    if inp.ndim != 2:
        raise ValueError(f"expected 2D input, got {tuple(inp.shape)}")
    acc = torch.zeros(inp.shape[1], dtype=torch.float32, device=inp.device)
    for start in range(0, inp.shape[0], ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        acc += inp[rows].float().sum(dim=0)
    return acc, acc.to(inp.dtype)


def make_inputs(m: int = M, n: int = N, seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    inp = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    return inp.to(device)


def _check_small() -> None:
    inp = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.bfloat16,
    )
    acc, stored = reference_reduce_abc(inp)
    expect = torch.tensor([5.0, 7.0, 9.0])
    if not torch.equal(acc, expect) or not torch.equal(stored, expect.to(torch.bfloat16)):
        raise SystemExit(f"column sum mismatch: {acc}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    inp = make_inputs(seed=seed, device=device)
    acc, stored = reference_reduce_abc(inp)
    max_abs = 0.0
    # Spot-check a few columns against an independent sum to avoid a second
    # full fp32 materialization of the 1 GiB activation.
    for col in (0, N // 2, N - 1):
        manual = inp[:, col].float().sum()
        max_abs = max(max_abs, abs(acc[col].item() - manual.item()))
    print(
        f"device={device} inp={tuple(inp.shape)} {inp.dtype} "
        f"out={tuple(stored.shape)} {stored.dtype}"
    )
    print(f"max_abs={max_abs:.6e}")
    if (
        stored.shape != (N,)
        or max_abs > 1e-2
        or not torch.equal(stored, acc.to(torch.bfloat16))
    ):
        raise SystemExit("ReduceABC column sum disagrees with the explicit sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_small()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
