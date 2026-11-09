"""Reference case for

musa::dnn::unary::kernel_unary<float, musa::dnn::unary::CastOp, float,
                                __mt_bfloat16, float, 128>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent ops (corr_op aten::_copy_from / .to(bfloat16)):
  RouterGatingLinearFunction Backward x128
  AttnFuncWithCPAndQKVOA2A Backward x64

The mangled signature is (bf16* dst, float const* src, numel, ...):
this specialization casts fp32 -> bf16. The template 128 is the vector
width used by the unary launcher (4 fp32 elements per thread), so

  grid.x = ceil(numel / (block.x * 4)), block.x = 1024.

In RouterGatingLinearFunction.backward the dgrad / wgrad GEMMs run in
fp32 (moe_router_dtype), then the results are stored back:

  grad_input  = gemm_fp32(...).to(input_dtype)   # bf16
  grad_weight = gemm_fp32(...).to(weight_dtype)  # bf16

  out[i] = cast_bf16_rne(inp[i])

PyTorch / muDNN use round-to-nearest-even on the discarded fp32 mantissa
bits (the low 16 bits). No scale, no alpha/beta beyond the cast.

Three traced problem sizes (64 launches each, 192 total):

  grid (32768,1,1)  numel 134217728 = 16384 * 8192
      router grad_input [16384, 8192] fp32 -> bf16
      duration 582.443..591.883 us
  grid (640,1,1)    numel 2621440   = 320 * 8192
      router grad_weight [320, 8192] fp32 -> bf16
      duration 12.800..16.560 us
  grid (98304,1,1)  numel 402653184 = 131072 * 16 * 192
      attention Q after Ulysses A2A [131072, 16, 192] fp32 -> bf16
      duration 1713.490..1748.131 us

This reference uses the router grad_input shape, the dominant
RouterGatingLinearFunction case by byte volume among the router casts.

Launch recorded on MTT S5000 (grad_input shape):
  grid (32768, 1, 1), block (1024, 1, 1), shared memory 12 bytes.
"""

from __future__ import annotations

import torch


M = 16384
N = 8192


def round_bf16_rne(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 with round-to-nearest-even on the truncated mantissa."""
    if x.dtype != torch.float32:
        raise ValueError(f"expected fp32 input, got {x.dtype}")
    bits = x.contiguous().view(torch.int32)
    # Add the RNE bias to bit 15, then keep the top 16 bits.
    lsb = (bits >> 16) & 1
    rounded = bits + (0x7FFF + lsb)
    # Reinterpret the high half as bf16 without going through fp32 again.
    high = (rounded >> 16).to(torch.int16)
    return high.view(torch.bfloat16).view_as(x)


def reference_cast_fp32_to_bf16(inp: torch.Tensor) -> torch.Tensor:
    """CastOp: out = bf16(inp), RNE."""
    return round_bf16_rne(inp.float())


def make_inputs(m: int = M, n: int = N, seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Values large enough that several mantissa bits sit in the rounded region.
    inp = torch.randn(m, n, generator=g, dtype=torch.float32) * 8
    return inp.to(device)


def _check_rne_ties() -> None:
    # Exact halfway cases on the bf16 boundary. bit pattern construction:
    # 1.0 fp32 = 0x3F800000; bf16 keeps 0x3F80. The next bf16 is 0x3F81.
    # Midpoint between them in fp32 is 0x3F808000.
    mid = torch.tensor([0x3F808000], dtype=torch.int32).view(torch.float32)
    # Even bf16 LSB (0) -> tie rounds down to 1.0.
    got = round_bf16_rne(mid)
    expect = torch.tensor([1.0], dtype=torch.bfloat16)
    if not torch.equal(got, expect):
        raise SystemExit(f"RNE tie-to-even failed for 1.0 midpoint: {got}")

    # Midpoint above 0x3F810000 (= 1.0 + 1 ulp in bf16): LSB becomes odd after
    # the high half, so the tie rounds up.
    mid_odd = torch.tensor([0x3F818000], dtype=torch.int32).view(torch.float32)
    got_odd = round_bf16_rne(mid_odd)
    expect_odd = torch.tensor([0x3F82], dtype=torch.int16).view(torch.bfloat16)
    if not torch.equal(got_odd, expect_odd):
        raise SystemExit(f"RNE tie-to-even failed for odd midpoint: {got_odd}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    inp = make_inputs(seed=seed, device=device)
    golden = reference_cast_fp32_to_bf16(inp)
    # On MUSA this is the muDNN unary CastOp selected by .to(bfloat16) /
    # aten::_copy_from into a bf16 tensor.
    got = inp.to(torch.bfloat16)

    diff = (got.float() - golden.float()).abs()
    max_abs = diff.max().item()
    bit_mismatch = int((got.view(torch.int16) != golden.view(torch.int16)).sum().item())
    print(f"device={device} input={tuple(inp.shape)} {inp.dtype} out={tuple(got.shape)} {got.dtype}")
    print(f"max_abs={max_abs:.6e} bit_mismatch={bit_mismatch}")
    if max_abs != 0 or bit_mismatch != 0:
        raise SystemExit("fp32->bf16 CastOp disagrees with the RNE reference")
    print("reference check passed")


if __name__ == "__main__":
    _check_rne_ties()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
