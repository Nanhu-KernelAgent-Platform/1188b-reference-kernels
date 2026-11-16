"""test_case25.py -- correctness & performance test for the generated MUSA kernel.

Operator (FlashAttention backward preprocess, per-token dot of out and dout):
    delta[h, s] = sum_d out[s, h, d] * dout[s, h, d]      (bf16 in, fp32 accum)

    out, dout : [SEQ=131072, HEADS=16, DIM=128] bf16
    delta     : [HEADS=16, SEQ=131072] fp32   ([head, seq] layout, same as LSE)

Trace reference (MTT S5000, grid (1024,16), block (1024,1), 64 launches/layer):
    duration 896..948 us   ->  the performance target used below: 0.93 ms

What is checked
  1. correctness  : full-tensor comparison against the fp32 reference dot product,
                    tolerance 1e-5 (abs + rel), plus the exact "ones" invariant
                    (delta of ones == DIM, atol = 0) and edge shapes.
  2. performance  : device-timed latency of the generated kernel vs
                    (a) the PyTorch reference implementation,
                    (b) the 0.93 ms traced duration.

Usage
    python test_case25.py              # full report, asserts both sections
    pytest test_case25.py -v           # same checks under pytest
    KA_BUNDLE=/path/to/bundle python test_case25.py     # override kernel bundle
    KA_ITERS=50 KA_PERF_REPEAT=5 ...   # tune benchmarking
Environment
    KA_BUNDLE        dir containing kernel.py/kernel.mu/binding.cpp/setup.py
                     (default: ./preprocess_delta_bundle, built in-place if needed)
    KA_SKIP_BUILD=1  do not try to build the extension, require an existing .so
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "preprocess_delta_bundle")
BUNDLE_DIR = os.environ.get("KA_BUNDLE", DEFAULT_BUNDLE)

SEQ = 131072
HEADS = 16
DIM = 128

TRACE_MS = 0.93          # traced duration of musa::dnn::preprocess_delta_kernel (896..948 us)
ATOL = 1e-5              # required precision error
RTOL = 1e-5
ITERS = int(os.environ.get("KA_ITERS", "30"))
REPEAT = int(os.environ.get("KA_PERF_REPEAT", "5"))
WARMUP = int(os.environ.get("KA_WARMUP", "10"))
STRICT_PERF = os.environ.get("KA_STRICT_PERF", "1") != "0"

DEVICE = "musa" if (hasattr(torch, "musa") and torch.musa.is_available()) else "cpu"


# --------------------------------------------------------------------------
# reference implementation, verbatim from the traced kernel's semantics
# --------------------------------------------------------------------------
def reference_preprocess_delta(out: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    """delta[h, s] = sum_d out[s, h, d] * dout[s, h, d], accumulated in fp32."""
    return (out.float() * dout.float()).sum(dim=-1).transpose(0, 1).contiguous()


# --------------------------------------------------------------------------
# kernel bundle loading (build the MUSA extension on demand)
# --------------------------------------------------------------------------
def _extension_so() -> str | None:
    hits = glob.glob(os.path.join(BUNDLE_DIR, "preprocess_delta_ext*.so"))
    return hits[0] if hits else None


def ensure_bundle(build: bool | None = None) -> str:
    """Return the bundle dir, building the extension in place when necessary."""
    if not os.path.isdir(BUNDLE_DIR):
        raise SystemExit(
            f"kernel bundle not found: {BUNDLE_DIR}\n"
            "set KA_BUNDLE=<dir with kernel.py/kernel.mu/binding.cpp/setup.py>"
        )
    if build is None:
        build = os.environ.get("KA_SKIP_BUILD", "0") != "1"
    if _extension_so() is None:
        if not build:
            raise SystemExit(f"extension not built in {BUNDLE_DIR} (KA_SKIP_BUILD=1)")
        print(f"[build] setup.py build_ext --inplace in {BUNDLE_DIR}", flush=True)
        t0 = time.time()
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=BUNDLE_DIR, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        print(f"[build] done in {time.time() - t0:.1f}s", flush=True)
    if _extension_so() is None:
        raise SystemExit(f"build finished but no .so appeared in {BUNDLE_DIR}")
    return BUNDLE_DIR


def load_kernel():
    """Import the generated kernel module and return its callable."""
    bundle = ensure_bundle()
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    import kernel as ka  # noqa: E402  (kernel.py -> preprocess_delta_ext)

    return ka.kernel_function


def make_inputs(seed: int = 0, seq: int = SEQ, heads: int = HEADS, dim: int = DIM,
                device: str | None = None):
    device = device or DEVICE
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out = torch.randn(seq, heads, dim, generator=g, dtype=torch.bfloat16).to(device)
    dout = torch.randn(seq, heads, dim, generator=g, dtype=torch.bfloat16).to(device)
    return out, dout


# --------------------------------------------------------------------------
# benchmarking
# --------------------------------------------------------------------------
def _sync():
    if DEVICE == "musa":
        torch.musa.synchronize()


def bench_ms(fn, iters: int = ITERS, warmup: int = WARMUP) -> float:
    """Device-timed latency per call; best of REPEAT batched measurements."""
    for _ in range(warmup):
        fn()
    _sync()
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        if DEVICE == "musa":
            start, end = torch.musa.Event(True), torch.musa.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            _sync()
            ms = start.elapsed_time(end) / iters
        else:
            for _ in range(iters):
                fn()
            ms = (time.perf_counter() - t0) * 1e3 / iters
        best = min(best, ms)
    return best


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def check_correctness(kernel_fn) -> dict:
    out, dout = make_inputs()
    ref = reference_preprocess_delta(out, dout)
    got = kernel_fn(out, dout)

    assert tuple(got.shape) == (HEADS, SEQ), f"bad shape {tuple(got.shape)}"
    assert got.dtype == torch.float32, f"bad dtype {got.dtype}"
    assert got.device == out.device, f"bad device {got.device}"

    diff = (got - ref).abs()
    max_abs = diff.max().item()
    denom = ref.abs().clamp_min(1e-30)
    max_rel = (diff / denom).max().item()

    finite = torch.isfinite(got).all().item()
    assert finite, "delta contains non-finite values"
    assert max_abs <= ATOL, f"max_abs={max_abs:.3e} > atol={ATOL:.1e}"
    assert max_rel <= RTOL or max_abs <= ATOL, f"max_rel={max_rel:.3e} > rtol={RTOL:.1e}"

    # spot check against the explicit per-token dot product (as in the reference case)
    spot = 0.0
    for s, h in ((0, 0), (SEQ // 2, HEADS // 2), (SEQ - 1, HEADS - 1)):
        manual = (out[s, h].float() * dout[s, h].float()).sum()
        spot = max(spot, abs(got[h, s].item() - manual.item()))
    assert spot <= ATOL, f"spot max_abs={spot:.3e} > atol={ATOL:.1e}"

    return {
        "shape": tuple(got.shape),
        "dtype": str(got.dtype),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "spot_max_abs": spot,
        "abs_mean": diff.mean().item(),
        "ref_abs_max": ref.abs().max().item(),
    }


def check_ones(kernel_fn) -> None:
    out = torch.ones(4, 2, DIM, dtype=torch.bfloat16, device=DEVICE)
    dout = torch.ones_like(out)
    delta = kernel_fn(out, dout)
    assert tuple(delta.shape) == (2, 4), f"bad shape {tuple(delta.shape)}"
    assert torch.equal(delta, torch.full_like(delta, float(DIM))), \
        "delta of ones is not the head dim exactly"


def check_edge_shapes(kernel_fn) -> list:
    """seq not a multiple of the 128-token tile, and the generic D != 128 path."""
    cases = [
        ("odd_seq", 1000, HEADS, DIM),
        ("generic_dim", 517, 4, 96),
        ("single_token", 1, 1, DIM),
    ]
    lines = []
    for name, seq, heads, dim in cases:
        out, dout = make_inputs(seed=7, seq=seq, heads=heads, dim=dim)
        ref = reference_preprocess_delta(out, dout)
        got = kernel_fn(out, dout)
        assert tuple(got.shape) == (heads, seq), f"{name}: bad shape {tuple(got.shape)}"
        max_abs = (got - ref).abs().max().item()
        assert max_abs <= ATOL, f"{name}: max_abs={max_abs:.3e} > atol={ATOL:.1e}"
        lines.append(f"{name} [seq={seq},heads={heads},dim={dim}] max_abs={max_abs:.3e}")
    return lines


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------
def check_performance(kernel_fn) -> dict:
    out, dout = make_inputs()
    ours = bench_ms(lambda: kernel_fn(out, dout))
    ref = bench_ms(lambda: reference_preprocess_delta(out, dout))

    bytes_moved = 2 * SEQ * HEADS * DIM * 2       # out + dout, bf16
    bandwidth = bytes_moved / (ours * 1e-3) / 1e12

    result = {
        "ours_ms": ours,
        "ref_ms": ref,
        "trace_ms": TRACE_MS,
        "speedup_vs_ref": ref / ours,
        "ratio_vs_trace": ours / TRACE_MS,
        "bandwidth_tbs": bandwidth,
    }
    if STRICT_PERF:
        assert ours <= TRACE_MS, (
            f"kernel {ours:.4f} ms slower than traced {TRACE_MS:.2f} ms"
        )
        assert ours < ref, f"kernel {ours:.4f} ms not faster than reference {ref:.4f} ms"
    return result


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def test_case25_ones():
    check_ones(load_kernel())


def test_case25_correctness():
    stats = check_correctness(load_kernel())
    assert stats["max_abs"] <= ATOL


def test_case25_edge_shapes():
    check_edge_shapes(load_kernel())


def test_case25_performance():
    res = check_performance(load_kernel())
    assert res["ours_ms"] <= TRACE_MS


# --------------------------------------------------------------------------
# standalone report
# --------------------------------------------------------------------------
def main() -> int:
    if DEVICE != "musa":
        print("MUSA device not available -- correctness only, performance skipped.")
    print(f"device={DEVICE}  bundle={BUNDLE_DIR}")
    kernel_fn = load_kernel()
    if DEVICE == "musa":
        print(f"gpu={torch.musa.get_device_name(0)}  torch={torch.__version__}\n")

    # 1) correctness -------------------------------------------------------
    check_ones(kernel_fn)
    print("correctness =================================================")
    st = check_correctness(kernel_fn)
    print(f"  out/dout      : [{SEQ}, {HEADS}, {DIM}] bfloat16")
    print(f"  delta         : {st['shape']} {st['dtype']} "
          f"(ref |max| = {st['ref_abs_max']:.3f})")
    print(f"  max_abs_err   : {st['max_abs']:.3e}   (atol {ATOL:.0e})")
    print(f"  max_rel_err   : {st['max_rel']:.3e}   (rtol {RTOL:.0e})")
    print(f"  mean_abs_err  : {st['abs_mean']:.3e}")
    print(f"  spot checks   : max_abs={st['spot_max_abs']:.3e}")
    print(f"  ones invariant: delta == {DIM} exactly  [ok]")
    for line in check_edge_shapes(kernel_fn):
        print(f"  edge case     : {line}")
    print("  VERDICT: PASS\n")

    if DEVICE != "musa":
        return 0

    # 2) performance -------------------------------------------------------
    print("performance =================================================")
    perf = check_performance(kernel_fn)
    print(f"  generated kernel : {perf['ours_ms']:.4f} ms")
    print(f"  pytorch reference: {perf['ref_ms']:.4f} ms "
          f"({perf['speedup_vs_ref']:.2f}x faster than it)")
    print(f"  traced (896-948us): {TRACE_MS:.2f} ms")
    print(f"  ours / traced     : {perf['ratio_vs_trace']:.3f}x "
          f"({'faster' if perf['ratio_vs_trace'] < 1 else 'slower'})")
    print(f"  effective BW      : {perf['bandwidth_tbs']:.2f} TB/s "
          f"({2 * SEQ * HEADS * DIM * 2 / 2**30:.2f} GiB moved)")
    ok = perf["ratio_vs_trace"] <= 1.0
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} "
          f"({perf['ours_ms']:.4f} ms vs {TRACE_MS:.2f} ms target)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
