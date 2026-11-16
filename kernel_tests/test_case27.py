"""test_case25.py -- correctness & performance test for the generated MUSA RMSNorm kernel.

Operator (reference case, MTT S5000 trace x1w_1188B_128k_ac-full_rank-8.json, rank 8):

    musa::dnn::LayerNormGlobalKernelVlen<
        __mt_bfloat16, float, __mt_bfloat16, float, 128, 4, 8,
        (musa::dnn::NormType)1, false>            # NormType 1 == RMSNorm

        rstd[m] = rsqrt(mean_i(x[m, i]^2) + eps)
        y[m, i] = x[m, i] * rstd[m] * gamma[i]

    x, gamma : bfloat16      accumulation / rstd : float32      y : bfloat16
    eps = 1e-6 (--norm-epsilon), zero_centered_gamma = false -> gamma is applied
    as stored.  No mean centering and no beta.

    M = 131072 / 8 (CP=8) = 16384 rows, N = 8192 hidden elements.

What is checked
  1. correctness : every output compared against the reference implementation
                   (chunked fp32 reference, ROW_CHUNK = 1024) with
                   atol = rtol = 1e-5 on the float32 outputs (y_fp32, rstd_fp32).
                   The bfloat16 store cannot meet 1e-5 -- its quantization floor
                   is 2^-8 relative -- so it is asserted at bfloat16 precision and
                   additionally reported in bf16 ULP against the reference store.
  2. closed form : the two analytic identities from the reference case
                   (constant row -> rstd = 1, y = gamma; x = [3, 4] -> rstd = 1/sqrt(12.5)).
  3. performance : device-timed latency vs
                   (a) the traced launch, 425.8 .. 480.1 us (median ~445 us),
                   (b) the PyTorch reference implementation.

Two kernels are exposed by the bundle and both are tested:
    kernel_function       -> (y_fp32, y_bf16, rstd_fp32)  reference signature
    kernel_function_lean  -> (y_bf16, rstd_fp32)         what the traced kernel stores
The traced 445 us kernel writes only y_bf16 + rstd_fp32 (512 MiB of traffic); the
reference signature additionally stores a 512 MiB fp32 copy of y, so only the lean
kernel is compared head-to-head against the trace latency, and the full kernel is
compared on achieved bandwidth instead.

Usage
    python test_case25.py                      # full report, asserts both sections
    pytest test_case25.py -v                   # same checks under pytest
    KA_BUNDLE=/path/to/bundle python test_case25.py
    KA_ITERS=50 KA_PERF_REPEAT=5 python test_case25.py
    KA_TRACE_MS=0.445 python test_case25.py    # override the traced target

Environment
    KA_BUNDLE        dir with kernel.py / kernel.mu / binding.cpp / setup.py
                     (default: ./rmsnorm_bundle, built in place if needed)
    KA_SKIP_BUILD=1  do not build the extension, require an existing .so
    KA_STRICT_PERF=0 report performance without asserting it
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "rmsnorm_bundle")
BUNDLE_DIR = os.environ.get("KA_BUNDLE", DEFAULT_BUNDLE)

M = 16384            # tokens per rank (131072 / CP=8)
N = 8192             # hidden size of this specialization
ROW_CHUNK = 1024     # chunking used by the reference implementation
EPS = 1e-6           # --norm-epsilon

TRACE_MS = float(os.environ.get("KA_TRACE_MS", "0.445"))   # 425.842 .. 480.083 us
ATOL = 1e-5          # required precision error (float32 outputs)
RTOL = 1e-5
BF16_RTOL = 1.6e-2   # torch bfloat16 defaults (what the engine verifies with)
BF16_ATOL = 1e-2

ITERS = int(os.environ.get("KA_ITERS", "50"))
REPEAT = int(os.environ.get("KA_PERF_REPEAT", "5"))
WARMUP = int(os.environ.get("KA_WARMUP", "10"))
STRICT_PERF = os.environ.get("KA_STRICT_PERF", "1") != "0"

DEVICE = "musa" if (hasattr(torch, "musa") and torch.musa.is_available()) else "cpu"


# --------------------------------------------------------------------------
# reference implementation, verbatim from the reference case
# --------------------------------------------------------------------------
def reference_rmsnorm_fwd(x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS):
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


def make_inputs(seed: int = 0, m: int = M, n: int = N, device: str | None = None):
    device = device or DEVICE
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.bfloat16)
    gamma = torch.randn(n, generator=g, dtype=torch.bfloat16)
    return x.to(device), gamma.to(device)


# --------------------------------------------------------------------------
# kernel bundle loading (build the MUSA extension on demand)
# --------------------------------------------------------------------------
def _extension_so() -> str | None:
    hits = glob.glob(os.path.join(BUNDLE_DIR, "rmsnorm_bundle_ext*.so"))
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
    """Import the bundle and return (kernel_function, kernel_function_lean)."""
    bundle = ensure_bundle()
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    import kernel as ka  # noqa: E402  (kernel.py -> rmsnorm_bundle_ext)

    lean = getattr(ka, "kernel_function_lean", None)
    return ka.kernel_function, lean


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
        if DEVICE == "musa":
            start, end = torch.musa.Event(True), torch.musa.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            _sync()
            ms = start.elapsed_time(end) / iters
        else:                                     # pragma: no cover
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ms = (time.perf_counter() - t0) * 1e3 / iters
        best = min(best, ms)
    return best


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def _stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-30)
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_abs": diff.mean().item(),
    }


def _bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    ia = a.view(torch.int16).to(torch.int32).to("cpu")
    ib = b.view(torch.int16).to(torch.int32).to("cpu")
    return int((ia - ib).abs().max().item())


def _check_fp32(got: torch.Tensor, ref: torch.Tensor, name: str) -> dict:
    st = _stats(got, ref)
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)
    st["name"] = name
    return st


def _check_bf16(got: torch.Tensor, ref: torch.Tensor, name: str) -> dict:
    """bf16 store: asserted at bf16 precision, reported in ULP against the reference."""
    torch.testing.assert_close(got, ref, rtol=BF16_RTOL, atol=BF16_ATOL)
    st = _stats(got, ref)
    st["max_ulp"] = _bf16_ulp(got, ref)
    st["name"] = name
    return st


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def check_correctness(kernel_fn, lean_fn=None) -> dict:
    x, gamma = make_inputs()
    y_acc_ref, y_ref, rstd_ref = reference_rmsnorm_fwd(x, gamma, EPS)

    out = kernel_fn(x, gamma, EPS)
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        f"kernel_function must return (y_fp32, y_bf16, rstd), got {type(out)}"
    y_acc, y_bf, rstd = out
    assert tuple(y_acc.shape) == (M, N) and y_acc.dtype == torch.float32, "bad y_fp32"
    assert tuple(y_bf.shape) == (M, N) and y_bf.dtype == torch.bfloat16, "bad y_bf16"
    assert tuple(rstd.shape) == (M,) and rstd.dtype == torch.float32, "bad rstd"
    assert y_acc.device == x.device, "bad device"

    full = [
        _check_fp32(y_acc, y_acc_ref, "y_fp32"),
        _check_bf16(y_bf, y_ref, "y_bf16"),
        _check_fp32(rstd, rstd_ref, "rstd_fp32"),
    ]

    report = {"full": full, "lean": []}
    if lean_fn is not None:
        y_bf2, rstd2 = lean_fn(x, gamma, EPS)
        assert y_bf2.dtype == torch.bfloat16 and rstd2.dtype == torch.float32, "bad lean dtypes"
        report["lean"] = [
            _check_bf16(y_bf2, y_ref, "lean.y_bf16"),
            _check_fp32(rstd2, rstd_ref, "lean.rstd_fp32"),
        ]

        finite = torch.isfinite(y_acc).all().item() and torch.isfinite(rstd).all().item()
        assert finite, "non-finite values in the output"
    return report


def check_closed_form(kernel_fn) -> list:
    """The two analytic identities of the reference case."""
    lines = []

    # constant row x = 1, gamma = 2, eps = 0  ->  rstd = 1, y = 2
    x = torch.ones(2, 4, dtype=torch.bfloat16, device=DEVICE)
    gamma = torch.full((4,), 2, dtype=torch.bfloat16, device=DEVICE)
    y_acc, y_bf, rstd = kernel_fn(x, gamma, 0.0)
    torch.testing.assert_close(rstd, torch.ones(2, device=DEVICE), rtol=0, atol=0)
    torch.testing.assert_close(y_acc, torch.full((2, 4), 2.0, device=DEVICE), rtol=0, atol=0)
    assert torch.equal(y_bf, y_acc.to(torch.bfloat16)), "bf16 store disagrees with the fp32 y"
    lines.append("constant row (x=1, gamma=2, eps=0): rstd=1, y=2 exactly")

    # x = [3, 4] -> mean(x^2) = 12.5, rstd = 1/sqrt(12.5), y = x * rstd * gamma
    x = torch.tensor([[3, 4]], dtype=torch.bfloat16, device=DEVICE)
    gamma = torch.tensor([1, 2], dtype=torch.bfloat16, device=DEVICE)
    y_acc, y_bf, rstd = kernel_fn(x, gamma, 0.0)
    expect_rstd = torch.rsqrt(torch.tensor([12.5], device=DEVICE))
    expect_y = x.float() * expect_rstd[:, None] * gamma.float()
    torch.testing.assert_close(rstd, expect_rstd, rtol=ATOL, atol=ATOL)
    torch.testing.assert_close(y_acc, expect_y, rtol=RTOL, atol=ATOL)
    lines.append(f"x=[3,4] -> rstd=1/sqrt(12.5) (got {rstd.item():.8f})")
    return lines


def check_edge_shapes(kernel_fn) -> list:
    """Odd row counts plus the generic N != 8192 path."""
    lines = []
    for m, n in ((1, 8192), (37, 8192), (517, 1024), (3, 96), (1024, 4096)):
        x, gamma = make_inputs(seed=11, m=m, n=n)
        y_acc_ref, y_ref, rstd_ref = reference_rmsnorm_fwd(x, gamma, EPS)
        y_acc, y_bf, rstd = kernel_fn(x, gamma, EPS)
        torch.testing.assert_close(y_acc, y_acc_ref, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(rstd, rstd_ref, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(y_bf, y_ref, rtol=BF16_RTOL, atol=BF16_ATOL)
        lines.append(f"m={m:5d} n={n:4d}: max_abs(y_fp32)={_stats(y_acc, y_acc_ref)['max_abs']:.3e}"
                     f" max_abs(rstd)={_stats(rstd, rstd_ref)['max_abs']:.3e}")
    return lines


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------
def check_performance(kernel_fn, lean_fn=None) -> dict:
    x, gamma = make_inputs()

    full_ms = bench_ms(lambda: kernel_fn(x, gamma, EPS))
    ref_ms = bench_ms(lambda: reference_rmsnorm_fwd(x, gamma, EPS))
    lean_ms = bench_ms(lambda: lean_fn(x, gamma, EPS)) if lean_fn is not None else None

    traffic_lean = 2 * M * N * 2 + M * 4          # read x + write y_bf16 + rstd
    traffic_full = traffic_lean + M * N * 4       # + the reference's fp32 y copy

    res = {
        "full_ms": full_ms,
        "lean_ms": lean_ms,
        "ref_ms": ref_ms,
        "trace_ms": TRACE_MS,
        "lean_vs_trace": (lean_ms / TRACE_MS) if lean_ms else None,
        "full_vs_ref": ref_ms / full_ms,
        "bw_lean_tbs": traffic_lean / (lean_ms * 1e-3) / 1e12 if lean_ms else None,
        "bw_full_tbs": traffic_full / (full_ms * 1e-3) / 1e12,
        "traffic_lean_mb": traffic_lean / 1e6,
        "traffic_full_mb": traffic_full / 1e6,
    }
    # The PyTorch reference is never supposed to win against the fused kernel.
    assert full_ms < ref_ms, f"kernel {full_ms:.4f} ms is not faster than the reference {ref_ms:.4f} ms"
    if STRICT_PERF and lean_ms is not None:
        assert lean_ms <= TRACE_MS, (
            f"lean kernel {lean_ms:.4f} ms is slower than the traced {TRACE_MS:.4f} ms"
        )
    return res


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def test_case25_closed_form():
    kernel_fn, _ = load_kernel()
    check_closed_form(kernel_fn)


def test_case25_correctness():
    kernel_fn, lean_fn = load_kernel()
    check_correctness(kernel_fn, lean_fn)


def test_case25_edge_shapes():
    kernel_fn, _ = load_kernel()
    check_edge_shapes(kernel_fn)


def test_case25_performance():
    kernel_fn, lean_fn = load_kernel()
    res = check_performance(kernel_fn, lean_fn)
    if res["lean_ms"] is not None:
        assert res["lean_ms"] <= TRACE_MS


# --------------------------------------------------------------------------
# standalone report
# --------------------------------------------------------------------------
def main() -> int:
    print(f"device={DEVICE}  bundle={BUNDLE_DIR}")
    kernel_fn, lean_fn = load_kernel()
    if DEVICE == "musa":
        print(f"gpu={torch.musa.get_device_name(0)}  torch={torch.__version__}")
    print(f"case: M={M} N={N} eps={EPS} x/gamma=bfloat16  y=bfloat16  rstd=float32\n")

    # 1) correctness -------------------------------------------------------
    print("correctness (atol=rtol=1e-5 on float32 outputs) =============")
    for line in check_closed_form(kernel_fn):
        print(f"  closed form   : {line}")
    rep = check_correctness(kernel_fn, lean_fn)
    for st in rep["full"]:
        print(f"  {st['name']:<14s} max_abs={st['max_abs']:.3e}  max_rel={st['max_rel']:.3e}"
              f"  mean_abs={st['mean_abs']:.3e}"
              + (f"  max_ulp={st['max_ulp']}" if "max_ulp" in st else ""))
    for st in rep["lean"]:
        print(f"  {st['name']:<14s} max_abs={st['max_abs']:.3e}  max_rel={st['max_rel']:.3e}"
              f"  mean_abs={st['mean_abs']:.3e}"
              + (f"  max_ulp={st['max_ulp']}" if "max_ulp" in st else ""))
    for line in check_edge_shapes(kernel_fn):
        print(f"  edge case     : {line}")
    print("  note: the bfloat16 store is asserted at bfloat16 precision (rtol "
          f"{BF16_RTOL:.0e}); 1e-5 cannot apply to an 8-bit-mantissa output.")
    print("  VERDICT: PASS\n")

    if DEVICE != "musa":
        print("MUSA device not available -- performance section skipped.")
        return 0

    # 2) performance -------------------------------------------------------
    print("performance =================================================")
    perf = check_performance(kernel_fn, lean_fn)
    print(f"  pytorch reference      : {perf['ref_ms']:.4f} ms")
    print(f"  full  kernel (3 outs)   : {perf['full_ms']:.4f} ms  "
          f"({perf['traffic_full_mb']:.0f} MB -> {perf['bw_full_tbs']:.2f} TB/s, "
          f"{perf['full_vs_ref']:.2f}x vs reference)")
    if perf["lean_ms"] is not None:
        print(f"  lean  kernel (2 outs)   : {perf['lean_ms']:.4f} ms  "
              f"({perf['traffic_lean_mb']:.0f} MB -> {perf['bw_lean_tbs']:.2f} TB/s)")
        print(f"  traced (425.8-480.1 us) : {perf['trace_ms']:.4f} ms")
        print(f"  lean / traced           : {perf['lean_vs_trace']:.3f}x "
              f"({'faster' if perf['lean_vs_trace'] < 1 else 'slower'})")
        ok = perf["lean_vs_trace"] <= 1.0 or not STRICT_PERF
    else:
        print(f"  traced (425.8-480.1 us) : {perf['trace_ms']:.4f} ms")
        ok = perf["full_ms"] <= perf["trace_ms"] or not STRICT_PERF
    print("  note: the traced kernel stores only y_bf16 + rstd (same IO as the lean kernel);"
          "\n        the reference signature adds a 512 MB fp32 copy of y.")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
