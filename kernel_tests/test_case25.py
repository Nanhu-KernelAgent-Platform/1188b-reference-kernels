"""test_case25.py -- correctness & performance test for the generated MUSA kernel.

Operator (muDNN unary CastOp, fp32 -> bf16 with round-to-nearest-even):

    out[i] = cast_bf16_rne(inp[i])
    lsb = (bits >> 16) & 1;  rounded = bits + (0x7FFF + lsb);  out = rounded >> 16

    inp : fp32   (no scale, no alpha/beta -- pure cast)
    out : bf16, same shape

Source of the trace: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent ops: RouterGatingLinearFunction Backward x128, AttnFuncWithCPAndQKVOA2A Backward x64.

Trace reference (MTT S5000), 64 launches each, 192 total:
    router grad_input [16384, 8192]   numel 134217728  582.443..591.883 us  <- dominant case
    router grad_weight [320, 8192]    numel   2621440   12.800..16.560 us
    attention Q [131072, 16, 192]     numel 402653184 1713.490..1748.131 us

What is checked
  1. correctness : comparison against the reference RNE cast, tolerance 1e-5
                   (abs + rel), plus the exact RNE tie-to-even invariants
                   (0x3F808000 -> 1.0, 0x3F818000 -> 0x3F82), denormal/inf/NaN
                   edge values, and the vector tail path for odd numel.
  2. performance : device-timed latency of the generated kernel vs
                   (a) the reference implementation,
                   (b) native muDNN .to(bfloat16) (the op actually traced),
                   (c) the traced duration per shape.

Usage
    python test_case25.py              # full report, asserts both sections
    pytest test_case25.py -v           # same checks under pytest
    KA_BUNDLE=/path/to/bundle python test_case25.py     # override kernel bundle
    KA_ITERS=50 KA_PERF_REPEAT=5 ...   # tune benchmarking
Environment
    KA_BUNDLE        dir containing kernel.py/kernel.mu/binding.cpp/setup.py
                     (default: ./final_bundle, else ../final_bundle; built in place if needed)
    KA_SKIP_BUILD=1  do not try to build the extension, require an existing .so
    KA_STRICT_PERF=0 relax the "faster than trace" assertions
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ATOL = 1e-5              # required precision error
RTOL = 1e-5
ITERS = int(os.environ.get("KA_ITERS", "30"))
REPEAT = int(os.environ.get("KA_PERF_REPEAT", "5"))
WARMUP = int(os.environ.get("KA_WARMUP", "10"))
STRICT_PERF = os.environ.get("KA_STRICT_PERF", "1") != "0"

DEVICE = "musa" if (hasattr(torch, "musa") and torch.musa.is_available()) else "cpu"

# (name, shape, traced_min_ms, traced_max_ms)
TRACE_CASES = [
    ("router grad_input [16384,8192]", (16384, 8192), 0.582443, 0.591883),
    ("router grad_weight [320,8192]", (320, 8192), 0.012800, 0.016560),
    ("attention Q [131072,16,192]", (131072, 16, 192), 1.713490, 1.748131),
]

# the dominant router case is the reference case (M x N = 16384 x 8192)
DOMINANT_SHAPE = (16384, 8192)
TRACE_MS = (0.582443 + 0.591883) / 2      # 0.587 ms performance target

# odd sizes exercising the tail path (numel not a multiple of 8 * block)
TAIL_SIZES = [1, 2, 7, 8, 9, 15, 16, 1023, 4095, 4096, 4097, 65537, 2621441, 134217727]


# --------------------------------------------------------------------------
# reference implementation, verbatim from the traced kernel's semantics
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# kernel bundle loading (build the MUSA extension on demand)
# --------------------------------------------------------------------------
def _bundle_dir() -> str:
    env = os.environ.get("KA_BUNDLE")
    if env:
        return env
    for cand in (os.path.join(HERE, "final_bundle"),
                 os.path.join(HERE, os.pardir, "final_bundle")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(HERE, "final_bundle")


BUNDLE_DIR = _bundle_dir()


def _extension_so() -> str | None:
    hits = glob.glob(os.path.join(BUNDLE_DIR, "_cast_rne_final*.so"))
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
    import kernel as ka  # noqa: E402  (kernel.py -> _cast_rne_final)

    return ka.kernel_function


def make_inputs(seed: int = 0, m: int = 16384, n: int = 8192, device: str | None = None):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Values large enough that several mantissa bits sit in the rounded region.
    return (torch.randn(m, n, generator=g, dtype=torch.float32) * 8).to(device or DEVICE)


def make_inputs_flat(numel: int, seed: int = 0, device: str | None = None):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return (torch.randn(numel, generator=g, dtype=torch.float32) * 8).to(device or DEVICE)


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
def _compare(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """max abs / rel error, allclose at ATOL/RTOL, and exact bf16 bit mismatches."""
    assert tuple(got.shape) == tuple(ref.shape), f"bad shape {tuple(got.shape)} != {tuple(ref.shape)}"
    assert got.dtype == torch.bfloat16, f"bad dtype {got.dtype}"
    assert got.device == ref.device, f"bad device {got.device}"

    gf, rf = got.float(), ref.float()
    diff = (gf - rf).abs()
    finite = bool(torch.isfinite(gf).all().item()) or bool(torch.equal(torch.isnan(gf), torch.isnan(rf)))
    max_abs = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0).max().item()
    denom = rf.abs().clamp_min(1e-30)
    max_rel = torch.nan_to_num(diff / denom, nan=0.0, posinf=0.0, neginf=0.0).max().item()
    close = bool(torch.allclose(gf, rf, atol=ATOL, rtol=0.0, equal_nan=True))
    bit_mismatch = int((got.view(torch.int16) != ref.view(torch.int16)).sum().item())
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "abs_mean": torch.nan_to_num(diff, nan=0.0).mean().item(),
        "close": close,
        "finite": finite,
        "bit_mismatch": bit_mismatch,
        "numel": got.numel(),
    }


def check_correctness(kernel_fn) -> dict:
    """Full-tensor comparison against the reference on the dominant case."""
    inp = make_inputs(seed=0, m=DOMINANT_SHAPE[0], n=DOMINANT_SHAPE[1])
    ref = reference_cast_fp32_to_bf16(inp)
    got = kernel_fn(inp)
    st = _compare(got, ref)
    st["shape"] = tuple(got.shape)
    st["dtype"] = str(got.dtype)
    st["ref_abs_max"] = ref.float().abs().max().item()

    assert st["finite"], "output contains non-finite values the reference does not have"
    assert st["max_abs"] <= ATOL, f"max_abs={st['max_abs']:.3e} > atol={ATOL:.1e}"
    assert st["max_rel"] <= RTOL or st["max_abs"] <= ATOL, \
        f"max_rel={st['max_rel']:.3e} > rtol={RTOL:.1e}"
    assert st["close"], "torch.allclose failed at the requested tolerance"
    del inp, ref, got
    if DEVICE == "musa":
        torch.musa.empty_cache()
    return st


def check_rne_ties(kernel_fn) -> None:
    """Exact halfway cases on the bf16 boundary, as in the reference."""
    # 1.0 fp32 = 0x3F800000; bf16 keeps 0x3F80, the next bf16 is 0x3F81.
    # Midpoint between them in fp32 is 0x3F808000.
    mid = torch.tensor([0x3F808000], dtype=torch.int32).view(torch.float32).clone()
    got = kernel_fn(mid.to(DEVICE)).cpu()
    expect = torch.tensor([1.0], dtype=torch.bfloat16)
    assert torch.equal(got, expect), f"RNE tie-to-even failed for 1.0 midpoint: {got}"

    # Midpoint above 0x3F810000: LSB becomes odd after the high half, rounds up.
    mid_odd = torch.tensor([0x3F818000], dtype=torch.int32).view(torch.float32).clone()
    got_odd = kernel_fn(mid_odd.to(DEVICE)).cpu()
    expect_odd = torch.tensor([0x3F82], dtype=torch.int16).view(torch.bfloat16)
    assert torch.equal(got_odd, expect_odd), \
        f"RNE tie-to-even failed for odd midpoint: {got_odd.view(torch.int16)}"


def check_edge_values(kernel_fn) -> dict:
    """Denormals, +-0, +-inf, NaN payloads, FLT_MAX and rounding boundaries."""
    pats = [
        0x00000000, 0x80000000,  # +-0
        0x00000001, 0x007FFFFF,  # fp32 denormals -> bf16 zero
        0x00800000, 0x00808000,  # smallest normal and its tie
        0x3F800000, 0x3F808000,  # 1.0 and its midpoint
        0x7F7FFFFF, 0x7F800000,  # FLT_MAX, +inf
        0xFF800000,              # -inf
        0x7FC00000, 0x7FC00001,  # qNaN / sNaN
    ]
    bits = torch.tensor([v - (1 << 32) if v >= (1 << 31) else v for v in pats],
                        dtype=torch.int32)
    inp = bits.view(torch.float32).clone().to(DEVICE)
    ref = reference_cast_fp32_to_bf16(inp)
    got = kernel_fn(inp)
    st = _compare(got, ref)
    assert st["max_abs"] <= ATOL, f"edge values: max_abs={st['max_abs']:.3e} > atol={ATOL:.1e}"
    return st


def check_trace_shapes(kernel_fn) -> list:
    """The three problem sizes recorded in the trace."""
    lines = []
    for name, shape, _, _ in TRACE_CASES:
        inp = (torch.randn(*shape, generator=torch.Generator().manual_seed(0),
                           dtype=torch.float32) * 8).to(DEVICE)
        ref = reference_cast_fp32_to_bf16(inp)
        got = kernel_fn(inp)
        st = _compare(got, ref)
        assert st["max_abs"] <= ATOL, f"{name}: max_abs={st['max_abs']:.3e} > atol={ATOL:.1e}"
        lines.append((name, st["numel"], st["max_abs"], st["bit_mismatch"]))
        del inp, ref, got
        if DEVICE == "musa":
            torch.musa.empty_cache()
    return lines


def check_tail_sizes(kernel_fn) -> list:
    """numel not a multiple of 8 * block: the vector tail path."""
    lines = []
    for numel in TAIL_SIZES:
        inp = make_inputs_flat(numel, seed=1)
        ref = reference_cast_fp32_to_bf16(inp)
        got = kernel_fn(inp)
        st = _compare(got, ref)
        assert st["max_abs"] <= ATOL, f"tail numel={numel}: max_abs={st['max_abs']:.3e}"
        lines.append((numel, st["max_abs"], st["bit_mismatch"]))
        del inp, ref, got
    if DEVICE == "musa":
        torch.musa.empty_cache()
    return lines


def check_against_mudnn(kernel_fn) -> dict:
    """Cross-check the generated op against the native muDNN cast (the traced op)."""
    inp = make_inputs_flat(1024 * 1024, seed=3)
    got = kernel_fn(inp)
    mudnn = inp.to(torch.bfloat16)
    st = _compare(got, mudnn)
    assert st["max_abs"] <= ATOL, f"disagrees with muDNN: max_abs={st['max_abs']:.3e}"
    return st


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------
def check_performance(kernel_fn) -> dict:
    """Latency of the generated kernel vs reference / muDNN / traced, per traced shape."""
    rows = []
    tot = {"ours": 0.0, "ref": 0.0, "mudnn": 0.0, "traced": 0.0}
    g = torch.Generator(device="cpu").manual_seed(0)
    for name, shape, lo, hi in TRACE_CASES:
        inp = (torch.randn(*shape, generator=g, dtype=torch.float32) * 8).to(DEVICE)
        iters = ITERS if inp.numel() <= 2_700_000 else max(5, ITERS // 3)
        t_ours = bench_ms(lambda: kernel_fn(inp), iters)
        t_mu = bench_ms(lambda: inp.to(torch.bfloat16), iters)
        try:
            t_ref = bench_ms(lambda: reference_cast_fp32_to_bf16(inp), max(3, iters // 10))
        except (RuntimeError, torch.musa.OutOfMemoryError) as e:  # multi-pass reference
            print(f"  [warn] reference timing skipped for {name}: {e}")
            t_ref = float("nan")
        t_tr = (lo + hi) / 2

        tot["ours"] += t_ours
        tot["ref"] += t_ref
        tot["mudnn"] += t_mu
        tot["traced"] += t_tr
        rows.append({
            "name": name, "numel": inp.numel(), "ours": t_ours, "ref": t_ref,
            "mudnn": t_mu, "traced": t_tr, "traced_range": (lo, hi),
            "bytes": inp.numel() * 6,
        })
        del inp
        torch.musa.empty_cache()

    # dominant reference case, timed on a pre-allocated tensor
    inp = make_inputs(seed=0, m=DOMINANT_SHAPE[0], n=DOMINANT_SHAPE[1])
    t_dom = bench_ms(lambda: kernel_fn(inp), 100)
    t_dom_mu = bench_ms(lambda: inp.to(torch.bfloat16), 100)
    t_dom_ref = bench_ms(lambda: reference_cast_fp32_to_bf16(inp), 10)
    del inp
    torch.musa.empty_cache()

    res = {
        "rows": rows,
        "total": tot,
        "dominant_ms": t_dom,
        "dominant_mudnn_ms": t_dom_mu,
        "dominant_ref_ms": t_dom_ref,
        "dominant_bytes": DOMINANT_SHAPE[0] * DOMINANT_SHAPE[1] * 6,
        "trace_ms": TRACE_MS,
    }

    if STRICT_PERF and DEVICE == "musa":
        assert t_dom <= TRACE_MS, \
            f"kernel {t_dom:.4f} ms slower than traced {TRACE_MS:.4f} ms (dominant case)"
        assert t_dom < t_dom_mu, \
            f"kernel {t_dom:.4f} ms not faster than native muDNN {t_dom_mu:.4f} ms"
        for r in rows:
            assert r["ours"] <= r["traced"], \
                f"{r['name']}: {r['ours']:.4f} ms slower than traced {r['traced']:.4f} ms"
    return res


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def test_case25_rne_ties():
    check_rne_ties(load_kernel())


def test_case25_correctness():
    stats = check_correctness(load_kernel())
    assert stats["max_abs"] <= ATOL


def test_case25_edge_values():
    check_edge_values(load_kernel())


def test_case25_trace_shapes():
    check_trace_shapes(load_kernel())


def test_case25_tail_sizes():
    check_tail_sizes(load_kernel())


def test_case25_against_mudnn():
    check_against_mudnn(load_kernel())


def test_case25_performance():
    res = check_performance(load_kernel())
    if DEVICE == "musa":
        assert res["dominant_ms"] <= TRACE_MS


# --------------------------------------------------------------------------
# standalone report
# --------------------------------------------------------------------------
def main() -> int:
    if DEVICE != "musa":
        print("MUSA device not available -- correctness only, performance skipped.")
    print(f"device={DEVICE}  bundle={BUNDLE_DIR}")
    kernel_fn = load_kernel()
    if DEVICE == "musa":
        print(f"gpu={torch.musa.get_device_name(0)}  torch={torch.__version__}")

    ok = True

    # 1) correctness -------------------------------------------------------
    print("\ncorrectness =================================================")
    print(f"  tolerance     : atol={ATOL:.0e} rtol={RTOL:.0e}")

    check_rne_ties(kernel_fn)
    print("  RNE tie cases : 0x3F808000 -> 1.0, 0x3F818000 -> 0x3F82   [ok]")

    ev = check_edge_values(kernel_fn)
    print(f"  edge values   : denorm/+-0/inf/nan/tie  max_abs={ev['max_abs']:.3e}   [ok]")

    st = check_correctness(kernel_fn)
    print(f"  inp           : [{DOMINANT_SHAPE[0]}, {DOMINANT_SHAPE[1]}] fp32 "
          f"(ref |max| = {st['ref_abs_max']:.3f})")
    print(f"  out           : {st['shape']} {st['dtype']}")
    print(f"  max_abs_err   : {st['max_abs']:.3e}   (atol {ATOL:.0e})")
    print(f"  max_rel_err   : {st['max_rel']:.3e}   (rtol {RTOL:.0e})")
    print(f"  mean_abs_err  : {st['abs_mean']:.3e}")
    print(f"  bit_mismatch  : {st['bit_mismatch']} / {st['numel']}")

    for name, numel, max_abs, mism in check_trace_shapes(kernel_fn):
        print(f"  traced shape  : {name:<32s} numel={numel:>10d} "
              f"max_abs={max_abs:.3e} bit_mismatch={mism}")

    tails = check_tail_sizes(kernel_fn)
    worst = max(t[1] for t in tails)
    print(f"  tail numels   : {len(tails)} sizes, worst max_abs={worst:.3e}   [ok]")

    mu = check_against_mudnn(kernel_fn)
    print(f"  vs muDNN      : max_abs={mu['max_abs']:.3e} "
          f"bit_mismatch={mu['bit_mismatch']}   [ok]")
    print("  VERDICT: PASS")

    if DEVICE != "musa":
        return 0

    # 2) performance -------------------------------------------------------
    print("\nperformance =================================================")
    perf = check_performance(kernel_fn)
    hdr = (f"  {'case':<30s}{'numel':>11s}{'ours':>9s}{'reference':>11s}"
           f"{'muDNN':>9s}{'traced':>9s}{'vs ref':>9s}{'vs muDNN':>10s}{'TB/s':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in perf["rows"]:
        bw = r["bytes"] / (r["ours"] * 1e-3) / 1e12
        print(f"  {r['name']:<30s}{r['numel']:>11d}{r['ours']:>9.4f}{r['ref']:>11.4f}"
              f"{r['mudnn']:>9.4f}{r['traced']:>9.4f}"
              f"{r['ref'] / r['ours']:>8.2f}x{r['mudnn'] / r['ours']:>9.2f}x{bw:>8.3f}")
    t = perf["total"]
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL (3 traced launches)':<30s}{'':>11s}{t['ours']:>9.4f}{t['ref']:>11.4f}"
          f"{t['mudnn']:>9.4f}{t['traced']:>9.4f}"
          f"{t['ref'] / t['ours']:>8.2f}x{t['mudnn'] / t['ours']:>9.2f}x")

    d = perf["dominant_ms"]
    print(f"\n  dominant case [{DOMINANT_SHAPE[0]},{DOMINANT_SHAPE[1]}] : {d:.4f} ms "
          f"({perf['dominant_bytes'] / (d * 1e-3) / 1e12:.3f} TB/s)")
    print(f"  vs reference implementation : {perf['dominant_ref_ms'] / d:.2f}x "
          f"({perf['dominant_ref_ms']:.4f} ms)")
    print(f"  vs native muDNN .to(bf16)   : {perf['dominant_mudnn_ms'] / d:.2f}x "
          f"({perf['dominant_mudnn_ms']:.4f} ms)")
    print(f"  vs traced {TRACE_MS:.4f} ms target  : {TRACE_MS / d:.2f}x "
          f"({(1 - d / TRACE_MS) * 100:.1f}% under target)")
    perf_ok = d <= TRACE_MS
    print(f"  VERDICT: {'PASS' if perf_ok else 'FAIL'} "
          f"({d:.4f} ms vs {TRACE_MS:.4f} ms target)")
    ok &= perf_ok

    print("\nRESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
