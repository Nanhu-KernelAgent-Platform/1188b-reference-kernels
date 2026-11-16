"""test_case25.py — correctness + performance test of the generated MUSA topk op
against the reference implementation.

Reference contract (from the traced router case):
    values[t, :], indices[t, :] = topk(input[t, :], k=8, largest=True, sorted=True)
    input fp32 [16384, 320]  ->  values fp32 [16384, 8], indices int64 [16384, 8]

Correctness
    golden  : reference_topk() == stable descending sort, take the first k
    tolerance: |values - golden| <= ATOL(=1e-5) + RTOL(=1e-5) * |golden|
    indices : must match exactly (stable tie-break: equal value -> smaller index)

Performance
    baseline: torch.topk(dim=1, largest=True, sorted=True) — the aten::topk /
              muDNN bitonic kernel recorded at 1482..1623 us in the trace
    candidate: kernel_function() from the generated bundle
    reported: median wall time of N timed iterations (MUSA events, warmup first)

Usage
    python3 test_case25.py                # default shape 16384 x 320, k=8
    python3 test_case25.py --iters 200 --shape 16384x320 --k 8
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

sys.path.insert(0, "/root/zf_bench/topk_final")
from kernel import kernel_function  # noqa: E402

# ----------------------------------------------------------------------------
# reference implementation (identical to the traced reference case)
# ----------------------------------------------------------------------------
M = 16384
NUM_EXPERTS = 320
TOPK = 8

ATOL = 1e-5
RTOL = 1e-5

DEVICE = "musa"


def reference_topk(inp: torch.Tensor, k: int = TOPK) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise top-k by a full stable descending sort, then take the first k."""
    if inp.ndim != 2:
        raise ValueError(f"expected 2D input, got {tuple(inp.shape)}")
    if not (1 <= k <= inp.shape[1]):
        raise ValueError(f"k={k} out of range for N={inp.shape[1]}")
    values, indices = torch.sort(inp, dim=1, descending=True, stable=True)
    return values[:, :k].contiguous(), indices[:, :k].contiguous()


def reference_torch_topk(inp: torch.Tensor, k: int = TOPK):
    """The aten::topk launch the reference trace attributes to muDNN bitonic."""
    return torch.topk(inp, k=k, dim=1, largest=True, sorted=True)


def make_inputs(seed: int = 0, device: str = DEVICE, m: int = M, n: int = NUM_EXPERTS,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    inp = torch.randn(m, n, generator=g, dtype=torch.float32)
    return inp.to(device=device, dtype=dtype)


# ----------------------------------------------------------------------------
# correctness
# ----------------------------------------------------------------------------
def check_case(inp: torch.Tensor, k: int, tag: str) -> bool:
    """Compare kernel_function() against reference_topk() with atol/rtol = 1e-5."""
    golden_v, golden_i = reference_topk(inp.cpu(), k)
    got_v, got_i = kernel_function(inp, k)

    if torch.musa.is_available():
        torch.musa.synchronize()

    problems: list[str] = []
    if tuple(got_v.shape) != tuple(golden_v.shape):
        problems.append(f"values shape {tuple(got_v.shape)} != {tuple(golden_v.shape)}")
    if tuple(got_i.shape) != tuple(golden_i.shape):
        problems.append(f"indices shape {tuple(got_i.shape)} != {tuple(golden_i.shape)}")
    if got_v.dtype != torch.float32:
        problems.append(f"values dtype {got_v.dtype} != float32")
    if got_i.dtype != torch.int64:
        problems.append(f"indices dtype {got_i.dtype} != int64")
    if got_v.device != inp.device:
        problems.append(f"values device {got_v.device} != {inp.device}")

    max_abs = max_rel = float("nan")
    idx_mismatch = -1
    if not problems:
        gv, gi = golden_v.to(inp.device), golden_i.to(inp.device)
        diff = (got_v - gv).abs()
        max_abs = diff.max().item()
        max_rel = (diff / gv.abs().clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        idx_mismatch = int((got_i != gi).sum().item())
        if max_abs > ATOL and max_rel > RTOL:
            problems.append(f"values out of tolerance (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})")
        if idx_mismatch != 0:
            problems.append(f"{idx_mismatch} mismatching indices")

    ok = not problems
    print(
        f"[{'PASS' if ok else 'FAIL'}] {tag:<38} shape={tuple(inp.shape)} k={k:<2} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx_mismatch={idx_mismatch}"
    )
    for p in problems:
        print(f"        -> {p}")
    return ok


def correctness_suite(seed: int, m: int, n: int, k: int) -> bool:
    print("=" * 104)
    print("CORRECTNESS  (reference = stable descending sort, tol atol=rtol=1e-5, indices exact)")
    print("=" * 104)
    ok = True

    # 1) the traced reference case (and a couple of extra seeds)
    for s in (seed, seed + 1, seed + 7):
        ok &= check_case(make_inputs(seed=s, m=m, n=n), k, f"traced case seed {s}")

    # 2) the reference file's small hand-checkable case
    small = torch.tensor([[0.1, 0.5, 0.3, 0.2], [4.0, 1.0, 3.0, 2.0]],
                         dtype=torch.float32, device=DEVICE)
    ok &= check_case(small, 2, "reference small case")

    # 3) other shapes and k values
    for mm, nn, kk in [(1, n, k), (17, n, k), (1000, n, k), (m, n, 1), (m, n, 4),
                       (256, 64, 8), (128, 33, 8), (64, 16, min(3, 16))]:
        ok &= check_case(make_inputs(seed=mm + nn, m=mm, n=nn), kk, f"shape {mm}x{nn}")

    # 4) tie-heavy input: worst case for the stable index tie-break
    g = torch.Generator(device="cpu").manual_seed(3)
    ties = torch.randint(0, 5, (4096, n), generator=g).float().to(DEVICE)
    ok &= check_case(ties, k, "heavy ties randint(0,5)")

    # 5) degenerate value ranges
    ok &= check_case(torch.zeros(64, n, device=DEVICE), k, "all zeros (full tie)")
    neg = -torch.rand(512, n, device=DEVICE)
    ok &= check_case(neg, k, "all negative")
    inf = torch.randn(64, n, device=DEVICE)
    inf[:, :40] = float("-inf")
    ok &= check_case(inf, k, "with -inf columns")
    return ok


# ----------------------------------------------------------------------------
# performance
# ----------------------------------------------------------------------------
def _time_ms(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.musa.synchronize()
    return start.elapsed_time(end) / iters


def benchmark(seed: int, m: int, n: int, k: int, iters: int, warmup: int,
              repeats: int = 5) -> tuple[float, float]:
    print()
    print("=" * 104)
    print("PERFORMANCE  (MUSA events, median of the elapsed means of repeated batches)")
    print("=" * 104)
    inp = make_inputs(seed=seed, m=m, n=n)

    ref_fn = lambda: reference_torch_topk(inp, k)       # noqa: E731
    new_fn = lambda: kernel_function(inp, k)            # noqa: E731

    # sanity: identical results before timing
    rv, ri = ref_fn()
    nv, ni = new_fn()
    torch.musa.synchronize()
    if not (torch.equal(rv, nv) and torch.equal(ri, ni)):
        d = (nv - rv).abs().max().item()
        print(f"[warn] candidate differs from torch.topk: max_abs={d:.3e} "
              f"idx_mismatch={int((ni != ri).sum().item())}")

    ref_runs = [_time_ms(ref_fn, iters, warmup) for _ in range(repeats)]
    new_runs = [_time_ms(new_fn, iters, warmup) for _ in range(repeats)]
    ref_ms = statistics.median(ref_runs)
    new_ms = statistics.median(new_runs)

    print(f"  shape={tuple(inp.shape)} dtype={inp.dtype} k={k}  "
          f"iters={iters} per batch, {repeats} batches")
    print(f"  reference torch.topk : median {ref_ms:8.4f} ms   "
          f"(runs: {', '.join(f'{t:.4f}' for t in ref_runs)})")
    print(f"  generated MUSA op    : median {new_ms:8.4f} ms   "
          f"(runs: {', '.join(f'{t:.4f}' for t in new_runs)})")
    speedup = ref_ms / new_ms
    print(f"  speedup vs reference : {speedup:.2f}x")
    gbs = inp.numel() * inp.element_size() / (new_ms * 1e-3) / 1e9
    print(f"  effective bandwidth  : {gbs:.1f} GB/s  "
          f"(input {inp.numel() * inp.element_size() / 1e6:.1f} MB)")
    if ref_ms > 0:
        print(f"  traced reference was : 1482.9..1623.7 us (median ~1565 us) "
              f"-> candidate is ~{1565.0 / (new_ms * 1e3):.2f}x faster")
    return ref_ms, new_ms


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="topk correctness + performance test")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shape", type=str, default=f"{M}x{NUM_EXPERTS}")
    parser.add_argument("--k", type=int, default=TOPK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    if not torch.musa.is_available():
        print("MUSA is not available", file=sys.stderr)
        return 2

    m, n = (int(x) for x in args.shape.lower().split("x"))
    print(f"device={DEVICE} ({torch.musa.get_device_name(0)}) torch={torch.__version__}")

    ok = correctness_suite(args.seed, m, n, args.k)
    if not args.skip_bench:
        benchmark(args.seed, m, n, args.k, args.iters, args.warmup, args.repeats)

    print()
    print("=" * 104)
    print("RESULT:", "PASS — correctness within 1e-5 and performance measured" if ok else "FAIL")
    print("=" * 104)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
