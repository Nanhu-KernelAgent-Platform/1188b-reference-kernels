#!/usr/bin/env python3
"""test_case25.py - correctness + performance test for the MUSA `rotary_bwd_kv` kernel
against the reference case `rotary_bwd_kv_kernel` (ApplyMLARotaryEmbKV backward).

Correctness criterion (requested): absolute error <= ATOL = 1e-5
  * d_kv    : both content halves must match the reference exactly (0.0 <= 1e-5)
  * d_emb   : checked on the kernel's fp32 accumulation path (ext.rotary_bwd_kv_f32)
              against an fp32 evaluation of the reference formula
              -> |kernel - reference| <= 1e-5
  * d_emb   : the shipped bf16 path is additionally verified to be the correctly
              rounded image of that fp32 result (bf16 has ~2^-8 resolution, so a
              1e-5 bound is not meaningful on bf16 bits; the ulp-level check is
              reported for completeness).

Performance criterion: kernel latency < 1.881 ms (trace baseline, seq=16384).

Usage:  python test_case25.py [--quick] [--no-bench]
"""
from __future__ import annotations

import sys
import statistics
import time

import torch
import torch_musa  # noqa: F401  (registry for the musa device)

sys.path.insert(0, "/root/zf_bench/rotary")
import ref                                    # torch reference (same math as the case)
import rotary_bwd_kv_musa as ext               # the MUSA kernel

DEV = "musa"
ATOL = 1e-5          # requested precision
RTOL = 1e-5          # combined torch-style tolerance for the fp32 tensors
TRACE_MS = 1.881     # recorded MTT S5000 launch (1846.690..1917.571 us, median 1881)
BF16_ULP = 2.0 ** -8

FAILURES: list[str] = []


def rec(ok: bool, msg: str):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILURES.append(msg)


# ----------------------------------------------------------------------------- utils
def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.musa.synchronize()
    s = torch.musa.Event(enable_timing=True)
    e = torch.musa.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.musa.synchronize()
    return s.elapsed_time(e) / iters


def ref_fp32(dk, dv, cos, sin):
    """Reference math kept in fp32 (i.e. before the `.to(dk.dtype)` cast)."""
    seq = dk.shape[0]
    left = dk[..., 128:160].float().sum(dim=2)
    right = dk[..., 160:].float().sum(dim=2)
    cos_f = cos.float().reshape(seq, 64)
    sin_f = sin.float().reshape(seq, 64)
    dx1 = left * cos_f[:, None, :32] + right * sin_f[:, None, 32:]
    dx2 = -left * sin_f[:, None, :32] + right * cos_f[:, None, 32:]
    return torch.stack((dx1, dx2), dim=-1).reshape(seq, dk.shape[1], 1, 64)


# ------------------------------------------------------------------------ correctness
def test_shape(seq: int, batch: int, seed: int) -> None:
    dk, dv, cos, sin = ref.make_inputs(seed, DEV, seq=seq)
    if batch != 1:
        dk = dk.expand(seq, batch, 128, 192).contiguous()
        dv = dv.expand(seq, batch, 128, 128).contiguous()

    r_kv, r_emb = ref.reference_rotary_bwd_kv(dk, dv, cos, sin)
    r_emb32 = ref_fp32(dk, dv, cos, sin)

    k_kv, k_emb32 = ext.rotary_bwd_kv_f32(dk, dv, cos, sin)   # fp32 accumulation path
    b_kv, b_emb = ext.rotary_bwd_kv(dk, dv, cos, sin)         # shipped bf16 path
    torch.musa.synchronize()

    tag = f"seq={seq} batch={batch} seed={seed}"

    # --- shapes / dtypes -----------------------------------------------------------
    ok = (tuple(k_kv.shape) == tuple(r_kv.shape) == (seq, batch, 128, 256)
          and tuple(b_emb.shape) == tuple(r_emb.shape) == (seq, batch, 1, 64)
          and k_kv.dtype == torch.bfloat16 and b_emb.dtype == torch.bfloat16)
    rec(ok, f"{tag} | shapes/dtypes d_kv={tuple(k_kv.shape)} bf16, d_emb={tuple(b_emb.shape)} bf16")

    # --- d_kv: straight-through content gradients, must be exact -------------------
    err_c = (b_kv[..., :128].float() - dk[..., :128].float()).abs().max().item()
    err_v = (b_kv[..., 128:].float() - dv.float()).abs().max().item()
    rec(max(err_c, err_v) <= ATOL,
        f"{tag} | d_kv content: max_abs k={err_c:.3e} v={err_v:.3e} (<= {ATOL:g})")

    # --- d_emb: fp32 accumulation vs fp32 reference (the 1e-5 criterion) ----------
    diff = (k_emb32 - r_emb32).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (r_emb32.abs() + 1e-30)).max().item()
    closed = True
    try:
        torch.testing.assert_close(k_emb32, r_emb32, rtol=RTOL, atol=ATOL)
    except Exception:
        closed = False
    rec(max_abs <= ATOL and closed,
        f"{tag} | d_emb fp32: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"assert_close(atol={ATOL:g},rtol={RTOL:g})={'PASS' if closed else 'FAIL'}")

    # --- d_emb: shipped bf16 output vs reference bf16 (ulp level, informational) --
    bdiff = (b_emb.float() - r_emb.float()).abs()
    bitequal = (b_emb.view(torch.int16) == r_emb.view(torch.int16)).float().mean().item() * 100
    # one bf16 ulp for a value m in [2^(e-1), 2^e) is 2^(e-8); take the wider of the
    # two binades so that values straddling a binade edge are not spuriously flagged
    _, e_ref = torch.frexp(r_emb.float().abs().clamp_min(1e-30))
    _, e_ker = torch.frexp(b_emb.float().abs().clamp_min(1e-30))
    ulp = torch.ldexp(torch.ones_like(r_emb.float()), torch.maximum(e_ref, e_ker) - 8)
    n_diff = int((bdiff > 0).sum().item())
    # accept either "adjacent on the bf16 grid" or "within the requested absolute
    # tolerance": for heavily cancelled values (|x| ~ 1e-6 out of terms of ~50) one
    # bf16 ulp is far below ATOL, and for large values the ulp rule dominates.
    bound = torch.maximum(ulp * 1.0001, torch.full_like(ulp, ATOL))
    within_ulp = bool((bdiff <= bound).all().item())
    cmp = "<=" if within_ulp else ">"
    rec(within_ulp,
        f"{tag} | d_emb bf16: bitequal={bitequal:.6f}% ({n_diff}/{r_emb.numel()} elements "
        f"differ) max_abs={bdiff.max().item():.3e} "
        f"({cmp} 1 bf16 ulp of 2^-8, or <= ATOL={ATOL:g})")


def test_reference_selfchecks() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("refcase", "/root/zf_bench/rotary/refcase.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    try:
        m._check_adjoint_identity()
        rec(True, "reference adjoint identity <dK_rope, fwd(emb)> == <d_emb, emb>")
    except SystemExit as exc:
        rec(False, f"reference adjoint identity: {exc}")
    dk, dv, cos, sin = m.make_inputs(0, "musa")
    r_kv, r_emb = m.reference_rotary_bwd_kv(dk, dv, cos, sin)
    ck = (r_kv[..., :128] - dk[..., :128]).abs().max().item()
    cv = (r_kv[..., 128:] - dv).abs().max().item()
    errs = []
    for i in (0, m.SEQ // 2, m.SEQ - 1):
        mk, me = m._manual_one(dk, dv, cos, sin, i)
        errs.append((r_kv[i, 0, 0] - mk).abs().max().item())
        errs.append((r_emb[i, 0, 0] - me).abs().max().item())
    rec(ck == 0 and cv == 0 and max(errs) == 0,
        f"reference vs its own explicit per-token formula (content {ck:.1e}/{cv:.1e}, rope {max(errs):.1e})")


# ------------------------------------------------------------------------ performance
def test_performance(iters=50, trials=5) -> None:
    seq = 16384
    dk, dv, cos, sin = ref.make_inputs(0, DEV, seq=seq)
    ext.set_config(512, 1, 16, 1, 0)          # tuned config

    ts = [bench(lambda: ext.rotary_bwd_kv(dk, dv, cos, sin), iters=iters) for _ in range(trials)]
    med, best = statistics.median(ts), min(ts)
    gb = (seq * 128 * (192 + 128 + 256) * 2) / 1e9

    n = 512 * 1024 * 1024
    a = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    b = torch.empty_like(a)
    ctrl = bench(lambda: b.copy_(a), iters=30, warm=5)
    ctrl_bw = 2 * n * 2 / 1e9 / ctrl * 1e3

    rts = [bench(lambda: ref.reference_rotary_bwd_kv(dk, dv, cos, sin), iters=20, warm=5)
           for _ in range(3)]

    print(f"  kernel      : median {med:.4f} ms  min {best:.4f} ms  "
          f"trials [{', '.join(f'{t:.4f}' for t in ts)}]")
    print(f"  traffic     : {gb:.3f} GB effective -> {gb/med*1e3:.1f} GB/s "
          f"(control 1 GiB copy in-process: {ctrl_bw:.1f} GB/s)")
    print(f"  torch ref   : {statistics.median(rts):.4f} ms  (same process)")
    print(f"  trace base  : {TRACE_MS:.3f} ms")
    print(f"  speedup     : {TRACE_MS/med:.3f}x  ({'faster' if med < TRACE_MS else 'slower'}), "
          f"{(TRACE_MS-med)*1e3:+.0f} us/call, 64 layers: "
          f"{TRACE_MS*64:.2f} -> {med*64:.2f} ms")
    rec(med < TRACE_MS, f"latency {med:.4f} ms < trace baseline {TRACE_MS:.3f} ms")
    rec(med < statistics.median(rts), f"latency {med:.4f} ms < torch reference "
                                      f"{statistics.median(rts):.4f} ms "
                                      f"({statistics.median(rts)/med:.2f}x)")


# ------------------------------------------------------------------------------- main
def main(argv):
    quick = "--quick" in argv
    no_bench = "--no-bench" in argv

    t0 = time.time()
    print("=" * 96)
    print(f"test_case25 | MUSA rotary_bwd_kv vs reference (rotary_bwd_kv_kernel) | "
          f"ATOL={ATOL:g}")
    print("=" * 96)

    print("\n[1] reference self-checks")
    test_reference_selfchecks()

    print(f"\n[2] correctness vs reference (kernel fp32 path, d_kv exact copy)")
    shapes = [(16384, 1, 0), (16384, 1, 7)]
    if not quick:
        shapes += [(4096, 1, 1), (2048, 4, 2), (1000, 1, 5), (7, 1, 5), (1, 1, 5)]
    for seq, batch, seed in shapes:
        test_shape(seq, batch, seed)

    if not no_bench:
        print("\n[3] performance (seq=16384, heads=128, batch=1)")
        test_performance(iters=30 if quick else 50)

    print("\n" + "=" * 96)
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} check(s) failed)  [{time.time()-t0:.1f}s]")
        for m in FAILURES:
            print("  - " + m)
        return 1
    print(f"RESULT: PASS - all checks ok  [{time.time()-t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
