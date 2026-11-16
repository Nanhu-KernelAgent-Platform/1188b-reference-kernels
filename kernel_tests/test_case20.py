#!/usr/bin/env python3
"""test_case20.py — case20 mla_rope_fwd_kv 独立复测
精度: allclose(atol=rtol=1e-5). 性能: isolated(25 warm+50 reps, 逐次sync+L2 flush, 中位数).
用法: cd kernel_tests && python3 test_case20.py      (PASS=exit 0 / FAIL=exit 1)
"""
import json, os, statistics, subprocess, sys
from pathlib import Path
import torch

CASE = 'case20_mla_rope_fwd_kv'
PROFILER_MS = 2.09
KIND = 'tol'

HERE = Path(__file__).resolve().parent
# case 文件解析顺序：KERNEL_CASES_DIR 环境变量 -> 仓库平级 ../kernel_cases
CASES_DIR = os.environ.get('KERNEL_CASES_DIR')


def _pick_case():
    cands = []
    if CASES_DIR:
        cands.append(Path(CASES_DIR) / (CASE + '.py'))
    cands += [HERE.parent / 'kernel_cases' / (CASE + '.py')]
    for c in cands:
        if c.exists():
            print('[case] 使用 %s' % c)
            return c
    raise SystemExit('未找到 case 文件: %s（仓库内经 ../kernel_cases 提供；本地可设 KERNEL_CASES_DIR 指定）' % (CASE + '.py'))


CASE_FILE = _pick_case()
def _pick_bundle():
    cands = [HERE.parent / 'kernel_results' / CASE, HERE.parent / CASE]
    for c in cands:
        if (c / 'kernel.py').exists():
            return c
    raise SystemExit('未找到 case 目录: ' + ' | '.join(str(c) for c in cands))

BUNDLE = _pick_bundle()
RESULTS = HERE / 'results'

BE = torch.musa if (hasattr(torch, 'musa') and torch.musa.is_available()) else torch.cuda
DEV = torch.device('musa' if BE is torch.musa else 'cuda')


def flush():
    d = torch.empty((32, 1024, 1024), dtype=torch.int64, device=DEV)
    d.fill_(42)
    del d


def bench(fn, warm=25, reps=50):
    for _ in range(warm):
        fn()
    BE.synchronize()
    ts = []
    for _ in range(reps):
        BE.synchronize(); flush()
        s = BE.Event(enable_timing=True); e = BE.Event(enable_timing=True)
        s.record(); fn(); e.record()
        BE.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def main():
    if not list(BUNDLE.glob('*.so')):
        print('[build] %s 无 .so，执行 setup.py build_ext --inplace' % CASE)
        r = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                           cwd=BUNDLE, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]

    g = {}
    exec(compile(open(CASE_FILE).read(), str(CASE_FILE), 'exec'), g)
    init = g['get_init_inputs']() if 'get_init_inputs' in g else []
    model = g['Model'](*init).to(DEV) if init else g['Model']().to(DEV)
    inputs = [t.to(DEV) if isinstance(t, torch.Tensor) else t for t in g['get_inputs']()]
    ref = model(*inputs)
    ref = list(ref) if isinstance(ref, (tuple, list)) else [ref]

    sys.path.insert(0, str(BUNDLE))
    import kernel as ka
    out = ka.kernel_function(*inputs)
    out = list(out) if isinstance(out, (tuple, list)) else [out]

    assert len(ref) == len(out), '输出个数不一致: %d vs %d' % (len(ref), len(out))
    eps_ok, exact_ok, maxd = True, True, 0.0
    for i, (a, b) in enumerate(zip(ref, out)):
        assert a.shape == b.shape, 'out%d shape %s vs %s' % (i, tuple(a.shape), tuple(b.shape))
        eps_ok &= bool(torch.allclose(b.float(), a.float(), atol=1e-5, rtol=1e-5))
        exact_ok &= bool(torch.equal(a, b))
        maxd = max(maxd, (a.float() - b.float()).abs().max().item())
    prec_ok = exact_ok if KIND in ('int', 'copy') else eps_ok

    def r_fn(): return model(*inputs)
    def k_fn(): return ka.kernel_function(*inputs)
    rts = bench(r_fn); kts = bench(k_fn)
    rmed, kmed = statistics.median(rts), statistics.median(kts)

    ok = bool(prec_ok and kmed < PROFILER_MS)
    print('case    : %s' % CASE)
    print('形状    : %s' % str([tuple(t.shape) for t in inputs]))
    print('精度    : eps@1e-5=%s  逐位等=%s  max|diff|=%.3e'
          % ('PASS' if eps_ok else 'FAIL', 'Y' if exact_ok else 'N', maxd))
    print('性能    : reference med %.4f ms | kernel med %.4f ms' % (rmed, kmed))
    print('加速比  : vs profiler %.3f ms -> %.2fx | vs reference %.2fx'
          % (PROFILER_MS, PROFILER_MS / kmed, rmed / kmed))
    print('VERDICT :', 'PASS' if ok else 'FAIL')

    RESULTS.mkdir(exist_ok=True)
    json.dump({'case': CASE, 'eps_1e-5': bool(eps_ok), 'exact_equal': bool(exact_ok),
               'max_abs_diff': maxd, 'reference_med': rmed, 'kernel_med': kmed,
               'vs_profiler': PROFILER_MS / kmed, 'vs_reference': rmed / kmed,
               'verdict': 'PASS' if ok else 'FAIL',
               'ref_samples': rts, 'ka_samples': kts},
              open(RESULTS / ('test_' + CASE + '.json'), 'w'), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
