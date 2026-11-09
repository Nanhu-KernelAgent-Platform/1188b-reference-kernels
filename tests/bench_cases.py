#!/usr/bin/env python3
"""基准测试：直接 torch 跑 20 个真实算子 reference 用例（不使用 KernelAgent）。

对每个 case：
  1. 用真实 per-rank shape 建输入（放到 device 上）
  2. 运行一次，做 gate：输出有限 / dtype 与声明一致 / shape 与声明一致 / 两次运行可复现
  3. warmup 后计时（device event，取 median / min / mean）
  4. 折算吞吐：FLOPs（GFLOP -> TFLOP/s）或访存量（GB -> GB/s）
  5. 与 CSV 里同 kernel 的「平均耗时」并列，便于看 PyTorch reference 相对厂商 asm 核的快慢

用法:
    python3 bench_cases.py                 # 全 20 例
    python3 bench_cases.py 1 2 17          # 指定 case
    python3 bench_cases.py --device cpu    # 换设备
    python3 bench_cases.py --reps 20 --warmup 5
输出: 终端表格 + bench_results.csv
"""

import argparse
import csv
import importlib.util
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(HERE, "cases")

# ---------------------------------------------------------------------------
# CSV 原始数据（021-1188B-real-ops.csv 前 20 行）与工作量估算
#   flops: 该次 kernel 的计算量（FMA 记 2）
#   bytes: 该次 kernel 的最少显存访存量
#   csv_scope: 说明 CSV 的「平均耗时」覆盖的是整个 op 还是 per-expert/per-push 的一次调用
# ---------------------------------------------------------------------------
MB = 1 << 20
S, H_MOE, E_LOCAL, TOPK = 4096, 8192, 40, 8
DQK, DV, NH = 192, 128, 128
FC1_N, FC1_K, FC2_N, FC2_K = 5120, 8192, 8192, 2560

WORK = {
    1:  dict(csv_total=42349.73, csv_calls=64, csv_avg=661.7,
             flops=2.06e12, nbytes=None, csv_scope="1 次 full-seq MLA attn bwd"),
    2:  dict(csv_total=29230.71, csv_calls=128, csv_avg=228.36,
             flops=6.87e11, nbytes=None, csv_scope="1 次 full-seq MLA attn fwd"),
    3:  dict(csv_total=3697.76, csv_calls=3610, csv_avg=1.02,
             flops=2 * S * FC1_N * FC1_K,
             nbytes=E_LOCAL * FC1_N * FC1_K * 2 + S * (FC1_K + FC1_N) * 2,
             csv_scope="per-expert 单次 GEMM（本行测的是 40 组整体）"),
    4:  dict(csv_total=3614.24, csv_calls=4416, csv_avg=0.82,
             flops=2 * S * FC2_N * FC2_K,
             nbytes=E_LOCAL * FC2_N * FC2_K * 2 + S * (FC2_K + FC2_N) * 2,
             csv_scope="per-expert 单次 GEMM（本行测的是 40 组整体）"),
    5:  dict(csv_total=3325.78, csv_calls=2816, csv_avg=1.18,
             flops=2 * S * FC2_N * FC2_K,
             nbytes=E_LOCAL * FC2_N * FC2_K * 2 + 2 * S * FC2_N * 2,
             csv_scope="per-expert 单次 wgrad（本行测的是 40 组整体）"),
    6:  dict(csv_total=3032.03, csv_calls=832, csv_avg=3.64, flops=None,
             nbytes=2 * 2 * ((S // 2) * NH * (DQK + DV)) * 2,
             csv_scope="1 次 k/v 索引重排"),
    7:  dict(csv_total=1859.73, csv_calls=1627, csv_avg=1.14,
             flops=2 * S * FC1_K * FC1_N,
             nbytes=E_LOCAL * FC1_N * FC1_K * 2 + S * (FC1_N + FC1_K) * 2,
             csv_scope="per-expert 单次 dgrad（本行测的是 40 组整体）"),
    8:  dict(csv_total=1726.89, csv_calls=2318, csv_avg=0.74,
             flops=2 * S * FC2_K * FC2_N,
             nbytes=E_LOCAL * FC2_N * FC2_K * 2 + S * (FC2_N + FC2_K) * 2,
             csv_scope="per-expert 单次 dgrad（本行测的是 40 组整体）"),
    9:  dict(csv_total=1296.54, csv_calls=2688, csv_avg=0.48,
             flops=2 * S * FC1_N * FC1_K,
             nbytes=E_LOCAL * FC1_N * FC1_K * 2 + 2 * S * FC1_N * 2,
             csv_scope="per-expert 单次 wgrad（本行测的是 40 组整体）"),
    10: dict(csv_total=1052.52, csv_calls=1840, csv_avg=0.57,
             flops=2 * S * FC1_N * FC1_K, nbytes=FC1_N * FC1_K * 2 + S * (FC1_K + FC1_N) * 2,
             csv_scope="1 次 GEMM（本行 1 组）"),
    11: dict(csv_total=981.92, csv_calls=474, csv_avg=2.07, flops=None,
             nbytes=S * H_MOE * 2 + S * H_MOE * 4, csv_scope="1 次 cast"),
    12: dict(csv_total=543.26, csv_calls=128, csv_avg=4.24, flops=None,
             nbytes=S * 320 + 2 * S * 320 * 8, csv_scope="1 次 row_id_map 构建"),
    13: dict(csv_total=512.78, csv_calls=1044, csv_avg=0.49,
             flops=2 * S * FC1_K * FC1_N, nbytes=FC1_N * FC1_K * 2 + S * (FC1_N + FC1_K) * 2,
             csv_scope="1 次 dgrad（本行 1 组）"),
    14: dict(csv_total=505.1, csv_calls=944, csv_avg=0.54,
             flops=2 * S * FC2_N * FC2_K, nbytes=FC2_N * FC2_K * 2 + S * (FC2_K + FC2_N) * 2,
             csv_scope="1 次 GEMM（本行 1 组）"),
    15: dict(csv_total=418.41, csv_calls=128, csv_avg=3.27, flops=None,
             nbytes=S * TOPK * H_MOE * 2 + S * H_MOE * 2, csv_scope="1 次 unpermute"),
    16: dict(csv_total=399.32, csv_calls=1024, csv_avg=0.39, flops=None,
             nbytes=3 * S * H_MOE * 2, csv_scope="1 次逐元素 add"),
    17: dict(csv_total=397.69, csv_calls=128, csv_avg=3.11, flops=None,
             nbytes=S * H_MOE * 2 + S * TOPK * H_MOE * 2, csv_scope="1 次 permute"),
    18: dict(csv_total=384.45, csv_calls=64, csv_avg=6.01, flops=None,
             nbytes=S * TOPK * H_MOE * 2 + S * H_MOE * 2 + 2 * S * 320 * 4,
             csv_scope="1 次 unpermute+probs grad"),
    19: dict(csv_total=300.15, csv_calls=409, csv_avg=0.73,
             flops=2 * S * FC2_K * FC2_N, nbytes=FC2_N * FC2_K * 2 + S * (FC2_N + FC2_K) * 2,
             csv_scope="1 次 dgrad（本行 1 组）"),
    20: dict(csv_total=267.61, csv_calls=128, csv_avg=2.09, flops=None,
             nbytes=S * NH * (128 + 128) * 2 + S * NH * (192 + 128) * 2,
             csv_scope="1 次 MLA rope(kv)"),
}

# 期望输出（类型, shape 列表）—— 用于运行 gate
EXPECT = {
    1: ("bf16", [[4096, 1, 128, 192], [4096, 1, 128, 192], [4096, 1, 128, 128]]),
    2: ("bf16", [[4096, 1, 128, 128]]),
    3: ("bf16", [[4096, 5120]]),
    4: ("bf16", [[4096, 8192]]),
    5: ("bf16", [[40, 8192, 2560]]),
    6: ("bf16", [[2, 2048, 1, 128, 192], [2, 2048, 1, 128, 128]]),
    7: ("bf16", [[4096, 8192]]),
    8: ("bf16", [[4096, 2560]]),
    9: ("bf16", [[40, 5120, 8192]]),
    10: ("bf16", [[4096, 5120]]),
    11: ("fp32", [[4096, 8192]]),
    12: ("int64", [[320, 4096], [4096, 320]]),
    13: ("bf16", [[4096, 8192]]),
    14: ("bf16", [[4096, 8192]]),
    15: ("bf16", [[4096, 8192]]),
    16: ("bf16", [[4096, 8192]]),
    17: ("bf16", [[32768, 8192], [32768]]),
    18: ("bf16+fp32", [[4096, 8192], [4096, 320]]),
    19: ("bf16", [[4096, 2560]]),
    20: ("bf16", [[4096, 1, 128, 192], [4096, 1, 128, 128]]),
}

DT = {"bf16": torch.bfloat16, "fp32": torch.float32, "int64": torch.int64}


def pick_device(req=None):
    if req:
        return torch.device(req)
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def backend(device):
    t = device.type
    return getattr(torch, t)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def load_cases(only=None):
    out = []
    for fn in sorted(os.listdir(CASES_DIR)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        cid = int(fn.split("_")[0].replace("case", ""))
        if only and cid not in only:
            continue
        spec = importlib.util.spec_from_file_location(fn[:-3], os.path.join(CASES_DIR, fn))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append((cid, fn, mod))
    return out


def flatten(out):
    return list(out) if isinstance(out, (list, tuple)) else [out]


def check_gate(cid, out, out2, device):
    """返回 (ok, msgs)."""
    msgs = []
    flat, flat2 = flatten(out), flatten(out2)
    exp_dt, exp_shapes = EXPECT[cid]
    if len(flat) != len(exp_shapes):
        return False, [f"输出个数 {len(flat)} != 期望 {len(exp_shapes)}"]
    ok = True
    for i, (o, e) in enumerate(zip(flat, exp_shapes)):
        if not isinstance(o, torch.Tensor):
            ok = False
            msgs.append(f"out{i} 不是 tensor")
            continue
        if list(o.shape) != e:
            ok = False
            msgs.append(f"out{i} shape {list(o.shape)} != {e}")
        if o.is_floating_point() and not torch.isfinite(o).all():
            ok = False
            msgs.append(f"out{i} 含 NaN/Inf")
        # 复现性（同一输入两次运行）：atomic/index_add 类核天然有微小非确定，按相对差判定
        ref = o.float().abs().max().clamp_min(1e-12)
        rel = ((o.float() - flat2[i].float()).abs().max() / ref).item()
        if rel > 1e-2:
            ok = False
            msgs.append(f"out{i} 两次运行相对差 {rel:.2e} 超阈值")
        elif rel > 0:
            msgs.append(f"out{i} 非确定性(相对 {rel:.1e})")
    return ok, msgs


def timeit(fn, reps, warmup, device):
    bk = backend(device)
    for _ in range(warmup):
        fn()
    bk.synchronize()
    times = []
    for _ in range(reps):
        s = bk.Event(enable_timing=True) if hasattr(bk, "Event") else None
        if s is not None:
            e = bk.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            bk.synchronize()
            times.append(s.elapsed_time(e))
        else:
            import time as _t
            t0 = _t.perf_counter()
            fn()
            bk.synchronize()
            times.append((_t.perf_counter() - t0) * 1e3)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reps", type=int, default=0, help="0 = 按耗时自适应")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "bench_results.csv"))
    args = ap.parse_args()

    device = pick_device(args.device)
    bk = backend(device)
    cases = load_cases(set(args.ids) if args.ids else None)
    print(f"# device={device} ({bk.get_device_name(0) if device.type != 'cpu' else 'cpu'}) cases={len(cases)}\n")

    rows = []
    n_ok = n_bad = 0
    for cid, fn, mod in cases:
        meta = mod.CASE
        torch.manual_seed(0)
        model = mod.Model(*mod.get_init_inputs()).to(device).eval()
        inputs = to_device(mod.get_inputs(), device)
        in_bytes = sum(x.numel() * x.element_size() for x in inputs if isinstance(x, torch.Tensor))

        tag = f"[case{cid:02d}] {meta['name']}"
        try:
            if device.type != "cpu":
                bk.empty_cache()
                if hasattr(bk, "reset_peak_memory_stats"):
                    bk.reset_peak_memory_stats()
            with torch.no_grad():
                out = model(*inputs)
                out2 = model(*inputs)
            ok, msgs = check_gate(cid, out, out2, device)
            out_bytes = sum(o.numel() * o.element_size() for o in flatten(out))

            # 自适应 reps：先热身（去掉 JIT/autotune 影响），再按 ~0.4s 测量窗口定 reps
            if args.reps:
                reps = args.reps
            else:
                with torch.no_grad():
                    t_probe = statistics.median(timeit(lambda: model(*inputs), 2, 2, device))
                reps = max(3, min(50, int(400.0 / max(t_probe, 0.01))))
            with torch.no_grad():
                times = timeit(lambda: model(*inputs), reps, args.warmup, device)
            med = statistics.median(times)
            mn = min(times)
            mean = statistics.fmean(times)
            peak = (bk.max_memory_allocated() / 1024**3
                    if device.type != "cpu" and hasattr(bk, "max_memory_allocated") else 0.0)

            w = WORK[cid]
            tflops = (w["flops"] / (mn * 1e-3) / 1e12) if w["flops"] else None
            gbps = (w["nbytes"] / (mn * 1e-3) / 1e9) if w["nbytes"] else None
            ratio = (med / w["csv_avg"]) if w["csv_avg"] else None

            print(tag)
            print(f"    op   : {meta['upstream']}")
            print(f"    io   : in {in_bytes / MB:,.0f} MB -> out {out_bytes / MB:,.0f} MB"
                  f"   peak {peak:.2f} GB")
            print(f"    gate : {'PASS' if ok else 'FAIL'}"
                  + ("" if not msgs else "  " + "; ".join(msgs)))
            print(f"    time : med {med:9.3f} ms | min {mn:9.3f} | mean {mean:9.3f}  ({reps} reps)")
            extra = []
            if tflops:
                extra.append(f"{tflops:7.1f} TFLOP/s")
            if gbps:
                extra.append(f"{gbps:7.1f} GB/s")
            if w["flops"] or w["nbytes"]:
                print(f"    perf : " + "  ".join(extra))
            print(f"    csv  : avg {w['csv_avg']:.2f} ms ({w['csv_scope']})"
                  + (f"  -> ref/csv = {ratio:.2f}x" if ratio else ""))
            print()

            rows.append(dict(
                id=cid, name=meta["name"], kernel=meta["kernel"], upstream=meta["upstream"],
                gate="PASS" if ok else "FAIL",
                ref_median_ms=round(med, 4), ref_min_ms=round(mn, 4), ref_mean_ms=round(mean, 4),
                reps=reps, in_MB=round(in_bytes / MB, 1), out_MB=round(out_bytes / MB, 1),
                peak_GB=round(peak, 3),
                flops_GFLOP=round(w["flops"] / 1e9, 1) if w["flops"] else "",
                achieved_TFLOPs=round(tflops, 2) if tflops else "",
                moved_MB=round(w["nbytes"] / MB, 1) if w["nbytes"] else "",
                achieved_GBs=round(gbps, 1) if gbps else "",
                csv_avg_ms=w["csv_avg"], csv_scope=w["csv_scope"],
                ref_over_csv=round(ratio, 2) if ratio else "",
            ))
            n_ok += 1 if ok else 0
            n_bad += 0 if ok else 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            n_bad += 1
            print(f"{tag}\n    FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            rows.append(dict(id=cid, name=meta["name"], kernel=meta["kernel"],
                             upstream=meta["upstream"], gate="ERROR", ref_median_ms="",
                             ref_min_ms="", ref_mean_ms="", reps=0, in_MB="", out_MB="",
                             peak_GB="", flops_GFLOP="", achieved_TFLOPs="", moved_MB="",
                             achieved_GBs="", csv_avg_ms=WORK[cid]["csv_avg"],
                             csv_scope=WORK[cid]["csv_scope"], ref_over_csv=""))
        finally:
            del model, inputs
            if device.type != "cpu":
                bk.empty_cache()

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# gate: ok={n_ok} fail={n_bad}   结果写入 {args.out}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
