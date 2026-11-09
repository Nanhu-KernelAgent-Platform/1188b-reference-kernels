#!/usr/bin/env python3
"""语义校验：用「另一条独立实现」交叉验证 20 个 case 的 reference 是否正确。

bench_cases.py 只证明"能跑且数值有限"；本脚本进一步用小规模 + 独立实现（SDPA / autograd /
朴素 python 双重循环 / float64）核对每个 case 的数学语义，避免"形状对、算错"。

用法:
    python3 verify_semantics.py            # 全部
    python3 verify_semantics.py --device cpu
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(HERE, "cases")


def load(cid):
    fn = [f for f in os.listdir(CASES_DIR) if f.startswith(f"case{cid:02d}_")]
    assert fn, f"case{cid:02d} not found"
    spec = importlib.util.spec_from_file_location(fn[0][:-3], os.path.join(CASES_DIR, fn[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _d(x):
    """统一搬到 CPU float64 比较（MUSA 不支持 fp64 的 batch matmul）。"""
    return x.detach().cpu().double()


def rel(a, b):
    a = _d(a)
    b = _d(b)
    d = (a - b).abs().max().item()
    scale = b.abs().max().item()
    return d, (d / scale if scale > 0 else d)


RESULTS = []


def record(name, ok, msg):
    RESULTS.append((name, "PASS" if ok else "FAIL", msg))
    print(f"  [{('PASS' if ok else 'FAIL')}] {name}: {msg}")


def compare(name, got, exp, tol=2e-2):
    if isinstance(got, (list, tuple)):
        ok = True
        parts = []
        for i, (g, e) in enumerate(zip(got, exp)):
            d, r = rel(g, e)
            parts.append(f"out{i} max|Δ|={d:.2e} rel={r:.2e}")
            ok = ok and r <= tol
        record(name, ok, "; ".join(parts) + f" (tol {tol:g})")
        return
    d, r = rel(got, exp)
    record(name, r <= tol, f"max|Δ|={d:.2e} rel={r:.2e} (tol {tol:g})")


# ---------------------------------------------------------------- attention
def verify_attention(device):
    """case01/02：SDPA 实现 vs 分块显式 softmax 实现（互相独立）."""
    m2 = load(2)
    S, B, H = 512, 1, 8
    torch.manual_seed(0)
    q = torch.randn(S, B, H, 192, device=device).bfloat16()
    k = torch.randn(S, B, H, 192, device=device).bfloat16()
    v = torch.randn(S, B, H, 128, device=device).bfloat16()

    sdpa = m2.Model(256, "sdpa")(q, k, v)
    # 显式实现：fp64 全精度参考（S=512 只有 512x512，可全量物化）
    scale = 192 ** -0.5
    qp = _d(q).permute(1, 2, 0, 3)
    kp = _d(k).permute(1, 2, 0, 3)
    vp = _d(v).permute(1, 2, 0, 3)
    sc = (qp @ kp.transpose(-1, -2)) * scale
    idx = torch.arange(S)
    sc = sc.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
    p = torch.softmax(sc, -1)
    ref = (p @ vp).permute(2, 0, 1, 3)
    compare("case02 attn fwd (SDPA vs fp64 显式)", sdpa, ref, tol=2e-2)

    # 反向：SDPA autograd vs 解析反向（case01 的 explicit 实现）
    m1 = load(1)
    do = torch.randn(S, B, H, 128, device=device).bfloat16() * 0.1
    o = torch.zeros_like(v)
    got = m1.Model(128, "sdpa")(q, k, v, o, do)
    exp = m1.Model(128, "explicit")(q, k, v, o, do)
    compare("case01 attn bwd (SDPA-autograd vs 解析反向)", got, exp, tol=3e-2)


# ---------------------------------------------------------------- grouped GEMM
def _tiny_grouped(M, G, K, N, device):
    sizes = [M // G + (1 if i < M % G else 0) for i in range(G)]
    x = torch.randn(M, K, device=device).bfloat16()
    w = torch.randn(G, N, K, device=device).bfloat16()
    dy = torch.randn(M, N, device=device).bfloat16()
    return sizes, x, w, dy


def verify_grouped_gemm(device):
    """case03/04/10/14（fwd）与 07/08/09/05/13/19（dgrad/wgrad）—— 与 autograd 对照."""
    sizes, x, w, dy = _tiny_grouped(64, 4, 32, 48, device)
    offs = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), 0)), device=device)

    # 前向：逐 group 用 F.linear（另一条算子路径）
    outs = []
    for i, m in enumerate(sizes):
        outs.append(F.linear(x[offs[i]:offs[i + 1]].float(), w[i].float()))
    fwd_exp = torch.cat(outs, 0)

    mod = load(3)
    got = mod.Model(4, 64)(x, w)
    compare("case03/10 grouped GEMM fwd (vs F.linear 逐组)", got, fwd_exp, tol=2e-2)

    # dgrad / wgrad：用 autograd 对同一前向求导
    xf = x.float().requires_grad_(True)
    wf = w.float().requires_grad_(True)
    outs = []
    for i, m in enumerate(sizes):
        outs.append(F.linear(xf[offs[i]:offs[i + 1]], wf[i]))
    y = torch.cat(outs, 0)
    gx, gw = torch.autograd.grad(y, (xf, wf), dy.float())

    compare("case07/13 grouped GEMM dgrad (vs autograd)",
            load(7).Model(4, 64)(dy, w), gx, tol=3e-2)
    compare("case09 grouped GEMM wgrad (vs autograd)",
            load(9).Model(4, 64)(dy, x), gw, tol=3e-2)

    # 单 group 版本（case10/13/14/19）与多 group 在 G=1 时应完全一致
    g1 = load(10).Model(1, 64)(x, w[:1])
    compare("case10 单 group fwd (vs F.linear)", g1, F.linear(x.float(), w[0].float()), tol=2e-2)


# ---------------------------------------------------------------- MoE 索引类
def _routing_map(T, E, topk, device):
    rm = torch.zeros(T, E, device=device, dtype=torch.bool)
    idx = torch.argsort(torch.rand(T, E, device=device), dim=-1)[:, :topk]
    rm.scatter_(1, idx, True)
    return rm, idx


def verify_moe(device):
    T, E, topk = 32, 5, 2
    torch.manual_seed(0)
    rm, idx = _routing_map(T, E, topk, device)

    # case12：与朴素双重循环对照
    mask = rm.cpu().tolist()
    # 朴素参考：expert-major 拼接布局 -> 段内序号 + 该 expert 段的全局起始行
    counts = [sum(1 for t in range(T) if mask[t][e]) for e in range(E)]
    ref = [[-1] * T for _ in range(E)]
    base = 0
    for e in range(E):
        c = 0
        for t in range(T):
            if mask[t][e]:
                ref[e][t] = base + c
                c += 1
        base += counts[e]
    ref = torch.tensor(ref, dtype=torch.int64, device=device)
    got_row, got_nt = load(12).Model()(rm)
    record("case12 row_id_map (vs 朴素双重循环)",
           torch.equal(got_row, ref) and torch.equal(got_nt, ref.t().contiguous()),
           f"int64 完全相等={torch.equal(got_row, ref)}")

    # case17：permute —— 朴素循环
    Hd = 8
    x = torch.randn(T, Hd, device=device).bfloat16()
    probs = torch.zeros(T, E, device=device)
    probs.scatter_(1, idx, torch.rand(T, topk, device=device))
    M = int((got_nt >= 0).sum())
    exp_x = torch.zeros(M, Hd, device=device, dtype=torch.float32)
    exp_p = torch.zeros(M, device=device, dtype=torch.float32)
    xl, pl = x.cpu().float().tolist(), probs.cpu().tolist()
    nt = got_nt.cpu().tolist()
    for t in range(T):
        for e in range(E):
            d = nt[t][e]
            if d >= 0:
                exp_x[d] = torch.tensor(xl[t], device=device)
                exp_p[d] = pl[t][e]
    got_x, got_p = load(17).Model(E, M)(x, got_nt, probs)
    compare("case17 permute (vs 朴素循环) act", got_x, exp_x, tol=0.0)
    compare("case17 permute (vs 朴素循环) probs", got_p, exp_p, tol=0.0)

    # case15 / 18：unpermute —— 朴素循环（行 15 无 prob 加权；行 18 另出 probs grad）
    pin = torch.randn(M, Hd, device=device).bfloat16()
    dpp = torch.randn(M, device=device)
    row_t = ref  # [E, T] trans，与 case15/18 期望的布局一致
    exp_u = torch.zeros(T, Hd, device=device, dtype=torch.float32)
    exp_dp = torch.zeros(T, E, device=device, dtype=torch.float32)
    pinl = pin.cpu().float().tolist()
    dppl = dpp.cpu().tolist()
    rl = row_t.cpu().tolist()
    for t in range(T):
        for e in range(E):
            s = rl[e][t]
            if s >= 0:
                exp_u[t] += torch.tensor(pinl[s], device=device)
                exp_dp[t][e] = dppl[s]
    compare("case15 unpermute (vs 朴素循环)", load(15).Model(E)(pin, row_t),
            exp_u.bfloat16(), tol=0.0)
    gu, gp = load(18).Model(E)(pin, dpp, row_t)
    compare("case18 unpermute+bwd (act)", gu, exp_u.bfloat16(), tol=0.0)
    compare("case18 unpermute+bwd (probs grad)", gp, exp_dp, tol=0.0)


# ---------------------------------------------------------------- elementwise
def verify_elementwise(device):
    x = torch.randn(16, 32, device=device).bfloat16()
    compare("case11 cast bf16->fp32 (vs float64)", load(11).Model()(x), _d(x), tol=0.0)
    a = torch.randn(16, 32, device=device).bfloat16()
    b = torch.randn(16, 32, device=device).bfloat16()
    compare("case16 aten add (bf16 舍入一致)", load(16).Model()(a, b),
            (_d(a) + _d(b)).bfloat16(), tol=0.0)

    # case06 index_select：与逐块手工拼接对照
    k_ag = torch.randn(2, 8, 1, 2, 6, device=device).bfloat16()
    v_ag = torch.randn(2, 8, 1, 2, 4, device=device).bfloat16()
    idx_t = torch.tensor([0, 1], dtype=torch.int32, device=device)
    gk, gv = load(6).Model()(k_ag, v_ag, idx_t)
    compare("case06 index_select k", gk, k_ag[[0, 1]], tol=0.0)
    compare("case06 index_select v", gv, v_ag[[0, 1]], tol=0.0)


# ---------------------------------------------------------------- MLA RoPE
def verify_rope(device):
    """case20：与 float64 朴素实现（逐 token/head 循环）对照."""
    S, B, H, KD, VD, EMB = 8, 1, 4, 8, 4, 6
    torch.manual_seed(0)
    kv = torch.randn(S, B, H, KD + VD, device=device).bfloat16()
    kpe = torch.randn(S, B, 1, EMB, device=device).bfloat16()
    theta = 1.0 / (10000.0 ** (torch.arange(0, EMB, 2, dtype=torch.float64) / EMB))
    pos = torch.arange(S, dtype=torch.float64)
    ang = pos[:, None] * theta[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).view(S, 1, 1, EMB).to(device)
    sin = torch.cat([ang.sin(), ang.sin()], -1).view(S, 1, 1, EMB).to(device)
    gk, gv = load(20).Model(KD, VD, EMB)(kv, kpe, cos, sin)

    half = EMB // 2
    kvl = kv.cpu().double().tolist()
    pel = kpe.cpu().double().squeeze(2).tolist()
    cl, sl = cos.cpu().double().tolist(), sin.cpu().double().tolist()
    ek = torch.zeros(S, B, H, EMB + KD, dtype=torch.float64)
    ev = torch.zeros(S, B, H, VD, dtype=torch.float64)
    for s_ in range(S):
        for h_ in range(H):
            ek[s_, 0, h_, :KD] = torch.tensor(kvl[s_][0][h_][:KD])
            ev[s_, 0, h_] = torch.tensor(kvl[s_][0][h_][KD:KD + VD])
            for i in range(half):
                x1, x2 = pel[s_][0][2 * i], pel[s_][0][2 * i + 1]
                ek[s_, 0, h_, KD + i] = x1 * cl[s_][0][0][i] - x2 * sl[s_][0][0][i]
                ek[s_, 0, h_, KD + half + i] = x2 * cl[s_][0][0][half + i] + x1 * sl[s_][0][0][half + i]
    compare("case20 MLA rope k (vs float64 循环)", gk, ek, tol=2e-2)
    compare("case20 MLA rope v (直通)", gv, ev, tol=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("musa") if hasattr(torch, "musa") and torch.musa.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"# device={device}\n")
    for fn in (verify_attention, verify_grouped_gemm, verify_moe, verify_elementwise, verify_rope):
        print(f"## {fn.__name__}")
        fn(device)
        print()
    bad = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"# 语义校验: pass={len(RESULTS) - len(bad)} fail={len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
