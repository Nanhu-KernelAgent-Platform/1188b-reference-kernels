# 20 个真实算子的运行 / 基准结果（纯 torch，未使用 KernelAgent）

**环境**：MTT S5000（MUSA，`torch 2.7.1`，8 卡 / 单卡 85.8 GB），单卡、`torch.no_grad()`，CUDA/MUSA event 计时。
**输入**：全部是 `model_config.json` 里的真实 per-rank shape（tp=1、cp=1、micro-batch=1）。
**脚本**：`bench_cases.py`（基准）、`verify_semantics.py`（语义交叉校验）。

---

## 1. 结论先说

| | |
|---|---|
| 运行是否通过 | **20/20 PASS**（输出 dtype/shape 与声明一致、无 NaN/Inf、同输入两次运行相对差 <1e-2；仅 unpermute 因 `index_add` 原子累加有 ~9e-4 的非确定性差异） |
| 语义是否正确 | **18/18 交叉校验 PASS**（SDPA / autograd / 朴素 python 双重循环 / fp64 对照） |
| 本机实测 roofline | 纯访存 ~**1.35 TB/s**；bf16 GEMM 大 M 时 ~**370 TFLOP/s**（单 group 用例实测 316–370） |
| 最大的性能缺口 | 40 组 × M≈102 的 **skinny grouped GEMM 只有 76–96 TFLOP/s（约为峰值的 21–26%）**，且被绑在 ~960 GB/s 的权重流带宽上；scattered 访存算子（row_id_map 91、unpermute 104 GB/s）离带宽天花板还有 8–13 倍 |

---

## 2. 基准结果总表

`median/min` 为 50 次（大 kernel 自适应 ≥3 次）重复的中位/最小耗时；`TFLOP/s`、`GB/s` 按该 kernel 一次调用的
最少 FLOPs / 显存搬运量折算（估算口径见 `bench_cases.py` 的 `WORK` 表）。

| id | case | **测试形状（输入 → 输出，per-rank）** | gate | median ms | min ms | TFLOP/s | GB/s | in MB | out MB | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mla_flash_attn_bwd_causal | q,k`[4096,1,128,192]` v,o,do`[4096,1,128,128]` → dq,dk`[4096,1,128,192]` dv`[4096,1,128,128]` | PASS | 10.436 | 10.183 | 202.3 | – | 768.0 | 512.0 | 3.379 |
| 2 | mla_flash_attn_fwd_causal | q,k`[4096,1,128,192]` v`[4096,1,128,128]` → o`[4096,1,128,128]` | PASS | 2.960 | 2.926 | 234.8 | – | 512.0 | 128.0 | 2.127 |
| 3 | moe_grouped_gemm_fc1_fwd | x`[4096,8192]` w`[40,5120,8192]` → y`[4096,5120]` | PASS | 3.611 | 3.563 | 96.42 | 972.2 | 3264.0 | 40.0 | 3.516 |
| 4 | moe_grouped_gemm_fc2_fwd | x`[4096,2560]` w`[40,8192,2560]` → y`[4096,8192]` | PASS | 2.251 | 2.216 | 77.54 | 797.0 | 1620.0 | 64.0 | 2.082 |
| 5 | moe_grouped_gemm_fc2_wgrad | dy`[4096,8192]` x`[4096,2560]` → dW`[40,8192,2560]` | PASS | 4.573 | 4.530 | 37.93 | 400.0 | 84.0 | 1600.0 | 12.582 |
| 6 | cp_seq_chunk_index_select | k_ag`[2,2048,1,128,192]` v_ag`[2,2048,1,128,128]` idx`[2]` → 同形状两个 | PASS | 3.244 | 3.242 | – | 207.0 | 320.0 | 320.0 | 3.75 |
| 7 | moe_grouped_gemm_fc1_dgrad | dy`[4096,5120]` w`[40,5120,8192]` → dx`[4096,8192]` | PASS | 3.695 | 3.615 | 95.04 | 958.3 | 3240.0 | 64.0 | 3.914 |
| 8 | moe_grouped_gemm_fc2_dgrad | dy`[4096,8192]` w`[40,8192,2560]` → dx`[4096,2560]` | PASS | 2.277 | 2.263 | 75.93 | 780.5 | 1664.0 | 20.0 | 1.789 |
| 9 | moe_grouped_gemm_fc1_wgrad | dy`[4096,5120]` x`[4096,8192]` → dW`[40,5120,8192]` | PASS | 8.349 | 8.309 | 41.35 | 414.0 | 104.0 | 3200.0 | 25.102 |
| 10 | grouped_gemm_single_group_fc1_fwd | x`[4096,8192]` w`[1,5120,8192]` → y`[4096,5120]` | PASS | 0.973 | 0.947 | 362.8 | 203.7 | 144.0 | 40.0 | 6.469 |
| 11 | router_bwd_cast_bf16_to_fp32 | x`[4096,8192]` bf16 → x`[4096,8192]` fp32 | PASS | 0.186 | 0.183 | – | 1098.0 | 64.0 | 128.0 | 0.563 |
| 12 | moe_row_id_map_pass2 | routing_map`[4096,320]` bool → row_id`[320,4096]` i64 + `[4096,320]` i64 | PASS | 0.254 | 0.244 | – | 91.4 | 1.2 | 20.0 | 0.301 |
| 13 | grouped_gemm_single_group_fc1_dgrad | dy`[4096,5120]` w`[1,5120,8192]` → dx`[4096,8192]` | PASS | 0.947 | 0.928 | 370.14 | 207.8 | 120.0 | 64.0 | 0.617 |
| 14 | grouped_gemm_single_group_fc2_fwd | x`[4096,2560]` w`[1,8192,2560]` → y`[4096,8192]` | PASS | 0.559 | 0.543 | 316.13 | 239.3 | 60.0 | 64.0 | 0.559 |
| 15 | moe_unpermute_mask_fwd | pin`[32768,8192]` map`[320,4096]` i64 → out`[4096,8192]` | PASS\* | 5.855 | 5.821 | – | 103.8 | 522.0 | 64.0 | 2.281 |
| 16 | aten_add_bf16 | a,b`[4096,8192]` bf16 → y`[4096,8192]` | PASS | 0.151 | 0.150 | – | 1345.0 | 128.0 | 64.0 | 0.625 |
| 17 | moe_permute_with_mask_map | x`[4096,8192]` map`[4096,320]` i64 probs`[4096,320]` fp32 → y`[32768,8192]` pp`[32768]` | PASS | 3.636 | 3.584 | – | 168.5 | 79.0 | 512.1 | 4.077 |
| 18 | moe_permute_bwd_unpermute_mask | dpin`[32768,8192]` dpp`[32768]` map`[320,4096]` i64 → dact`[4096,8192]` dprobs`[4096,320]` | PASS\* | 5.727 | 5.686 | – | 108.1 | 522.1 | 69.0 | 3.156 |
| 19 | grouped_gemm_single_group_fc2_dgrad | dy`[4096,8192]` w`[1,8192,2560]` → dx`[4096,2560]` | PASS | 0.496 | 0.490 | 350.92 | 265.6 | 104.0 | 20.0 | 0.275 |
| 20 | mla_rope_fwd_kv | kv`[4096,1,128,256]` kpe`[4096,1,1,64]` cos,sin`[4096,1,1,64]` → key`[4096,1,128,192]` value`[4096,1,128,128]` | PASS | 0.979 | 0.967 | – | 624.7 | 258.5 | 320.0 | 1.752 |

说明：所有 shape 都是 **per-rank**（tp=1 / cp=1 / micro-batch=1，seq 4096、hidden 8192、320 experts 分到 EP=8 后每卡 40 个、topk=8 → 每 expert ≈102 行、permute 展开 4096×8=32768 行）；`w[G,N,K]` 的 G=40 表示 40 个 local expert 各一份权重（单 group 用量例是 G=1）；dtype 除标注外均为 bf16（softmax / probs / router 与 cast 输出为 fp32，索引为 int64）。

`PASS*`：仅"两次运行"有 ~9e-4 的相对差，来自 fp32 `index_add_` 的原子累加顺序（非确定性但数值正确）。

机读版：`bench_results.csv`。复跑：

```bash
python3 bench_cases.py              # 全 20 例，写 bench_results.csv
python3 bench_cases.py 1 2 17       # 只跑指定 case
python3 verify_semantics.py         # 语义交叉校验（18 项）
```

---

## 3. 性能解读

### 3.1 本机 roofline（用同一批用例标定）
* **纯访存天花板 ≈ 1.35 TB/s**：`aten::add`（192 MB / 0.151 ms）= 1345 GB/s，`bf16→fp32 cast` = 1098 GB/s。
* **bf16 GEMM 实用峰值 ≈ 370 TFLOP/s**：单 group、M=4096 的 GEMM 实测 316–370 TFLOP/s
  （case13 370、case10 363、case19 351、case14 316），可视为达到硬件上限。

### 3.2 两类 GEMM 形态差了 4 倍
| 形态 | 用例 | 实测 | 瓶颈 |
|---|---|---|---|
| 单 group / 大 M（M=4096） | 10/13/14/19 | **316–370 TFLOP/s** | 计算受限，接近峰值 |
| 40 组 / skinny M（M≈102/expert） | 3/4/7/8 | **76–96 TFLOP/s** | 权重流带宽受限：40×84 MB 权重 / 3.6 ms ≈ 972 GB/s（带宽天花板的 72%），算力只用到 26% |
| 反向 wgrad | 5/9 | **38–41 TFLOP/s** | 输出是 dW 本体（1.6–3.2 GB 写），且 K=102 极短，tile 严重欠填充 |

⇒ **真实训练里 MoE 的 grouped GEMM（每 expert ~102 行）是最大的算力浪费点**：同样 FLOPs 换成大 M 能快 4 倍。
这是后续"针对真实 shape 生成专用 kernel"最该盯的目标（对 40 组做 persistent/多 expert 融合，或把 m_splits 补齐到对齐值）。

### 3.3 访存类算子：scattered 访问是主要缺口
| 算子 | 实测带宽 | 相对 1.35 TB/s |
|---|---|---|
| aten::add | 1345 GB/s | 100%（连续访问，达标） |
| bf16→fp32 cast | 1098 GB/s | 81% |
| MLA rope(kv) | 625 GB/s | 46% |
| index_select（CP 重排，恒等拷贝） | 207 GB/s | 15% |
| permute（scatter 到 32768 行） | 169 GB/s | 12% |
| unpermute（gather + 近似原子累加） | 104 GB/s | 8% |
| row_id_map（两趟 block scan） | 91 GB/s | 7%（且本身只有 20 MB 数据，属延迟受限） |

⇒ MoE 的 permute/unpermute 族（合计在 CSV 里也是前列热点）是**访存效率最低**的一块，
真实训练里每个 MoE 层都要跑两遍，值得专门做 kernel（合并 row_id_map 生成、向量化行搬移、去掉冗余的两次遍历）。

### 3.4 attention
SDPA 前向 2.96 ms（235 TFLOP/s）、反向 10.44 ms（202 TFLOP/s），相对 GEMM 峰值约 55–63%，
在 seq=4096、causal、192/128 头维的组合下属于正常水平；要做 MLA 专用 kernel 的话，这个就是 baseline。

---

## 4. 语义校验（`verify_semantics.py`）

| 校验项 | 对照实现 | 结果 |
|---|---|---|
| case02 attn fwd | SDPA vs fp64 显式 softmax | PASS（rel 2.4e-3） |
| case01 attn bwd | SDPA-autograd vs 分块解析反向 | PASS（dq/dk/dv rel ≤7.4e-3） |
| case03/10 grouped GEMM fwd | 逐组 `F.linear` | PASS（rel 2.7e-3） |
| case07/13 grouped GEMM dgrad | autograd | PASS（rel 2.0e-3） |
| case09 grouped GEMM wgrad | autograd | PASS（rel 1.8e-3） |
| case12 row_id_map | 朴素双重循环（int64 完全相等） | PASS |
| case17 permute（act / probs） | 朴素双重循环（逐元素完全相等） | PASS |
| case15 / case18 unpermute（含 probs grad） | 朴素双重循环（逐元素完全相等） | PASS |
| case11 cast / case16 add / case06 index_select | fp64 精确值 | PASS（完全相等） |
| case20 MLA rope（k / v） | fp64 逐 token/head 循环 | PASS（rel 1.9e-3） |

---

## 5. 与 CSV 里厂商 kernel 耗时的关系（不要直接对比）

CSV 的「平均耗时」在**量纲上**与本次实测的差距极不一致，说明它不能当 kernel 时延用：

| CSV 行 | 该项 op | CSV 平均耗时 | 本次 torch 实测 | 比值 |
|---|---|---|---|---|
| 16 | aten::add | 0.39 ms | 0.151 ms | 2.6× |
| 11 | bf16→fp32 cast | 2.07 ms | 0.186 ms | 11× |
| 17 | permute | 3.11 ms | 3.64 ms | 0.85× |
| 15 | unpermute | 3.27 ms | 5.86 ms | 0.56× |
| 6 | index_select | 3.64 ms | 3.24 ms | 1.12× |
| 20 | MLA rope | 2.09 ms | 0.98 ms | 2.1× |
| 2 | attn fwd | **228 ms** | **2.96 ms** | **77×** |
| 1 | attn bwd | **662 ms** | **10.4 ms** | **63×** |
| 3 | fc1 grouped GEMM | 1.02 ms/次 | 3.61 ms（40 组整体） | 见下 |

* 访存类算子两边同量级（0.6–2.6×），**互相印证了 shape 假定是对的**；
* attention 差 60–80 倍、grouped GEMM 若按 per-expert 解读也差 10 倍以上，这在同一块卡上不可能
  ⇒ CSV 的「调用次数/平均耗时」统计口径与"一次真实 kernel 时延"不一致（profiler 窗口、per-rank 聚合、
  或 profiling 期间被 launch-blocking/限频影响），**需要重新抓一次干净的 profiler 才能对齐**；
* 因此在拿到干净数据之前，**本文件的 torch 实测值就作为"当前真实算子的基准性能"**。

---

## 6. 复现与文件

```bash
cd ../real_1188B_top20
python3 bench_cases.py                 # 20 例跑测 + 写 bench_results.csv
python3 verify_semantics.py            # 18 项语义交叉校验
```

* `bench_cases.py` — 基准：建真实 shape 输入 → gate（dtype/shape/有限性/复现性）→ event 计时 → 折算 TFLOP/s、GB/s
* `verify_semantics.py` — 语义交叉校验（SDPA / autograd / 朴素循环 / fp64）
* `bench_results.csv` — 本次跑测的机读结果
* `cases/case01..case20.py` — 20 个 reference 用例（attention 默认走 SDPA，`impl="explicit"` 保留显式实现）
