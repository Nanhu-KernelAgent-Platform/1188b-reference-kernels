# kernel_tests — 优化 kernel 独立复测目录

为 `../kernel_results/` 下各优化 kernel 提供**单算子可独立复现**的精度 + 性能验证。
每个 `test_caseNN.py` 自包含：加载被测程序（按 `KERNEL_CASES_DIR` → 仓库平级
`../kernel_cases/caseNN_*.py` 顺序解析）→ 调用对应优化 kernel →
按标准口径校验精度 → 计时 reference 与 kernel → 打印判定并落盘样本。

## 目录内容

| 文件 / 目录 | 说明 |
|---|---|
| `test_case06.py` | case06 `cp_seq_chunk_index_select` 测试（e-5 容差口径） |
| `test_case12.py` | case12 `moe_row_id_map_pass2` 测试（int64 逐位等口径） |
| `test_case15.py` | case15 `moe_unpermute_mask_fwd` 测试（e-5 容差口径） |
| `test_case17.py` | case17 `moe_permute_with_mask_map` 测试（bf16/fp32 位等比口径） |
| `test_case20.py` | case20 `mla_rope_fwd_kv` 测试（e-5 容差口径） |
| `results/` | 运行时自动创建；存放逐次计时样本与判定 JSON（属产物，不入库） |

> 被测程序（eager PyTorch reference + 真实形状输入构造器，固定 seed 可复现）**不随本目录**：
> 仓库内统一由平级目录 `../kernel_cases/caseNN_*.py` 提供（`kernel_tests/` 与
> `kernel_cases/` 平级，克隆后默认命中）；在本目录单独使用时，可用
> `KERNEL_CASES_DIR` 环境变量显式指定 case 文件所在目录。

## 测试输入

- 被测优化 kernel 来自 `../kernel_results/<case>/`（脚本内带路径回退：先查 `../kernel_results/`，再查 `../`）。
- 输入张量由 case 文件（`../kernel_cases/caseNN_*.py`）按固定 seed 生成，为真实 profiler 形状：
  seq 131072 / CP 8 / TP 8 / 每卡 16384 tokens；MoE 有效行数 R=86,016。
  各 case 输入：case06 `k_ag(2,65536,16,192)+v_ag(2,65536,16,128)+idx(2)`；
  case12 `routing_map(16384,320) bool`；case15 `pin(86016,8192) bf16 + row_id_map(320,16384) int64`；
  case17 `x(16384,8192) bf16 + row_id_map(16384,320) int64 + probs(16384,320) fp32`；
  case20 `kv(16384,1,128,256) + k_pos_emb(16384,1,1,64) + cos/sin(16384,1,1,64)`。

## 测试条件

- 硬件/栈：MTT S5000，PyTorch + torch_musa（`setup.py build_ext --inplace` 编译环境）。
- 精度口径：`allclose(atol=rtol=1e-5)`（输入经 `.float()` 后比较，形状先逐一核对）；case12/17 因语义为
  整数索引图 / 纯位拷贝，额外要求 `torch.equal` 逐位一致。
- 性能口径（isolated）：25 次 warmup + 50 次计时 reps；每次 rep 前 `synchronize` + 256MB L2 冲刷；
  Event 对计时，取 50 次**中位数**。
- 达标判定（VERDICT）：精度通过 **且** kernel 中位数快于 profiler 真实平均
  （case06 3.64 / case12 4.24 / case15 3.27 / case17 3.11 / case20 2.09 ms）。脚本以退出码 0=PASS、1=FAIL 返回。

## 测试步骤

```bash
cd kernel_tests
python3 test_case06.py        # 单算子复现（其余 4 个同理：test_case12/15/17/20）
for t in test_case06 test_case12 test_case15 test_case17 test_case20; do python3 $t.py; done   # 全量

# 若 case 文件不在 ../kernel_cases（例如只拉了 kernel_tests），显式指定目录：
KERNEL_CASES_DIR=/path/to/case_dir python3 test_case06.py
```

- 首次运行若对应 case 目录无已编译扩展，脚本自动执行 `setup.py build_ext --inplace`（需 torch_musa）。

## 测试输出

屏幕摘要（每算子一块）：输入形状、精度结果（eps@1e-5 / 逐位等 / max|diff|）、
reference 与 kernel 的中位耗时、**vs profiler 与 vs reference 加速比**、VERDICT。
同时写盘 `results/test_caseNN.json`：逐次 50 条计时样本 + 全部判定键值，可据此自行汇总性能表格。

## 预期结果（提交时实测）

| case | 精度@1e-5 | kernel (ms) | vs profiler | vs reference |
|---|---|---|---|---|
| case06 | ✅ 逐位等 | 2.045 | 1.78x | 6.33x |
| case12 | ✅ 逐位等 | 0.458 | 9.26x | 2.10x |
| case15 | ✅（max\|diff\|≤1.2e-07） | 1.304 | 2.51x | 12.32x |
| case17 | ✅ 位等比 | 2.560 | 1.22x | 3.58x |
| case20 | ✅ 逐位等 | 2.031 | 1.03x | 1.66x |

注：case15 的微小差异来自 reference 侧 fp32 `index_add_` 累加顺序的逐次非确定（1 个 bf16 LSB）；
case20 加速余量小系训练 kernel 已贴 HBM 带宽极限（约 1.16 TB/s）所致。不同机器/驱动下
绝对耗时会有正常漂移，但精度判定与"快于 profiler"结论应保持一致。
