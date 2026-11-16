# 1188B 真实算子优化 kernel 关键信息总结

目前有5个1188B 训练热点算子的 MUSA 优化 kernel。每个子目录含且仅含 4 个源文件：
`binding.cpp`（pybind 绑定）/ `kernel.mu`（MUSA 设备核）/ `kernel.py`（`kernel_function` 入口）/ `setup.py`（torch_musa 扩展编译）。

## 性能与精度（验收实测）

- 形状口径：真实 profiler 形状（seq 131072 / CP 8 / TP 8 / 每卡 16384 tokens；MoE 有效行数 R=86,016）。
- 性能口径：isolated（25 warm + 50 reps，逐次 sync + L2 flush，中位数），8×MTT S5000。
- 精度口径：`allclose(atol=rtol=1e-5)`（`.float()` 后比较，形状先核同），附逐位等状态。

| case | 算子（真实训练 kernel） | reference (ms) | **本 kernel (ms)** | vs profiler 平均 | vs reference | 精度@1e-5 |
|---|---|---|---|---|---|---|
| case06_cp_seq_chunk_index_select | triton_poi_fused_index_select_0（3.64ms） | 12.9356 | **2.0448** | **1.78x** | 6.33x | ✅（逐位等） |
| case12_moe_row_id_map_pass2 | _row_id_map_pass_2_kernel（4.24ms） | 0.9719 | **0.4580** | **9.26x** | 2.12x | ✅（int64 逐位等） |
| case15_moe_unpermute_mask_fwd | moe_unpermute_mask<...,8>（3.27ms） | 16.0839 | **1.3131** | **2.49x** | 12.25x | ✅（max\|diff\|≤1.19e-07） |
| case17_moe_permute_with_mask_map | permute_with_mask_map<...,true>（3.11ms） | 9.1297 | **2.5681** | **1.21x** | 3.56x | ✅（bf16/fp32 位等比） |
| case20_mla_rope_fwd_kv | rotary_fwd_kv_kernel（2.09ms） | 3.3719 | **2.0304** | **1.03x** | 1.66x | ✅（逐位等） |

## 形状通用性

- case12 / case17：T/E/H/R 全部运行时推导（R = max(row_id_map)+1），无形状硬编码；
- case15 / case20：按输入张量运行维度取形；
- case06：纯位拷贝 kernel，天然形状无关。

## 构建与使用

```bash
# 每个 case 目录下（需 torch_musa 环境，MTT S5000 已验证）：
python3 setup.py build_ext --inplace
```

```python
import sys
sys.path.insert(0, 'kernel_results/case17_moe_permute_with_mask_map')
from kernel import kernel_function   # (x, row_id_map, probs) -> (out[R,H], permuted_probs[R])
```
