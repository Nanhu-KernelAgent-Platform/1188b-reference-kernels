"""case15 — MoE unpermute with mask map（forward 方向，无 merging probs）:
          moe_unpermute_mask<bf16,float,long,false,false,true,8>

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 15 行
kernel: moe_unpermute_mask<__mt_bfloat16, float, long, false, false, true, 8>
        模板参数顺序（permutation_mask.mu:134）:
        Dtype, P_Dtype, IdxDtype, with_merging_probs=false, with_permuted_probs=false,
        trans_row_id_map=true, vlen=8
上层   : _moe_unpermute_mask_map（transformer_engine/pytorch/permutation.py:434）
Kernel : transformer_engine/musa/common/permutation/permutation_mask.mu:134-215

真实形状（per-rank）:
  in (permuted act) [32768, 8192]  bf16     # 32768 = 4096*topk，expert-order 的行排布
  row_id_map        [320, 4096]    int64    # trans 布局，-1 = 未选中，否则 = in 里的源行号
  ----
  out               [4096, 8192]   bf16     # out[t] = Σ_{e: row_id_map[e,t]>=0} in[row_id_map[e,t]]

说明: with_merging_probs=false ⇒ 不做 prob 加权（这两个 CSV 实例都是 false）；
      with_permuted_probs=false ⇒ 不产出 probs 反向梯度（对比 case18）。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
TOKENS = 4096
HID = 8192
NUM_EXPERTS = 320
TOPK = 8
NUM_OUT_TOKENS = TOKENS * TOPK

CASE = {
    "id": 15,
    "name": "moe_unpermute_mask_fwd",
    "kernel": "moe_unpermute_mask<__mt_bfloat16, float, long, false, false, true, 8>",
    "upstream": "_moe_unpermute_mask_map",
    "csv_row": 15,
    "source": "transformer_engine/pytorch/permutation.py:434; "
              "transformer_engine/musa/common/permutation/permutation_mask.mu:134",
    "dtype": "bf16 act / int64 map",
    "provenance": "形状由源码+配置直接给出（高）；模板参数解码=源码直读（高）",
    "backward": False,
}


def make_row_id_map_trans(routing_map):
    """routing_map [T,E] bool -> row_id_map [E,T] int64（trans，-1 = 未选中）。

    = 每个 expert 段内的序号 + 该 expert 段的全局起始行（各 expert token 数的前缀和）。
    """
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1                 # [T,E]
    counts = mask.sum(dim=0)                               # [E]
    offset = torch.cumsum(counts, dim=0) - counts          # [E]
    row = torch.where(mask > 0, within + offset.unsqueeze(0), torch.full_like(within, -1))
    return row.t().contiguous()



class Model(nn.Module):
    """out[t] = Σ_e in[row_id_map[e, t]]（只累加被选中的 expert）。"""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, pin: torch.Tensor, row_id_map: torch.Tensor) -> torch.Tensor:
        # 向量化：把所有 (expert, token) 有效项一次性 index_add，避免按 expert 循环
        valid = row_id_map >= 0                    # [E, T]
        src = row_id_map[valid]                    # [num_selected] in 里的源行
        dst = valid.nonzero(as_tuple=True)[1]       # 对应的 token 下标
        out = torch.zeros(row_id_map.shape[1], pin.shape[1], dtype=torch.float32, device=pin.device)
        out.index_add_(0, dst, pin.index_select(0, src).float())
        return out.to(pin.dtype)


def get_init_inputs():
    return [NUM_EXPERTS]


def get_inputs():
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    idx = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :TOPK]
    rm.scatter_(1, idx, True)
    row_id_map = make_row_id_map_trans(rm)
    pin = torch.randn(NUM_OUT_TOKENS, HID).to(torch.bfloat16)
    return [pin, row_id_map]
