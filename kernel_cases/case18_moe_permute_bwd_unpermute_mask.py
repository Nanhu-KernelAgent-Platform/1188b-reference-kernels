"""case18 — MoE unpermute with mask map + probs 反向（permute 的 backward）:
          moe_unpermute_mask<bf16,float,long,false,true,true,8>

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 18 行
kernel: moe_unpermute_mask<__mt_bfloat16, float, long, false, true, true, 8>
        模板参数（permutation_mask.mu:134）:
        Dtype, P_Dtype, IdxDtype, with_merging_probs=false, with_permuted_probs=true,
        trans_row_id_map=true, vlen=8
上层   : _moe_permute_mask_mapBackward（transformer_engine/pytorch/permutation.py:396 调
         tex.moe_unpermute_mask(..., permuted_probs_grad ...)）

真实形状（per-rank）:
  dperm_act  [32768, 8192]  bf16    # 对 permute 输出的梯度
  dpp        [32768]        fp32    # 对 permuted_probs 的梯度
  row_id_map [320, 4096]    int64
  ----
  dact       [4096, 8192]   bf16    # dact[t] = Σ_e dperm_act[row_id_map[e,t]]
  dprobs     [4096, 320]    fp32    # dprobs[t,e] = dpp[row_id_map[e,t]]，未选中处 = 0

与 case15 的唯一区别: with_permuted_probs=true ⇒ 额外产出 [num_tokens, num_experts] 的 probs 梯度。
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
    "id": 18,
    "name": "moe_permute_bwd_unpermute_mask",
    "kernel": "moe_unpermute_mask<__mt_bfloat16, float, long, false, true, true, 8>",
    "upstream": "_moe_permute_mask_mapBackward",
    "csv_row": 18,
    "source": "transformer_engine/pytorch/permutation.py:396 (moe_permute_mask backward); "
              "transformer_engine/musa/common/permutation/permutation_mask.mu:134",
    "dtype": "bf16 act / fp32 probs / int64 map",
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
    """返回 (dact [T,H] bf16, dprobs [T,E] fp32)。"""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, dperm_act: torch.Tensor, dpp: torch.Tensor, row_id_map: torch.Tensor):
        e, n = row_id_map.shape
        h = dperm_act.shape[1]
        valid = row_id_map >= 0                     # [E, T]
        e_idx, dst = valid.nonzero(as_tuple=True)   # expert 下标 / token 下标
        src = row_id_map[e_idx, dst]                # [num_selected]
        dact = torch.zeros(n, h, dtype=torch.float32, device=dperm_act.device)
        dact.index_add_(0, dst, dperm_act.index_select(0, src).float())
        dprobs = torch.zeros(n * e, dtype=torch.float32, device=dperm_act.device)
        dprobs.index_copy_(0, dst * e + e_idx, dpp.index_select(0, src))
        return dact.to(dperm_act.dtype), dprobs.view(n, e)


def get_init_inputs():
    return [NUM_EXPERTS]


def get_inputs():
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    idx = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :TOPK]
    rm.scatter_(1, idx, True)
    row_id_map = make_row_id_map_trans(rm)
    dperm_act = torch.randn(NUM_OUT_TOKENS, HID).to(torch.bfloat16)
    dpp = torch.randn(NUM_OUT_TOKENS, dtype=torch.float32)
    return [dperm_act, dpp, row_id_map]
