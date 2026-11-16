import torch
import moe_permute_rt_ext

def kernel_function(x, row_id_map, probs):
    # (out [R,H] bf16, permuted_probs [R] fp32)，R = max(row_id_map)+1
    return moe_permute_rt_ext.permute_mask_map(x, row_id_map, probs)
