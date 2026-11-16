import torch
import mla_rope_kv_ext

def kernel_function(kv, k_pos_emb, cos, sin):
    return mla_rope_kv_ext.mla_rope_kv(kv, k_pos_emb, cos, sin)
