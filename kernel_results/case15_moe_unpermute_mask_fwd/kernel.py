import torch
import moe_unpermute_ext

def kernel_function(pin, row_id_map):
    return moe_unpermute_ext.moe_unpermute(pin, row_id_map)
