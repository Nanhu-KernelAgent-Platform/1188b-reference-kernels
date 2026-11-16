import torch
import musa_index_select_ext as _ext

def kernel_function(k_ag, v_ag, index):
    return (_ext.index_select_bf16(k_ag, index),
            _ext.index_select_bf16(v_ag, index))
