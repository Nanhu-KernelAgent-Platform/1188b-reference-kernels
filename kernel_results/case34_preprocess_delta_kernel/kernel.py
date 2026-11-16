import torch
import torch_musa
import preprocess_delta_ext

def kernel_function(out, dout):
    return preprocess_delta_ext.preprocess_delta(out, dout)
