import torch
import _cast_fp32_to_bf16_rne_musa


def kernel_function(inp):
    if inp.dtype != torch.float32:
        inp = inp.float()
    if not inp.is_contiguous():
        inp = inp.contiguous()
    return _cast_fp32_to_bf16_rne_musa.cast_fp32_to_bf16_rne(inp)
