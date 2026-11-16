import torch
import rmsnorm_musa


def kernel_function(x, gamma, eps=1e-6):
    return rmsnorm_musa.rmsnorm_forward(x, gamma, eps)
