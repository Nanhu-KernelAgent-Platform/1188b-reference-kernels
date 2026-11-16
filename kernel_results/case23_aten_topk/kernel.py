"""Row-wise top-k (largest, sorted) for fp32 router scores on MUSA.

Replaces the aten::topk call in the Megatron router path:
    values[t, :], indices[t, :] = topk(input[t, :], k=8, largest=True, sorted=True)
Contract matches torch.topk(dim=1, largest=True, sorted=True) exactly, including
the stable tie-break (equal values -> smaller expert index first).
"""
import torch
import topk_musa

MAX_K = 8


def kernel_function(input, k=8, dim=1, largest=True, sorted=True):
    if not largest:
        raise ValueError("only largest=True is supported")
    if dim not in (1, -1):
        raise ValueError("only dim=1 is supported")
    if input.dim() != 2:
        raise ValueError(f"expected 2D input, got {input.dim()}D")
    n = int(input.shape[1])
    k = min(int(k), n)
    if k > MAX_K:
        raise ValueError(f"k={k} exceeds the {MAX_K} supported by this kernel")
    return topk_musa.topk_forward(input, k)
