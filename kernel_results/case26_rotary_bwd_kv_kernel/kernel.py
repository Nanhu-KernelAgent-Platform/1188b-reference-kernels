"""Drop-in entry point: forward(dk, dv, cos, sin) -> (d_kv, d_emb) on MUSA.

Thin wrapper over the compiled extension (src/kernel.mu + src/binding.cpp).
Tuned for MTT S5000: 512 threads/block, split k/v copy passes, 16 B vector
accesses, one (seq*batch) row per block.
"""
import torch
import rotary_bwd_kv_musa as _ext

_ext.set_config(512, 1, 16, 1, 0)  # nt, mode, vec bytes, rows/block, cache hint


def kernel_function(dk, dv, cos, sin):
    out = _ext.rotary_bwd_kv(dk, dv, cos, sin)
    return out[0], out[1]
