"""Reference case for

permute_with_mask_map_trans<__mt_bfloat16, float, long, false>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: _moe_unpermute_mask_map Backward (TransformerEngine mask-map
unpermute, merging_probs is None).

This is the TE muDNN launch in permutation_mask.mu. The bool template is
with_permuted_probs: false does not read probs and does not write
permuted_probs. The float in the template is still P_Dtype because the
Python wrapper passes torch.empty(0) (default fp32) into moe_permute_mask,
which selects the high-precision-probs entry point even when probs are
unused. IdxDtype is long. The _trans specialization is chosen when
row_id_map.shape[0] == num_experts (expert-major map).

  row_id_map[e, t] == -1  -> skip
  otherwise               -> output[row_id_map[e, t]] = input[t]

input / output are bf16. No scale. row_id_map is int64 [num_experts,
num_tokens]. Same scatter math as the deep_ep permute false kernel; this
file tracks the TE launch and the variable token count of this parent.

Shape evidence from the same trace:
  hidden = 8192 (smem = hidden * sizeof(bf16) = 16384, profiler reports
  16512 with a 128-byte pad).
  num_experts = 320, topk = 8 (1188B MoE). Without drop, num_out = M * 8.
  grid.x = num_tokens varies across the 64 launches: 80820..99179,
  median 87663. A full median-sized bf16 output is about 11 GiB, so the
  self-check uses M = 16384 (the CP-local token count used by the other
  MoE references) with the same map layout. Launch numbers below still
  quote the traced grids.

Launch recorded on MTT S5000:
  grid (num_tokens, 1, 1), block (32, 1, 1), shared memory 16512 bytes.
  64 launches (one unpermute backward per layer call).
  Duration 3035.457..3910.263 us (median ~3313 us).
"""

from __future__ import annotations

import torch


# Runnable self-check size. Traced grids: 80820..99179, median 87663.
M = 16384
HIDDEN = 8192
NUM_EXPERTS = 320
TOPK = 8


def reference_permute_with_mask_map_trans(
    inp: torch.Tensor, row_id_map: torch.Tensor
) -> torch.Tensor:
    """Scatter inp[token] to output[row_id_map[expert, token]] when the id is not -1."""
    if row_id_map.ndim != 2 or inp.ndim != 2 or row_id_map.shape[1] != inp.shape[0]:
        raise ValueError(
            f"shapes inp {tuple(inp.shape)} row_id_map {tuple(row_id_map.shape)}"
        )
    valid = row_id_map >= 0
    if not bool(valid.any()):
        return inp.new_empty((0, inp.shape[1]))
    dest = row_id_map[valid]
    num_out = int(dest.max().item()) + 1
    token = torch.arange(inp.shape[0], device=inp.device).view(1, -1).expand_as(row_id_map)[valid]
    out = inp.new_zeros((num_out, inp.shape[1]))
    out[dest] = inp[token]
    return out


def make_row_id_map(
    num_tokens: int = M,
    num_experts: int = NUM_EXPERTS,
    topk: int = TOPK,
) -> torch.Tensor:
    """Expert-major int64 map. Token t is routed to experts (t + k) mod num_experts."""
    choice = (torch.arange(num_tokens)[:, None] + torch.arange(topk)) % num_experts
    selected = torch.zeros(num_experts, num_tokens, dtype=torch.bool)
    selected.scatter_(0, choice.t(), True)
    counts = selected.sum(dim=1)
    offset = torch.cumsum(counts, 0) - counts
    within = selected.long().cumsum(dim=1) - 1
    empty = torch.full((num_experts, num_tokens), -1, dtype=torch.int64)
    return torch.where(selected, within + offset[:, None], empty)


def _check_scatter_no_probs() -> None:
    inp = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4) + 1
    # token 0 -> experts 0 and 2, token 1 -> expert 1. Expert 3 is dropped.
    row_id_map = torch.tensor(
        [
            [0, -1],
            [-1, 2],
            [1, -1],
            [-1, -1],
        ],
        dtype=torch.int64,
    )
    out = reference_permute_with_mask_map_trans(inp, row_id_map)
    if out.shape != (3, 4):
        raise SystemExit(f"unexpected output shape {tuple(out.shape)}")
    if not torch.equal(out[0], inp[0]) or not torch.equal(out[1], inp[0]) or not torch.equal(out[2], inp[1]):
        raise SystemExit("scatter did not copy the source token")


def run_case(device: str = "cpu") -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    inp = torch.randn(M, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    row_id_map = make_row_id_map().to(device)
    out = reference_permute_with_mask_map_trans(inp, row_id_map)
    max_abs = 0.0
    nvalid = 0
    for token in (0, M // 2, M - 1):
        dest = row_id_map[:, token]
        dest = dest[dest >= 0]
        nvalid += int(dest.numel())
        err = (out[dest].float() - inp[token].float()).abs().max().item()
        max_abs = max(max_abs, err)
    num_out = M * TOPK
    print(
        f"device={device} inp={tuple(inp.shape)} {inp.dtype} "
        f"out={tuple(out.shape)} {out.dtype} copied_rows={nvalid}"
    )
    print(f"max_abs={max_abs:.6e}")
    if out.shape != (num_out, HIDDEN) or row_id_map[TOPK, 0].item() != -1 or max_abs != 0:
        raise SystemExit("permute_with_mask_map_trans disagrees with the source token")
    print("reference check passed")


if __name__ == "__main__":
    _check_scatter_no_probs()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
