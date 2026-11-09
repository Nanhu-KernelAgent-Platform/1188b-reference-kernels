"""Reference case for

deep_ep::intranode_ace::moe_permute_mask_kernel_tme<__mt_bfloat16, true>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: FusedDispatch. The template bool is with_permuted_probs. true
scatters the token activation and also copies the fp32 routing probability
onto the matching permuted row. The sibling <__mt_bfloat16, false> is
FusedCombine Backward and only scatters the hidden gradient.

  row_id_map[e, t] == -1  -> skip
  otherwise               -> output[row_id_map[e, t]] = input[t]
                             permuted_probs[row_id_map[e, t]] = probs[t, e]

input / output are bf16. probs is [num_tokens, num_experts] fp32.
permuted_probs is [num_out_tokens] fp32. The hidden row is not scaled by
the probability; the expert MLP applies the weight later. row_id_map is
int64, shape [num_experts, num_tokens], expert-major destination rows
(same layout as the TE mask-permute map).

Local tokens are the CP slice of the 131072 sequence:
  M = 131072 / 8 = 16384, hidden = 8192, num_experts = 320, topk = 8.
With that routing the permuted buffer has 16384 * 8 = 131072 rows.
One block per token, so the grid is the token count. The 32-thread block
walks the expert map; TME moves the hidden row and thread 0 writes the
prob.

Launch recorded on MTT S5000:
  grid (16384, 1, 1), block (32, 1, 1), shared memory 17792 bytes.
  17792 = 8192 * 2 + 320 * 4 + 128: the bf16 row, the fp32 prob row, and
  a 128-byte pad.
  128 launches (forward dispatch runs twice under full recompute).
  Duration 1565.489..1631.810 us (median ~1603 us).
"""

from __future__ import annotations

import torch


M = 16384
HIDDEN = 8192
NUM_EXPERTS = 320
TOPK = 8


def reference_permute_mask_with_probs(
    inp: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter inp[token] and probs[token, expert] to the destination rows."""
    if row_id_map.ndim != 2 or inp.ndim != 2 or row_id_map.shape[1] != inp.shape[0]:
        raise ValueError(
            f"shapes inp {tuple(inp.shape)} row_id_map {tuple(row_id_map.shape)}"
        )
    experts, tokens = row_id_map.shape
    if probs.shape != (tokens, experts):
        raise ValueError(f"probs {tuple(probs.shape)} expected {(tokens, experts)}")
    valid = row_id_map >= 0
    if not bool(valid.any()):
        return inp.new_empty((0, inp.shape[1])), probs.new_empty((0,))
    dest = row_id_map[valid]
    num_out = int(dest.max().item()) + 1
    token = torch.arange(tokens, device=inp.device).view(1, -1).expand_as(row_id_map)[valid]
    expert = torch.arange(experts, device=inp.device).view(-1, 1).expand_as(row_id_map)[valid]
    out = inp.new_zeros((num_out, inp.shape[1]))
    out[dest] = inp[token]
    permuted_probs = probs.new_zeros((num_out,))
    permuted_probs[dest] = probs[token, expert]
    return out, permuted_probs


def make_row_id_map(
    num_tokens: int = M,
    num_experts: int = NUM_EXPERTS,
    topk: int = TOPK,
) -> torch.Tensor:
    """Expert-major map. Token t is routed to experts (t + k) mod num_experts, k < topk."""
    choice = (torch.arange(num_tokens)[:, None] + torch.arange(topk)) % num_experts
    selected = torch.zeros(num_experts, num_tokens, dtype=torch.bool)
    selected.scatter_(0, choice.t(), True)
    counts = selected.sum(dim=1)
    offset = torch.cumsum(counts, 0) - counts
    within = selected.long().cumsum(dim=1) - 1
    empty = torch.full((num_experts, num_tokens), -1, dtype=torch.int64)
    return torch.where(selected, within + offset[:, None], empty)


def make_probs(
    row_id_map: torch.Tensor,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """fp32 probs on the routed (token, expert) pairs; zeros elsewhere."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    experts, tokens = row_id_map.shape
    probs = torch.zeros(tokens, experts, dtype=torch.float32)
    valid = row_id_map >= 0
    token = torch.arange(tokens).view(1, -1).expand_as(row_id_map)[valid]
    expert = torch.arange(experts).view(-1, 1).expand_as(row_id_map)[valid]
    raw = torch.rand(int(valid.sum().item()), generator=g, dtype=torch.float32) + 0.1
    probs[token, expert] = raw
    return probs.to(device)


def _check_scatter_and_probs() -> None:
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
    probs = torch.tensor(
        [
            [10.0, 0.0, 20.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    out, permuted_probs = reference_permute_mask_with_probs(inp, row_id_map, probs)
    if out.shape != (3, 4) or permuted_probs.shape != (3,):
        raise SystemExit(f"unexpected shapes out {tuple(out.shape)} probs {tuple(permuted_probs.shape)}")
    if not torch.equal(out[0], inp[0]) or not torch.equal(out[1], inp[0]) or not torch.equal(out[2], inp[1]):
        raise SystemExit("scatter did not copy the source token")
    if not torch.equal(permuted_probs, torch.tensor([10.0, 20.0, 30.0])):
        raise SystemExit(f"permuted probs mismatch: {permuted_probs}")
    # Hidden is not scaled by the routing weight inside this kernel.
    if torch.equal(out[0], (inp[0].float() * 10).to(inp.dtype)):
        raise SystemExit("hidden row was unexpectedly scaled by the prob")


def run_case(device: str = "cpu") -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    inp = torch.randn(M, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    row_id_map = make_row_id_map().to(device)
    probs = make_probs(row_id_map.cpu(), seed=1, device=device)
    out, permuted_probs = reference_permute_mask_with_probs(inp, row_id_map, probs)
    max_abs = 0.0
    max_prob_abs = 0.0
    nvalid = 0
    for token in (0, M // 2, M - 1):
        dest = row_id_map[:, token]
        experts = torch.arange(NUM_EXPERTS, device=device)[dest >= 0]
        dest = dest[dest >= 0]
        nvalid += int(dest.numel())
        err = (out[dest].float() - inp[token].float()).abs().max().item()
        max_abs = max(max_abs, err)
        prob_err = (permuted_probs[dest] - probs[token, experts]).abs().max().item()
        max_prob_abs = max(max_prob_abs, prob_err)
    num_out = M * TOPK
    print(
        f"device={device} inp={tuple(inp.shape)} {inp.dtype} "
        f"out={tuple(out.shape)} {out.dtype} "
        f"permuted_probs={tuple(permuted_probs.shape)} {permuted_probs.dtype} "
        f"copied_rows={nvalid}"
    )
    print(f"max_abs={max_abs:.6e} max_prob_abs={max_prob_abs:.6e}")
    if (
        out.shape != (num_out, HIDDEN)
        or permuted_probs.shape != (num_out,)
        or row_id_map[TOPK, 0].item() != -1
        or max_abs != 0
        or max_prob_abs != 0
    ):
        raise SystemExit("permute-with-probs disagrees with the source token / prob")
    print("reference check passed")


if __name__ == "__main__":
    _check_scatter_and_probs()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
