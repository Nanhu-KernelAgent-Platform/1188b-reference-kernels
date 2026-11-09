"""Reference case for

deep_ep::intranode_ace::moe_unpermute_mask_kernel_tme<__mt_bfloat16, 8, false>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: FusedCombine. Combine's forward unpermute restores each token
from its expert rows. Routing weights were already applied upstream in the
expert MLP, so this kernel only sums the hidden rows. The template bool is
with_permuted_probs; false does not read or write probs. The sibling
<__mt_bfloat16, 8, true> is FusedDispatch Backward and also gathers the
fp32 prob gradients.

  src = row_id_map[e, t]
  src < 0                 -> skip
  otherwise               -> grad_x[t] += grad_in[src]

The hidden sum is fp32 and stored bf16. It is not multiplied by a routing
weight and not divided by topk. row_id_map is int32, shape
[num_experts, num_tokens].

The template 8 is the bf16 vector width. block 128 covers 128 * 8 = 1024
hidden elements, so grid.x = 8192 / 1024 = 8. grid.y is the token count.

Local tokens are the CP slice of the 131072 sequence:
  M = 131072 / 8 = 16384, hidden = 8192, num_experts = 320, topk = 8.
With that routing the permuted buffer has 16384 * 8 = 131072 rows.

Launch recorded on MTT S5000:
  grid (8, 16384, 1), block (128, 1, 1), shared memory 1836 bytes.
  128 launches (combine forward runs twice under full recompute).
  Duration 1157.527..1237.247 us (median ~1202 us).
"""

from __future__ import annotations

import torch


M = 16384
HIDDEN = 8192
NUM_EXPERTS = 320
TOPK = 8


def reference_unpermute_mask(
    grad_in: torch.Tensor, row_id_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum grad_in rows selected by row_id_map.

    Returns the fp32 hidden sum and the bf16 stored grad_x.
    """
    if row_id_map.ndim != 2 or grad_in.ndim != 2:
        raise ValueError(
            f"shapes grad_in {tuple(grad_in.shape)} row_id_map {tuple(row_id_map.shape)}"
        )
    experts, tokens = row_id_map.shape
    if int(row_id_map.max().item()) >= grad_in.shape[0]:
        raise ValueError("row_id_map points past grad_in")
    acc = torch.zeros(tokens, grad_in.shape[1], dtype=torch.float32, device=grad_in.device)
    token_ids = torch.arange(tokens, device=grad_in.device)
    for expert in range(experts):
        src_row = row_id_map[expert]
        mask = src_row >= 0
        if not bool(mask.any()):
            continue
        rows = src_row[mask]
        acc.index_add_(0, token_ids[mask], grad_in.index_select(0, rows).float())
    return acc, acc.to(grad_in.dtype)


def make_row_id_map(
    num_tokens: int = M,
    num_experts: int = NUM_EXPERTS,
    topk: int = TOPK,
) -> torch.Tensor:
    """Expert-major int32 map. Token t goes to experts (t + k) mod num_experts."""
    choice = (torch.arange(num_tokens)[:, None] + torch.arange(topk)) % num_experts
    selected = torch.zeros(num_experts, num_tokens, dtype=torch.bool)
    selected.scatter_(0, choice.t(), True)
    counts = selected.sum(dim=1)
    offset = torch.cumsum(counts, 0) - counts
    within = selected.long().cumsum(dim=1) - 1
    empty = torch.full((num_experts, num_tokens), -1, dtype=torch.int64)
    return torch.where(selected, within + offset[:, None], empty).to(torch.int32)


def _check_unweighted_sum() -> None:
    grad_in = torch.tensor([[4, 1], [4, 1], [7, 0]], dtype=torch.bfloat16)
    # token 0 -> rows 0 and 1, token 1 -> row 2.
    row_id_map = torch.tensor(
        [
            [0, -1],
            [-1, 2],
            [1, -1],
            [-1, -1],
        ],
        dtype=torch.int32,
    )
    acc, stored = reference_unpermute_mask(grad_in, row_id_map)
    expect = torch.tensor([[8.0, 2.0], [7.0, 0.0]])
    if not torch.equal(acc, expect) or not torch.equal(stored, expect.to(torch.bfloat16)):
        raise SystemExit(f"hidden grad is not the unweighted sum: {acc}")


def run_case(device: str = "cpu") -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    num_out = M * TOPK
    grad_in = torch.randn(num_out, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    row_id_map = make_row_id_map().to(device)
    acc, stored = reference_unpermute_mask(grad_in, row_id_map)
    max_abs = 0.0
    for token in (0, M // 2, M - 1):
        manual = torch.zeros(HIDDEN, dtype=torch.float32, device=grad_in.device)
        for expert in range(NUM_EXPERTS):
            src_row = int(row_id_map[expert, token].item())
            if src_row < 0:
                continue
            manual += grad_in[src_row].float()
        max_abs = max(max_abs, (acc[token] - manual).abs().max().item())
        if not torch.equal(stored[token], acc[token].to(torch.bfloat16)):
            raise SystemExit(f"bf16 store disagrees with the fp32 sum at token {token}")
    print(
        f"device={device} grad_in={tuple(grad_in.shape)} {grad_in.dtype} "
        f"grad_x={tuple(stored.shape)} {stored.dtype}"
    )
    print(f"max_abs={max_abs:.6e}")
    if stored.shape != (M, HIDDEN) or max_abs > 1e-4:
        raise SystemExit("unpermute disagrees with the explicit row sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_unweighted_sum()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
