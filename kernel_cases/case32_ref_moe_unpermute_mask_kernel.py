"""Reference case for

deep_ep::intranode_ace::moe_unpermute_mask_kernel_tme<__mt_bfloat16, 8, true>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: FusedDispatchBackward. Dispatch forward copied each token into
its expert rows and copied the fp32 routing probs alongside. This backward
is the matching unpermute: sum the hidden gradients from those rows, and
gather the permuted prob gradients back onto the token.

  src = row_id_map[e, t]
  src < 0                 -> skip; grad_probs[t, e] = 0
  otherwise               -> grad_x[t] += grad_in[src]
                             grad_probs[t, e] = prob_grad[src]

The hidden sum is fp32 and stored bf16. It is not multiplied by the prob
gradient and not divided by topk. prob_grad and grad_probs are fp32.
row_id_map is int32, shape [num_experts, num_tokens].

The template 8 is the bf16 vector width. block 128 covers 128 * 8 = 1024
hidden elements, so grid.x = 8192 / 1024 = 8. grid.y is the token count.
The bool selects this prob-gather specialization. The sibling
<__mt_bfloat16, 8, false> is FusedCombine and only reduces the hidden rows.

Local tokens are the CP slice of the 131072 sequence:
  M = 131072 / 8 = 16384, hidden = 8192, num_experts = 320, topk = 8.
With that routing the permuted buffer has 16384 * 8 = 131072 rows.

Launch recorded on MTT S5000:
  grid (8, 16384, 1), block (128, 1, 1), shared memory 1836 bytes.
  64 launches (one dispatch backward per layer).
  Duration 1200.087..1250.607 us.
"""

from __future__ import annotations

import torch


M = 16384
HIDDEN = 8192
NUM_EXPERTS = 320
TOPK = 8


def reference_unpermute_mask(
    grad_in: torch.Tensor, row_id_map: torch.Tensor, prob_grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum grad_in rows selected by row_id_map and gather prob_grad.

    Returns the fp32 hidden sum, the bf16 stored grad_x, and grad_probs
    laid out [num_tokens, num_experts].
    """
    if row_id_map.ndim != 2 or grad_in.ndim != 2:
        raise ValueError(
            f"shapes grad_in {tuple(grad_in.shape)} row_id_map {tuple(row_id_map.shape)}"
        )
    experts, tokens = row_id_map.shape
    if prob_grad.ndim != 1 or prob_grad.shape[0] != grad_in.shape[0]:
        raise ValueError(
            f"prob_grad {tuple(prob_grad.shape)} does not match grad_in {tuple(grad_in.shape)}"
        )
    if int(row_id_map.max().item()) >= grad_in.shape[0]:
        raise ValueError("row_id_map points past grad_in")
    acc = torch.zeros(tokens, grad_in.shape[1], dtype=torch.float32, device=grad_in.device)
    gathered = torch.zeros(tokens, experts, dtype=torch.float32, device=grad_in.device)
    token_ids = torch.arange(tokens, device=grad_in.device)
    for expert in range(experts):
        src_row = row_id_map[expert]
        mask = src_row >= 0
        if not bool(mask.any()):
            continue
        rows = src_row[mask]
        acc.index_add_(0, token_ids[mask], grad_in.index_select(0, rows).float())
        gathered[mask, expert] = prob_grad.index_select(0, rows).float()
    return acc, acc.to(grad_in.dtype), gathered


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


def _check_sum_and_prob_grad() -> None:
    grad_in = torch.tensor([[4, 1], [4, 1], [7, 0]], dtype=torch.bfloat16)
    row_id_map = torch.tensor(
        [
            [0, -1],
            [-1, 2],
            [1, -1],
            [-1, -1],
        ],
        dtype=torch.int32,
    )
    prob_grad = torch.tensor([10.0, 20.0, 30.0])
    acc, stored, gathered = reference_unpermute_mask(grad_in, row_id_map, prob_grad)
    expect_acc = torch.tensor([[8.0, 2.0], [7.0, 0.0]])
    expect_prob = torch.tensor(
        [
            [10.0, 0.0, 20.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
        ]
    )
    if not torch.equal(acc, expect_acc) or not torch.equal(stored, expect_acc.to(torch.bfloat16)):
        raise SystemExit("hidden grad is not the unweighted sum of the source rows")
    if not torch.equal(gathered, expect_prob):
        raise SystemExit("prob grad was not gathered from the source row")


def run_case(device: str = "cpu") -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    num_out = M * TOPK
    grad_in = torch.randn(num_out, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    prob_grad = torch.randn(num_out, generator=g, dtype=torch.float32).to(device)
    row_id_map = make_row_id_map().to(device)
    acc, stored, gathered = reference_unpermute_mask(grad_in, row_id_map, prob_grad)
    max_abs = 0.0
    for token in (0, M // 2, M - 1):
        manual = torch.zeros(HIDDEN, dtype=torch.float32, device=grad_in.device)
        for expert in range(NUM_EXPERTS):
            src_row = int(row_id_map[expert, token].item())
            if src_row < 0:
                if gathered[token, expert].item() != 0:
                    raise SystemExit(f"dropped expert {expert} of token {token} wrote a prob grad")
                continue
            manual += grad_in[src_row].float()
            if gathered[token, expert].item() != prob_grad[src_row].item():
                raise SystemExit(f"prob grad mismatch at token {token} expert {expert}")
        max_abs = max(max_abs, (acc[token] - manual).abs().max().item())
        if not torch.equal(stored[token], acc[token].to(torch.bfloat16)):
            raise SystemExit(f"bf16 store disagrees with the fp32 sum at token {token}")
    print(
        f"device={device} grad_in={tuple(grad_in.shape)} {grad_in.dtype} "
        f"grad_x={tuple(stored.shape)} {stored.dtype} "
        f"grad_probs={tuple(gathered.shape)} {gathered.dtype}"
    )
    print(f"max_abs={max_abs:.6e}")
    if stored.shape != (M, HIDDEN) or gathered.shape != (M, NUM_EXPERTS) or max_abs > 1e-4:
        raise SystemExit("unpermute disagrees with the explicit row sum")
    print("reference check passed")


if __name__ == "__main__":
    _check_sum_and_prob_grad()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
