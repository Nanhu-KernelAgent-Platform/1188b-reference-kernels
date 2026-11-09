"""Reference case for

deep_ep::intranode_ace::moe_permute_mask_kernel_tme<__mt_bfloat16, false>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: FusedCombineBackward. Combine's forward unpermute already
applied the routing weights, so this backward only scatters the token
gradient back to the expert rows. The template bool is with_permuted_probs.
false does not read probs and does not write permuted_probs. The sibling
<__mt_bfloat16, true> is FusedDispatch and does copy probs.

  row_id_map[e, t] == -1  -> skip
  otherwise               -> output[row_id_map[e, t]] = grad[t]

grad and output are bf16. No scale. row_id_map is int64, shape
[num_experts, num_tokens], the same layout as the mask-permute map
(expert-major destination rows).

Local tokens are the CP slice of the 131072 sequence:
  M = 131072 / 8 = 16384, hidden = 8192, num_experts = 320, topk = 8.
One block per token, so the grid is the token count. The 32-thread block
walks the expert map; TME moves the hidden row.

Launch recorded on MTT S5000:
  grid (16384, 1, 1), block (32, 1, 1), shared memory 17792 bytes.
  17792 = 8192 * 2 + 320 * 4 + 128: the bf16 row, the fp32 prob row
  reserved for the true specialization, and a 128-byte pad.
  64 launches (one combine backward per layer; the forward dispatch
  kernel runs twice because of full recompute).
  Duration 1508.809..1558.169 us.
"""

from __future__ import annotations

import torch


M = 16384
HIDDEN = 8192
NUM_EXPERTS = 320
TOPK = 8


def reference_permute_mask(grad: torch.Tensor, row_id_map: torch.Tensor) -> torch.Tensor:
    """Scatter grad[token] to output[row_id_map[expert, token]] when the id is not -1."""
    if row_id_map.ndim != 2 or grad.ndim != 2 or row_id_map.shape[1] != grad.shape[0]:
        raise ValueError(
            f"shapes grad {tuple(grad.shape)} row_id_map {tuple(row_id_map.shape)}"
        )
    valid = row_id_map >= 0
    if not bool(valid.any()):
        return grad.new_empty((0, grad.shape[1]))
    dest = row_id_map[valid]
    num_out = int(dest.max().item()) + 1
    token = torch.arange(grad.shape[0], device=grad.device).view(1, -1).expand_as(row_id_map)[valid]
    out = grad.new_zeros((num_out, grad.shape[1]))
    out[dest] = grad[token]
    return out


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


def _check_scatter_ignores_probs() -> None:
    grad = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4) + 1
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
    probs = torch.full((2, 4), 0.25)
    out = reference_permute_mask(grad, row_id_map)
    if out.shape != (3, 4):
        raise SystemExit(f"unexpected output shape {tuple(out.shape)}")
    if not torch.equal(out[0], grad[0]) or not torch.equal(out[1], grad[0]) or not torch.equal(out[2], grad[1]):
        raise SystemExit("scatter did not copy the source token")
    scaled = reference_permute_mask(grad * probs[:, :1].to(grad.dtype), row_id_map)
    if torch.equal(scaled, out):
        raise SystemExit("prob scale was not observable on the input")
    del probs


def run_case(device: str = "cpu") -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    grad = torch.randn(M, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    row_id_map = make_row_id_map().to(device)
    out = reference_permute_mask(grad, row_id_map)
    max_abs = 0.0
    nvalid = 0
    for token in (0, M // 2, M - 1):
        dest = row_id_map[:, token]
        dest = dest[dest >= 0]
        nvalid += int(dest.numel())
        err = (out[dest].float() - grad[token].float()).abs().max().item()
        max_abs = max(max_abs, err)
    num_out = M * TOPK
    print(
        f"device={device} grad={tuple(grad.shape)} {grad.dtype} "
        f"out={tuple(out.shape)} {out.dtype} copied_rows={nvalid}"
    )
    print(f"max_abs={max_abs:.6e}")
    if out.shape != (num_out, HIDDEN) or row_id_map[TOPK, 0].item() != -1 or max_abs != 0:
        raise SystemExit("permute scatter disagrees with the source token")
    print("reference check passed")


if __name__ == "__main__":
    _check_scatter_ignores_probs()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
