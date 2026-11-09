"""Reference case for

aten::topk

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent stack: CheckpointFunction > Torch-Compiled Region: 3/0.
The GPU body attributed to this op is

  musa::dnn::bitonic_sort_kernel<float, long, true,
    BitonicOp<float, float, long, true, (SortMode)0>, 512, FastDivmodU32, true>

  values[t, :], indices[t, :] = topk(input[t, :], k=8, largest=True, sorted=True)

input is fp32 [num_tokens, num_experts]. values stay fp32. indices are
int64. dim is the expert axis (dim=1). largest=True matches the first
bool in the bitonic template. With training / autograd enabled the
Megatron router path uses sorted=True
(torch.topk(..., sorted=torch.is_grad_enabled())).

This is the plain aten::topk inside the compiled router region. The
sibling FusedTopkScoreFunction / TE fused_topk kernel fuses the sigmoid
score with top-k and is a different launch.

Shape evidence from the same trace and the 1188B script:
  num_experts = 320 (--num-experts 320). The bitonic CapN=512 fits one
  expert row per block.
  M = 16384 = 131072 / CP 8. grid.x is exactly the token count.
  k = 8 (--moe-router-topk 8).
  Router dtype is fp32 (--moe-router-dtype fp32), so the sort key is float.

Launch recorded on MTT S5000:
  grid (16384, 1, 1), block (256, 1, 1), shared memory 6668 bytes.
  128 launches (forward top-k runs twice under full recompute).
  Duration 1482.928..1623.729 us (median ~1565 us).
"""

from __future__ import annotations

import torch


M = 16384
NUM_EXPERTS = 320
TOPK = 8


def reference_topk(
    inp: torch.Tensor, k: int = TOPK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise top-k by a full stable descending sort, then take the first k.

    Returns (values, indices) with values descending along dim 1, the same
    contract as torch.topk(..., dim=1, largest=True, sorted=True).
    """
    if inp.ndim != 2:
        raise ValueError(f"expected 2D input, got {tuple(inp.shape)}")
    if not (1 <= k <= inp.shape[1]):
        raise ValueError(f"k={k} out of range for N={inp.shape[1]}")
    # Stable full sort is the CPU-checkable golden; the device kernel is a
    # bitonic partial sort with the same largest/sorted contract.
    values, indices = torch.sort(inp, dim=1, descending=True, stable=True)
    return values[:, :k].contiguous(), indices[:, :k].contiguous()


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # fp32 router scores. randn avoids ties so bitonic and stable sort agree
    # on indices as well as values.
    inp = torch.randn(M, NUM_EXPERTS, generator=g, dtype=torch.float32)
    return inp.to(device)


def _check_small() -> None:
    inp = torch.tensor(
        [
            [0.1, 0.5, 0.3, 0.2],
            [4.0, 1.0, 3.0, 2.0],
        ],
        dtype=torch.float32,
    )
    values, indices = reference_topk(inp, k=2)
    expect_v = torch.tensor([[0.5, 0.3], [4.0, 3.0]])
    expect_i = torch.tensor([[1, 2], [0, 2]])
    if not torch.equal(values, expect_v) or not torch.equal(indices, expect_i):
        raise SystemExit(f"small topk mismatch: values={values} indices={indices}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    inp = make_inputs(seed=seed, device=device)
    golden_v, golden_i = reference_topk(inp, k=TOPK)
    # On MUSA this is the muDNN bitonic topk selected by aten::topk.
    got_v, got_i = torch.topk(inp, k=TOPK, dim=1, largest=True, sorted=True)

    val_diff = (got_v - golden_v).abs()
    max_abs = val_diff.max().item()
    idx_mismatch = int((got_i != golden_i).sum().item())
    print(
        f"device={device} input={tuple(inp.shape)} {inp.dtype} "
        f"values={tuple(got_v.shape)} {got_v.dtype} "
        f"indices={tuple(got_i.shape)} {got_i.dtype}"
    )
    print(f"max_abs={max_abs:.6e} idx_mismatch={idx_mismatch}")
    if max_abs != 0 or idx_mismatch != 0:
        raise SystemExit("aten::topk disagrees with the stable-sort reference")
    print("reference check passed")


if __name__ == "__main__":
    _check_small()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
