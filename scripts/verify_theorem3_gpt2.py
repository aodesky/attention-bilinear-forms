#!/usr/bin/env python3
"""
verify_theorem3_gpt2.py

Empirical verification of Theorem 3 ("Heads have balanced attention") for
every attention head in GPT-2 small.

Theorem 3 states: for any non-degenerate, non-big head H in a multi-head
biaffine attention block on residual stream R^D, the residual-stream
symmetric form

    S_H = (W_K^T W_Q + W_Q^T W_K) / 2  in R^{DxD}

has inertia (d_i, d_i, D - 2*d_i), where d_i is the head dimension.

For GPT-2 small: D = d_model = 768, d_i = head_dim = 64. The prediction is
inertia (64, 64, 640) on every head. We compute the inertia of S_H for all
n_layers * n_heads heads and check the count of (n_+, n_-, n_0) eigenvalues
against the prediction.

This script also reports the head-space symmetric-form inertia of

    S_head = (W_K W_Q^T + W_Q W_K^T) / 2  in R^{d_i x d_i},

which is a *different* matrix not directly constrained by Theorem 3.
Migliarini's "self-hating attention head" analysis of L1H5 reports
spectrum information for the head-space form, so we print head-space
inertia for the labelled head families and L1H5 in particular for
comparison.

Usage:
    python verify_theorem3_gpt2.py

Outputs:
    - prints per-head Theorem-3 check; expects 0 deviations
    - prints head-space inertia summary by published category

Dependencies: torch, transformers, numpy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM


# Published head labels for GPT-2 small.
# Sources cited in attention.tex: wang2022interpretability, olsson2022induction,
# zukowski2024inspected, migliarini2025selfhating.
INDUCTION_HEADS: set = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10), (3, 10)}
PREV_TOKEN_HEADS: set = {(2, 2), (4, 10), (4, 11)}
DUPLICATE_HEADS: set = {(0, 5), (1, 11), (3, 0), (4, 7)}
MONOSEM_OTHER_HEADS: set = {
    (1, 5), (2, 3), (2, 9), (3, 2), (3, 3), (4, 2),
    (6, 3), (6, 11), (7, 9), (8, 1),
}
SELF_HATING_HEAD: Tuple[int, int] = (1, 5)  # L1H5; Migliarini 2025.
# L1H5 deliberately appears in MONOSEM_OTHER_HEADS as well: it is both
# the self-hating head and one of the monosemantic heads of the sources.

# Relative tolerance for treating an eigenvalue as zero.
EIG_TOL_REL: float = 1e-7


def category_of(layer: int, head: int) -> str:
    """Return the published category label for the head (L, H), or 'other'."""
    key = (int(layer), int(head))
    if key in DUPLICATE_HEADS:
        return "duplicate"
    if key in PREV_TOKEN_HEADS:
        return "previous_token"
    if key in INDUCTION_HEADS:
        return "induction"
    if key in MONOSEM_OTHER_HEADS:
        return "monosem_other"
    return "other"


def extract_head_qk(model, layer: int, head: int, n_heads: int,
                    d_model: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (W_Q, W_K) for the given head from a GPT-2-style model.

    GPT-2's attention uses a fused linear layer ``c_attn`` of shape
    [3*d_model, d_model] when accessed as W @ x (Conv1D internally stores
    [in, out]; we transpose first). The three blocks along the row dimension
    are Q, K, V. Each block reshapes to (n_heads, head_dim, d_model).

    Returns:
        Wq_h: [head_dim, d_model]
        Wk_h: [head_dim, d_model]
    """
    block = model.transformer.h[layer]
    # c_attn.weight shape is [d_model, 3*d_model] (Conv1D's [in, out]); transpose
    # to the convention output = W @ input, giving [3*d_model, d_model].
    W = block.attn.c_attn.weight.detach().cpu().to(torch.float64).T
    Wq_all, Wk_all, _Wv_all = W.split(d_model, dim=0)
    Wq_layer = Wq_all.reshape(n_heads, head_dim, d_model)
    Wk_layer = Wk_all.reshape(n_heads, head_dim, d_model)
    return Wq_layer[head].numpy(), Wk_layer[head].numpy()


def inertia(evals: np.ndarray, tol_rel: float = EIG_TOL_REL) -> Tuple[int, int, int]:
    """
    Inertia of a real symmetric matrix from its eigenvalues.

    Returns (n_+, n_-, n_0): counts of positive, negative, and approximately
    zero eigenvalues. The threshold for "zero" is tol_rel * max(|eigenvalue|).
    Returns (0, 0, len(evals)) if all eigenvalues are zero.
    """
    if evals.size == 0:
        return (0, 0, 0)
    max_abs = float(np.max(np.abs(evals)))
    if max_abs == 0.0:
        return (0, 0, int(evals.size))
    tol = tol_rel * max_abs
    n_pos = int((evals > tol).sum())
    n_neg = int((evals < -tol).sum())
    n_zero = int(evals.size - n_pos - n_neg)
    return (n_pos, n_neg, n_zero)


def residual_stream_inertia(Wq_h: np.ndarray, Wk_h: np.ndarray) -> Tuple[int, int, int]:
    """
    Inertia of S_H = (W_K^T W_Q + W_Q^T W_K) / 2  on R^D.

    Wq_h, Wk_h have shape [d_head, d_model]. The product W_K^T W_Q lives in
    R^{D x D}, of rank at most d_head; its symmetrization has at most 2*d_head
    nonzero eigenvalues.
    """
    L = Wk_h.T @ Wq_h  # [D, D]
    S = 0.5 * (L + L.T)
    evals = np.linalg.eigvalsh(S)
    return inertia(evals)


def head_space_inertia(Wq_h: np.ndarray, Wk_h: np.ndarray) -> Tuple[int, int, int]:
    """
    Inertia of S_head = (W_K W_Q^T + W_Q W_K^T) / 2  on R^{d_head}.

    This is the matrix Migliarini analyzes in his L1H5 writeup. It is *not*
    directly constrained by Theorem 3, which is about the residual-stream
    symmetrization. Returns (n_+, n_-, n_0) for the d_head x d_head matrix.
    """
    M = Wk_h @ Wq_h.T  # [d_head, d_head]
    S = 0.5 * (M + M.T)
    evals = np.linalg.eigvalsh(S)
    return inertia(evals)


def main() -> None:
    print("Loading gpt2 ...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    n_layers = int(cfg.n_layer)
    n_heads = int(cfg.n_head)
    d_model = int(cfg.n_embd)
    head_dim = d_model // n_heads

    predicted_inertia = (head_dim, head_dim, d_model - 2 * head_dim)
    print(f"  n_layers={n_layers}  n_heads={n_heads}  d_model={d_model}  head_dim={head_dim}")
    print(f"  Theorem 3 prediction for each head: (n_+, n_-, n_0) = {predicted_inertia}")
    print()

    # Theorem 3 verification on the residual stream.
    deviations: List[Tuple[int, int, Tuple[int, int, int]]] = []
    head_space_rows: List[Dict] = []

    for layer in range(n_layers):
        for head in range(n_heads):
            Wq_h, Wk_h = extract_head_qk(model, layer, head, n_heads, d_model, head_dim)
            rs = residual_stream_inertia(Wq_h, Wk_h)
            if rs != predicted_inertia:
                deviations.append((layer, head, rs))
            hs = head_space_inertia(Wq_h, Wk_h)
            head_space_rows.append({
                "layer": layer,
                "head": head,
                "category": category_of(layer, head),
                "rs_inertia": rs,
                "hs_n_pos": hs[0],
                "hs_n_neg": hs[1],
                "hs_n_zero": hs[2],
            })

    total = n_layers * n_heads
    print(f"=== Residual-stream Theorem-3 check ===")
    print(f"  Heads checked: {total}")
    print(f"  Heads matching {predicted_inertia}: {total - len(deviations)}")
    if deviations:
        print(f"  Deviations: {len(deviations)}")
        for (l, h, iner) in deviations[:20]:
            print(f"    L{l}H{h}: {iner}")
    else:
        print(f"  All {total} heads satisfy Theorem 3 exactly.")
    print()

    # Head-space inertia summary by category. This is not predicted by
    # Theorem 3; we report it because Migliarini's L1H5 claim ("33 of 64
    # negative eigenvalues") is about this matrix.
    print(f"=== Head-space inertia summary (not constrained by Theorem 3) ===")
    print(f"  S_head = (W_K W_Q^T + W_Q W_K^T)/2  has shape {head_dim} x {head_dim}.")
    print(f"  All entries below report mean and range of n_- = #negative eigvals (out of {head_dim}).")
    print()
    cat_order = ["duplicate", "previous_token", "induction", "monosem_other", "other"]
    print(f"  {'category':16s}  {'n':>4s}  {'mean n_-':>9s}  {'range':>13s}")
    for cat in cat_order:
        sub = [r["hs_n_neg"] for r in head_space_rows if r["category"] == cat]
        if not sub:
            continue
        print(f"  {cat:16s}  {len(sub):>4d}  {np.mean(sub):>9.2f}  "
              f"[{min(sub):>3d}, {max(sub):>3d}]")
    print()

    # Spotlight the self-hating head L1H5.
    layer_sh, head_sh = SELF_HATING_HEAD
    row = next(r for r in head_space_rows
               if r["layer"] == layer_sh and r["head"] == head_sh)
    print(f"=== L1H5 (Migliarini's self-hating head) ===")
    print(f"  Residual-stream inertia: {row['rs_inertia']} -> matches Theorem 3: "
          f"{row['rs_inertia'] == predicted_inertia}")
    print(f"  Head-space inertia: "
          f"(n_+, n_-, n_0) = ({row['hs_n_pos']}, {row['hs_n_neg']}, {row['hs_n_zero']})")


if __name__ == "__main__":
    main()
