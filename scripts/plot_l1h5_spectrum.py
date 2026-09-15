#!/usr/bin/env python3
"""Plot the nonzero spectrum of the residual-stream symmetric form of L1H5.

L1H5 is Migliarini's "self-hating" head in GPT-2 small. By Theorem 4 its
residual-stream symmetrization S_H = (W_K^T W_Q + W_Q^T W_K)/2 has inertia
(64, 64, 640); this script plots a histogram of the 128 nonzero
eigenvalues and verifies the claims of the paper's "Reconciling
Migliarini's count" paragraph: the inertia is (64, 64), and exactly 33
of the largest 64 eigenvalues by magnitude are negative, recovering
Migliarini's count "33 of 64".

Saves figures/qk_geometry/l1h5_spectrum.{pdf,png} (used in the
introduction of attention.tex).
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

LAYER, HEAD = 1, 5
EIG_TOL_REL = 1e-7

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading gpt2 ...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    n_heads, d_model = int(cfg.n_head), int(cfg.n_embd)
    head_dim = d_model // n_heads

    block = model.transformer.h[LAYER]
    W = block.attn.c_attn.weight.detach().cpu().to(torch.float64).T
    Wq_all, Wk_all, _ = W.split(d_model, dim=0)
    Wq = Wq_all.reshape(n_heads, head_dim, d_model)[HEAD].numpy()
    Wk = Wk_all.reshape(n_heads, head_dim, d_model)[HEAD].numpy()

    L = Wk.T @ Wq
    S = 0.5 * (L + L.T)
    evals = np.linalg.eigvalsh(S)

    tol = EIG_TOL_REL * float(np.max(np.abs(evals)))
    nonzero = evals[np.abs(evals) > tol]
    pos = nonzero[nonzero > 0]
    neg = nonzero[nonzero < 0]

    neg_energy_frac = float((neg**2).sum() / (nonzero**2).sum())
    top64 = nonzero[np.argsort(-np.abs(nonzero))][:head_dim]
    neg_in_top64 = int((top64 < 0).sum())

    print(f"nonzero eigenvalues: {nonzero.size}  (n_+={pos.size}, n_-={neg.size})")
    print(f"negative energy fraction: {neg_energy_frac:.4f}")
    print(f"negatives among top {head_dim} by magnitude: {neg_in_top64}")
    print(f"range: [{nonzero.min():.3f}, {nonzero.max():.3f}]")

    # Verify the claims of the "Reconciling Migliarini's count"
    # paragraph of the paper: the symmetrization has exactly
    # head_dim positive and head_dim negative eigenvalues, and
    # Migliarini's count "33 of 64" is recovered as the negative
    # count among the largest head_dim eigenvalues by magnitude.
    assert pos.size == head_dim and neg.size == head_dim, \
        f"inertia ({pos.size}, {neg.size}) != ({head_dim}, {head_dim})"
    assert neg_in_top64 == 33, \
        f"negatives among top {head_dim}: {neg_in_top64} != 33"
    print("verified: inertia (64, 64) and Migliarini's count 33 of 64")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(nonzero.min(), nonzero.max(), 41)
    ax.hist(nonzero, bins=bins, color="#cccccc", edgecolor="black",
            linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    # Center the zero line: symmetric x-limits padded past the extremes.
    xmax = 1.05 * float(np.max(np.abs(nonzero)))
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("count")

    fig.tight_layout()
    pdf = OUT / "l1h5_spectrum.pdf"
    png = OUT / "l1h5_spectrum.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
