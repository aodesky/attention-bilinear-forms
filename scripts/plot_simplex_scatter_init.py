#!/usr/bin/env python3
"""Weights of the heads of a pretrained model on Delta^3.

The heads are drawn with the type markers of the cluster figures.

The script also reports the weights of the heads of the same
architecture randomly initialized, for comparison in the text; they
are not plotted, since all of them coincide with the midpoint of
the edge joining the antisymmetric and hyperbolic vertices to
within 1e-3.  The initialization is not reimplemented here: the
model is instantiated from its configuration with
AutoModelForCausalLM.from_config, which runs the library's own
_init_weights, so the heads are initialized exactly as they would
be at the start of training.  For gpt2 this means each entry of
the c_attn matrix holding W_Q and W_K is an independent sample
from N(0, initializer_range^2), with initializer_range = 0.02;
only the c_proj matrices receive the additional
1/sqrt(2 n_layer) residual rescaling, and those do not enter the
bilinear form.

For each head the bilinear form L = W_K^T W_Q is restricted to the
reduced subspace im(W_K^T) + im(W_Q^T) and its weights (a,b,c,d)
are computed as in the paper.

Usage:
    python plot_simplex_scatter_init.py               # gpt2
    python plot_simplex_scatter_init.py --model gpt2-large
    python plot_simplex_scatter_init.py --seed 1

Saves figures/qk_geometry/abcd_simplex_trained_init.{pdf,png}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from transformers import AutoConfig, AutoModelForCausalLM

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qk_geometry_diagnostics import iter_qk_heads, load_model
from cluster_head_spectra import CLUSTER_NAMES, marker_kwargs
from plot_simplex_scatter import load_data

MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
}

# projected tetrahedron vertices, as in plot_simplex_faces.py
VA = np.array([0.42, 1.00])   # a = 1
VB = np.array([0.00, 0.12])   # b = 1
VC = np.array([0.78, 0.00])   # c = 1
VD = np.array([1.00, 0.48])   # d = 1


def head_weights(model, model_id):
    """(a, b, c, d) for every head of the given model."""
    out = []
    for hw in iter_qk_heads(model, model_id):
        Wq = hw.Wq.to(torch.float64).numpy()
        Wk = hw.Wk.to(torch.float64).numpy()
        Q = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
        L = (Wk @ Q).T @ (Wq @ Q)
        S = 0.5 * (L + L.T)
        T = 0.5 * (L - L.T)
        nL2 = (L**2).sum()
        ev = np.linalg.eigvalsh(S)
        alpha = np.sort(ev[ev > 0])[::-1]
        beta = np.sort(-ev[ev < 0])[::-1]
        m = max(len(alpha), len(beta))
        alpha = np.pad(alpha, (0, m - len(alpha)))
        beta = np.pad(beta, (0, m - len(beta)))
        u = alpha - beta
        out.append(((T**2).sum() / nL2,
                    2 * alpha @ beta / nL2,
                    (u[u > 0]**2).sum() / nL2,
                    (u[u < 0]**2).sum() / nL2))
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    model_id = MODELS[args.model]

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    W = head_weights(model, model_id)
    a, b, c, d = W.T
    print(f"{args.model}: {len(W)} randomly initialized heads (seed {args.seed})")
    print(f"  a: median {np.median(a):.4f}   b: median {np.median(b):.4f}")
    print(f"  c: median {np.median(c):.4f}   d: median {np.median(d):.4f}")
    print(f"  a - b: median {np.median(a - b):+.4f}")
    print(f"  min(c,d)/max(c,d): median "
          f"{np.median(np.minimum(c, d) / np.maximum(c, d)):.4f}")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    ax.add_patch(Polygon([VB, VC, VD], closed=True,
                         facecolor="#eeeeee", edgecolor="none", zorder=0))
    for x, y in [(VA, VB), (VA, VC), (VA, VD), (VB, VC), (VC, VD)]:
        ax.plot([x[0], y[0]], [x[1], y[1]], color="black",
                linewidth=1.1, zorder=1)
    ax.plot([VB[0], VD[0]], [VB[1], VD[1]], color="black",
            linewidth=0.9, linestyle="--", dashes=(4, 3), zorder=1)
    for v in (VA, VB, VC, VD):
        ax.plot(*v, marker="o", markersize=4, color="black", zorder=3)
    ax.text(VA[0], VA[1] + 0.03, r"$a = 1$: antisymmetric",
            ha="center", va="bottom", fontsize=11)
    ax.text(VB[0] - 0.02, VB[1], r"$b = 1$: hyperbolic",
            ha="right", va="center", fontsize=11)
    ax.text(VC[0], VC[1] - 0.04, r"$c = 1$: positive semidefinite",
            ha="center", va="top", fontsize=11)
    ax.text(VD[0] + 0.02, VD[1], r"$d = 1$: negative semidefinite",
            ha="left", va="center", fontsize=11)

    def embed(a, b, c, d):
        return (np.outer(a, VA) + np.outer(b, VB)
                + np.outer(c, VC) + np.outer(d, VD))

    labels, bt, ct, dt, st = load_data()
    at = 1 - st
    Pt = embed(at, st * bt, st * ct, st * dt)
    for cl in (1, 2, 3):
        m = labels == cl
        ax.scatter(Pt[m, 0], Pt[m, 1], **marker_kwargs(cl, s=30),
                   label=f"{CLUSTER_NAMES[cl - 1]} ($n={m.sum()}$)",
                   zorder=2)

    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    pdf = OUT / "abcd_simplex_trained_init.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "abcd_simplex_trained_init.png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
