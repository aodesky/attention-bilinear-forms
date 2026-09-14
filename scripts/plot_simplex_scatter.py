#!/usr/bin/env python3
"""Scatter of the gpt2 heads on the archetype-weight simplices.

Figure 1 (bcd_simplex_scatter): the heads' symmetric parts on the
2-simplex {b + c + d = 1} in barycentric coordinates.
Uses the spectral weights of S/||S|| (equivalently (b,c,d)/(1-a)).

Figure 2 (abcd_simplex_scatter): the heads' full bilinear forms on
the 3-simplex {a + b + c + d = 1}, embedded in a regular tetrahedron,
with a = ||T||^2 / ||L||^2 taken from the diagnostics summary.

Marker coding matches the cluster figures (+ / o / - for Types
I+, II, I-).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
GEOM = PROJECT / "scripts" / "qk_geometry_results" / "gpt2" / "head_summary.csv"
OUT = PROJECT / "figures" / "qk_geometry"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_head_spectra import CLUSTER_NAMES, marker_kwargs

# Threshold of the zoomed sub-triangle {b >= ZOOM_B}; used by
# sample_gamma_spectra.py.
ZOOM_B = 0.7


def load_data():
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    keys = [tuple(k) for k in np.load(RESULTS / "gpt2_head_keys.npy")]

    _, raw = kmeans2(X, 3, minit="++", seed=0)
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum()
                    for r in X])
    order = sorted(range(3), key=lambda cl: -imb[raw == cl].mean())
    labels = np.empty_like(raw)
    for new, old in enumerate(order, start=1):
        labels[raw == old] = new

    b = np.empty(len(X))
    c = np.empty(len(X))
    d = np.empty(len(X))
    for i, r in enumerate(X):
        lp = np.sort(r[r > 0])[::-1]
        ln = np.sort(-r[r < 0])[::-1]
        m = max(lp.size, ln.size)
        lp = np.pad(lp, (0, m - lp.size))
        ln = np.pad(ln, (0, m - ln.size))
        u = lp - ln
        b[i] = 2 * lp @ ln
        c[i] = (u[u > 0] ** 2).sum()
        d[i] = (u[u < 0] ** 2).sum()

    df = pd.read_csv(GEOM)
    s_energy = {(int(r.layer), int(r.head)): r.S_energy_ratio
                for r in df.itertuples()}
    s = np.array([s_energy[k] for k in keys])
    return labels, b, c, d, s


def initialized_bcd(model_tag="gpt2", seed=0):
    """(b, c, d) of the symmetric parts of randomly initialized heads."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from qk_geometry_diagnostics import iter_qk_heads
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_tag)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    out = []
    for hw in iter_qk_heads(model, model_tag):
        Wq = hw.Wq.to(torch.float64).numpy()
        Wk = hw.Wk.to(torch.float64).numpy()
        Q = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
        L = (Wk @ Q).T @ (Wq @ Q)
        S = 0.5 * (L + L.T)
        ev = np.linalg.eigvalsh(S)
        nS2 = (ev**2).sum()
        alpha = np.sort(ev[ev > 0])[::-1]
        beta = np.sort(-ev[ev < 0])[::-1]
        m = max(len(alpha), len(beta))
        alpha = np.pad(alpha, (0, m - len(alpha)))
        beta = np.pad(beta, (0, m - len(beta)))
        u = alpha - beta
        out.append((2 * alpha @ beta / nS2,
                    (u[u > 0]**2).sum() / nS2,
                    (u[u < 0]**2).sum() / nS2))
    return np.array(out)


def draw_triangle(ax, Pb, Pc, Pd, fontsize=11, vertex_labels=True):
    ax.plot([Pb[0], Pc[0]], [Pb[1], Pc[1]], color="black", linewidth=1.0)
    ax.plot([Pb[0], Pd[0]], [Pb[1], Pd[1]], color="black", linewidth=1.0)
    ax.plot([Pc[0], Pd[0]], [Pc[1], Pd[1]], color="black", linewidth=0.9,
            linestyle="--", dashes=(4, 3))
    if vertex_labels:
        ax.text(Pb[0], Pb[1] + 0.03, r"$b = 1$", ha="center",
                va="bottom", fontsize=fontsize)
        ax.text(Pc[0] - 0.02, Pc[1] - 0.02, r"$c = 1$", ha="right",
                va="top", fontsize=fontsize)
        ax.text(Pd[0] + 0.02, Pd[1] - 0.02, r"$d = 1$", ha="left",
                va="top", fontsize=fontsize)
    ax.set_aspect("equal")
    ax.axis("off")


def bary(bb, cc, dd, Pb, Pc, Pd):
    x = bb * Pb[0] + cc * Pc[0] + dd * Pd[0]
    y = bb * Pb[1] + cc * Pc[1] + dd * Pd[1]
    return x, y


def figure_bcd(labels, b, c, d):
    Pb = (0.5, np.sqrt(3) / 2)
    Pc = (0.0, 0.0)
    Pd = (1.0, 0.0)

    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    draw_triangle(ax, Pb, Pc, Pd, fontsize=13, vertex_labels=False)
    ax.text(Pb[0], Pb[1] + 0.075, r"Type $\mathrm{II}$",
            ha="center", va="bottom", fontsize=13)
    ax.text(Pb[0], Pb[1] + 0.03, r"$b = 1$: hyperbolic",
            ha="center", va="bottom", fontsize=12)
    ax.text(Pc[0] - 0.02, Pc[1] - 0.03,
            r"$c = 1$: positive semidefinite",
            ha="left", va="top", fontsize=12)
    ax.text(Pd[0] + 0.02, Pd[1] - 0.03,
            r"$d = 1$: negative semidefinite",
            ha="right", va="top", fontsize=12)
    ax.text(0.205, 0.47,
            r"\begin{tabular}{c}Type $\mathrm{I}_+$ ($d = 0$)\\"
            r"hyperbolic $+$ positive semidef.\end{tabular}",
            ha="right", va="center", fontsize=12, rotation=60)
    ax.text(0.795, 0.47,
            r"\begin{tabular}{c}Type $\mathrm{I}_-$ ($c = 0$)\\"
            r"hyperbolic $+$ negative semidef.\end{tabular}",
            ha="left", va="center", fontsize=12, rotation=-60)
    for cl in (1, 2, 3):
        m = labels == cl
        x, y = bary(b[m], c[m], d[m], Pb, Pc, Pd)
        ax.scatter(x, y, **marker_kwargs(cl, s=42),
                   label=f"{CLUSTER_NAMES[cl - 1]} ($n={m.sum()}$)",
                   zorder=2)
    ax.legend(fontsize=11, frameon=True, loc="upper left")

    fig.tight_layout()
    pdf = OUT / "bcd_simplex_scatter.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "bcd_simplex_scatter.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


def figure_abcd(labels, b, c, d, s):
    # full-form weights: a = 1 - s, and (b, c, d) scaled by s
    a_w = 1 - s
    b_w, c_w, d_w = s * b, s * c, s * d

    # Same projected tetrahedron as the schematic figure
    # (plot_simplex_faces.py); barycentric combination commutes with
    # the linear projection, so plotting in the projected vertices is
    # exact.
    A = np.array([0.42, 1.00])   # a = 1 (apex)
    B = np.array([0.00, 0.12])   # b = 1
    C = np.array([0.78, 0.00])   # c = 1
    D = np.array([1.00, 0.48])   # d = 1 (back vertex)

    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.add_patch(Polygon([B, C, D], closed=True, facecolor="#eeeeee",
                         edgecolor="none", zorder=0))
    for p, q in [(A, B), (A, C), (A, D), (B, C), (C, D)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="black",
                linewidth=1.1, zorder=1)
    ax.plot([B[0], D[0]], [B[1], D[1]], color="black", linewidth=0.9,
            linestyle="--", dashes=(4, 3), zorder=1)
    for p in (A, B, C, D):
        ax.plot(*p, marker="o", markersize=4, color="black", zorder=3)
    ax.text(A[0], A[1] + 0.03, r"$a = 1$: antisymmetric",
            ha="center", va="bottom", fontsize=11)
    ax.text(B[0] - 0.02, B[1], r"$b = 1$: hyperbolic",
            ha="right", va="center", fontsize=11)
    ax.text(C[0], C[1] - 0.04, r"$c = 1$: positive semidefinite",
            ha="center", va="top", fontsize=11)
    ax.text(D[0] + 0.02, D[1], r"$d = 1$: negative semidefinite",
            ha="left", va="center", fontsize=11)

    P = (np.outer(a_w, A) + np.outer(b_w, B)
         + np.outer(c_w, C) + np.outer(d_w, D))
    for cl in (1, 2, 3):
        m = labels == cl
        ax.scatter(P[m, 0], P[m, 1], **marker_kwargs(cl, s=30),
                   label=f"{CLUSTER_NAMES[cl - 1]} ($n={m.sum()}$)",
                   zorder=2)

    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    pdf = OUT / "abcd_simplex_scatter.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "abcd_simplex_scatter.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


def main() -> None:
    labels, b, c, d, s = load_data()
    figure_bcd(labels, b, c, d)
    figure_abcd(labels, b, c, d, s)


if __name__ == "__main__":
    main()
