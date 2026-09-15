#!/usr/bin/env python3
"""Fitted gamma archetypes of the three types of balanced bimodal forms.

Three panels in the layout of the cluster tail-histogram figure
(spectra_cluster_barycenters). For each cluster of the paper's k = 3
clustering of the gpt2 spectra (seed 0, renumbered by signed
asymmetry), the averaged spectrum of its ten most extreme members is
recomputed exactly as in cluster_head_spectra.py, its negative and
positive parts are fitted separately by gamma distributions with the
origin fixed at zero, and the two fitted densities are drawn. The
fitted mode and shape of each lobe are printed.

Saves figures/qk_geometry/type_schematics.{pdf,png}.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.vq import kmeans2

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT = PROJECT / "figures" / "qk_geometry"

N_EXTREME = 10

TITLES = [
    r"Type I$_+$ (positive-definite-like)",
    r"Type II (approximately symmetric spectrum)",
    r"Type I$_-$ (negative-definite-like)",
]


def extreme_averages():
    """The per-cluster averaged extreme spectra of the k = 3 figure."""
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    _, raw = kmeans2(X, 3, minit="++", seed=0)
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum()
                    for r in X])
    order = sorted(range(3), key=lambda c: -imb[raw == c].mean())
    labels = np.empty_like(raw)
    for new, old in enumerate(order, start=1):
        labels[raw == old] = new

    gmean = X.mean(axis=0)
    avgs = []
    for c in (1, 2, 3):
        idx = np.where(labels == c)[0]
        direction = X[idx].mean(axis=0) - gmean
        direction /= np.linalg.norm(direction)
        scores = (X[idx] - gmean) @ direction
        reps = idx[np.argsort(-scores)[:N_EXTREME]]
        avgs.append(X[reps].mean(axis=0))
    return avgs


def fit_lobe(samples):
    """Gamma fit (origin fixed at zero) to one side of a spectrum.
    Returns (mode, shape, scale) for the fitted density."""
    k, _, scale = stats.gamma.fit(samples, floc=0)
    return (k - 1.0) * scale, k, scale


def main() -> None:
    avgs = extreme_averages()

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 3.8))
    for ax, title, avg in zip(axes, TITLES, avgs):
        neg = -avg[avg < 0]
        pos = avg[avg > 0]
        lobes = []
        for samples, sign in [(neg, -1), (pos, +1)]:
            mode, k, scale = fit_lobe(samples)
            lobes.append((sign * mode, k, scale, sign))
            side = "negative" if sign < 0 else "positive"
            print(f"{title}: {side} lobe  mode = {sign * mode:+.4f}, "
                  f"shape k = {k:.2f}")

        xlim = 1.05 * float(np.max(np.abs(avg)))
        grid = np.linspace(-xlim, xlim, 2001)
        peak = 0.0
        for mode, k, scale, sign in lobes:
            color = "#1f77b4" if sign < 0 else "#d62728"
            pdf = stats.gamma.pdf(sign * grid, k, scale=scale)
            peak = max(peak, pdf.max())
            ax.plot(grid, pdf, color=color, linewidth=1.8)
            # The curves are drawn exactly as fitted; only the labels
            # are nudged away from the axis when a fitted mode is too
            # close to zero to letter legibly.
            label_x = sign * max(abs(mode), 0.07 * xlim)
            label = r"$\gamma_\mu$" if sign < 0 else r"$\gamma_\nu$"
            ax.text(label_x, pdf.max() * 1.06, label,
                    ha="center", va="bottom", fontsize=13)
            ax.text(label_x, -0.06, r"$\mu$" if sign < 0 else r"$\nu$",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=13)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--",
                   alpha=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_yticks([])
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, 1.25 * peak)
        ax.set_xticks([])

    fig.tight_layout()
    pdf_path = OUT / "type_schematics.pdf"
    png_path = OUT / "type_schematics.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
