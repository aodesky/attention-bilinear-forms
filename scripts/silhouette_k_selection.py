#!/usr/bin/env python3
"""Silhouette-based selection of k for the head-spectra clustering.

For each k in 2..6, runs k-means (scipy.cluster.vq.kmeans2, minit="++")
on the rows of the gpt2 sorted unit-normalized spectra matrix X from
fifty random initializations, keeps the solution with the smallest
objective (within-cluster sum of squares), and reports its silhouette
coefficient (Rousseeuw 1987) and cluster sizes.

Also renders the k = 4 analogue of the cluster tail-histogram figure
(per cluster, the averaged histogram of the ten most extreme members
along the cluster direction, exactly as in cluster_head_spectra.py)
and prints the cross-tabulation of the k = 4 solution against the
paper's k = 3 clustering (seed 0), showing that k = 4 refines k = 3.

Source of the numbers quoted in the k = 3 remark of the mechanistic
interpretability section of attention.tex.

Outputs:
    figures/qk_geometry/spectra_cluster_barycenters_k4.{pdf,png}
"""

from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT = PROJECT / "figures" / "qk_geometry"

N_SEEDS = 50
K_RANGE = range(2, 7)
N_EXTREME = 10
PAPER_SEED = 0  # seed of the k = 3 clustering used in the paper


def best_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Labels of the smallest-objective solution over N_SEEDS restarts."""
    best = None
    for seed in range(N_SEEDS):
        cent, labels = kmeans2(X, k, minit="++", seed=seed)
        if len(np.unique(labels)) < k:
            continue
        inertia = ((X - cent[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    if best is None:
        raise RuntimeError(f"all restarts degenerate for k={k}")
    return best[1]


def main() -> None:
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    keys = [tuple(k) for k in np.load(RESULTS / "gpt2_head_keys.npy")]
    print(f"X shape: {X.shape}")

    for k in K_RANGE:
        labels = best_kmeans(X, k)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        s = silhouette_score(X, labels)
        print(f"k={k}  silhouette={s:.3f}  sizes={sizes}")

    # --- k = 4 vs the paper's k = 3 clustering (seed 0) ---
    _, lab3 = kmeans2(X, 3, minit="++", seed=PAPER_SEED)
    lab4 = best_kmeans(X, 4)
    # Here clusters are numbered by decreasing size (unlike the paper's
    # k = 3 figure, which orders by signed asymmetry): in the cross-tab
    # below, the n=64 row is Type I+, n=69 is Type I-, n=11 is Type II.
    order = np.argsort(-np.bincount(lab4))
    relab = np.empty_like(lab4)
    for new, old in enumerate(order):
        relab[lab4 == old] = new
    lab4 = relab

    print("\ncross-tab: paper k=3 clusters (rows, by size) x k=4 "
          "clusters (cols, by size):")
    for r in np.argsort(-np.bincount(lab3)):
        row = [int(np.sum((lab3 == r) & (lab4 == c))) for c in range(4)]
        print(f"  k3 cluster n={np.sum(lab3 == r):3d}: {row}")

    # --- k = 4 tail histograms, same construction as the k = 3 figure ---
    gmean = X.mean(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(5.5 * 4, 3.8))
    print()
    for c, ax in zip(range(4), axes):
        idx = np.where(lab4 == c)[0]
        direction = X[idx].mean(axis=0) - gmean
        direction /= np.linalg.norm(direction)
        scores = (X[idx] - gmean) @ direction
        reps = idx[np.argsort(-scores)[:N_EXTREME]]
        rep_names = [f"L{keys[i][0]}H{keys[i][1]}" for i in reps]
        avg = X[reps].mean(axis=0)

        pos, neg = avg[avg > 0], avg[avg < 0]
        bins = np.linspace(avg.min(), avg.max(), 41)
        ax.hist(neg, bins=bins, color="#1f77b4", edgecolor="black",
                linewidth=0.4)
        ax.hist(pos, bins=bins, color="#d62728", edgecolor="black",
                linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--",
                   alpha=0.6)
        neg_frac = float((neg**2).sum())
        ax.set_title(f"cluster {c + 1} ($n={idx.size}$, "
                     f"negative energy ${neg_frac:.2f}$)", fontsize=10)
        ax.set_xlabel("eigenvalue")
        print(f"cluster {c + 1}: n={idx.size}  neg_energy={neg_frac:.3f}  "
              f"extreme members: {rep_names}")
    axes[0].set_ylabel("count")

    fig.tight_layout()
    pdf = OUT / "spectra_cluster_barycenters_k4.pdf"
    png = OUT / "spectra_cluster_barycenters_k4.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"\nWrote {pdf}")


if __name__ == "__main__":
    main()
