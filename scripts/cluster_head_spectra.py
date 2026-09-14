#!/usr/bin/env python3
"""Clustering of the 144 gpt2 head spectra.

Each head is represented by the sorted vector of the 128 nonzero
eigenvalues of S_H = (W_K^T W_Q + W_Q^T W_K)/2, normalized to unit
energy; stacking these gives a 144 x 128 matrix X.

The heads are clustered by k-means (k = 3) on the rows of X: the
partition locally minimizing the sum of squared Euclidean distances
from each row to its cluster mean. The scatter plots use the SVD
X = U S V^T (uncentered): each head's coordinates are the leading
entries of its row of U S. The first coordinate is nearly constant
across heads (the first right-singular vector points along the common
shape of the spectra), which the 3-D plot shows directly; the 2-D plot
projects onto components 2 and 3.

Run once per model (--model TAG); the gpt2 outputs carry no suffix,
the others are suffixed _TAG.

Outputs (as .pdf and .png):
    figures/qk_geometry/spectra_cluster_combined{_TAG}
        left: components 1-3 of the uncentered SVD; right: the
        projection onto components 2-3 (with O/E level sets for the
        non-gpt2 models), marker-coded by cluster
    figures/qk_geometry/spectra_cluster_scatter
        (gpt2 only) components 2-3 with O/E level sets
    figures/qk_geometry/spectra_cluster_barycenters{_TAG}
        per cluster, the averaged histogram of its most extreme heads
    scripts/cluster_results/{TAG}_sorted_spectra.npy   (heads, 2n) cache
    scripts/cluster_results/{TAG}_head_keys.npy        (layer, head) rows
    printed: SVD variance shares, first-coordinate statistics,
        cluster sizes and extreme members

The silhouette table justifying k = 3 is produced by
silhouette_k_selection.py.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}\usepackage{amssymb}"

# Cluster markers, matching the type names: + for Type I+, o for
# Type II, - for Type I-.  The shapes alone distinguish the types, so
# the figures remain readable in grayscale; the colors match the
# interactive plots.
MARKERS = ["+", "o", "_"]
MARKER_TEX = [r"$+$", r"$\circ$", r"$-$"]
MARKER_COLORS = ["#1f77b4", "#e07a00", "#5b5b5b"]
# The one-parameter family of limiting weights.
FAMILY_COLOR = "#d62728"
# Cluster names, indexed by the renumbered cluster (1 = positive-dominant,
# 2 = the near-balanced heavy-tailed family, 3 = negative-dominant).
CLUSTER_NAMES = [r"Type I$_+$", r"Type II", r"Type I$_-$"]


def marker_kwargs(c, s=30):
    """Scatter style for renumbered cluster c.  Open markers get a white
    face with a colored edge; line markers (+, -) ignore edgecolor in
    matplotlib, so they are stroked in the cluster color instead."""
    color = MARKER_COLORS[c - 1]
    if MARKERS[c - 1] in ("+", "_"):
        return dict(marker=MARKERS[c - 1], s=int(1.8 * s),
                    color=color, linewidth=0.9)
    return dict(marker=MARKERS[c - 1], s=s, facecolor="none",
                edgecolor=color, linewidth=0.9)


K = 3
# Each cluster is portrayed by its most extreme members: the heads with
# the largest projection of (spectrum - global mean) onto the cluster's
# defining direction (cluster barycenter - global mean). Override the
# count with --n-extreme.
DEFAULT_N_EXTREME = 10

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "distilgpt2": "distilgpt2",
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
}


def load_spectra(tag):
    spec_path = RESULTS / f"{tag}_sorted_spectra.npy"
    keys_path = RESULTS / f"{tag}_head_keys.npy"
    if spec_path.exists() and keys_path.exists():
        X = np.load(spec_path)
        keys = [tuple(k) for k in np.load(keys_path)]
        return X, keys

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qk_geometry_diagnostics import (compute_reduced_spectrum,
                                         iter_qk_heads, load_model)

    model = load_model(MODELS[tag])
    model.eval()
    spectra, keys = [], []
    for hw in iter_qk_heads(model, MODELS[tag]):
        # compute_reduced_spectrum returns the 2n eigenvalues of the
        # symmetric part on the reduced subspace (all nonzero).
        evals, _ = compute_reduced_spectrum(hw)
        spec = np.sort(evals)
        spectra.append(spec / np.linalg.norm(spec))
        keys.append((hw.layer, hw.head))
    sizes = {s.size for s in spectra}
    assert len(sizes) == 1, f"unequal spectrum sizes: {sizes}"
    X = np.array(spectra)
    np.save(spec_path, X)
    np.save(keys_path, np.array(keys))
    return X, keys


def svd_scores(X, dim=3):
    """First `dim` SVD coordinates of the rows of X (uncentered).

    Take the SVD X = U S V^T and return the first `dim` columns of
    U S (each head's coordinates along the top right-singular
    directions), together with the per-component variance shares
    s_i^2 / sum_j s_j^2 and the first `dim` right-singular vectors.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    emb = U[:, :dim] * s[:dim]
    shares = s**2 / (s**2).sum()
    # Fix the sign ambiguity of each singular vector so the axes have a
    # consistent orientation: component 1 points along the common shape
    # of the spectra (positive coordinates), component 2 increases with
    # the signed asymmetry (positive minus negative energy), component 3
    # with the spectral effective rank. Flip the corresponding
    # right-singular vector together with each coordinate.
    if emb[:, 0].mean() < 0:
        emb[:, 0] = -emb[:, 0]
        Vt[0] = -Vt[0]
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum() for r in X])
    p = X**2 / (X**2).sum(axis=1, keepdims=True)
    effrank = np.exp(-(p * np.log(p + 1e-300)).sum(axis=1))
    for j, stat in [(1, imb), (2, effrank)]:
        if j < dim and np.corrcoef(emb[:, j], stat)[0, 1] < 0:
            emb[:, j] = -emb[:, j]
            Vt[j] = -Vt[j]
    return emb, shares, Vt[:dim]


def draw_oe_levels(ax, emb, X, V):
    """Draw level sets of O (solid) and E (dotted) on a components-2/3
    axis. Analytic level sets in the plane, via the rank-3 model
    lambda(c2, c3) = c1bar*v1 + c2*v2 + c3*v3 with c1 frozen at its
    mean, normalized to unit energy. The two quantities are the sums
    of the odd and even moments of the normalized spectrum:
    O = sum lambda/(1-lambda^2) and E = sum lambda^2/(1-lambda^2).
    Returns the axis limits (x_lo, x_hi, y_lo, y_hi)."""
    margin = 0.06
    x_lo, x_hi = emb[:, 1].min() - margin, emb[:, 1].max() + margin
    y_lo, y_hi = emb[:, 2].min() - margin, emb[:, 2].max() + margin
    c1bar = emb[:, 0].mean()
    grid2, grid3 = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                               np.linspace(y_lo, y_hi, 300))
    lam = (c1bar * V[0][None, None, :]
           + grid2[..., None] * V[1][None, None, :]
           + grid3[..., None] * V[2][None, None, :])
    lam = lam / np.sqrt((lam**2).sum(axis=2, keepdims=True))
    o_grid = (lam / (1 - lam**2)).sum(axis=2)
    e_grid = (lam**2 / (1 - lam**2)).sum(axis=2)

    o_heads = (X / (1 - X**2)).sum(axis=1)
    o_levels = sorted(set(np.round(
        np.percentile(o_heads, [5, 20, 35, 50, 65, 80, 95]), 1).tolist()))
    cs_o = ax.contour(grid2, grid3, o_grid, levels=o_levels,
                      colors="black", linewidths=0.5, linestyles="solid",
                      alpha=0.45, zorder=1)
    ax.clabel(cs_o, fontsize=6, fmt=lambda v: rf"$O={v:g}$")
    e_heads = (X**2 / (1 - X**2)).sum(axis=1)
    e_levels = sorted(set(np.round(
        np.percentile(e_heads, [10, 30, 50, 70, 90]), 3).tolist()))
    cs_e = ax.contour(grid2, grid3, e_grid, levels=e_levels,
                      colors="black", linewidths=0.5, linestyles="dotted",
                      alpha=0.45, zorder=1)
    ax.clabel(cs_e, fontsize=6, fmt=lambda v: rf"$E={v:g}$")
    return x_lo, x_hi, y_lo, y_hi


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-extreme", type=int, default=None,
                        help="number of most extreme heads (largest "
                             "projection onto the cluster direction) to "
                             "average per cluster; default scales "
                             "proportionally with the number of heads "
                             "(10 per 144 heads, floor 10)")
    parser.add_argument("--model", choices=sorted(MODELS), default="gpt2")
    args = parser.parse_args()
    tag = args.model
    suffix = "" if tag == "gpt2" else f"_{tag}"

    print(f"=== {tag} ===")
    X, keys = load_spectra(tag)
    n_extreme = args.n_extreme
    if n_extreme is None:
        n_extreme = max(DEFAULT_N_EXTREME,
                        round(DEFAULT_N_EXTREME * len(X) / 144))
    print(f"n_extreme = {n_extreme} ({len(X)} heads)")

    emb, shares, V = svd_scores(X, dim=3)
    print(f"\nSVD variance shares: "
          + ", ".join(f"component {j+1}: {100*shares[j]:.1f}%" for j in range(3))
          + f"  (top three: {100*shares[:3].sum():.1f}%)")
    print(f"coordinate 1 across heads: "
          f"{emb[:, 0].mean():.3f} +/- {emb[:, 0].std():.3f}")

    _, raw = kmeans2(X, K, minit="++", seed=0)
    # Renumber clusters by decreasing mean signed asymmetry (positive
    # minus negative energy of the spectrum), so that cluster 1 is the
    # positive-dominant type (Type I+), cluster 3 the negative-dominant
    # type (Type I-), and cluster 2 the near-balanced family (Type II).
    assert K == 3
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum()
                    for r in X])
    order = sorted(range(K), key=lambda c: -imb[raw == c].mean())
    labels = np.empty_like(raw)
    for new, old in enumerate(order, start=1):
        labels[raw == old] = new
    print("cluster mean signed asymmetry (renumbered): "
          + ", ".join(f"{c}: {imb[labels == c].mean():+.3f}"
                      for c in range(1, K + 1)))

    # --- combined figure: 3-D scatter and plain 2-D projection ---
    fig = plt.figure(figsize=(13.5, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2],
                   **marker_kwargs(c))
    ranges = emb.max(axis=0) - emb.min(axis=0)
    ax.set_box_aspect(tuple(ranges))
    ax.set_proj_type("persp", focal_length=0.35)
    ax.view_init(elev=15, azim=-35)
    if tag == "gpt2":
        ax.set_xticks([0.8, 0.9, 1.0])
    ax.set_xlabel("component 1", labelpad=8)
    ax.set_ylabel("component 2", labelpad=6)
    ax.set_zlabel("component 3", labelpad=12)
    ax.tick_params(labelsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    # The gpt2 combined figure (introduction) keeps a plain right
    # panel; the gpt2 O/E level sets appear in the standalone 2-D
    # figure below. For the other models (appendix) the right panel
    # carries the O/E level sets directly.
    if tag != "gpt2":
        lims = draw_oe_levels(ax2, emb, X, V)
        ax2.set_xlim(lims[0], lims[1])
        ax2.set_ylim(lims[2], lims[3])
    for c in np.unique(labels):
        m = labels == c
        ax2.scatter(emb[m, 1], emb[m, 2], **marker_kwargs(c, s=36),
                    label=f"{CLUSTER_NAMES[c - 1]} ($n={m.sum()}$)",
                    zorder=2)
    ax2.set_xlabel("component 2")
    ax2.set_ylabel("component 3")
    ax2.legend(fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.11, top=0.96,
                        wspace=0.08)
    combined_pdf = OUT / f"spectra_cluster_combined{suffix}.pdf"
    combined_png = OUT / f"spectra_cluster_combined{suffix}.png"
    fig.savefig(combined_pdf)
    fig.savefig(combined_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {combined_pdf}")

    # --- 2-D scatter (components 2, 3) with O/E level sets ---
    # Only the gpt2 version appears in the paper (the moments
    # section); for the other models the level sets are already on
    # the combined figure's right panel.
    if tag == "gpt2":
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        x_lo, x_hi, y_lo, y_hi = draw_oe_levels(ax, emb, X, V)

        for c in np.unique(labels):
            m = labels == c
            ax.scatter(emb[m, 1], emb[m, 2], **marker_kwargs(c, s=36),
                       label=f"{CLUSTER_NAMES[c - 1]} ($n={m.sum()}$)",
                       zorder=2)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("component 2")
        ax.set_ylabel("component 3")
        ax.legend(fontsize=9, frameon=True)
        fig.tight_layout()
        scatter_pdf = OUT / "spectra_cluster_scatter.pdf"
        scatter_png = OUT / "spectra_cluster_scatter.png"
        fig.savefig(scatter_pdf, bbox_inches="tight")
        fig.savefig(scatter_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {scatter_pdf}")

    # --- tail histograms: most extreme members along the cluster direction ---
    gmean = X.mean(axis=0)
    fig, axes = plt.subplots(1, K, figsize=(5.5 * K, 3.8))
    print()
    for c, ax in zip(np.unique(labels), np.atleast_1d(axes)):
        idx = np.where(labels == c)[0]
        bary = X[idx].mean(axis=0)
        direction = bary - gmean
        direction /= np.linalg.norm(direction)
        scores = (X[idx] - gmean) @ direction
        reps = idx[np.argsort(-scores)[:n_extreme]]
        rep_names = [f"L{keys[i][0]}H{keys[i][1]}" for i in reps]
        avg = X[reps].mean(axis=0)  # average of the extreme sorted spectra

        pos, neg = avg[avg > 0], avg[avg < 0]
        bins = np.linspace(avg.min(), avg.max(), 41)
        ax.hist(neg, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.4)
        ax.hist(pos, bins=bins, color="#d62728", edgecolor="black", linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        neg_frac = float((neg**2).sum())
        ax.set_title(f"{CLUSTER_NAMES[c - 1]} ({MARKER_TEX[c - 1]}, $n={idx.size}$, "
                     f"negative energy ${neg_frac:.2f}$)", fontsize=10)
        ax.set_xlabel("eigenvalue")
        print(f"cluster {c}: n={idx.size}  extreme members: {rep_names}  "
              f"neg_energy={neg_frac:.3f}")
    np.atleast_1d(axes)[0].set_ylabel("count")

    fig.tight_layout()
    bary_pdf = OUT / f"spectra_cluster_barycenters{suffix}.pdf"
    bary_png = OUT / f"spectra_cluster_barycenters{suffix}.png"
    fig.savefig(bary_pdf, bbox_inches="tight")
    fig.savefig(bary_png, dpi=200, bbox_inches="tight")
    print(f"\nWrote {bary_pdf}")


if __name__ == "__main__":
    main()
