#!/usr/bin/env python3
"""Absolute values of the eigenvalues of the full bilinear forms of gpt2.

The nonzero eigenvalues of L = W_K^T W_Q equal the eigenvalues of
the n x n matrix W_Q W_K^T, and are complex in general. For each of
the 144 heads of gpt2 this script computes these n = 64 eigenvalues,
sorts their absolute values in decreasing order, and rescales the list to
unit Euclidean norm (cached to
scripts/cluster_results/gpt2_sorted_fullmod.npy).

The experiment corroborating the gamma observation: fit a gamma
distribution (origin fixed at zero) to the entries of the mean
sorted profile and to each head's profile, and report the fitted
shape parameters, the common-shape energy of the profile matrix,
and the Kolmogorov-Smirnov distance of the pooled entries from the
pooled fit.

Outputs:
    figures/qk_geometry/full_spectrum_mean.{pdf,png}
        histogram of the mean profile with the fitted gamma density
    printed: SVD shares, per-head and pooled gamma fits, KS distance
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT = PROJECT / "figures" / "qk_geometry"

CACHE = RESULTS / "gpt2_sorted_fullmod.npy"


def compute_profiles():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qk_geometry_diagnostics import iter_qk_heads, load_model

    model = load_model("gpt2")
    model.eval()
    mods = []
    for hw in iter_qk_heads(model, "gpt2"):
        Wq = hw.Wq.to(torch.float64).numpy()
        Wk = hw.Wk.to(torch.float64).numpy()
        # nonzero spectrum of L = Wk^T Wq (complex in general)
        ev = np.linalg.eigvals(Wq @ Wk.T)
        m = np.sort(np.abs(ev))[::-1]
        mods.append(m / np.linalg.norm(m))
    return np.array(mods)


def main() -> None:
    if CACHE.exists():
        X = np.load(CACHE)
    else:
        X = compute_profiles()
        np.save(CACHE, X)
    print(f"{X.shape[0]} heads, {X.shape[1]} eigenvalue absolute values each")

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shares = s**2 / (s**2).sum()
    print(f"SVD shares: {100*shares[0]:.1f}% / {100*shares[1]:.1f}% / "
          f"{100*shares[2]:.1f}%")

    mean_row = X.mean(axis=0)
    k_mean, _, scale_mean = stats.gamma.fit(mean_row, floc=0)
    print(f"gamma fit to mean profile: shape k = {k_mean:.2f}")

    ks = [stats.gamma.fit(r, floc=0)[0] for r in X]
    print(f"per-head fitted shapes: median {np.median(ks):.2f}, "
          f"IQR [{np.percentile(ks, 25):.2f}, "
          f"{np.percentile(ks, 75):.2f}]")

    pooled = X.ravel()
    k_p, loc_p, scale_p = stats.gamma.fit(pooled, floc=0)
    D, _ = stats.kstest(pooled, "gamma", args=(k_p, loc_p, scale_p))
    print(f"pooled gamma fit: k = {k_p:.2f}, KS distance D = {D:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    counts, bins, _ = ax.hist(mean_row, bins=24, color="#bbbbbb",
                              edgecolor="black", linewidth=0.4,
                              label="mean over heads of sorted "
                                    r"$|\lambda|$")
    # Scale the fitted density to the count histogram: total count
    # times bin width.
    area = mean_row.size * (bins[1] - bins[0])
    grid = np.linspace(0, mean_row.max() * 1.1, 400)
    ax.plot(grid, area * stats.gamma.pdf(grid, k_mean, 0, scale_mean),
            color="black", linewidth=1.6,
            label=rf"fitted gamma distribution ($k = {k_mean:.1f}$)")
    ax.set_xlabel(r"eigenvalue absolute value $|\lambda|$")
    ax.set_ylabel("count")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    pdf = OUT / "full_spectrum_mean.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "full_spectrum_mean.png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
