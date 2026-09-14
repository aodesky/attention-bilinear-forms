#!/usr/bin/env python3
"""Histogram of the mean sorted spectrum of the gpt2 heads.

Plots the distribution of the 128 entries of the average row of the
matrix X of sorted unit-normalized spectra (the common shape recorded
by the first right-singular vector).

Saves figures/qk_geometry/mean_spectrum.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT = PROJECT / "figures" / "qk_geometry"


def main() -> None:
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    xbar = X.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xmax = 1.05 * float(np.max(np.abs(xbar)))
    bins = np.linspace(-xmax, xmax, 37)
    ax.hist(xbar, bins=bins, color="#cccccc", edgecolor="black",
            linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("count")
    fig.tight_layout()
    pdf = OUT / "mean_spectrum.pdf"
    png = OUT / "mean_spectrum.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
