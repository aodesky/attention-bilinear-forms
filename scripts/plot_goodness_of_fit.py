#!/usr/bin/env python3
"""How well three families fit the spectra, against model size.

One panel per spectral sample.  The horizontal axis is the number of
parameters of the model, on a logarithmic scale; the vertical axis is
the median error between the observed histogram and the fitted density
over the heads of that model, so lower is a closer fit.  The three families,
gamma, lognormal and Weibull, each have two parameters once the location
is held at zero, so the comparison between them is even.

Reads the per-head fits written by gamma_goodness_of_fit.py.

Usage:
    python plot_goodness_of_fit.py

Saves figures/qk_geometry/goodness_of_fit.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["text.usetex"] = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

RESULTS = ms.PROJECT / "scripts" / "survey_results"
OUT = ms.PROJECT / "figures" / "qk_geometry"

# the four samples, with the symbols the paper uses
COMPONENTS = {
    "full": r"$|\lambda(L)|$",
    "S_positive": r"$\lambda_+(S)$",
    "S_negative": r"$\lambda_-(S)$",
    "T": r"$\sigma(T)$",
}
FAMILIES = {
    "gamma": ("gamma", "#1f77b4", "o"),
    "lognormal": ("lognormal", "#e07a00", "s"),
    "weibull": ("Weibull", "#5b5b5b", "^"),
}


def main() -> None:
    path = RESULTS / "gamma_goodness_of_fit.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run gamma_goodness_of_fit.py first")
    fits = pd.read_csv(path)

    missing = set(ms.TAGS) - set(fits.model.unique())
    if missing:
        raise ValueError(f"no fits for {sorted(missing)}")
    unexpected = set(fits.family.unique()) - set(FAMILIES)
    if unexpected:
        raise ValueError(f"unexpected families in the fits: {unexpected}")

    size = {i.tag: i.n_parameters for i in ms.all_model_info()}

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey=True,
                             sharex=True)
    for ax, (component, symbol) in zip(axes.ravel(), COMPONENTS.items()):
        subset = fits[fits.component == component]
        for family, (label, colour, marker) in FAMILIES.items():
            rows = subset[subset.family == family]
            medians = rows.groupby("model").bin_error.median()
            x = np.array([size[t] for t in medians.index], dtype=float)
            # a scatter, not a curve: the models are not a sequence, and
            # joining them would suggest a trend through points that
            # differ in architecture as much as in size
            ax.scatter(x, medians.to_numpy(), s=22, marker=marker,
                       facecolor="none", edgecolor=colour, linewidth=0.9,
                       label=label)
        ax.set_xscale("log")
        ax.set_title(symbol, fontsize=11)
        ax.set_xlabel("parameters", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.4)

    for ax in axes[:, 0]:
        ax.set_ylabel("median error of fit", fontsize=9)
    for ax in axes[0, :]:
        ax.set_xlabel("")

    # margin enough that no marker is clipped by the frame
    low = fits.bin_error.groupby(
        [fits.model, fits.component, fits.family]).median().min()
    high = fits.bin_error.groupby(
        [fits.model, fits.component, fits.family]).median().max()
    pad = 0.08 * (high - low)
    axes[0, 0].set_ylim(low - pad, high + pad)

    # below the axes, so it covers no data in any panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pdf = OUT / "goodness_of_fit.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "goodness_of_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{'sample':14s} " +
          " ".join(f"{label:>10s}" for label, _, _ in FAMILIES.values()))
    for component, symbol in COMPONENTS.items():
        subset = fits[fits.component == component]
        cells = []
        for family in FAMILIES:
            cells.append(subset[subset.family == family].bin_error.median())
        print(f"{component:14s} " + " ".join(f"{c:10.4f}" for c in cells))
    print(f"\nWrote {pdf}")


if __name__ == "__main__":
    main()
