#!/usr/bin/env python3
"""Sanity check: the fitted densities drawn over the observed spectra.

For each model this draws a random sample of heads and, for each of the
four positive spectral samples, plots the histogram of the observed
values with the three fitted densities laid over it: gamma, lognormal
and Weibull, each fitted by maximum likelihood with the location held
at zero, exactly as in gamma_goodness_of_fit.py.

This is a diagnostic, not a figure of the paper.  It exists so that the
goodness-of-fit comparison can be checked by eye: a family that wins on
the Kolmogorov--Smirnov distance should visibly follow the data.

Usage:
    python check_fits.py                    # every model, 30 heads each
    python check_fits.py --model gpt2 --heads 12
    python check_fits.py --component T

Writes one page per model to figures/fit_checks/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms
from gamma_goodness_of_fit import CANDIDATES, load_samples

OUT = ms.PROJECT / "figures" / "fit_checks"
SEED = 0
COLOURS = {"gamma": "#1f77b4", "lognormal": "#e07a00", "weibull": "#5b5b5b"}
LABELS = {"gamma": "gamma", "lognormal": "lognormal", "weibull": "Weibull"}
COMPONENT_TITLES = {
    "full": r"$|\lambda(L)|$",
    "S_positive": r"$\lambda_+(S)$",
    "S_negative": r"$\lambda_-(S)$",
    "T": r"$\sigma(T)$",
}


def panel(ax, sample: np.ndarray) -> dict[str, float]:
    """Histogram of one head's sample, with the three fits over it.

    The horizontal range is clipped to the central 99% of the sample.  A
    few heads carry one eigenvalue far above the rest, which would
    otherwise compress the bulk of the distribution into the leftmost
    bins and leave the panel unreadable; the fits themselves use the
    whole sample.
    """
    upper = float(np.quantile(sample, 0.99))
    ax.hist(sample, bins=24, density=True, color="#d9d9d9",
            edgecolor="#9a9a9a", linewidth=0.3,
            range=(float(sample.min()), upper))
    grid = np.linspace(sample.min() * 0.5, upper * 1.05, 400)
    distances = {}
    for name, distribution in CANDIDATES.items():
        parameters = distribution.fit(sample, floc=0)
        ax.plot(grid, distribution.pdf(grid, *parameters),
                color=COLOURS[name], linewidth=1.2, label=LABELS[name])
        distances[name] = float(stats.kstest(
            sample, distribution.cdf, args=parameters).statistic)
    best = min(distances, key=distances.get)
    ax.set_title(f"KS {min(distances.values()):.3f} ({LABELS[best]})",
                 fontsize=6, pad=2)
    ax.set_xlim(0.0, upper * 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    return distances


def check_model(tag: str, n_heads: int, components: list[str]) -> None:
    """One page per model: a grid of heads for each spectral sample."""
    samples = load_samples(tag)
    total = len(samples[components[0]])
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(total, size=min(n_heads, total), replace=False)
    chosen.sort()

    # a grid rather than one long row, so the panels stay legible
    cols = 6
    rows = int(np.ceil(len(chosen) / cols))
    wins = {name: 0 for name in CANDIDATES}

    OUT.mkdir(parents=True, exist_ok=True)
    for component in components:
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(1.7 * cols, 1.5 * rows),
                                 squeeze=False)
        for index, head in enumerate(chosen):
            ax = axes[index // cols][index % cols]
            distances = panel(ax, samples[component][head])
            wins[min(distances, key=distances.get)] += 1
        for index in range(len(chosen), rows * cols):
            axes[index // cols][index % cols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{tag}, {COMPONENT_TITLES[component]}: "
                     f"{len(chosen)} heads drawn at random", fontsize=11)
        fig.tight_layout(rect=(0, 0.015, 1, 0.97))
        fig.savefig(OUT / f"{tag}_{component}.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)

    won = ", ".join(f"{LABELS[k]} {v}" for k, v in wins.items())
    print(f"  {tag:26s} {len(chosen):3d} heads   closest fit: {won}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ms.TAGS)
    parser.add_argument("--heads", type=int, default=30)
    parser.add_argument("--component", choices=list(COMPONENT_TITLES))
    args = parser.parse_args()

    tags = [args.model] if args.model else ms.TAGS
    components = ([args.component] if args.component
                  else list(COMPONENT_TITLES))
    for tag in tags:
        check_model(tag, args.heads, components)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
