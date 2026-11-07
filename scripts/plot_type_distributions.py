#!/usr/bin/env python3
"""Combine the measured GPT-2 type histograms with the rank-one family.

Uses exactly the extreme-head selection and 40 bins of the original
spectra_cluster_barycenters figure: each panel is a histogram of the mean
sorted, unit-length spectrum of ten heads, not a fitted density. Type I-
and Type I+ sit above their respective branches of the family, while
Type II sits below its midpoint, with each label centered on its histogram.

Run from the repository root with:
    MPLCONFIGDIR=/tmp/ollm-mpl venv/bin/python scripts/plot_type_distributions.py

Writes figures/qk_geometry/type_distributions.{pdf,png}.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_type_schematics import extreme_averages


OUT = Path(__file__).resolve().parent.parent / "figures" / "qk_geometry"


def main() -> None:
    mpl.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.5})
    positive, balanced, negative = extreme_averages()
    spectra = [negative, balanced, positive]
    titles = [r"Type $\mathrm{I}_-$", r"Type $\mathrm{II}$",
              r"Type $\mathrm{I}_+$"]

    # Keep the family central: branch histograms above, midpoint histogram
    # below. Labels share the histogram centers.
    fig = plt.figure(figsize=(5.5, 2.65))
    centers = [0.25, 0.5, 0.75]
    # Reduce the insets by 20%, keeping their edges nearest the family
    # fixed so the labels retain their spacing.
    inset_width, inset_height = 0.24, 0.224
    bottoms = [0.70, 0.34 - inset_height, 0.70]
    for i, (center, bottom, avg) in enumerate(zip(centers, bottoms, spectra)):
        ax = fig.add_axes([center - inset_width / 2, bottom,
                           inset_width, inset_height])
        bins = np.linspace(avg.min(), avg.max(), 41)
        ax.hist(avg[avg < 0], bins=bins, color="#1f77b4",
                edgecolor="black", linewidth=0.3)
        ax.hist(avg[avg > 0], bins=bins, color="#d62728",
                edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.5,
                   linestyle="--", alpha=0.6)
        # Keep the original per-panel bin edges and x limits; share the
        # count scale so bar heights remain comparable across types.
        ax.set_ylim(0, 20)
        ax.set_yticks([0, 10, 20])
        ax.set_xticks([-0.1, 0, 0.1] if i != 1 else [-0.2, 0, 0.2, 0.4])
        ax.tick_params(labelsize=7, length=2, width=0.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        if i != 2:
            ax.set_ylabel("count", fontsize=8, labelpad=2)
        ax.text(1.035, -0.025, r"$\lambda$", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)

    family = fig.add_axes([0, 0, 1, 1], frameon=False)
    family.set(xlim=(0, 1), ylim=(0, 1))
    family.set_axis_off()
    line_y = 0.535
    family.plot([0.05, 0.5, 0.95], [line_y] * 3, color="black", linewidth=1.1,
                marker="o", markersize=3)
    label_box = dict(facecolor="white", edgecolor="none", pad=1.5)
    for i in (0, 2):
        x = centers[i]
        family.text(x, 0.59, titles[i], ha="center", va="center",
                    fontsize=10, bbox=label_box)
    family.text(0.5, 0.445, titles[1], ha="center", va="center",
                fontsize=10, bbox=label_box)
    family.text(0.5, 0.385, "symmetric spectrum", ha="center", va="center",
                fontsize=8, bbox=label_box)
    family.text(0.5, 0.56, r"$\Theta_\circ$", ha="center", va="bottom", fontsize=10)
    for x, label in [(0.05, r"$d=1$"), (0.25, r"$\Theta_-$"),
                     (0.75, r"$\Theta_+$"), (0.95, r"$c=1$")]:
        family.text(x, 0.51, label, ha="center", va="top", fontsize=10)
    for x, label in [(0.25, r"$S=H-P$"), (0.75, r"$S=H+P$")]:
        family.text(x, 0.425, label, ha="center", va="top", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    # Trim the empty outer margins released by the smaller insets.
    export_box = mpl.transforms.Bbox.from_bounds(0, 0.15, 5.5, 2.38)
    for extension in ("pdf", "png"):
        path = OUT / f"type_distributions.{extension}"
        fig.savefig(path, dpi=240, facecolor="white", bbox_inches=export_box)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
