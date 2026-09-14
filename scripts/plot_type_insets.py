#!/usr/bin/env python3
"""Schematic bimodal spectra for the three types, one file per type.

These are idealizations, not fits: each panel draws a pair of gamma
densities gamma_mu (negative lobe, blue) and gamma_nu (positive lobe,
red) with parameters chosen by hand so that the defining feature of the
type is legible at the size of a figure inset.  The three types are
distinguished by the relative weight of the two lobes
(Observation on the three types, and the definition of the types):

    Type I+   |mu| << nu     the positive lobe dominates
    Type II   |mu| ~= nu     the two lobes balance
    Type I-   nu << |mu|     the negative lobe dominates

Each lobe is drawn with area proportional to its share of the spectral
energy, so the visual weight of a lobe tracks the quantity that defines
the type; the common vertical scale is shared across the three panels.
The experimental fits are plotted separately by plot_type_schematics.py.

The panels carry no titles or axis labels: in the parity figure of
attention.tex the segment above them supplies the names, with Type I+
under the c = 1 end, Type II under the midpoint, and Type I- under the
d = 1 end.

Saves figures/qk_geometry/type_inset_{Iplus,II,Iminus}.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"

NEG_COLOR = "#1f77b4"
POS_COLOR = "#d62728"

# (file name, |mu|, nu, weight of the negative lobe).  The modes are
# chosen so that the dominant lobe sits further from the origin, and the
# weights so that the two lobes carry the stated share of the energy.
# Shape k is held fixed, so the three panels differ only in the contrast
# between the lobes, which is the feature that defines the types.
SHAPE = 3.0
TYPES = [
    ("Iplus",  0.30, 1.90, 0.25),   # |mu| << nu : positive dominates
    ("II",     0.75, 0.75, 0.50),   # |mu| == nu : balanced
    ("Iminus", 1.90, 0.30, 0.75),   # nu << |mu| : negative dominates
]


def lobe(mode, k):
    """A gamma density with the given mode and shape, as a callable."""
    scale = mode / (k - 1.0)
    return lambda x: stats.gamma.pdf(x, k, scale=scale)


def main() -> None:
    # Wide enough that every lobe, including the long tail of the
    # dominant one, returns to the axis inside the window.
    xlim = 5.5
    grid = np.linspace(-xlim, xlim, 2001)

    # a common vertical scale, so the panels are directly comparable
    curves = []
    for name, mu, nu, wneg in TYPES:
        neg = wneg * lobe(mu, SHAPE)(-grid)
        pos = (1.0 - wneg) * lobe(nu, SHAPE)(grid)
        curves.append((name, neg, pos))
    peak = max(max(neg.max(), pos.max()) for _, neg, pos in curves)

    for name, neg, pos in curves:
        fig, ax = plt.subplots(figsize=(2.5, 1.15))
        ax.plot(grid, neg, color=NEG_COLOR, linewidth=1.5)
        ax.plot(grid, pos, color=POS_COLOR, linewidth=1.5)
        ax.fill_between(grid, neg, color=NEG_COLOR, alpha=0.18, linewidth=0)
        ax.fill_between(grid, pos, color=POS_COLOR, alpha=0.18, linewidth=0)
        ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--",
                   alpha=0.6)

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, 1.12 * peak)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.7)

        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.98)
        pdf_path = OUT / f"type_inset_{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
        fig.savefig(OUT / f"type_inset_{name}.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
