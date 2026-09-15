#!/usr/bin/env python3
"""Schematic of the archetype-weight simplices and their fibers.

Left: the 3-simplex Delta^3 of weights (a, b, c, d) of nonzero
bilinear forms, with the four archetype vertices labeled and the
face {a = 0} (symmetric forms) shaded.
Right: the face {a = 0}, the 2-simplex of symmetric forms, with the
fibers over the vertices, edges, and interior labeled in terms of
bilinear forms (Propositions on archetype energies and dominance,
and the empty-edge corollary).

Saves figures/qk_geometry/simplex_faces.{pdf,png}.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ------------------------------------------------------- 3-simplex
    A = (0.42, 1.00)   # a = 1 : antisymmetric (apex)
    B = (0.00, 0.12)   # b = 1 : equal spectral lobes
    C = (0.78, 0.00)   # c = 1 : positive semidefinite
    D = (1.00, 0.48)   # d = 1 : negative semidefinite

    ax1.add_patch(Polygon([B, C, D], closed=True, facecolor="#dddddd",
                          edgecolor="none", alpha=0.7, zorder=0))
    for p, q in [(A, B), (A, C), (A, D), (B, C), (C, D)]:
        ax1.plot([p[0], q[0]], [p[1], q[1]], color="black",
                 linewidth=1.2, zorder=2)
    ax1.plot([B[0], D[0]], [B[1], D[1]], color="black", linewidth=1.0,
             linestyle="--", dashes=(4, 3), zorder=1)
    for p in (A, B, C, D):
        ax1.plot(*p, marker="o", markersize=4, color="black", zorder=3)

    ax1.text(A[0], A[1] + 0.04, r"$a = 1$: antisymmetric",
             ha="center", va="bottom", fontsize=11)
    ax1.text(B[0] - 0.03, B[1], r"$b = 1$: $\lambda_+=\lambda_-$",
             ha="right", va="center", fontsize=11)
    ax1.text(C[0], C[1] - 0.05, r"$c = 1$: positive semidefinite",
             ha="center", va="top", fontsize=11)
    ax1.text(D[0] + 0.03, D[1], r"$d = 1$: negative semidefinite",
             ha="left", va="center", fontsize=11)
    ax1.text(0.54, 0.16, r"$a = 0$",
             ha="center", va="center", fontsize=11, zorder=4)

    ax1.set_xlim(-0.55, 1.75)
    ax1.set_ylim(-0.30, 1.16)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ------------------------------------------------------- 2-simplex
    Pb = (0.50, 0.90)  # b = 1 vertex
    Pc = (0.02, 0.06)  # positive semidefinite vertex
    Pd = (0.98, 0.06)  # negative semidefinite vertex

    ax2.plot([Pb[0], Pc[0]], [Pb[1], Pc[1]], color="black", linewidth=1.2)
    ax2.plot([Pb[0], Pd[0]], [Pb[1], Pd[1]], color="black", linewidth=1.2)
    ax2.plot([Pc[0], Pd[0]], [Pc[1], Pd[1]], color="black", linewidth=1.0,
             linestyle="--", dashes=(4, 3))
    for p in (Pb, Pc, Pd):
        ax2.plot(*p, marker="o", markersize=4, color="black", zorder=3)

    ax2.text(Pb[0], Pb[1] + 0.04,
             r"$b = 1$: $\lambda_+=\lambda_-$ (Type $\mathrm{II}$)",
             ha="center", va="bottom", fontsize=11)
    ax2.text(Pc[0] - 0.04, Pc[1] - 0.04, r"$c = 1$: positive semidefinite",
             ha="right", va="top", fontsize=11)
    ax2.text(Pd[0] + 0.04, Pd[1] - 0.04, r"$d = 1$: negative semidefinite",
             ha="left", va="top", fontsize=11)

    ax2.text(0.16, 0.52,
             r"\begin{tabular}{c}$d = 0$: Type $\mathrm{I}_+$"
             r"\\ $S=H+P$\end{tabular}",
             ha="right", va="center", fontsize=10, rotation=60)
    ax2.text(0.84, 0.52,
             r"\begin{tabular}{c}$c = 0$: Type $\mathrm{I}_-$"
             r"\\ $S=H-P$\end{tabular}",
             ha="left", va="center", fontsize=10, rotation=-60)
    ax2.text(0.50, -0.11,
             r"$b = 0$: empty fiber over the open edge",
             ha="center", va="top", fontsize=10)

    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(-0.30, 1.05)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.tight_layout()
    pdf = OUT / "simplex_faces.pdf"
    png = OUT / "simplex_faces.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
