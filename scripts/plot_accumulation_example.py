#!/usr/bin/env python3
"""Illustration of the accumulation proposition.

Samples spectra of gamma-heads with modes mu = -1/2, nu = 1
(rho = 1/2) and parameter k = 2, computes their weights (b, c, d),
and plots each sample at its coordinate b on the edge {d = 0} of the
triangle Delta, drawn horizontally, for two sample sizes, together
with the predicted accumulation point
b = 2 rho / (1 + rho^2) = 0.8.

Saves figures/qk_geometry/accumulation_example.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"

K = 2.0
RHO = 0.5          # modes mu = -1/2, nu = 1
RUNS = 100
SEED = 0


def sample_weights(rng, n):
    scale = 1.0 / (K - 1.0)            # nu = 1
    lp = np.sort(rng.gamma(K, scale, n))[::-1]
    ln = np.sort(rng.gamma(K, RHO * scale, n))[::-1]
    norm2 = (lp**2).sum() + (ln**2).sum()
    lp, ln = lp / np.sqrt(norm2), ln / np.sqrt(norm2)
    u = lp - ln
    return 2 * lp @ ln, (u[u > 0] ** 2).sum(), (u[u < 0] ** 2).sum()


def main() -> None:
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(8.0, 1.9))
    # the edge {d = 0}, drawn horizontally, parameterized by b
    ax.plot([0, 1], [0, 0], color="black", linewidth=1.2, zorder=1)
    for xv, lab in [(0.0, r"$c = 1$"), (1.0, r"$b = 1$")]:
        ax.plot(xv, 0, marker="o", markersize=4, color="black",
                zorder=3)
        ax.text(xv, -0.55, lab, ha="center", va="top", fontsize=11)

    for n, color, yv, label in [(64, "#bbbbbb", 0.45, r"$n = 64$"),
                                (1024, "#555555", 0.9,
                                 r"$n = 1024$")]:
        w = np.array([sample_weights(rng, n) for _ in range(RUNS)])
        ax.scatter(w[:, 0], np.full(RUNS, yv), s=14, facecolor=color,
                   edgecolor="none", zorder=2, label=label)

    den = 1 + RHO**2
    ax.plot(2 * RHO / den, 0, marker="x", markersize=11,
            color="black", linestyle="none", markeredgewidth=2.0,
            zorder=3, label=r"accumulation point")
    ax.legend(fontsize=9, frameon=True, loc="center left",
              bbox_to_anchor=(0.02, 0.72))
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-1.1, 1.5)
    ax.axis("off")

    fig.tight_layout()
    pdf = OUT / "accumulation_example.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "accumulation_example.png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
