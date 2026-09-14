#!/usr/bin/env python3
"""Null model for the Type II locus on the (b, c, d) simplex.

Samples symmetric-form spectra from the balanced mixture
(1/2)(gamma_{-1} + gamma_{+1}): n eigenvalues drawn from a gamma
distribution with mode +1 and n from its reflection with mode -1
(equal counts on both sides, as required by balance). The archetype
weights (b, c, d) are scale-invariant, so only the gamma shape
parameter matters; SHAPE matches the Type II panel of the schematic
figure. Each run yields one point on the 2-simplex; the figure shows
RUNS points, full triangle and the sub-triangle b >= ZOOM_B
magnified.

Saves figures/qk_geometry/gamma_null_simplex.{pdf,png}.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_simplex_scatter import ZOOM_B, bary, draw_triangle

RUNS = 100
N_SIDE = 64      # eigenvalues per side, as in a gpt2 head
SHAPE = 1.8      # gamma shape of the Type II schematic
SEED = 0


def sample_weights(rng):
    scale = 1.0 / (SHAPE - 1.0)          # mode at 1
    lp = np.sort(rng.gamma(SHAPE, scale, N_SIDE))[::-1]
    ln = np.sort(rng.gamma(SHAPE, scale, N_SIDE))[::-1]
    norm2 = (lp**2).sum() + (ln**2).sum()
    lp = lp / np.sqrt(norm2)
    ln = ln / np.sqrt(norm2)
    u = lp - ln
    return 2 * lp @ ln, (u[u > 0] ** 2).sum(), (u[u < 0] ** 2).sum()


def main() -> None:
    rng = np.random.default_rng(SEED)
    w = np.array([sample_weights(rng) for _ in range(RUNS)])
    b, c, d = w[:, 0], w[:, 1], w[:, 2]

    rel = np.abs(c - d) / (c + d)
    print(f"{RUNS} samples of (1/2)(gamma_-1 + gamma_+1), "
          f"n = {N_SIDE} per side, shape = {SHAPE}:")
    print(f"  b: median {np.median(b):.4f}  "
          f"range [{b.min():.4f}, {b.max():.4f}]")
    print(f"  c + d: median {np.median(c + d):.4f}")
    print(f"  |c-d|/(c+d): median {np.median(rel):.2f}  "
          f"range [{rel.min():.2f}, {rel.max():.2f}]")

    Pb = (0.5, np.sqrt(3) / 2)
    Pc = (0.0, 0.0)
    Pd = (1.0, 0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.4))

    draw_triangle(ax1, Pb, Pc, Pd)
    t = ZOOM_B
    Qc = bary(t, 1 - t, 0, Pb, Pc, Pd)
    Qd = bary(t, 0, 1 - t, Pb, Pc, Pd)
    ax1.plot([Qc[0], Qd[0]], [Qc[1], Qd[1]], color="#888888",
             linewidth=0.8, zorder=1)
    x, y = bary(b, c, d, Pb, Pc, Pd)
    ax1.scatter(x, y, s=14, facecolor="#555555", edgecolor="none",
                zorder=2)

    # No vertex labels: the corners of the magnified sub-triangle are
    # (b,c,d) = (t, 1-t, 0) and (t, 0, 1-t), not c = 1 and d = 1.
    draw_triangle(ax2, Pb, Pc, Pd, vertex_labels=False)
    inside = b >= t
    print(f"  zoom {{b >= {t}}}: {inside.sum()} of {RUNS} inside")
    x, y = bary((b[inside] - t) / (1 - t), c[inside] / (1 - t),
                d[inside] / (1 - t), Pb, Pc, Pd)
    ax2.scatter(x, y, s=14, facecolor="#555555", edgecolor="none",
                zorder=2)
    ax2.text(0.5, -0.05, rf"the sub-triangle $b \geq {t}$, magnified",
             ha="center", va="top", fontsize=10, transform=ax2.transAxes)

    fig.tight_layout()
    pdf = OUT / "gamma_null_simplex.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "gamma_null_simplex.png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
