#!/usr/bin/env python3
"""Attracting subloci of the model family lambda ~ (gamma_mu + gamma_nu)/2.

Modes mu < 0 < nu. The archetype weights are scale-invariant and
gamma_{|mu|} is a pure rescaling of gamma_nu (same shape parameter),
so the family depends only on the mode ratio rho = |mu| / nu. As
n -> infinity the sorted lists become proportional,
(lambda_-)_j ~ rho (lambda_+)_j, which forces d = 0 and predicts the
attracting point

    (b, c, d) = (2 rho / (1 + rho^2), (1 - rho)^2 / (1 + rho^2), 0)

on the edge {d = 0} of the simplex, degenerating to the b = 1
vertex at rho = 1 (Proposition prop:accumulation of the paper).

Outputs:
  figures/qk_geometry/gamma_family_simplex.{pdf,png}
      clouds of RUNS samples for several rho on the triangle, with the
      predicted attracting points marked
  figures/qk_geometry/gamma_family_dominance.{pdf,png}
      fraction of samples with d = 0 exactly, and median d, vs rho
  printed: sample means vs the predicted attractors
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
from plot_simplex_scatter import bary, draw_triangle

RUNS = 100        # samples per rho in the cloud figure
RUNS_SWEEP = 200  # samples per rho in the d = 0 sweep
N_SIDE = 64
SHAPE = 1.8
SEED = 0

RHOS_CLOUD = [1.0, 0.9, 0.75, 0.5, 0.25]
GRAYS = ["#333333", "#555555", "#777777", "#999999", "#bbbbbb"]


def sample_weights(rng, rho):
    scale = 1.0 / (SHAPE - 1.0)
    lp = np.sort(rng.gamma(SHAPE, scale, N_SIDE))[::-1]
    ln = np.sort(rng.gamma(SHAPE, rho * scale, N_SIDE))[::-1]
    norm2 = (lp**2).sum() + (ln**2).sum()
    lp = lp / np.sqrt(norm2)
    ln = ln / np.sqrt(norm2)
    u = lp - ln
    return 2 * lp @ ln, (u[u > 0] ** 2).sum(), (u[u < 0] ** 2).sum()


def predicted(rho):
    den = 1 + rho**2
    return 2 * rho / den, (1 - rho) ** 2 / den, 0.0


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- figure 1: clouds on the triangle with predicted attractors
    Pb = (0.5, np.sqrt(3) / 2)
    Pc = (0.0, 0.0)
    Pd = (1.0, 0.0)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    draw_triangle(ax, Pb, Pc, Pd)

    print(f"{'rho':>5s} {'mean (b,c,d)':>28s} {'predicted (b,c,d)':>28s} "
          f"{'P(d=0)':>7s}")
    for rho, gray in zip(RHOS_CLOUD, GRAYS):
        w = np.array([sample_weights(rng, rho) for _ in range(RUNS)])
        x, y = bary(w[:, 0], w[:, 1], w[:, 2], Pb, Pc, Pd)
        ax.scatter(x, y, s=12, facecolor=gray, edgecolor="none",
                   zorder=2, label=rf"$\rho = {rho:g}$")
        pb, pc, pd = predicted(rho)
        px, py = bary(pb, pc, pd, Pb, Pc, Pd)
        ax.plot(px, py, marker="x", markersize=9, color="black",
                markeredgewidth=1.6, zorder=3)
        mean = w.mean(axis=0)
        print(f"{rho:5.2f} "
              f"({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.4f})"
              f"{'':>6s}({pb:.3f}, {pc:.3f}, {pd:.3f})"
              f"{(w[:, 2] == 0).mean():9.2f}")

    # the limiting curve: the whole edge {d = 0}
    ts = np.linspace(0, 1, 100)
    ex, ey = bary(ts, 1 - ts, 0 * ts, Pb, Pc, Pd)
    ax.plot(ex, ey, color="black", linewidth=0.6, alpha=0.4, zorder=1)
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.set_title(r"$\lambda \sim (\gamma_\mu + \gamma_\nu)/2$: "
                 rf"${RUNS}$ samples per ratio $\rho = |\mu|/\nu$, "
                 r"predicted attractors "
                 r"($\times$)", fontsize=11)

    fig.tight_layout()
    pdf = OUT / "gamma_family_simplex.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "gamma_family_simplex.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")

    # ---- figure 2: attraction to the cone {d = 0} as rho varies
    rhos = np.linspace(0.0, 1.0, 21)
    frac0 = np.empty_like(rhos)
    med_d = np.empty_like(rhos)
    for i, rho in enumerate(rhos):
        w = np.array([sample_weights(rng, rho)
                      for _ in range(RUNS_SWEEP)])
        frac0[i] = (w[:, 2] == 0).mean()
        med_d[i] = np.median(w[:, 2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax1.plot(rhos, frac0, marker="o", markersize=4, color="black",
             linewidth=1.0)
    ax1.set_xlabel(r"$\rho = |\mu|/\nu$")
    ax1.set_ylabel(r"fraction of samples with $d = 0$ exactly")
    ax2.semilogy(rhos, np.maximum(med_d, 1e-12), marker="o",
                 markersize=4, color="black", linewidth=1.0)
    ax2.set_xlabel(r"$\rho = |\mu|/\nu$")
    ax2.set_ylabel(r"median $d$")
    fig.tight_layout()
    pdf = OUT / "gamma_family_dominance.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "gamma_family_dominance.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
