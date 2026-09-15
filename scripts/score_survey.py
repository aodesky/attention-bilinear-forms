#!/usr/bin/env python3
"""Symmetric-energy and pairing figures, and the balanced-spectrum check.

One panel per model, laid out on a common grid, for the scores of the
paper; balance is checked directly from eigenvalue counts:

* the symmetric energy S, against a random baseline of the same shape;
* equal positive and negative eigenvalue counts, as predicted by the
  balanced-attention theorem;
* the pairing score P, against the same random baseline, and as a
  histogram over the heads of each model.

The trained scores are read from ``survey_results/head_statistics.csv``
(written by ``head_statistics.py``).  The baseline draws W_K and W_Q with
independent standard normal entries at each model's own shape (N, n) and
forms L = W_K^T W_Q, which is the ensemble of the corollary on random
QK products; it is redrawn here rather than cached, from a fixed seed.

Writes, to figures/qk_geometry/:
    combined_qk_diagnostics.pdf            symmetric energy, trained vs random
    combined_qk_diagnostics_random.pdf     pairing score, trained vs random
    pairing_score_histogram.pdf            pairing score, per-model histograms

Also reports the lobe proportionality error and pairing-bound gap in
Observation 5, from the same stored statistics used in Section 6.2.1.

Usage:
    python score_survey.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

OUT = ms.PROJECT / "figures" / "qk_geometry"
STATS = ms.PROJECT / "scripts" / "survey_results" / "head_statistics.csv"
SEED = 0
CMAP = "RdBu_r"


def grid(n_panels: int) -> tuple[int, int]:
    """Rows and columns for a panel grid of the given size."""
    cols = 4 if n_panels > 9 else 3
    return (n_panels + cols - 1) // cols, cols


def random_baseline(info: ms.ModelInfo, rng: np.random.Generator,
                    score: str) -> np.ndarray:
    """The score at heads of a random QK product of the model's shape."""
    out = np.empty(info.n_heads_total)
    for i in range(info.n_heads_total):
        Wk = rng.standard_normal((info.head_dim, info.d_model))
        Wq = rng.standard_normal((info.head_dim, info.d_model))
        # restrict to the reduced subspace, as the extraction does
        basis = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
        L = (Wk @ basis).T @ (Wq @ basis)
        S, T = ms.symmetric_antisymmetric(L)
        if score == "symmetric_energy":
            out[i] = (S ** 2).sum() / (L ** 2).sum()
        elif score == "pairing":
            lam_p, lam_m = ms.sorted_lobes(S)
            out[i] = 1.0 - np.linalg.norm(lam_p - lam_m) / np.sqrt((S ** 2).sum())
        else:
            raise ValueError(f"unknown score {score!r}")
    return out


def heatmap_figure(df: pd.DataFrame, column: str, path: Path, *,
                   centre: float | None, label: str,
                   baseline: str | None = None) -> None:
    """One heat map per model, heads arranged by layer and head index.

    With ``baseline`` set, each model contributes two panels: the
    trained scores and a random baseline at the same shape.
    """
    infos = ms.all_model_info()
    rng = np.random.default_rng(SEED)
    panels: list[tuple[str, np.ndarray, ms.ModelInfo]] = []
    for info in infos:
        d = df[df.model == info.tag]
        panels.append((info.tag, d[column].to_numpy(), info))
        if baseline is not None:
            panels.append((f"{info.tag} (random)",
                           random_baseline(info, rng, baseline), info))

    rows, cols = grid(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.5 * rows),
                             squeeze=False)
    values = np.concatenate([v for _, v, _ in panels])
    lo, hi = float(values.min()), float(values.max())
    if centre is not None and lo < centre < hi:
        norm, lo, hi = TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi), None, None
    elif hi - lo < 1e-12:
        # a constant score (the balance is one at every head): show it as
        # the centre of the scale rather than letting imshow pick limits
        norm, lo, hi = None, lo - 0.5, hi + 0.5
    else:
        norm = None

    for ax, (title, values_, info) in zip(axes.ravel(), panels):
        image = values_.reshape(info.n_layers, info.n_heads)
        im = ax.imshow(image, aspect="auto", cmap=CMAP, norm=norm,
                       vmin=lo, vmax=hi)
        ax.set_title(rf"\texttt{{{title}}}", fontsize=8)
        ax.set_xlabel("head", fontsize=7)
        ax.set_ylabel("layer", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label(
        label, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def histogram_figure(df: pd.DataFrame, path: Path) -> None:
    """The distribution of the pairing score over the heads of each model."""
    infos = ms.all_model_info()
    rows, cols = grid(len(infos))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.1 * rows),
                             squeeze=False)
    bins = np.linspace(0.0, 1.0, 41)
    cmap = plt.get_cmap(CMAP)

    for ax, info in zip(axes.ravel(), infos):
        values = df.loc[df.model == info.tag, "pairing"].to_numpy()
        counts, edges = np.histogram(values, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=np.diff(edges), align="center",
               color=[cmap(c) for c in centres], edgecolor="black",
               linewidth=0.2)
        ax.axvline(float(np.median(values)), color="black", linewidth=0.8,
                   linestyle="--")
        ax.set_title(rf"\texttt{{{info.tag}}}", fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(r"$\mathscr{P}$", fontsize=8)
        ax.set_ylabel("heads", fontsize=7)
    for ax in axes.ravel()[len(infos):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def print_balance_summary(df: pd.DataFrame) -> None:
    """Check Observation 1 directly from positive and negative eigenvalue counts."""
    print("Balanced symmetric spectra (equal positive and negative counts):")
    for tag, rows in df.groupby("model", sort=False):
        balanced = int((rows.n_pos == rows.n_neg).sum())
        print(f"  {tag:26s} {balanced}/{len(rows)} heads")


def print_pairing_summary(df: pd.DataFrame) -> None:
    """Report the quantities in Observation 5 and the proportional-lobes discussion."""
    columns = ("lobe_departure", "pairing_gap", "pairing")
    print("Proportional lobes: median [10%, 90%] over heads")
    print("lobe_departure = ||lambda_- - rho lambda_+|| / ||lambda_-||")
    print("pairing_gap = bound at the head's rho minus its pairing score")
    print(f"{'model':26s} {'heads':>6s}  " +
          "  ".join(f"{column:>25s}" for column in columns))
    groups = [(tag, df[df.model == tag]) for tag in ms.TAGS]
    groups.append(("pooled", df))
    for tag, rows in groups:
        cells = []
        for column in columns:
            low, median, high = rows[column].quantile([0.1, 0.5, 0.9])
            cells.append(f"{median:.3f} [{low:.3f}, {high:.3f}]")
        print(f"{tag:26s} {len(rows):6d}  " +
              "  ".join(f"{cell:>25s}" for cell in cells))
    print("Per-model median ranges:")
    for column in columns:
        medians = df.groupby("model")[column].median()
        print(f"  {column}: {medians.min():.3f} to {medians.max():.3f}")


def main() -> None:
    if not STATS.exists():
        raise FileNotFoundError(
            f"{STATS} not found; run head_statistics.py first")
    df = pd.read_csv(STATS)
    missing = set(ms.TAGS) - set(df.model.unique())
    if missing:
        raise ValueError(f"head statistics are missing for {sorted(missing)}")

    heatmap_figure(df, "symmetric_energy",
                   OUT / "combined_qk_diagnostics.pdf",
                   centre=0.5, label=r"$\mathscr{S}$",
                   baseline="symmetric_energy")
    print_balance_summary(df)
    print_pairing_summary(df)
    heatmap_figure(df, "pairing", OUT / "combined_qk_diagnostics_random.pdf",
                   centre=0.5, label=r"$\mathscr{P}$", baseline="pairing")
    histogram_figure(df, OUT / "pairing_score_histogram.pdf")


if __name__ == "__main__":
    main()
