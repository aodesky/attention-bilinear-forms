"""Histogram of per-head pairing scores across the eight models.

Reads scripts/qk_geometry_results/<model>/head_summary.csv for each of
the eight models and renders one histogram panel per model. Bars are
colored by their pairing score using the same RdBu_r colormap centered
at 1/2 as the pairing score heatmap figure
(regenerate_qk_random_baseline_figure.py); the normalization here uses
the trained scores only, so the color scale can differ slightly from
the heatmap's (whose normalization includes the random baselines).

Saves figures/qk_geometry/pairing_score_histogram.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "qk_geometry_results"
OUT = PROJECT / "figures" / "qk_geometry"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("distilgpt2", "distilgpt2"),
    ("gpt2", "gpt2"),
    ("gpt2-medium", "gpt2-medium"),
    ("gpt2-large", "gpt2-large"),
    ("facebook__opt-125m", "opt-125m"),
    ("facebook__opt-350m", "opt-350m"),
    ("facebook__opt-1.3b", "opt-1.3b"),
    ("EleutherAI__pythia-70m-deduped", "pythia-70m-deduped"),
]

CMAP = "RdBu_r"
BINS = np.linspace(0.0, 1.0, 21)  # bin width 0.05


def main():
    cmap = plt.get_cmap(CMAP)

    scores = {}
    for model_dir, _ in MODELS:
        df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
        scores[model_dir] = df["S_pair_score"].dropna().values

    # Same normalization convention as the pairing score heatmap:
    # diverging colormap symmetric around the midpoint 1/2.
    all_vals = np.concatenate(list(scores.values()))
    half_width = float(np.nanmax(np.abs(all_vals - 0.5)))
    norm = mpl.colors.Normalize(vmin=0.5 - half_width, vmax=0.5 + half_width)

    n_cols = 4
    n_rows = (len(MODELS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols + 1.5, 2.8 * n_rows),
        gridspec_kw={"wspace": 0.3, "hspace": 0.5},
    )

    centers = 0.5 * (BINS[:-1] + BINS[1:])
    for ax, (model_dir, label) in zip(axes.ravel(), MODELS):
        counts, edges = np.histogram(scores[model_dir], bins=BINS)
        ax.bar(
            edges[:-1], counts, width=np.diff(edges), align="edge",
            color=[cmap(norm(c)) for c in centers],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"pairing score $\mathscr{P}$", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

    for row in axes:
        row[0].set_ylabel("count of heads", fontsize=9)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        shrink=0.9, pad=0.02, fraction=0.03,
    )
    cbar.set_label(r"pairing score $\mathscr{P}$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(r"Per-head pairing score $\mathscr{P}$", y=1.06, fontsize=12)

    pdf_path = OUT / "pairing_score_histogram.pdf"
    png_path = OUT / "pairing_score_histogram.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")

    print()
    print("Per-model pairing score summary:")
    print(f"{'model':25s}  {'mean P':>8s}  {'min P':>8s}  {'max P':>8s}")
    for model_dir, label in MODELS:
        vals = scores[model_dir]
        print(f"{label:25s}  {vals.mean():>8.4f}  {vals.min():>8.4f}  {vals.max():>8.4f}")


if __name__ == "__main__":
    main()
