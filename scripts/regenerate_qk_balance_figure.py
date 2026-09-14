"""Build a 2x4 figure of the balance score B across the eight models.

The balance is identically 1 on every head by Proposition prop:forced-balance,
so this figure is a numerical sanity check of the proposition rather than a
diagnostic that varies.

Saves figures/qk_geometry/combined_qk_diagnostics_balance.pdf (and .png).
"""

from pathlib import Path

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


def load_pivot(model_dir: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
    return df.pivot(index="layer", columns="head", values=column).sort_index()


def main():
    pivots = {m: load_pivot(m, "S_inertia_balance") for m, _ in MODELS}

    n_cols = 4
    n_rows = (len(MODELS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols + 1.5, 2.4 * n_rows),
        gridspec_kw={
            "width_ratios": [1] * n_cols,
            "wspace": 0.25,
            "hspace": 0.55,
            "left": 0.05,
        },
    )

    last_im = None
    for i, (model_dir, model_label) in enumerate(MODELS):
        col_idx = i % n_cols
        ax = axes[i // n_cols, col_idx]
        pivot = pivots[model_dir]
        last_im = ax.imshow(
            pivot.values, aspect="auto", interpolation="nearest",
            cmap=CMAP, vmin=0.0, vmax=1.0,
        )
        ax.set_title(model_label, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("layer", fontsize=8)
        ax.set_xlabel("head", fontsize=8)
        ax.tick_params(labelsize=7)

    cbar = fig.colorbar(
        last_im, ax=axes.ravel().tolist(),
        shrink=0.85, pad=0.02, fraction=0.03,
    )
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(r"Balance score $\mathscr{B}$", y=1.10, fontsize=12)

    pdf_path = OUT / "combined_qk_diagnostics_balance.pdf"
    png_path = OUT / "combined_qk_diagnostics_balance.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
