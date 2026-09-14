"""Build a 4x4 figure of symmetric energy ratios for the eight models.

Rows 1-2: trained per-head symmetric energy for each of the eight
       models (distilgpt2, gpt2, gpt2-medium, gpt2-large, opt-125m,
       opt-350m, opt-1.3b, pythia-70m-deduped).
Rows 3-4: a random baseline of the same shape (same D, d, num_layers,
       num_heads) as each trained model, with W_K, W_Q drawn iid from
       N(0,1).

All panels share a single colormap and colorbar so the trained heads
can be directly compared to the random baseline.

Saves figures/qk_geometry/combined_qk_diagnostics.pdf (and .png).
"""

from pathlib import Path
import math

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
SEED = 0


def load_pivot(model_dir: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
    return df.pivot(index="layer", columns="head", values=column).sort_index()


def model_shape(model_dir: str) -> tuple[int, int, int, int]:
    """Return (num_layers, num_heads, d_model, head_dim) for the model."""
    df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
    num_layers = int(df["layer"].max()) + 1
    num_heads = int(df["head"].max()) + 1
    d_model = int(df["d_model"].iloc[0])
    head_dim = int(df["head_dim"].iloc[0])
    return num_layers, num_heads, d_model, head_dim


def random_sym_energy_grid(num_layers: int, num_heads: int,
                           d_model: int, head_dim: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample ||S||_F^2 / ||L||_F^2 for random W_K, W_Q ~ N(0,1)."""
    grid = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            Wq = rng.standard_normal((head_dim, d_model))
            Wk = rng.standard_normal((head_dim, d_model))
            U = np.concatenate([Wk.T, Wq.T], axis=1)
            Q, _ = np.linalg.qr(U, mode="reduced")  # columns: orthonormal basis of im(Wk^T)+im(Wq^T)
            KR = Wk @ Q
            QR = Wq @ Q
            Sred = 0.5 * (KR.T @ QR + QR.T @ KR)
            Ared = 0.5 * (KR.T @ QR - QR.T @ KR)
            S_frob_sq = float(np.linalg.norm(Sred, "fro") ** 2)
            A_frob_sq = float(np.linalg.norm(Ared, "fro") ** 2)
            L_frob_sq = S_frob_sq + A_frob_sq
            grid[layer, head] = S_frob_sq / L_frob_sq if L_frob_sq > 0 else np.nan
    return grid


def main():
    trained_pivots = {m: load_pivot(m, "S_energy_ratio") for m, _ in MODELS}

    rng = np.random.default_rng(SEED)
    random_grids = {}
    for model_dir, _ in MODELS:
        num_layers, num_heads, d_model, head_dim = model_shape(model_dir)
        random_grids[model_dir] = random_sym_energy_grid(
            num_layers, num_heads, d_model, head_dim, rng,
        )

    # Joint colormap range across all sixteen panels.
    # Symmetric range around 0.5 so the diverging colormap places white at the
    # random-baseline midpoint. Half-width is set to the largest deviation from
    # 0.5 observed in either the trained or random panels.
    all_vals = np.concatenate(
        [p.values.ravel() for p in trained_pivots.values()]
        + [g.ravel() for g in random_grids.values()]
    )
    half_width = float(np.nanmax(np.abs(all_vals - 0.5)))
    vmin = 0.5 - half_width
    vmax = 0.5 + half_width

    # 4 models per row: trained panels fill the top rows, the random
    # baseline the bottom rows.
    n_cols = 4
    group_rows = math.ceil(len(MODELS) / n_cols)
    n_rows = 2 * group_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols + 1.5, 2.4 * n_rows),
        gridspec_kw={
            "width_ratios": [1] * n_cols,
            "wspace": 0.25,
            "hspace": 0.55,
            "left": 0.10,
        },
    )

    last_im = None
    for i, (model_dir, model_label) in enumerate(MODELS):
        for group, grid in enumerate(
            [trained_pivots[model_dir].values, random_grids[model_dir]]
        ):
            row_idx = group * group_rows + i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(
                grid, aspect="auto", interpolation="nearest",
                cmap=CMAP, vmin=vmin, vmax=vmax,
            )
            ax.set_title(model_label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("layer", fontsize=8)
            ax.set_xlabel("head", fontsize=8)
            ax.tick_params(labelsize=7)

    # Single shared colorbar.
    cbar = fig.colorbar(
        last_im, ax=axes.ravel().tolist(),
        shrink=0.85, pad=0.02, fraction=0.03,
    )
    cbar.ax.tick_params(labelsize=7)

    # Group labels.
    for group, label in enumerate(["trained", "random baseline"]):
        top = axes[group * group_rows, 0].get_position()
        bot = axes[group * group_rows + group_rows - 1, 0].get_position()
        y_center = 0.5 * (bot.y0 + top.y1)
        fig.text(
            0.02, y_center, label,
            rotation=90, va="center", ha="center", fontsize=10,
        )

    fig.suptitle(
        r"Symmetric energy $\mathscr{S}$",
        y=0.995, fontsize=12,
    )

    pdf_path = OUT / "combined_qk_diagnostics.pdf"
    png_path = OUT / "combined_qk_diagnostics.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
