"""Build a 4x4 figure of the pairing score across the eight models.

Rows 1-2: trained pairing score of S for each of the eight models
       (distilgpt2, gpt2, gpt2-medium, gpt2-large, opt-125m,
       opt-350m, opt-1.3b, pythia-70m-deduped).
Rows 3-4: a random baseline at each model's (D, d, layers, heads)
       shape, with W_K, W_Q ~ N(0, 1) iid (low-rank Gaussian QK
       products).

All panels share a single colormap and colorbar.

Saves figures/qk_geometry/combined_qk_diagnostics_random.pdf (and .png).
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
SEED = 0
EIG_EPS_REL = 1e-7


def load_pivot(model_dir: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
    return df.pivot(index="layer", columns="head", values=column).sort_index()


def model_shape(model_dir: str) -> tuple[int, int, int, int]:
    df = pd.read_csv(RESULTS / model_dir / "head_summary.csv")
    num_layers = int(df["layer"].max()) + 1
    num_heads = int(df["head"].max()) + 1
    d_model = int(df["d_model"].iloc[0])
    head_dim = int(df["head_dim"].iloc[0])
    return num_layers, num_heads, d_model, head_dim


def pairing_score_for_head(d_model: int, head_dim: int,
                           rng: np.random.Generator) -> float:
    """Compute the pairing score P_H for one random head W_K, W_Q ~ N(0,1)
    of shape (head_dim, d_model). Uses the low-rank reduction: S = (L+L^T)/2
    with L = W_K^T W_Q has rank <= 2*head_dim, and its nonzero spectrum
    equals the spectrum of the reduced matrix on span(W_K^T, W_Q^T)."""
    Wq = rng.standard_normal((head_dim, d_model))
    Wk = rng.standard_normal((head_dim, d_model))
    U = np.concatenate([Wk.T, Wq.T], axis=1)
    Q, _ = np.linalg.qr(U, mode="reduced")  # columns: orthonormal basis of im(Wk^T)+im(Wq^T)
    KR = Wk @ Q
    QR = Wq @ Q
    Sred = 0.5 * (KR.T @ QR + QR.T @ KR)
    S_frob = float(np.linalg.norm(Sred, "fro"))
    if S_frob <= 0:
        return float("nan")
    evals = np.linalg.eigvalsh(Sred)
    max_abs = float(np.max(np.abs(evals))) if evals.size else 0.0
    tol = EIG_EPS_REL * max_abs
    pos = np.sort(evals[evals > tol])[::-1]
    neg = np.sort(-evals[evals < -tol])[::-1]
    # Pad the shorter list with zeros.
    m = max(len(pos), len(neg))
    pos_padded = np.concatenate([pos, np.zeros(m - len(pos))])
    neg_padded = np.concatenate([neg, np.zeros(m - len(neg))])
    diff_norm = float(np.linalg.norm(pos_padded - neg_padded))
    return 1.0 - diff_norm / S_frob


def random_pairing_grid(num_layers: int, num_heads: int,
                        d_model: int, head_dim: int,
                        rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            grid[layer, head] = pairing_score_for_head(d_model, head_dim, rng)
    return grid


def main():
    trained_pivots = {m: load_pivot(m, "S_pair_score") for m, _ in MODELS}

    rng = np.random.default_rng(SEED)
    random_grids = {}
    for model_dir, _ in MODELS:
        num_layers, num_heads, d_model, head_dim = model_shape(model_dir)
        random_grids[model_dir] = random_pairing_grid(
            num_layers, num_heads, d_model, head_dim, rng,
        )

    # Joint colormap range across all sixteen panels, symmetric around 0.5
    # so that diverging colormap places white at the natural midpoint.
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
    group_rows = (len(MODELS) + n_cols - 1) // n_cols
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

    cbar = fig.colorbar(
        last_im, ax=axes.ravel().tolist(),
        shrink=0.85, pad=0.02, fraction=0.03,
    )
    cbar.ax.tick_params(labelsize=7)

    row_labels = ["trained", "random baseline"]
    for group, label in enumerate(row_labels):
        top = axes[group * group_rows, 0].get_position()
        bot = axes[group * group_rows + group_rows - 1, 0].get_position()
        y_center = 0.5 * (bot.y0 + top.y1)
        fig.text(
            0.02, y_center, label,
            rotation=90, va="center", ha="center", fontsize=10,
        )

    fig.suptitle(r"Pairing score $\mathscr{P}$", y=0.995, fontsize=12)

    pdf_path = OUT / "combined_qk_diagnostics_random.pdf"
    png_path = OUT / "combined_qk_diagnostics_random.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Print summary stats for the body text.
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print()
    print("Per-model pairing score summary:")
    print(f"{'model':25s}  {'trained mean':>12s}  {'trained min':>12s}  {'random mean':>12s}  {'random std':>10s}")
    for model_dir, label in MODELS:
        tp = trained_pivots[model_dir].values.ravel()
        rp = random_grids[model_dir].ravel()
        print(f"{label:25s}  {np.nanmean(tp):>12.4f}  {np.nanmin(tp):>12.4f}  {np.nanmean(rp):>12.4f}  {np.nanstd(rp):>10.4f}")


if __name__ == "__main__":
    main()
