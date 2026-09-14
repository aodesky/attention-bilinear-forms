#!/usr/bin/env python3
"""Overview diagnostics for the antisymmetric parts T of the heads.

Computes, for gpt2, the sorted unit-normalized singular value
profiles sigma(T) of the reduced antisymmetric parts (cached to
scripts/cluster_results/gpt2_sorted_asym.npy), and applies the
paper's existing diagnostics to them: mean profile, sample
histograms, SVD of the profile matrix, k-means clustering with
silhouette scores, cross-tabulation against the S-side clusters
(Types I+/II/I-), effective ranks, and the even-moment statistic
E_T = sum sigma^2/(1 - sigma^2) of the normalized profile.

Figures are written to figures/qk_geometry/, statistics printed to
stdout (consumed by reports/antisymmetric_parts.tex and by the
antisymmetric-components section of attention.tex).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
GEOM = PROJECT / "scripts" / "qk_geometry_results"
FIGS = PROJECT / "figures" / "qk_geometry"
FIGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_head_spectra import CLUSTER_NAMES, MARKER_TEX, marker_kwargs


def compute_asym_profiles():
    cache = RESULTS / "gpt2_sorted_asym.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    from qk_geometry_diagnostics import iter_qk_heads, load_model
    model = load_model("gpt2")
    model.eval()
    keys_ref = [tuple(k) for k in np.load(RESULTS / "gpt2_head_keys.npy")]
    profiles = {}
    with torch.no_grad():
        for hw in iter_qk_heads(model, "gpt2"):
            Wk = hw.Wk.double()
            Wq = hw.Wq.double()
            U = torch.cat([Wk.T, Wq.T], dim=1)
            Qb, _ = torch.linalg.qr(U, mode="reduced")
            Kr = Wk @ Qb
            Qr = Wq @ Qb
            Lred = Kr.T @ Qr
            Tred = 0.5 * (Lred - Lred.T)
            sv = torch.linalg.svdvals(Tred).numpy()
            sv = np.sort(sv)[::-1]
            profiles[(hw.layer, hw.head)] = sv / np.linalg.norm(sv)
    X = np.array([profiles[k] for k in keys_ref])
    np.save(cache, X)
    return X


def best_kmeans(X, k, seeds=50):
    best = None
    for seed in range(seeds):
        cent, lab = kmeans2(X, k, minit="++", seed=seed)
        if len(np.unique(lab)) < k:
            continue
        inertia = ((X - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    if best is None:
        raise RuntimeError(f"all restarts degenerate for k={k}")
    return best[1]


def s_type_labels():
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    _, raw = kmeans2(X, 3, minit="++", seed=0)
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum()
                    for r in X])
    order = sorted(range(3), key=lambda c: -imb[raw == c].mean())
    lab = np.empty_like(raw)
    for new, old in enumerate(order, start=1):
        lab[raw == old] = new
    return X, lab


def main() -> None:
    XT = compute_asym_profiles()
    XS, s_lab = s_type_labels()
    keys = [tuple(k) for k in np.load(RESULTS / "gpt2_head_keys.npy")]
    df = pd.read_csv(GEOM / "gpt2" / "head_summary.csv")
    df = df.set_index(df.apply(
        lambda r: (int(r["layer"]), int(r["head"])), axis=1)).loc[keys]

    print("=== basic statistics (gpt2) ===")
    a_w = df["A_energy_ratio"].values
    print(f"a = ||T||^2/||L||^2: mean {a_w.mean():.3f}, "
          f"range [{a_w.min():.3f}, {a_w.max():.3f}]")
    print(f"rank(T_red): unique {sorted(df['A_rank'].unique())}")
    print(f"pairing rel. error: max {df['A_pair_rel_error'].max():.1e}")
    effT = df["A_effective_rank"].values
    effS = df["S_effective_rank"].values
    print(f"eff rank sigma(T): median {np.median(effT):.1f}, "
          f"range [{effT.min():.1f}, {effT.max():.1f}]")
    print(f"corr(effT, effS) = {np.corrcoef(effT, effS)[0,1]:.3f}")
    top_share = df["A_max_sval"].values**2 * 2 / (df["A_frob"].values**2)
    print(f"top-pair share 2*smax^2/||T||^2: median {np.median(top_share):.3f}, "
          f"max {top_share.max():.3f}")

    ET = (XT**2 / (1 - XT**2)).sum(axis=1)
    ES = (XS**2 / (1 - XS**2)).sum(axis=1)
    print(f"E_T: median {np.median(ET):.3f}, range "
          f"[{ET.min():.3f}, {ET.max():.3f}]; corr(E_T, E_S) = "
          f"{np.corrcoef(ET, ES)[0,1]:.3f}")

    # SVD of profile matrix
    U, sv, Vt = np.linalg.svd(XT, full_matrices=False)
    shares = sv**2 / (sv**2).sum()
    emb = U[:, :3] * sv[:3]
    # Orient the axes as in cluster_head_spectra.svd_scores.
    if emb[:, 0].mean() < 0:
        emb[:, 0] *= -1
    for j, stat in [(1, effT), (2, ET)]:
        if np.corrcoef(emb[:, j], stat)[0, 1] < 0:
            emb[:, j] *= -1
    print(f"SVD shares: {100*shares[0]:.1f}% / {100*shares[1]:.1f}% / "
          f"{100*shares[2]:.1f}%  coord1 {emb[:,0].mean():.3f} "
          f"+/- {emb[:,0].std():.3f}")

    print("=== silhouette (best of 50 restarts) ===")
    for k in range(2, 7):
        lab = best_kmeans(XT, k)
        s = silhouette_score(XT, lab)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        print(f"k={k}  silhouette={s:.3f}  sizes={sizes}")

    labT = best_kmeans(XT, 3)
    order = sorted(range(3), key=lambda c: -effT[labT == c].mean())
    relab = np.empty_like(labT)
    for new, old in enumerate(order, start=1):
        relab[labT == old] = new
    labT = relab
    print("=== cross-tab: S-types (rows) x T-clusters (cols) ===")
    for st in (1, 2, 3):
        row = [int(np.sum((s_lab == st) & (labT == tc))) for tc in (1, 2, 3)]
        print(f"{CLUSTER_NAMES[st-1]:12s} {row}")
    for tc in (1, 2, 3):
        m = labT == tc
        print(f"T-cluster {tc}: n={m.sum()}, mean effT "
              f"{effT[m].mean():.1f}, mean E_T {ET[m].mean():.2f}")

    # ---------- figures ----------
    # mean profile
    xbar = XT.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1.05 * xbar.max(), 37)
    ax.hist(xbar, bins=bins, color="#cccccc", edgecolor="black",
            linewidth=0.5)
    ax.set_xlabel("singular value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGS / "asym_mean_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    # sample heads
    rng = np.random.default_rng(0)
    pick = rng.choice(len(XT), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.2))
    for ax, i in zip(axes.ravel(), pick):
        bins = np.linspace(0, 1.05 * XT[i].max(), 31)
        ax.hist(XT[i], bins=bins, color="#cccccc", edgecolor="black",
                linewidth=0.4)
        l, h = keys[i]
        ax.set_title(rf"L{l}H{h} ({MARKER_TEX[s_lab[i]-1]})", fontsize=9)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / "asym_samples.pdf", bbox_inches="tight")
    plt.close(fig)

    # T-cluster extreme averages
    gmean = XT.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for tc, ax in zip((1, 2, 3), axes):
        idx = np.where(labT == tc)[0]
        dvec = XT[idx].mean(axis=0) - gmean
        dvec /= np.linalg.norm(dvec)
        reps = idx[np.argsort(-(XT[idx] - gmean) @ dvec)[:10]]
        avg = XT[reps].mean(axis=0)
        bins = np.linspace(0, 1.05 * avg.max(), 41)
        ax.hist(avg, bins=bins, color="#cccccc", edgecolor="black",
                linewidth=0.4)
        ax.set_title(rf"$T$-cluster {tc} ($n={idx.size}$)", fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("singular value")
    fig.tight_layout()
    fig.savefig(FIGS / "asym_barycenters.pdf", bbox_inches="tight")
    plt.close(fig)

    # SVD scatter with S-type markers
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for st in (1, 2, 3):
        m = s_lab == st
        ax.scatter(emb[m, 1], emb[m, 2], **marker_kwargs(st, s=36),
                   label=f"{CLUSTER_NAMES[st-1]} ($n={m.sum()}$)")
    ax.set_xlabel("component 2")
    ax.set_ylabel("component 3")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGS / "asym_svd_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    # effective ranks S vs T
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for st in (1, 2, 3):
        m = s_lab == st
        ax.scatter(effS[m], effT[m], **marker_kwargs(st, s=36),
                   label=f"{CLUSTER_NAMES[st-1]} ($n={m.sum()}$)")
    lo = min(effS.min(), effT.min()) - 4
    hi = max(effS.max(), effT.max()) + 4
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.6,
            linestyle="--")
    ax.set_xlabel(r"effective rank of $\mathrm{spec}(S)$")
    ax.set_ylabel(r"effective rank of $\sigma(T)$")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGS / "asym_effrank_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    # E_S vs E_T
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for st in (1, 2, 3):
        m = s_lab == st
        ax.scatter(ES[m], ET[m], **marker_kwargs(st, s=36),
                   label=f"{CLUSTER_NAMES[st-1]} ($n={m.sum()}$)")
    ax.set_xlabel(r"$E$ of $\mathrm{spec}(S)$")
    ax.set_ylabel(r"$E_T$ of $\sigma(T)$")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGS / "asym_E_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    # cross-model summary
    print("=== cross-model summary ===")
    dirs = [("distilgpt2", "distilgpt2"), ("gpt2", "gpt2"),
            ("gpt2-medium", "gpt2-medium"), ("gpt2-large", "gpt2-large"),
            ("facebook__opt-125m", "opt-125m"),
            ("facebook__opt-350m", "opt-350m"),
            ("facebook__opt-1.3b", "opt-1.3b"),
            ("EleutherAI__pythia-70m-deduped", "pythia-70m-deduped")]
    for mdir, label in dirs:
        f = GEOM / mdir / "head_summary.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        full = (m["A_rank"] == 2 * m["head_dim"]).mean()
        print(f"{label:20s} mean a {1 - m['S_energy_ratio'].mean():.3f}  "
              f"effT median {m['A_effective_rank'].median():6.1f}  "
              f"full rank frac {full:.2f}  "
              f"pair err max {m['A_pair_rel_error'].max():.0e}")

    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()
