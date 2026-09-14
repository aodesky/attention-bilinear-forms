#!/usr/bin/env python3
"""
qk_geometry_diagnostics.py

Download the pretrained Hugging Face models analyzed in the paper,
extract the per-head Q/K projection matrices, and compute the spectral
statistics of the bilinear form

    L_h = W_K,h^T W_Q,h.

For each attention head, the script computes:
  - symmetric part S = (L + L^T)/2
  - antisymmetric part A = (L - L^T)/2   (denoted T in the paper)
  - energy ratios ||S||_F^2 / ||L||_F^2 and ||A||_F^2 / ||L||_F^2
  - inertia of S: number of positive / negative / near-zero eigenvalues
  - pairing of the positive and negative eigenvalue lists of S
  - singular values of A: rank, effective rank, and pairing

The computations use a low-rank reduction. Since W_Q,h and W_K,h have
shape (head_dim, d_model), the matrices S and A have rank at most
2*head_dim; their nonzero spectra are computed on an orthonormal basis
of span(W_K^T, W_Q^T) instead of forming d_model x d_model matrices.

One directory per model is written under --out, each containing the
table head_summary.csv of per-head statistics consumed by the other
scripts and by the paper.

Install:
    pip install torch transformers pandas tqdm safetensors

The shipped data (scripts/qk_geometry_results/) was produced by
running with the default arguments, i.e. all eight models of the
paper:
    python qk_geometry_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoModelForCausalLM


# The eight models of the paper (all head dimension 64).
DEFAULT_MODELS = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "EleutherAI/pythia-70m-deduped",
]


@dataclass
class HeadWeights:
    model_id: str
    model_type: str
    layer: int
    head: int
    kv_head: int
    Wq: torch.Tensor  # shape: [head_dim, d_model]
    Wk: torch.Tensor  # shape: [head_dim, d_model]


@dataclass
class HeadStats:
    model_id: str
    model_type: str
    layer: int
    head: int
    kv_head: int
    d_model: int
    head_dim: int

    L_frob: float
    S_frob: float
    A_frob: float
    S_energy_ratio: float
    A_energy_ratio: float

    S_pos: int
    S_neg: int
    S_zero: int
    S_inertia_balance: float
    S_negative_energy_frac: float
    S_positive_energy_frac: float
    S_pair_rel_error: float
    S_pair_score: float
    S_max_abs_eig: float
    S_effective_rank: float

    A_rank: int
    A_max_sval: float
    A_effective_rank: float
    A_pair_rel_error: float
    A_pair_score: float


# ---------------------------------------------------------------------------
# Model loading and Q/K extraction
# ---------------------------------------------------------------------------

def projection_matrix(module: torch.nn.Module) -> torch.Tensor:
    """
    Return a projection matrix in the convention:
        output = W @ input
    with W of shape [out_features, in_features].

    torch.nn.Linear stores weight as [out, in].
    GPT-style Conv1D stores weight as [in, out], so we transpose it.
    """
    cls = module.__class__.__name__
    if isinstance(module, torch.nn.Linear):
        return module.weight.detach().cpu().to(torch.float64)
    if cls == "Conv1D" and hasattr(module, "weight"):
        return module.weight.detach().cpu().to(torch.float64).T
    raise TypeError(f"Unsupported projection module type: {cls}")


def split_heads(W: torch.Tensor, n_heads: int, head_dim: Optional[int] = None) -> torch.Tensor:
    """
    Split W [n_heads * head_dim, d_model] into [n_heads, head_dim, d_model].
    """
    out_dim, d_model = W.shape
    if head_dim is None:
        assert out_dim % n_heads == 0, (out_dim, n_heads)
        head_dim = out_dim // n_heads
    return W.reshape(n_heads, head_dim, d_model)


def get_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def iter_qk_heads(model, model_id: str) -> Generator[HeadWeights, None, None]:
    """
    Yield per-head Q/K matrices for the architectures of the paper's
    models.

    Supported:
      - GPT-2 / DistilGPT2: transformer.h.*.attn.c_attn
      - GPT-NeoX / Pythia: gpt_neox.layers.*.attention.query_key_value
      - OPT: model.decoder.layers.*.self_attn.q_proj/k_proj
    """
    cfg = model.config
    mt = getattr(cfg, "model_type", "unknown")

    # GPT-2 / DistilGPT2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = list(model.transformer.h)
        if blocks and hasattr(blocks[0].attn, "c_attn"):
            n_heads = int(getattr(cfg, "n_head"))
            d_model = int(getattr(cfg, "n_embd"))
            head_dim = d_model // n_heads
            for layer, block in enumerate(blocks):
                W = projection_matrix(block.attn.c_attn)  # [3*d_model, d_model]
                Wq, Wk, _Wv = W.split(d_model, dim=0)
                Q = split_heads(Wq, n_heads, head_dim)
                K = split_heads(Wk, n_heads, head_dim)
                for h in range(n_heads):
                    yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
            return

    # GPT-NeoX / Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = list(model.gpt_neox.layers)
        n_heads = int(getattr(cfg, "num_attention_heads"))
        d_model = int(getattr(cfg, "hidden_size"))
        head_dim = d_model // n_heads
        for layer, block in enumerate(layers):
            attn = block.attention
            W = projection_matrix(attn.query_key_value)  # [3*d_model, d_model]
            # GPT-NeoX packs as [Q,K,V] by head in many versions. For Pythia/HF this
            # reshape is [n_heads, 3*head_dim, d_model], then split inside each head.
            W_by_head = W.reshape(n_heads, 3 * head_dim, d_model)
            Q = W_by_head[:, :head_dim, :]
            K = W_by_head[:, head_dim:2 * head_dim, :]
            for h in range(n_heads):
                yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
        return

    # OPT
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = list(model.model.decoder.layers)
        if layers and hasattr(layers[0].self_attn, "q_proj"):
            n_heads = int(getattr(cfg, "num_attention_heads"))
            d_model = int(getattr(cfg, "hidden_size"))
            head_dim = d_model // n_heads
            for layer, block in enumerate(layers):
                attn = block.self_attn
                Q = split_heads(projection_matrix(attn.q_proj), n_heads, head_dim)
                K = split_heads(projection_matrix(attn.k_proj), n_heads, head_dim)
                for h in range(n_heads):
                    yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
            return

    raise ValueError(
        f"Could not find a supported Q/K layout for model_type={mt}. "
        "Add an extractor in iter_qk_heads()."
    )


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------

def effective_rank_from_svals(s: np.ndarray, eps: float = 1e-30) -> float:
    s = np.asarray(s, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    # Entropy effective rank.
    p = s / (s.sum() + eps)
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def stable_rank_from_svals(s: np.ndarray, eps: float = 1e-30) -> float:
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s[0] <= eps:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2 + eps))


def symmetric_pair_error(evals: np.ndarray, tol: float) -> Tuple[float, float]:
    """
    Relative error for pairing positive eigenvalues with negative eigenvalues
    of equal magnitude. Returns (error, score=1-error).
    """
    pos = np.sort(evals[evals > tol])[::-1]
    neg = np.sort(-evals[evals < -tol])[::-1]
    total = float((pos ** 2).sum() + (neg ** 2).sum())
    if total <= 0:
        return (np.nan, np.nan)
    m = min(len(pos), len(neg))
    diff = 0.0
    if m:
        diff += float(((pos[:m] - neg[:m]) ** 2).sum())
    if len(pos) > m:
        diff += float((pos[m:] ** 2).sum())
    if len(neg) > m:
        diff += float((neg[m:] ** 2).sum())
    err = math.sqrt(diff / total)
    return (err, 1.0 - err)


def skew_singular_pair_error(svals: np.ndarray, tol: float) -> Tuple[float, float]:
    """
    For a real skew-symmetric matrix, singular values should occur in equal
    pairs. This is mostly a numerical sanity check and rank/effective-rank
    summary; any exact real skew matrix has this structure.
    """
    s = np.sort(svals[svals > tol])[::-1]
    if s.size < 2:
        return (np.nan, np.nan)
    total = float((s ** 2).sum())
    diffs = []
    i = 0
    while i + 1 < len(s):
        diffs.append((s[i] - s[i + 1]) ** 2)
        i += 2
    if i < len(s):
        diffs.append(s[i] ** 2)
    err = math.sqrt(float(np.sum(diffs)) / (total + 1e-30))
    return (err, 1.0 - err)


def spectral_stats_for_head(hw: HeadWeights, eig_eps_rel: float = 1e-7, eig_eps_abs: float = 0.0) -> HeadStats:
    """
    Compute statistics for one head.

    Wq, Wk have shape [head_dim, d_model].
    L = Wk.T @ Wq.
    S = (L + L.T)/2.
    A = (L - L.T)/2.

    We reduce S and A to span(Wk.T, Wq.T), dimension at most 2*head_dim.
    """
    Wq = hw.Wq.to(torch.float64)
    Wk = hw.Wk.to(torch.float64)
    head_dim, d_model = Wq.shape

    # Basis for the range of S and A.
    U = torch.cat([Wk.T, Wq.T], dim=1)  # [d_model, 2*head_dim]
    # QR may be slightly unstable if U is rank-deficient, but the resulting
    # reduced matrices simply have extra near-zero eigenvalues.
    R = torch.linalg.qr(U, mode="reduced").Q  # [d_model, r]

    KR = Wk @ R
    QR = Wq @ R
    Sred = 0.5 * (KR.T @ QR + QR.T @ KR)
    Ared = 0.5 * (KR.T @ QR - QR.T @ KR)

    S_frob = float(torch.linalg.matrix_norm(Sred, ord="fro").item())
    A_frob = float(torch.linalg.matrix_norm(Ared, ord="fro").item())
    L_frob = math.sqrt(S_frob ** 2 + A_frob ** 2)

    if L_frob > 0:
        S_energy_ratio = S_frob ** 2 / (L_frob ** 2)
        A_energy_ratio = A_frob ** 2 / (L_frob ** 2)
    else:
        S_energy_ratio = np.nan
        A_energy_ratio = np.nan

    evals = torch.linalg.eigvalsh(Sred).cpu().numpy()
    max_abs_eig = float(np.max(np.abs(evals))) if evals.size else 0.0
    tol = max(eig_eps_abs, eig_eps_rel * max_abs_eig)

    pos_mask = evals > tol
    neg_mask = evals < -tol
    S_pos = int(pos_mask.sum())
    S_neg = int(neg_mask.sum())
    # Zero count relative to the ambient dimension d_model (so it
    # includes the d_model - 2*head_dim structural zeros outside the
    # reduced subspace): this is the third entry of the signature of S
    # on the full residual stream.
    S_zero = int(d_model - S_pos - S_neg)

    nz = S_pos + S_neg
    S_inertia_balance = float(1.0 - abs(S_pos - S_neg) / nz) if nz > 0 else np.nan

    pos_energy = float((evals[pos_mask] ** 2).sum())
    neg_energy = float((evals[neg_mask] ** 2).sum())
    total_sym_energy = pos_energy + neg_energy
    S_positive_energy_frac = pos_energy / total_sym_energy if total_sym_energy > 0 else np.nan
    S_negative_energy_frac = neg_energy / total_sym_energy if total_sym_energy > 0 else np.nan

    S_pair_rel_error, S_pair_score = symmetric_pair_error(evals, tol)

    abs_evals = np.sort(np.abs(evals[np.abs(evals) > tol]))[::-1]
    S_effective_rank = effective_rank_from_svals(abs_evals)

    # Skew diagnostics.
    # Use singular values of the reduced skew matrix.
    A_svals = torch.linalg.svdvals(Ared).cpu().numpy()
    max_A_sval = float(A_svals.max()) if A_svals.size else 0.0
    Atol = max(eig_eps_abs, eig_eps_rel * max_A_sval)
    A_rank = int((A_svals > Atol).sum())
    A_effective_rank = effective_rank_from_svals(A_svals[A_svals > Atol])
    A_pair_rel_error, A_pair_score = skew_singular_pair_error(A_svals, Atol)

    return HeadStats(
        model_id=hw.model_id,
        model_type=hw.model_type,
        layer=hw.layer,
        head=hw.head,
        kv_head=hw.kv_head,
        d_model=d_model,
        head_dim=head_dim,
        L_frob=L_frob,
        S_frob=S_frob,
        A_frob=A_frob,
        S_energy_ratio=S_energy_ratio,
        A_energy_ratio=A_energy_ratio,
        S_pos=S_pos,
        S_neg=S_neg,
        S_zero=S_zero,
        S_inertia_balance=S_inertia_balance,
        S_negative_energy_frac=S_negative_energy_frac,
        S_positive_energy_frac=S_positive_energy_frac,
        S_pair_rel_error=S_pair_rel_error,
        S_pair_score=S_pair_score,
        S_max_abs_eig=max_abs_eig,
        S_effective_rank=S_effective_rank,
        A_rank=A_rank,
        A_max_sval=max_A_sval,
        A_effective_rank=A_effective_rank,
        A_pair_rel_error=A_pair_rel_error,
        A_pair_score=A_pair_score,
    )


# ---------------------------------------------------------------------------
# Output naming and reduced spectra
# ---------------------------------------------------------------------------

def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def compute_reduced_spectrum(hw: HeadWeights) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the nonzero-region eigenvalues of S and singular values
    of A on the reduced subspace (used by the clustering scripts).
    """
    Wq = hw.Wq.to(torch.float64)
    Wk = hw.Wk.to(torch.float64)
    U = torch.cat([Wk.T, Wq.T], dim=1)
    R = torch.linalg.qr(U, mode="reduced").Q
    KR = Wk @ R
    QR = Wq @ R
    Sred = 0.5 * (KR.T @ QR + QR.T @ KR)
    Ared = 0.5 * (KR.T @ QR - QR.T @ KR)
    evals = torch.linalg.eigvalsh(Sred).cpu().numpy()
    svals = torch.linalg.svdvals(Ared).cpu().numpy()
    return evals, svals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(model_id: str, trust_remote_code: bool = False):
    print(f"\nLoading {model_id} ...")
    # Try causal LM first. If that fails, fall back to base AutoModel.
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
        )
    except Exception as e_lm:
        print(f"AutoModelForCausalLM failed for {model_id}: {e_lm}")
        print("Trying AutoModel ...")
        return AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
        )


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "S_energy_ratio", "A_energy_ratio",
        "S_inertia_balance", "S_negative_energy_frac", "S_positive_energy_frac",
        "S_pair_score", "S_pair_rel_error", "S_effective_rank",
        "A_effective_rank", "A_pair_score", "A_pair_rel_error",
    ]
    aggs = df.groupby("model_id")[cols].agg(["mean", "std", "median"])
    aggs.columns = [f"{a}_{b}" for a, b in aggs.columns]
    aggs = aggs.reset_index()
    aggs["num_heads"] = df.groupby("model_id").size().values
    return aggs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face model IDs.")
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent / "qk_geometry_results"),
                        help="Output directory.")
    parser.add_argument("--max-heads", type=int, default=None, help="Analyze at most this many heads per model.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--eig-eps-rel", type=float, default=1e-7, help="Relative eigenvalue/singular-value threshold.")
    parser.add_argument("--eig-eps-abs", type=float, default=0.0, help="Absolute eigenvalue/singular-value threshold.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    metadata = {
        "models": args.models,
        "eig_eps_rel": args.eig_eps_rel,
        "eig_eps_abs": args.eig_eps_abs,
        "notes": {
            "L": "L = W_K^T W_Q per head, with W_Q/W_K shape [head_dim, d_model].",
            "S": "S = (L + L^T)/2, the symmetric part.",
            "A": "A = (L - L^T)/2, the antisymmetric part (T in the paper).",
            "low_rank": "Spectra are computed in span(W_K^T, W_Q^T), preserving nonzero eigen/singular values of S and A.",
        }
    }

    for model_id in args.models:
        model_dir = out_dir / safe_name(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = load_model(model_id, trust_remote_code=args.trust_remote_code)
            model.eval()
            model.cpu()

            heads_iter = list(iter_qk_heads(model, model_id))
            if args.max_heads is not None:
                heads_iter = heads_iter[: args.max_heads]

            print(f"Found {len(heads_iter)} heads for {model_id}")

            model_rows = []
            for hw in tqdm(heads_iter, desc=f"Analyzing {model_id}"):
                try:
                    st = spectral_stats_for_head(hw, eig_eps_rel=args.eig_eps_rel, eig_eps_abs=args.eig_eps_abs)
                    row = asdict(st)
                    all_rows.append(row)
                    model_rows.append(row)
                except Exception as e:
                    print(f"Failed head layer={hw.layer} head={hw.head}: {e}")

            model_df = pd.DataFrame(model_rows)
            model_df.to_csv(model_dir / "head_summary.csv", index=False)

            # Free memory between models.
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"ERROR processing {model_id}: {e}")

    full_df = pd.DataFrame(all_rows)
    summary = summarize_by_model(full_df)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Wrote results to: {out_dir.resolve()}")
    if not full_df.empty:
        print("\nBy-model summary:")
        cols = [
            "model_id", "num_heads",
            "S_energy_ratio_mean", "A_energy_ratio_mean",
            "S_inertia_balance_mean", "S_pair_score_mean", "A_effective_rank_mean",
        ]
        existing = [c for c in cols if c in summary.columns]
        print(summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
