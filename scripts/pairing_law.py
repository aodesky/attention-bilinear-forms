#!/usr/bin/env python3
"""The pairing score across the heads of the eight models.

With lambda_+ and lambda_- the positive eigenvalues of S and the
absolute values of its negative eigenvalues, both sorted decreasing
and padded with zeros to a common length, the pairing score is

    P = 1 - ||lambda_+ - lambda_-|| / ||S||,

which is 1 exactly when the spectrum is symmetric about zero.
Writing rho = ||lambda_-|| / ||lambda_+|| for the ratio of the two
lobe norms, and replacing rho by 1/rho if it exceeds 1, the reverse
triangle inequality gives

    ||lambda_+ - lambda_-|| >= (1 - rho) ||lambda_+||,

while ||S||^2 = (1 + rho^2) ||lambda_+||^2, so that

    P <= 1 - (1 - rho) / sqrt(1 + rho^2),                       (*)

with equality exactly when lambda_- = rho lambda_+, that is when the
two sorted lobes are proportional.  The script reports:

  - the distribution of P in each model, and a beta law fitted to it
    on [0,1] by maximum likelihood;
  - the bound (*) evaluated at each head's own rho, its correlation
    with the head's P, the proportion of heads at which it holds, and
    the gap between the bound and P;
  - the distribution of rho.

Usage:
    python pairing_law.py
    python pairing_law.py --model gpt2
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats as st

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qk_geometry_diagnostics import iter_qk_heads, load_model

MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
}


def head_row(Wq, Wk):
    """(pairing score, lobe norm ratio) for one head."""
    Q = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
    L = (Wk @ Q).T @ (Wq @ Q)
    S = 0.5 * (L + L.T)
    ev = np.linalg.eigvalsh(S)
    nz = ev[np.abs(ev) > 1e-9 * np.abs(ev).max()]
    pos = np.sort(nz[nz > 0])[::-1]
    neg = np.sort(-nz[nz < 0])[::-1]
    m = max(len(pos), len(neg))
    pos = np.pad(pos, (0, m - len(pos)))
    neg = np.pad(neg, (0, m - len(neg)))
    P = 1.0 - np.linalg.norm(pos - neg) / np.linalg.norm(nz)
    rho = np.linalg.norm(neg) / np.linalg.norm(pos)
    return P, rho


def bound(rho):
    """The right side of (*), at r = min(rho, 1/rho)."""
    r = np.minimum(rho, 1.0 / rho)
    return 1.0 - (1.0 - r) / np.sqrt(1.0 + r ** 2)


def analyze(tag, model_id):
    model = load_model(model_id)
    model.eval()
    rows = [head_row(hw.Wq.to(torch.float64).numpy(),
                     hw.Wk.to(torch.float64).numpy())
            for hw in iter_qk_heads(model, model_id)]
    del model
    r = np.array(rows)
    P, rho = r[:, 0], r[:, 1]
    a, b, _, _ = st.beta.fit(P, floc=0, fscale=1)
    ks = st.kstest(P, "beta", args=(a, b, 0, 1)).pvalue
    ub = bound(rho)
    print(f"{tag:22s} heads={len(P):4d}")
    print(f"    P: median {np.median(P):.4f}  "
          f"[10%,90%] = [{np.quantile(P, .1):.3f}, {np.quantile(P, .9):.3f}]")
    print(f"    beta fit on [0,1]: a = {a:.2f}, b = {b:.2f}  "
          f"(Kolmogorov-Smirnov p = {ks:.3f})")
    print(f"    bound (*): median {np.median(ub):.4f}, "
          f"correlation with P {np.corrcoef(P, ub)[0, 1]:+.3f}, "
          f"holds for {np.mean(ub >= P - 1e-12) * 100:.0f}% of heads")
    gap = ub - P
    print(f"    gap bound - P: median {np.median(gap):.4f}  "
          f"[10%,90%] = [{np.quantile(gap, .1):.4f}, {np.quantile(gap, .9):.4f}]")
    print(f"    rho: median {np.median(rho):.3f}  "
          f"[10%,90%] = [{np.quantile(rho, .1):.3f}, {np.quantile(rho, .9):.3f}]")
    return P


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    args = ap.parse_args()
    tags = [args.model] if args.model else list(MODELS)
    allP = [analyze(t, MODELS[t]) for t in tags]
    if len(allP) > 1:
        P = np.concatenate(allP)
        print(f"\npooled: {len(P)} heads, median {np.median(P):.4f}")
        a, b, _, _ = st.beta.fit(P, floc=0, fscale=1)
        print(f"    beta fit on [0,1]: a = {a:.2f}, b = {b:.2f}  "
              f"(Kolmogorov-Smirnov p = "
              f"{st.kstest(P, 'beta', args=(a, b, 0, 1)).pvalue:.4f})")
        print("    the pooled fit mixes models with different parameters")


if __name__ == "__main__":
    main()
