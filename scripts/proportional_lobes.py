#!/usr/bin/env python3
"""Empirical proportionality of sorted lobes and related profile predictions.

Observation 5 reports the error in lambda_- = rho lambda_+, where
rho = ||lambda_-|| / ||lambda_+||.  Section 6.2.1 relates it to the
proportionality that arises asymptotically from independent samples
of a common distribution in the accumulation theorem.  That theorem
concerns the symmetric part S and predicts

  (b, c, d) = (2 rho, (1 - rho)_+^2, (rho - 1)_+^2) / (1 + rho^2).

This script reports, per model and pooled:

  (i)  ||lambda_- - rho lambda_+|| / ||lambda_-||;
  (ii) ||T||^2 / (2 <lambda_+, lambda_->) = a / b;
  (iii) the distance of the full profile to the candidate below; and
  (iv) the distance of the symmetric profile to the theorem's limit.

The full-profile candidate

  (a, b, c, d) = (2 rho, 2 rho, (1 - rho)_+^2, (rho - 1)_+^2) / (1 + rho)^2

requires the additional condition a = b, besides proportional lobes.
It is a separate diagnostic, not a conclusion of the accumulation
theorem for S.  Quantity (i) is the lobe_departure statistic quoted in
Observation 5. Medians and central 80 percent intervals are reported.

Usage:
    python proportional_lobes.py
    python proportional_lobes.py --model gpt2
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

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

COLUMNS = ("prop", "ratio", "dist_L", "dist_S", "rho")


def predicted(rho):
    """Limit points of the theorem for L and for S alone."""
    cp, dm = max(1 - rho, 0.0) ** 2, max(rho - 1, 0.0) ** 2
    wL = np.array([2 * rho, 2 * rho, cp, dm]) / (1 + rho) ** 2
    wS = np.array([2 * rho, cp, dm]) / (1 + rho ** 2)
    return wL, wS


def head_row(Wq, Wk):
    Q = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
    L = (Wk @ Q).T @ (Wq @ Q)
    S = 0.5 * (L + L.T)
    T = 0.5 * (L - L.T)
    w = np.linalg.eigvalsh(S)
    alpha = np.sort(w[w > 0])[::-1]
    beta = np.sort(-w[w < 0])[::-1]
    m = max(len(alpha), len(beta))
    alpha = np.pad(alpha, (0, m - len(alpha)))
    beta = np.pad(beta, (0, m - len(beta)))
    nS2 = alpha @ alpha + beta @ beta
    nT2 = (T ** 2).sum()
    nL2 = nS2 + nT2
    ab = alpha @ beta
    diff = alpha - beta
    wL = np.array([nT2, 2 * ab, (diff.clip(0) ** 2).sum(),
                   ((-diff).clip(0) ** 2).sum()]) / nL2
    wS = wL[1:] / (1 - wL[0])
    rho = np.sqrt(beta @ beta) / np.sqrt(alpha @ alpha)
    pL, pS = predicted(rho)
    prop = np.linalg.norm(beta - rho * alpha) / np.linalg.norm(beta)
    return (prop, nT2 / (2 * ab), np.linalg.norm(wL - pL),
            np.linalg.norm(wS - pS), rho)


def collect(tags):
    out = {}
    for tag in tags:
        model = load_model(MODELS[tag])
        model.eval()
        rows = [head_row(hw.Wq.to(torch.float64).numpy(),
                         hw.Wk.to(torch.float64).numpy())
                for hw in iter_qk_heads(model, MODELS[tag])]
        del model
        out[tag] = np.array(rows)
    return out


def summary(A):
    cells = []
    for j in range(4):
        v = A[:, j]
        lo, hi = np.quantile(v, [0.1, 0.9])
        cells.append(f"{np.median(v):.3f} [{lo:.3f}, {hi:.3f}]")
    return cells


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    args = ap.parse_args()
    tags = [args.model] if args.model else list(MODELS)
    data = collect(tags)

    head = (f"{'model':20s} {'heads':>5s}  "
            f"{'(i) ||b - rho a||/||b||':>24s}  "
            f"{'(ii) ||T||^2/2<a,b>':>24s}  "
            f"{'|w(L) - pred|':>24s}  "
            f"{'|w(S) - pred|':>24s}")
    print("median [10%, 90%] over heads")
    print(head)
    for tag, A in data.items():
        print(f"{tag:20s} {len(A):5d}  " + "  ".join(f"{c:>24s}" for c in summary(A)))
    P = np.vstack(list(data.values()))
    if len(data) > 1:
        print(f"{'pooled':20s} {len(P):5d}  " + "  ".join(f"{c:>24s}" for c in summary(P)))

    for tag, A in data.items():
        within = np.mean(A[:, 2] < 0.1) * 100
        print(f"{tag:20s} weights within 0.1 of the predicted point: "
              f"{within:5.1f}%   median rho {np.median(A[:, 4]):.3f}")


if __name__ == "__main__":
    main()
