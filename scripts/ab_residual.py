#!/usr/bin/env python3
"""The residual b - a across the heads of all eight models.

For each head the bilinear form L = W_K^T W_Q is restricted to the
reduced subspace im(W_K^T) + im(W_Q^T), split as L = S + T, and the
weights a = ||T||^2/||L||^2 and b = 2<alpha,beta>/||L||^2 are
computed, alpha and beta being the positive eigenvalues of S and the
absolute values of its negative eigenvalues.

Reported, for the pooled population:

  - the centre of b - a by four estimators: the median with a
    bootstrap interval, the mode of a Gaussian kernel estimate, the
    location of a fitted Student t, and the 25 per cent trimmed
    mean.  The plain mean is also printed; it is not a stable
    summary, the fitted t having about 1.3 degrees of freedom;
  - the counts beyond two, three, four and six standard deviations
    against the Gaussian expectation, and the Shapiro-Wilk,
    D'Agostino and Kolmogorov-Smirnov tests of normality;
  - the Kolmogorov-Smirnov p-value of a fitted Student t.

Usage:
    python ab_residual.py
    python ab_residual.py --model gpt2
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
SEED = 0
BOOTSTRAP = 4000


def head_weights(Wq, Wk):
    """(a, b) for one head, on the reduced subspace."""
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
    nL2 = (S ** 2).sum() + (T ** 2).sum()
    return (T ** 2).sum() / nL2, 2 * (alpha @ beta) / nL2


def collect(tags):
    out = {}
    for tag in tags:
        model = load_model(MODELS[tag])
        model.eval()
        rows = [head_weights(hw.Wq.to(torch.float64).numpy(),
                             hw.Wk.to(torch.float64).numpy())
                for hw in iter_qk_heads(model, MODELS[tag])]
        del model
        out[tag] = np.array(rows)
        print(f"{tag:22s} {len(rows):4d} heads")
    return out


def report(e):
    rng = np.random.default_rng(SEED)
    m, sd = e.mean(), e.std()
    boot = [np.median(rng.choice(e, len(e), replace=True))
            for _ in range(BOOTSTRAP)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    kde = st.gaussian_kde(e, bw_method=0.15)
    grid = np.linspace(-0.2, 0.2, 20001)
    nu, loc, scale = st.t.fit(e)

    print(f"\npooled: {len(e)} heads,  e = b - a")
    print(f"  median              {np.median(e):+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  mode (kernel)       {grid[np.argmax(kde(grid))]:+.4f}")
    print(f"  location (Student t){loc:+.4f}")
    print(f"  25% trimmed mean    {st.trim_mean(e, 0.25):+.4f}")
    print(f"  mean                {m:+.4f}   (not stable: see below)")

    print(f"\n  tail counts against a Gaussian of the same variance:")
    for k in (2, 3, 4, 6):
        obs = int(np.sum(np.abs(e - m) > k * sd))
        exp = len(e) * 2 * st.norm.sf(k)
        print(f"    beyond {k} sd: {obs:5d} observed, {exp:11.4g} expected")

    print(f"\n  normality of e:")
    print(f"    Shapiro-Wilk      p = {st.shapiro(e).pvalue:.3e}")
    print(f"    D'Agostino K^2    p = {st.normaltest(e).pvalue:.3e}")
    print(f"    Kolmogorov-Smirnov p = "
          f"{st.kstest(e, 'norm', args=(m, sd)).pvalue:.3e}")
    print(f"  Student t fit: nu = {nu:.2f}, scale = {scale:.4f}, "
          f"KS p = {st.kstest(e, 't', args=(nu, loc, scale)).pvalue:.3f}")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    args = ap.parse_args()
    tags = [args.model] if args.model else list(MODELS)
    data = collect(tags)

    print(f"\n{'model':22s} {'heads':>6s} {'median b-a':>11s} {'% b > a':>8s}")
    for tag, A in data.items():
        v = A[:, 1] - A[:, 0]
        print(f"{tag:22s} {len(v):6d} {np.median(v):+11.4f} "
              f"{np.mean(v > 0) * 100:7.1f}%")
    report(np.vstack(list(data.values()))[:, 1]
           - np.vstack(list(data.values()))[:, 0])


if __name__ == "__main__":
    main()
