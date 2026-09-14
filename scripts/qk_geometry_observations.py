#!/usr/bin/env python3
"""Row and column space geometry of the bilinear forms of attention heads.

Computes the three quantities of the corresponding observations in
attention.tex, for each of the eight models:

  1. the cosines of the principal angles between im(L) = im(W_K^T)
     and im(L^T) = im(W_Q^T), compared with the same quantity for
     two uniformly random n-dimensional subspaces of R^N;
  2. the normalized trace |tr L| / ||L||, where
     tr L = sum_ij (W_K)_ij (W_Q)_ij, and its correlation across
     heads with the pairing score of the symmetric part;
  3. the correlation, within each head, between the sorted singular
     values of T and the sorted absolute values of the eigenvalues
     of S;
  4. gamma fits (maximum likelihood, origin fixed at zero) to the
     singular values of T and to the two lobes of spec(S), and the
     comparison of the fitted mode of T with the geometric mean of
     the two lobe modes, and of the fitted shapes.  Heads for which
     any fitted shape falls below 1 have no interior mode and are
     omitted from that comparison.

Spectra are computed on the reduced subspace
im(W_K^T) + im(W_Q^T), as elsewhere in the paper.

Usage:
    python qk_geometry_observations.py               # all eight models
    python qk_geometry_observations.py --model gpt2

The random-subspace baseline is seeded.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

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
BASELINE_TRIALS = 10


def pairing_score(evals):
    """1 - ||lambda_+ - lambda_-|| / ||S||, on the nonzero spectrum."""
    nz = evals[np.abs(evals) > 1e-9 * np.abs(evals).max()]
    pos = np.sort(nz[nz > 0])[::-1]
    neg = np.sort(-nz[nz < 0])[::-1]
    m = max(len(pos), len(neg))
    pos = np.pad(pos, (0, m - len(pos)))
    neg = np.pad(neg, (0, m - len(neg)))
    return 1.0 - np.linalg.norm(pos - neg) / np.linalg.norm(nz)


def random_overlap(N, n, rng):
    """Mean principal cosine of two uniform n-subspaces of R^N."""
    out = []
    for _ in range(BASELINE_TRIALS):
        A = np.linalg.qr(rng.normal(size=(N, n)))[0]
        B = np.linalg.qr(rng.normal(size=(N, n)))[0]
        out.append(np.linalg.svd(A.T @ B, compute_uv=False).mean())
    return float(np.mean(out))


def analyze(tag, model_id):
    model = load_model(model_id)
    model.eval()
    overlap, trace, pairing, coupling = [], [], [], []
    fits = []
    N = n = None
    for hw in iter_qk_heads(model, model_id):
        Wq = hw.Wq.to(torch.float64).numpy()
        Wk = hw.Wk.to(torch.float64).numpy()
        n, N = Wq.shape
        A = np.linalg.qr(Wk.T)[0]
        B = np.linalg.qr(Wq.T)[0]
        overlap.append(np.linalg.svd(A.T @ B, compute_uv=False).mean())

        Q = np.linalg.qr(np.concatenate([Wk.T, Wq.T], axis=1))[0]
        L = (Wk @ Q).T @ (Wq @ Q)
        S = 0.5 * (L + L.T)
        T = 0.5 * (L - L.T)
        trace.append(abs(np.trace(L)) / np.linalg.norm(L))
        evals = np.linalg.eigvalsh(S)
        pairing.append(pairing_score(evals))
        a = np.sort(np.abs(evals))[::-1]
        b = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]
        coupling.append(np.corrcoef(a, b)[0, 1])

        # gamma fits: distinct singular values of T, and the two lobes
        sig = b[0::2]
        pos = evals[evals > 0]
        neg = -evals[evals < 0]
        kT, _, sT = stats.gamma.fit(sig, floc=0)
        kp, _, sp = stats.gamma.fit(pos, floc=0)
        kn, _, sn = stats.gamma.fit(neg, floc=0)
        fits.append((kT, (kT - 1) * sT, kp, (kp - 1) * sp,
                     kn, (kn - 1) * sn))
    del model

    rng = np.random.default_rng(SEED)
    base = random_overlap(N, n, rng)
    print(f"{tag:22s} N={N:5d} n={n:3d} heads={len(overlap):4d}")
    print(f"    principal cosines: median {np.median(overlap):.3f}  "
          f"(uniform subspaces: {base:.3f})")
    print(f"    |tr L| / ||L||   : median {np.median(trace):.3f}  "
          f"corr with pairing score {np.corrcoef(pairing, trace)[0, 1]:+.3f}")
    print(f"    corr(sorted sigma(T), sorted |spec S|): "
          f"median {np.median(coupling):.4f}")

    f = np.array(fits)
    kT, mT, kp, mp, kn, mn = f.T
    ok = (kT > 1) & (kp > 1) & (kn > 1)
    geo = np.sqrt(mp[ok] * mn[ok])
    kS = np.sqrt(kp[ok] * kn[ok])
    print(f"    gamma fits ({ok.sum()} of {len(f)} heads with all shapes > 1):")
    print(f"      mode(T) / sqrt(mode(+) mode(-)): "
          f"median {np.median(mT[ok] / geo):.3f}, "
          f"corr {np.corrcoef(mT[ok], geo)[0, 1]:.3f}")
    print(f"      shape(T) / sqrt(shape(+) shape(-)): "
          f"median {np.median(kT[ok] / kS):.3f}, "
          f"corr of logs {np.corrcoef(np.log(kT[ok]), np.log(kS))[0, 1]:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default=None)
    args = p.parse_args()
    tags = [args.model] if args.model else list(MODELS)
    for tag in tags:
        analyze(tag, MODELS[tag])


if __name__ == "__main__":
    main()
