#!/usr/bin/env python3
"""How closely the heads satisfy the hypothesis of the symmetric-bias theorem.

For each head the bilinear form L = W_K^T W_Q, already restricted to
the reduced subspace im(W_K^T) + im(W_Q^T) by the extraction database,
is split as L = S + T.  Let V_+ and V_- be the spans of the
eigenvectors of S with positive and negative eigenvalues, P_+ and P_-
the orthogonal projections onto them, and write T in blocks

    C = P_+ T P_+,   X = P_+ T P_-,   D = P_- T P_-,

so that ||T||^2 = ||C||^2 + 2 ||X||^2 + ||D||^2.  The theorem assumes
C = D = 0 (T maps V_+ into V_- and V_- into V_+) and concludes

    ||T||^2 <= 2 <lambda_+, lambda_->  <=  ||S||^2,

lambda_+ and lambda_- being the positive eigenvalues of S and the
absolute values of its negative eigenvalues, both sorted decreasing.

For each head the script computes

    delta = (||C||^2 + ||D||^2) / ||T||^2,

the share of ||T||^2 carried by the diagonal blocks (delta = 0 is the
hypothesis), and the ratios ||T||^2 / 2<lambda_+, lambda_-> and
||T||^2 / ||S||^2.  It reports, per model and pooled: the median and
central 80 per cent interval of delta; the proportion of heads at
which each of the two inequalities holds; the correlation of delta
with log(||T||^2 / 2<lambda_+, lambda_->); and the median of delta
among the heads violating the first inequality against the median
among the rest.

The per-head values are written to
``scripts/survey_results/symmetric_bias.csv`` and the printed report to
``scripts/survey_results/symmetric_bias_summary.txt``.

Usage:
    python symmetric_bias.py
    python symmetric_bias.py --model gpt2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

OUT = ms.PROJECT / "scripts" / "survey_results"


def head_row(L: np.ndarray) -> tuple[float, float, float]:
    """(delta, ||T||^2 / 2<l+, l->, ||T||^2 / ||S||^2) for one form."""
    S, T = ms.symmetric_antisymmetric(L)
    w, V = np.linalg.eigh(S)
    Vp, Vm = V[:, w > 0], V[:, w < 0]
    C = Vp.T @ T @ Vp
    D = Vm.T @ T @ Vm
    nT2 = float((T ** 2).sum())
    delta = (float((C ** 2).sum()) + float((D ** 2).sum())) / nT2
    lp, lm = ms.sorted_lobes(S)
    return delta, nT2 / float(2.0 * lp @ lm), nT2 / float((S ** 2).sum())


def collect(tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        for layer, head, _kv, L in ms.iter_forms(tag):
            delta, ratio_pair, ratio_norm = head_row(L)
            rows.append(dict(model=tag, layer=layer, head=head, delta=delta,
                             ratio_pair=ratio_pair, ratio_norm=ratio_norm))
    return pd.DataFrame(rows)


def report(tag: str, df: pd.DataFrame) -> str:
    d = df["delta"].to_numpy()
    r1 = df["ratio_pair"].to_numpy()
    r2 = df["ratio_norm"].to_numpy()
    lo, hi = np.quantile(d, [0.1, 0.9])
    viol = r1 > 1
    corr = np.corrcoef(d, np.log(r1))[0, 1]
    med_viol = np.median(d[viol]) if viol.any() else float("nan")
    med_rest = np.median(d[~viol]) if (~viol).any() else float("nan")
    return (f"{tag:24s} {len(df):5d}  "
            f"delta {np.median(d):.3f} [{lo:.3f}, {hi:.3f}]  "
            f"||T||^2<=2<l+,l->: {np.mean(r1 <= 1) * 100:5.1f}%  "
            f"||T||^2<=||S||^2: {np.mean(r2 <= 1) * 100:5.1f}%  "
            f"corr(delta, log ratio) {corr:+.3f}  "
            f"delta | violating {med_viol:.3f}  "
            f"| rest {med_rest:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ms.TAGS, default=None)
    args = ap.parse_args()
    tags = [args.model] if args.model else list(ms.TAGS)

    df = collect(tags)
    lines = ["median [10%, 90%] over heads"]
    for tag in tags:
        lines.append(report(tag, df[df["model"] == tag]))
    if len(tags) > 1:
        lines.append(report("pooled", df))
    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.model is None:
        OUT.mkdir(exist_ok=True)
        df.to_csv(OUT / "symmetric_bias.csv", index=False)
        (OUT / "symmetric_bias_summary.txt").write_text(text)


if __name__ == "__main__":
    main()
