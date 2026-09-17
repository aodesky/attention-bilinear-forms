#!/usr/bin/env python3
"""Effective ranks of the bilinear forms L, from their singular values.

Reports whether the accumulation of head profiles near Theta could be
an artifact of forms that are nearly rank one.  With h_1 >= h_2 >= ...
the singular values of L, the participation ratio

    (sum_i h_i^2)^2 / sum_i h_i^4

equals 1 exactly when L has rank one and n when its spectrum is flat,
and it weights the singular values as the profile does, whose
coordinates are quadratic in L and normalized by ||L||^2 = sum h_i^2.
The stable rank sum_i h_i^2 / h_1^2, the reciprocal of the leading
energy share, is reported alongside.

Reads survey_results/head_statistics.csv (run head_statistics.py first)
and prints per-model and pooled quantiles, together with the number of
heads whose participation ratio lies below 2 and below 5.

Usage:
    python effective_rank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

STATS = ms.PROJECT / "scripts" / "survey_results" / "head_statistics.csv"


def main() -> None:
    if not STATS.exists():
        raise FileNotFoundError(f"{STATS} not found; run head_statistics.py")
    df = pd.read_csv(STATS)
    if "participation_ratio" not in df.columns:
        raise KeyError("head_statistics.csv predates the effective ranks; "
                       "rerun head_statistics.py")

    print(f"{'model':26s} {'heads':>5s} {'n':>4s}   "
          f"{'participation ratio':^24s}   {'stable rank':^24s}")
    print(f"{'':26s} {'':>5s} {'':>4s}   "
          f"{'p10':>6s} {'median':>8s} {'p90':>8s}   {'p10':>6s} {'median':>8s} {'p90':>8s}")
    for tag in ms.TAGS:
        f = df[df.model == tag]
        n = ms.model_info(tag).head_dim
        pr, sr = f.participation_ratio, f.stable_rank
        print(f"{tag:26s} {len(f):5d} {n:4d}   "
              f"{pr.quantile(.1):6.1f} {pr.median():8.1f} {pr.quantile(.9):8.1f}   "
              f"{sr.quantile(.1):6.1f} {sr.median():8.1f} {sr.quantile(.9):8.1f}")

    pr, sr = df.participation_ratio, df.stable_rank
    print(f"\n{'all heads':26s} {len(df):5d} {'':4s}   "
          f"{pr.quantile(.1):6.1f} {pr.median():8.1f} {pr.quantile(.9):8.1f}   "
          f"{sr.quantile(.1):6.1f} {sr.median():8.1f} {sr.quantile(.9):8.1f}")
    for cut in (2, 5):
        k = int((pr < cut).sum())
        print(f"participation ratio below {cut}: {k} heads ({100 * k / len(df):.2f}%)")
    k = int((sr < 2).sum())
    print(f"stable rank below 2 (leading share above one half): "
          f"{k} heads ({100 * k / len(df):.2f}%)")


if __name__ == "__main__":
    main()
