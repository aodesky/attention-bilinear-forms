#!/usr/bin/env python3
"""The three-type partition of the heads, from the profile map.

The paper classifies a head using the profile (0, b, c, d) of its
symmetric part. Type II means c + d <= RADIUS, a neighborhood N of
(0, 1, 0, 0) in the face Delta. Outside N, Type I+ means c > d;
Type I- is the remaining case (including exact ties).

Only the parity is canonical.  The sign of c - d is determined by the
head, but the heads fill a continuum around the hyperbolic vertex with
no gap at which to cut, so the separation of Type II from the rest
depends on the choice of N.  This script therefore reports the partition
at the radius the paper fixes, the parity on its own, and the
sensitivity of the partition to the radius, so that the reader can see
which statements depend on the choice and which do not.

On the symmetric profile, 1 - b = c + d; the default tolerance is 0.02.

Usage:
    python type_partition.py
    python type_partition.py --radius 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

STATS = ms.PROJECT / "scripts" / "survey_results" / "head_statistics.csv"
RADIUS = 0.02                      # the paper's N = {1 - b <= 1/50}
NAMES = {1: "Type I+", 2: "Type II", 3: "Type I-"}


def assign(frame: pd.DataFrame, radius: float) -> np.ndarray:
    """The type of every head, from the profile of its symmetric part."""
    c = frame.c_sym.to_numpy()
    d = frame.d_sym.to_numpy()
    inside = (c + d) <= radius     # 1 - b = c + d
    return np.where(inside, 2, np.where(c > d, 1, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=RADIUS,
                        help="radius of N, in the distance 1 - b")
    args = parser.parse_args()

    if not STATS.exists():
        raise FileNotFoundError(f"{STATS} not found; run head_statistics.py")
    df = pd.read_csv(STATS)
    labels = assign(df, args.radius)

    print(f"N = {{1 - b <= {args.radius}}}\n")
    print(f"{'model':26s} {'heads':>6s} " +
          " ".join(f"{NAMES[k]:>9s}" for k in (1, 2, 3)) +
          f" {'median 1-b':>11s}")
    for tag in ms.TAGS:
        mask = (df.model == tag).to_numpy()
        sub = labels[mask]
        distance = (df.c_sym + df.d_sym).to_numpy()[mask]
        counts = " ".join(f"{int((sub == k).sum()):4d} "
                          f"{100 * (sub == k).mean():4.1f}%" for k in (1, 2, 3))
        print(f"{tag:26s} {mask.sum():6d} {counts} {np.median(distance):11.4f}")

    print(f"\npooled over {len(df)} heads: " +
          ", ".join(f"{NAMES[k]} {100 * (labels == k).mean():.1f}%"
                    for k in (1, 2, 3)))

    # the parity alone, which does not depend on N
    positive = (df.c_sym > df.d_sym).to_numpy()
    share = pd.Series(positive).groupby(df.model.values).mean()
    print(f"\nparity alone (no choice of N): {100 * positive.mean():.1f}% of "
          f"heads have c > d; per model {100 * share.min():.0f}% to "
          f"{100 * share.max():.0f}%")

    print("\nsensitivity of the partition to the radius:")
    print(f"{'radius':>7s} " + " ".join(f"{NAMES[k]:>9s}" for k in (1, 2, 3)))
    for radius in (0.01, 0.02, 0.03, 0.05, 0.10):
        other = assign(df, radius)
        print(f"{radius:7.2f} " +
              " ".join(f"{100 * (other == k).mean():8.1f}%" for k in (1, 2, 3)))


if __name__ == "__main__":
    main()
