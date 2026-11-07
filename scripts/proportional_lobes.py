#!/usr/bin/env python3
"""Empirical proportionality of sorted lobes and related profile predictions.

Observation 5 reports the error in lambda_- = rho lambda_+, where
rho = ||lambda_-|| / ||lambda_+||.  Section 5.2.1 relates it to the
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

Reads the extracted forms through model_set, so no model weights and no
network connection are needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

COLUMN_TITLES = (
    r"(i) ||l- - rho l+||/||l-||",
    r"(ii) ||T||^2 / 2<l+,l->",
    "|π(L) - predicted|",
    "|π(S) - predicted|",
)


def predicted(rho: float) -> tuple[np.ndarray, np.ndarray]:
    """The full-profile candidate and the symmetric-profile limit."""
    c, d = max(1.0 - rho, 0.0) ** 2, max(rho - 1.0, 0.0) ** 2
    profile_L = np.array([2 * rho, 2 * rho, c, d]) / (1 + rho) ** 2
    profile_S = np.array([2 * rho, c, d]) / (1 + rho ** 2)
    return profile_L, profile_S


def head_row(L: np.ndarray) -> tuple[float, float, float, float, float]:
    """The two departures, the two distances and rho, for one head."""
    S, T = ms.symmetric_antisymmetric(L)
    lam_p, lam_m = ms.sorted_lobes(S)
    norm_p = float(np.linalg.norm(lam_p))
    norm_m = float(np.linalg.norm(lam_m))
    if norm_p == 0.0 or norm_m == 0.0:
        return (np.nan,) * 5

    profile = np.array(ms.profile(L))
    rho = norm_m / norm_p
    profile_L, profile_S = predicted(rho)
    inner = float(lam_p @ lam_m)
    departure = float(np.linalg.norm(lam_m - rho * lam_p)) / norm_m
    ratio = float((T ** 2).sum()) / (2 * inner) if inner > 0 else np.inf
    return (departure, ratio,
            float(np.linalg.norm(profile - profile_L)),
            float(np.linalg.norm(profile[1:] / (1 - profile[0]) - profile_S)),
            rho)


def collect(tag: str) -> np.ndarray:
    return np.array([head_row(L) for _, _, _, L in ms.iter_forms(tag)])


def summary(rows: np.ndarray) -> list[str]:
    cells = []
    for column in range(4):
        values = rows[:, column]
        values = values[np.isfinite(values)]
        low, high = np.quantile(values, [0.1, 0.9])
        cells.append(f"{np.median(values):.3f} [{low:.3f}, {high:.3f}]")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ms.TAGS)
    args = parser.parse_args()

    tags = [args.model] if args.model else ms.TAGS
    data = {tag: collect(tag) for tag in tags}

    print("median [10%, 90%] over heads\n")
    print(f"{'model':26s} {'heads':>6s}  " +
          "  ".join(f"{title:>28s}" for title in COLUMN_TITLES))
    for tag, rows in data.items():
        print(f"{tag:26s} {len(rows):6d}  " +
              "  ".join(f"{cell:>28s}" for cell in summary(rows)))

    pooled = np.vstack(list(data.values()))
    if len(data) > 1:
        print(f"{'pooled':26s} {len(pooled):6d}  " +
              "  ".join(f"{cell:>28s}" for cell in summary(pooled)))

    print("\nheads whose profile lies within 0.1 of the predicted point:")
    for tag, rows in data.items():
        within = 100 * float(np.mean(rows[:, 2] < 0.1))
        print(f"  {tag:26s} {within:5.1f}%   median rho "
              f"{np.median(rows[:, 4]):.3f}")


if __name__ == "__main__":
    main()
