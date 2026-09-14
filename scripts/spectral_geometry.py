#!/usr/bin/env python3
"""Relations between the symmetric and antisymmetric spectra of a head.

For every head of every model in the survey this compares the two lobes
of the spectrum of S with the singular values of T, fitting a gamma
distribution to each of the three positive samples with the origin held
at zero:

* the mode tau of sigma(T) against the geometric mean of the two modes
  of the lobes, sqrt(|mu| nu);
* the shape k_T of sigma(T) against the geometric mean sqrt(k_+ k_-) of
  the two fitted shapes; and
* the sorted singular values of T against the sorted absolute values of
  the eigenvalues of S, head by head, by their correlation.

A fitted gamma whose shape is below one has no interior mode, so its
mode is not defined; heads where any of the three fitted shapes falls
below one are excluded from the comparison of modes and of shapes, and
counted in the output.  The correlation of the sorted lists uses every
head.

Reads the spectra cached by head_statistics.py.

Usage:
    python spectral_geometry.py
    python spectral_geometry.py --model gpt2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

RESULTS = ms.PROJECT / "scripts" / "survey_results"


def gamma_mode_shape(sample: np.ndarray) -> tuple[float, float]:
    """The mode and shape of a gamma fitted with the origin at zero.

    The mode is ``(k - 1) * scale``, which is positive only when the
    fitted shape exceeds one; below that the density is decreasing and
    has no interior mode, and the mode is returned as ``nan``.
    """
    shape, _, scale = stats.gamma.fit(sample, floc=0)
    mode = (shape - 1.0) * scale if shape > 1.0 else np.nan
    return mode, shape


def model_rows(tag: str) -> list[dict]:
    with np.load(RESULTS / f"{tag}_spectra.npz") as z:
        eigenvalues = z["eigenvalues"]
        sigma_T = z["sigma_T"]
        layers = z["layer"]
        heads = z["head"]

    rows = []
    for i in range(eigenvalues.shape[0]):
        ev = eigenvalues[i]
        pos = ev[ev > 0]
        neg = -ev[ev < 0]
        # each singular value of an antisymmetric matrix occurs twice
        sigma = np.sort(sigma_T[i])[::-1][::2]

        mode_pos, k_pos = gamma_mode_shape(pos)
        mode_neg, k_neg = gamma_mode_shape(neg)
        mode_T, k_T = gamma_mode_shape(sigma)

        # The two full sorted lists, compared entry by entry: every
        # singular value of T, against every eigenvalue magnitude of S.
        magnitudes = np.sort(np.abs(ev))[::-1]
        singular = np.sort(sigma_T[i])[::-1]
        correlation = float(np.corrcoef(magnitudes, singular)[0, 1])

        rows.append(dict(
            model=tag, layer=int(layers[i]), head=int(heads[i]),
            mu=-mode_neg, nu=mode_pos, tau=mode_T,
            k_plus=k_pos, k_minus=k_neg, k_T=k_T,
            mode_ratio=mode_T / np.sqrt(mode_neg * mode_pos),
            shape_ratio=k_T / np.sqrt(k_pos * k_neg),
            sigma_lambda_correlation=correlation))
    return rows


def report(df: pd.DataFrame) -> None:
    defined = df.dropna(subset=["mode_ratio"])
    dropped = len(df) - len(defined)
    print(f"heads: {len(df)}; excluded from the mode and shape "
          f"comparison for want of an interior mode: {dropped}")

    def summary(frame: pd.DataFrame, column: str) -> str:
        lo, med, hi = np.quantile(frame[column].dropna(), [0.1, 0.5, 0.9])
        return f"median {med:.3f}, 80% in [{lo:.2f}, {hi:.2f}]"

    gpt2 = defined[defined.model == "gpt2"]
    print("\ngpt2:")
    print(f"  tau / sqrt(|mu| nu):  {summary(gpt2, 'mode_ratio')}")
    print(f"  k_T / sqrt(k_+ k_-):  {summary(gpt2, 'shape_ratio')}")
    print("  corr(sigma(T), |lambda(S)|):  "
          f"{summary(df[df.model == 'gpt2'], 'sigma_lambda_correlation')}")

    print("\nper-model medians:")
    print(f"{'model':26s} {'mode ratio':>11s} {'shape ratio':>12s} "
          f"{'corr':>7s} {'heads used':>11s}")
    for tag in ms.TAGS:
        d = defined[defined.model == tag]
        full = df[df.model == tag]
        print(f"{tag:26s} {d.mode_ratio.median():11.3f} "
              f"{d.shape_ratio.median():12.3f} "
              f"{full.sigma_lambda_correlation.median():7.3f} "
              f"{len(d):6d}/{len(full):<5d}")

    med = defined.groupby("model").median(numeric_only=True)
    corr = df.groupby("model").sigma_lambda_correlation.median()
    print(f"\nacross the {df.model.nunique()} models:")
    print(f"  mode ratio median in "
          f"[{med.mode_ratio.min():.2f}, {med.mode_ratio.max():.2f}]")
    print(f"  shape ratio median in "
          f"[{med.shape_ratio.min():.2f}, {med.shape_ratio.max():.2f}]")
    print(f"  correlation median in [{corr.min():.3f}, {corr.max():.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ms.TAGS)
    args = parser.parse_args()
    tags = [args.model] if args.model else ms.TAGS

    rows: list[dict] = []
    for tag in tags:
        rows.extend(model_rows(tag))
    df = pd.DataFrame(rows)

    path = RESULTS / "spectral_geometry.csv"
    df.to_csv(path, index=False)
    report(df)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
