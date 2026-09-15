#!/usr/bin/env python3
"""The goodness-of-fit tables of Section 3, as LaTeX.

Reads the summary written by ``gamma_goodness_of_fit.py`` and prints the
two tables of the paper: the share of heads best fit by each of the two
candidate families, averaged over the models, and the same shares model
by model.  Printing them from the results keeps the tables and the data
in step; paste the output into attention.tex.

Usage:
    python gamma_gof_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

SUMMARY = (ms.PROJECT / "scripts" / "survey_results"
           / "gamma_goodness_of_fit_summary.csv")

# the four spectral samples, with the symbols the paper uses
COMPONENTS = {
    "full": r"$|\lambda(L)|$",
    "S_positive": r"$\lambda_+(S)$",
    "S_negative": r"$\lambda_-(S)$",
    "T": r"$\sigma(T)$",
}
SHARE = "generalized_gamma_best_fit_share"


def mean_table(summary: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"sample & gen.\ gamma & lognormal \\", r"\midrule"]
    for key, symbol in COMPONENTS.items():
        rows = summary[summary.component == key]
        share = 100.0 * rows[SHARE]
        lines.append(
            f"{symbol} & ${share.mean():.1f} \\pm {share.std():.1f}\\%$ "
            f"& ${100 - share.mean():.1f} \\pm {share.std():.1f}\\%$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_model_table(summary: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             "model & " + " & ".join(COMPONENTS.values()) + r" \\",
             r"\midrule"]
    for tag in ms.TAGS:
        cells = []
        for key in COMPONENTS:
            row = summary[(summary.model == tag) & (summary.component == key)]
            if row.empty:
                raise ValueError(f"no result for {tag}, {key}")
            share = 100.0 * float(row[SHARE].iloc[0])
            cells.append(f"${share:.1f}$")
        lines.append(rf"\texttt{{{tag}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"{SUMMARY} not found; run gamma_goodness_of_fit.py first")
    summary = pd.read_csv(SUMMARY)
    missing = set(ms.TAGS) - set(summary.model.unique())
    if missing:
        raise ValueError(f"the summary is missing {sorted(missing)}")

    print("% mean shares across the models\n")
    print(mean_table(summary))
    print("\n% shares model by model (percentage best fit by "
          "generalized gamma)\n")
    print(per_model_table(summary))


if __name__ == "__main__":
    main()
