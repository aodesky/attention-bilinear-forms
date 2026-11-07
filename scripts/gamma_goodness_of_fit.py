#!/usr/bin/env python3
"""Goodness-of-fit comparison for the spectral samples of Section 3.

For each head of the models in the paper's survey (model_set.MODELS),
fit a gamma, a lognormal and a Weibull distribution, each with location
fixed at zero, to four positive spectral samples:

  * magnitudes of the nonzero eigenvalues of the full QK form;
  * positive eigenvalues of the symmetric part S;
  * magnitudes of the negative eigenvalues of S;
  * the positive imaginary parts of the eigenvalues of the
    antisymmetric part T.

The three families all have two parameters once the location is fixed,
so comparing them by maximum likelihood is a fair comparison: none can
win by having more freedom than the others.  (A generalized gamma has
three parameters and contains both the gamma and the Weibull as special
cases, so it can never fit worse and is not a candidate here.)

For each fit the script records the average error between the observed
histogram and the fitted density, and which family fits best.  These
errors are descriptive: eigenvalues within a head need not be
independent, and the parameters are estimated from the same
observations.

The spectra come from scripts/survey_results/, written by
head_statistics.py from the extraction database; no model weights are
loaded here.

Outputs:
    scripts/survey_results/gamma_goodness_of_fit.csv
    scripts/survey_results/gamma_goodness_of_fit_summary.csv
"""

from pathlib import Path
import argparse
import gc
import sys

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

SPECTRA = PROJECT / "scripts" / "survey_results"
OUTPUT = SPECTRA

MODELS = dict(ms.MODELS)


# The three two-parameter families compared, each fitted with the
# location held at zero.
CANDIDATES = {
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
}


def load_samples(tag):
    """The four positive spectral samples of every head of a model.

    Read from the per-head spectra saved by head_statistics.py: the
    eigenvalues of the symmetric part and the singular values of the
    antisymmetric part.  The magnitudes of the eigenvalues of the full
    form L are recomputed here, since only its symmetric and
    antisymmetric spectra are cached.
    """
    with np.load(SPECTRA / f"{tag}_spectra.npz") as z:
        eigenvalues = z["eigenvalues"]

    positive = np.stack([row[row > 0] for row in eigenvalues])
    negative = np.stack([-row[row < 0] for row in eigenvalues])

    # The form has rank n, so only n of its 2n eigenvalues are nonzero;
    # the sample is those n magnitudes, as in the paper.  The eigenvalues
    # of the antisymmetric part are purely imaginary and come in
    # conjugate pairs, and the sample is their positive imaginary parts:
    # taken this way, two that happen to coincide are both kept.
    rank = ms.model_info(tag).head_dim
    full, imaginary = [], []
    for _, _, _, L in ms.iter_forms(tag):
        full.append(np.sort(np.abs(np.linalg.eigvals(L)))[::-1][:rank])
        _, antisymmetric = ms.symmetric_antisymmetric(L)
        parts = np.linalg.eigvals(antisymmetric).imag
        imaginary.append(np.sort(parts[parts > 0])[::-1])
    full = np.stack(full)
    if not (full > 0).all():
        raise ValueError(f"{tag}: a full-form eigenvalue magnitude vanished")

    widths = {len(row) for row in imaginary}
    if len(widths) != 1:
        raise ValueError(
            f"{tag}: heads disagree on the number of positive imaginary "
            f"parts ({sorted(widths)})")

    return {
        "full": full,
        "S_positive": positive,
        "S_negative": negative,
        "T": np.stack(imaginary),
    }


BINS = 20


def bin_error(sample, distribution, parameters):
    """Average error between the histogram and the fitted density.

    The sample is binned into a histogram of BINS equal bins scaled to
    unit area, and the result is the root mean square difference between
    the height of each bin and the fitted density at its midpoint.  The
    sample is rescaled to unit mean first, so that the measure is
    dimensionless: without that it would carry the units of the
    spectrum, and the value for a model with small eigenvalues would not
    be comparable with one whose eigenvalues are large.
    """
    scaled = sample / sample.mean()
    rescaled_parameters = distribution.fit(scaled, floc=0)
    edges = np.histogram_bin_edges(scaled, bins=BINS)
    observed, _ = np.histogram(scaled, bins=edges, density=True)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    predicted = distribution.pdf(midpoints, *rescaled_parameters)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def evaluate_fit(sample, distribution, parameters):
    log_likelihood = float(np.sum(distribution.logpdf(sample, *parameters)))
    return parameters, log_likelihood, bin_error(sample, distribution,
                                                 parameters)


def fit_candidate(sample, distribution):
    parameters = distribution.fit(sample, floc=0)
    return evaluate_fit(sample, distribution, parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS),
                        help="only this model")
    args = parser.parse_args()
    models = [args.model] if args.model else list(MODELS)

    records = []
    for model in models:
        for component, samples in load_samples(model).items():
            for head, sample in enumerate(samples):
                fits = {name: fit_candidate(sample, distribution)
                        for name, distribution in CANDIDATES.items()}
                best_log_likelihood = max(value[1] for value in fits.values())
                winner = max(CANDIDATES,
                             key=lambda family: fits[family][1])
                for family, (_, log_likelihood, error) in fits.items():
                    records.append(
                        {
                            "model": model,
                            "component": component,
                            "head_index": head,
                            "sample_size": len(sample),
                            "family": family,
                            "bin_error": error,
                            "log_likelihood": log_likelihood,
                            "fit_gap": best_log_likelihood - log_likelihood,
                            "best_fit": family == winner,
                        }
                    )

    results = pd.DataFrame.from_records(records)
    results.to_csv(OUTPUT / "gamma_goodness_of_fit.csv", index=False)

    summaries = []
    for model in models:
        for component in load_samples(model):
            subset = results[
                (results["model"] == model)
                & (results["component"] == component)
            ]
            reference = subset[subset["family"] == "gamma"]
            row = {
                "model": model,
                "component": component,
                "heads": len(reference),
                "sample_size_per_head": int(reference["sample_size"].iloc[0]),
                "gamma_median_error": reference["bin_error"].median(),
                "gamma_iqr_error_low": reference["bin_error"].quantile(0.25),
                "gamma_iqr_error_high": reference["bin_error"].quantile(0.75),
            }
            for family in CANDIDATES:
                family_rows = subset[subset["family"] == family]
                row[f"{family}_best_fit_share"] = (
                    family_rows["best_fit"].mean()
                )
            summaries.append(row)

    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(
        OUTPUT / "gamma_goodness_of_fit_summary.csv", index=False
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
