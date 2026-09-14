#!/usr/bin/env python3
"""Goodness-of-fit diagnostics for the spectral claims in Section 4.

For each head of the models in the paper's survey (model_set.MODELS),
compare a generalized gamma fit with a lognormal fit, both with
location fixed at zero, on four positive spectral samples:

  * magnitudes of the nonzero eigenvalues of the full QK form;
  * positive eigenvalues of the symmetric part S;
  * magnitudes of the negative eigenvalues of S;
  * one singular value from each repeated pair of the antisymmetric part T.

Only these two families are reported.  Gamma and Weibull are special
cases of generalized gamma, so with no penalty for the number of fitted
parameters they can never fit better; they are still fitted here, since
their maximum likelihood estimates seed the generalized gamma fit, but
they are not among the candidates compared.

The two families are compared by maximum likelihood alone.  The script
also records the Kolmogorov--Smirnov distance of each fit.  The KS
distances are descriptive: eigenvalues within a head need not be
independent, and parameters are estimated from the same observations.

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


# The families whose shares the paper reports.  Gamma and Weibull are
# nested in generalized gamma and so are not compared; see the note above.
CANDIDATES = {
    "generalized_gamma": (stats.gengamma, 3),
    "lognormal": (stats.lognorm, 2),
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
        sigma_T = z["sigma_T"]

    positive = np.stack([row[row > 0] for row in eigenvalues])
    negative = np.stack([-row[row < 0] for row in eigenvalues])

    # Each singular value of a real antisymmetric matrix occurs twice;
    # sigma_T is sorted decreasing, so every second entry is distinct.
    distinct_antisymmetric = np.sort(sigma_T, axis=1)[:, ::-1][:, ::2]

    # The form has rank n, so only n of its 2n eigenvalues are nonzero;
    # the sample is those n magnitudes, as in the paper.
    rank = ms.model_info(tag).head_dim
    full = np.stack([
        np.sort(np.abs(np.linalg.eigvals(L)))[::-1][:rank]
        for _, _, _, L in ms.iter_forms(tag)
    ])
    if not (full > 0).all():
        raise ValueError(f"{tag}: a full-form eigenvalue magnitude vanished")

    return {
        "full": full,
        "S_positive": positive,
        "S_negative": negative,
        "T": distinct_antisymmetric,
    }


def evaluate_fit(sample, distribution, parameters):
    log_likelihood = float(np.sum(distribution.logpdf(sample, *parameters)))
    ks_distance = float(
        stats.kstest(sample, distribution.cdf, args=parameters).statistic
    )
    return parameters, log_likelihood, ks_distance


def fit_candidate(sample, distribution):
    parameters = distribution.fit(sample, floc=0)
    return evaluate_fit(sample, distribution, parameters)


def fit_generalized_gamma(sample, gamma_fit, weibull_fit):
    """Fit generalized gamma, retaining its nested gamma/Weibull fits."""
    candidates = [fit_candidate(sample, stats.gengamma)]
    gamma_parameters = gamma_fit[0]
    weibull_parameters = weibull_fit[0]
    candidates.append(
        evaluate_fit(
            sample,
            stats.gengamma,
            (gamma_parameters[0], 1.0, 0.0, gamma_parameters[-1]),
        )
    )
    candidates.append(
        evaluate_fit(
            sample,
            stats.gengamma,
            (1.0, weibull_parameters[0], 0.0, weibull_parameters[-1]),
        )
    )
    return max(candidates, key=lambda fit: fit[1])


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
                # The nested gamma and Weibull fits seed the generalized
                # gamma optimization but are not themselves candidates.
                gamma_fit = fit_candidate(sample, stats.gamma)
                weibull_fit = fit_candidate(sample, stats.weibull_min)
                fits = {
                    "generalized_gamma": fit_generalized_gamma(
                        sample, gamma_fit, weibull_fit
                    ),
                    "lognormal": fit_candidate(sample, stats.lognorm),
                }
                best_log_likelihood = max(value[1] for value in fits.values())
                # Prefer the more general family should the two fits tie.
                winner = max(("generalized_gamma", "lognormal"),
                             key=lambda family: fits[family][1])
                for family, (_, log_likelihood, ks_distance) in fits.items():
                    records.append(
                        {
                            "model": model,
                            "component": component,
                            "head_index": head,
                            "sample_size": len(sample),
                            "family": family,
                            "ks_distance": ks_distance,
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
            reference = subset[subset["family"] == "generalized_gamma"]
            row = {
                "model": model,
                "component": component,
                "heads": len(reference),
                "sample_size_per_head": int(reference["sample_size"].iloc[0]),
                "gengamma_median_ks": reference["ks_distance"].median(),
                "gengamma_iqr_ks_low": reference["ks_distance"].quantile(0.25),
                "gengamma_iqr_ks_high": reference["ks_distance"].quantile(0.75),
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
