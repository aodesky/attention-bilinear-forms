#!/usr/bin/env python3
"""Observation 9: the symmetric and antisymmetric spectra correlate."""

from observation_common import Observation
from spectral_geometry import SpectralGeometry


class SigmaLambdaObservation(Observation):
    number = 9
    label = "sigma-lambda"

    def run(self) -> None:
        SpectralGeometry().report_metric("sigma_lambda_correlation")


if __name__ == "__main__":
    SigmaLambdaObservation().run()
