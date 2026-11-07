#!/usr/bin/env python3
"""Observation 6: full and antisymmetric spectra are gamma-like."""

from observation_common import ModuleStep, ObservationPipeline


class FullGammaObservation(ObservationPipeline):
    number = 6
    label = "full-gamma"
    steps = (
        ModuleStep("gamma_goodness_of_fit"),
        ModuleStep("plot_goodness_of_fit"),
    )


if __name__ == "__main__":
    FullGammaObservation().run()

