#!/usr/bin/env python3
"""Observation 3: symmetric spectra have two gamma-like lobes."""

from observation_common import ModuleStep, ObservationPipeline


class BimodalGammaObservation(ObservationPipeline):
    number = 3
    label = "bimodal-gamma"
    steps = (
        ModuleStep("cluster_head_spectra"),
        ModuleStep("plot_mean_spectrum"),
    )


if __name__ == "__main__":
    BimodalGammaObservation().run()

