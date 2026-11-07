#!/usr/bin/env python3
"""Observation 7: the antisymmetric mode is a geometric mean."""

from observation_common import Observation
from spectral_geometry import SpectralGeometry


class ModesObservation(Observation):
    number = 7
    label = "modes"

    def run(self) -> None:
        SpectralGeometry().report_metric("mode_ratio")


if __name__ == "__main__":
    ModesObservation().run()

