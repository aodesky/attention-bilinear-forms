#!/usr/bin/env python3
"""Observation 8: the antisymmetric shape is a geometric mean."""

from observation_common import Observation
from spectral_geometry import SpectralGeometry


class ShapesObservation(Observation):
    number = 8
    label = "shapes"

    def run(self) -> None:
        SpectralGeometry().report_metric("shape_ratio")


if __name__ == "__main__":
    ShapesObservation().run()

