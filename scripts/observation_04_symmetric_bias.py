#!/usr/bin/env python3
"""Observation 4: attention heads are biased toward symmetry."""

from observation_common import Observation
from score_survey import ScoreSurvey


class SymmetricBiasObservation(Observation):
    number = 4
    label = "symmetric-bias"

    def run(self) -> None:
        ScoreSurvey().symmetric_bias()


if __name__ == "__main__":
    SymmetricBiasObservation().run()

