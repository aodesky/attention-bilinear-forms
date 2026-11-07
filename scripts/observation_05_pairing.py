#!/usr/bin/env python3
"""Observation 5: the sorted spectral lobes are approximately proportional."""

from observation_common import Observation
from score_survey import ScoreSurvey


class PairingObservation(Observation):
    number = 5
    label = "pairing"

    def run(self) -> None:
        ScoreSurvey().pairing()


if __name__ == "__main__":
    PairingObservation().run()

