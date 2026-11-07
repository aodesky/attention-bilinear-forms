#!/usr/bin/env python3
"""Observation 1: every symmetric part is balanced."""

from observation_common import Observation
from score_survey import ScoreSurvey


class BalanceObservation(Observation):
    number = 1
    label = "balance"

    def run(self) -> None:
        ScoreSurvey().balance()


if __name__ == "__main__":
    BalanceObservation().run()

