#!/usr/bin/env python3
"""Observation 2: head profiles accumulate near Theta."""

from observation_common import ModuleStep, ObservationPipeline


class HeadProfilesObservation(ObservationPipeline):
    number = 2
    label = "head-profiles"
    steps = (ModuleStep("type_partition"), ModuleStep("effective_rank"),
             ModuleStep("plot_simplex_scatter"))


if __name__ == "__main__":
    HeadProfilesObservation().run()

