"""Reusable building blocks for the paper's observation entry points."""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence


class Observation(ABC):
    """One reproducible empirical observation from the paper."""

    number: int
    label: str

    @abstractmethod
    def run(self) -> None:
        """Regenerate the evidence and print the reported summary."""


@contextmanager
def _arguments(program: str, arguments: Sequence[str]) -> Iterator[None]:
    old = sys.argv
    sys.argv = [program, *arguments]
    try:
        yield
    finally:
        sys.argv = old


@dataclass(frozen=True)
class ModuleStep:
    """An importable legacy analysis step in an observation pipeline."""

    module: str
    arguments: tuple[str, ...] = ()

    def run(self) -> None:
        component = importlib.import_module(self.module)
        with _arguments(self.module, self.arguments):
            component.main()


class ObservationPipeline(Observation):
    """An observation assembled from reusable analysis components."""

    steps: tuple[ModuleStep, ...] = ()

    def run(self) -> None:
        for step in self.steps:
            step.run()

