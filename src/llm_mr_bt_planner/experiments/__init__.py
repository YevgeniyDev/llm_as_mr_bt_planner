"""Reproducible multi-trial experiments and reporting."""

from __future__ import annotations

from .report import aggregate, to_csv, to_markdown_table
from .runner import ExperimentReport, TrialRecord, run_experiment

__all__ = [
    "run_experiment",
    "ExperimentReport",
    "TrialRecord",
    "aggregate",
    "to_csv",
    "to_markdown_table",
]
