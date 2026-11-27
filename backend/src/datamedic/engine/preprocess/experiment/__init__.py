"""Preprocessing Experiment Lab engine (Phase 4 Part D): candidate pipelines evaluated on
identical frozen folds, with data-quality changes, model metrics and selection stability.
Imports pandas and scikit-learn: used by the worker only, never by the API."""

from datamedic.engine.preprocess.experiment.runner import (
    Cancelled,
    Candidate,
    ExperimentSpec,
    cohort,
    run_experiment,
)

__all__ = ["Cancelled", "Candidate", "ExperimentSpec", "cohort", "run_experiment"]
