"""Executing validated steps as a scikit-learn ``Pipeline``.

The executor is the leakage primitive used by experiments, impact previews and commits:

    executor = PreprocessExecutor(validation_result, seed=0)
    executor.fit(frame.loc[train_mask])         # learns from these rows only
    result = executor.transform(frame)           # never learns; returns frame + step reports

Fitting a step sees the fitting rows *after* the earlier steps (scikit-learn ``Pipeline``
semantics), so later steps never see rows the earlier ones did not.
"""

import importlib
import time
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.base import OP_INVALID, DMTransformer
from datamedic.engine.preprocess.catalog import SAMPLE_WEIGHT
from datamedic.engine.preprocess.report import (
    StepReport,
    column_changes,
    feature_lineage,
    redact,
    redact_effects,
)
from datamedic.engine.preprocess.validate import ValidatedStep, ValidationResult

POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
MODEL_STAGE_NOT_COMMITTABLE = "MODEL_STAGE_NOT_COMMITTABLE"
INTERNAL_PREFIX = "_dm_"  # row identifier, sample weight: never model inputs


def load_transformer(v: ValidatedStep, seed: int) -> DMTransformer:
    module_name, _, class_name = v.spec.impl.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, DMTransformer)) or cls.op != v.spec.op:
        raise EngineError(OP_INVALID, "The operation's implementation is misconfigured.")
    transformer: DMTransformer = cls(
        columns=tuple(v.step.columns), params=dict(v.step.params), seed=seed
    )
    return transformer


def _check_row_ids(frame: pd.DataFrame) -> None:
    if ROW_ID not in frame.columns:
        raise EngineError(OP_INVALID, "Steps run on canonical tables with a row identifier.")
    if not frame[ROW_ID].is_unique:
        raise EngineError(OP_INVALID, "Row identifiers must be unique.")
    if not frame.columns.is_unique:
        raise EngineError(OP_INVALID, "Column names must be unique.")


@dataclass
class TransformResult:
    frame: pd.DataFrame
    reports: list[StepReport]
    # Per column: origin, the steps that transformed, assessed, added or removed it, and
    # whether it is still present (feature lineage).
    lineage: list[dict[str, Any]] = field(default_factory=list)

    def report_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.reports]


class PreprocessExecutor:
    def __init__(
        self,
        validation: ValidationResult,
        *,
        seed: int = 0,
        sensitive: frozenset[str] | None = None,
    ) -> None:
        if not validation.valid:
            raise EngineError(OP_INVALID, "The pipeline has validation errors and cannot run.")
        self.validation = validation
        self.seed = seed
        self.sensitive = validation.sensitive if sensitive is None else sensitive
        self.pipeline = Pipeline(
            [(f"step_{v.index}", load_transformer(v, seed)) for v in validation.steps]
        )
        self._fitted = False

    @property
    def steps(self) -> list[ValidatedStep]:
        return self.validation.steps

    def transformers(self) -> list[DMTransformer]:
        return [t for _, t in self.pipeline.steps]

    def fit(self, frame: pd.DataFrame) -> "PreprocessExecutor":
        """Learn every step from ``frame`` only (e.g. the training rows of one fold)."""
        _check_row_ids(frame)
        if self.pipeline.steps:
            start = time.perf_counter()
            self.pipeline.fit(frame)
            self._fit_ms = (time.perf_counter() - start) * 1000
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> TransformResult:
        """Apply the fitted steps to any rows. Equivalent to ``Pipeline.transform`` plus a
        report per step and post-condition checks."""
        if not self._fitted:
            raise EngineError(OP_INVALID, "The pipeline must be fitted before it is applied.")
        _check_row_ids(frame)
        current = frame
        reports: list[StepReport] = []
        lineage_steps: list[dict[str, Any]] = []
        for v, (_, transformer) in zip(self.steps, self.pipeline.steps, strict=True):
            start = time.perf_counter()
            after = transformer.transform(current)
            duration = (time.perf_counter() - start) * 1000
            self._check_postconditions(v, current, after, self.validation.protected)
            added = [c for c in after.columns if c not in current.columns]
            removed = [c for c in current.columns if c not in after.columns]
            lineage_steps.append(
                {
                    "index": v.index,
                    "op": v.spec.op,
                    "columns": list(v.step.columns),
                    "added": added,
                    "removed": removed,
                    "decisions": _decisions(transformer),
                }
            )
            reports.append(
                StepReport(
                    index=v.index,
                    op=v.spec.op,
                    op_version=v.spec.version,
                    step_hash=v.step_hash,
                    rows_in=len(current),
                    rows_out=len(after),
                    columns_added=added,
                    columns_removed=removed,
                    columns=column_changes(current, after, list(v.step.columns)),
                    learned=redact(transformer.learned_params(), self.sensitive),
                    warnings=list(transformer.warnings_) + list(transformer.transform_warnings_),
                    n_fit_rows=int(transformer.n_fit_rows_),
                    effects=redact_effects(
                        dict(getattr(transformer, "effects_", {})), self.sensitive
                    ),
                    duration_ms=round(duration, 3),
                )
            )
            current = after
        input_columns = [c for c in frame.columns if c != ROW_ID]
        return TransformResult(current, reports, feature_lineage(input_columns, lineage_steps))

    def fit_transform(self, frame: pd.DataFrame) -> TransformResult:
        """Fit on all of ``frame`` and apply to it (evaluation helpers and code-generation
        equivalence). Not a commit path: committing must use ``commit_transform``, which
        refuses model-stage steps. Training-only steps leave the rows unchanged here."""
        return self.fit(frame).transform(frame)

    def commit_transform(self, frame: pd.DataFrame) -> TransformResult:
        """The only executor method a commit (new dataset version) may use. Refuses, before
        any data is read, every step that is not ``data`` or ``either`` stage: scaling,
        encoding, feature selection, resampling and class weights belong inside a model's
        training folds and are never written into a version (owner decision D1). Training-only
        steps are model stage, so resampled rows can never be committed."""
        model_steps = [v for v in self.steps if v.spec.stage == "model"]
        if model_steps:
            raise EngineError(
                MODEL_STAGE_NOT_COMMITTABLE,
                "Model-stage steps (scaling, encoding, feature selection, resampling, class "
                "weights) run only inside model training and can never be committed into a "
                "dataset version.",
                {
                    "steps": [v.index for v in model_steps],
                    "ops": sorted({v.spec.op for v in model_steps}),
                },
            )
        return self.fit(frame).transform(frame)

    def fit_resample(self, frame: pd.DataFrame) -> TransformResult:
        """Fit every step on ``frame`` (the training rows of one fold) and return the table a
        model is trained on: every step applied, then the training-only step (resampling or
        class weights, always the last step) applied to these rows only.

        This is the only way to obtain resampled rows. Apply the fitted steps to validation
        and test rows with ``transform``, which never resamples. Added rows have negative
        ``_dm_row_id`` values and the index is a fresh range (rows were added or removed)."""
        if self.steps and self.steps[-1].spec.kind == "training_only":
            _check_row_ids(frame)
            if (frame[ROW_ID] < 0).any():
                raise EngineError(
                    OP_INVALID,
                    "Training rows must have non-negative row identifiers; negative identifiers "
                    "mark rows added by resampling.",
                )
        self.fit(frame)
        result = self.transform(frame)
        if not self.steps or self.steps[-1].spec.kind != "training_only":
            return result
        v = self.steps[-1]
        transformer = self.pipeline.steps[-1][1]
        before = result.frame
        start = time.perf_counter()
        out = transformer.resample(before)
        duration = (time.perf_counter() - start) * 1000
        self._check_training_table(v, before, out, self.validation.target)
        lineage = list(result.lineage)
        if SAMPLE_WEIGHT in out.columns and SAMPLE_WEIGHT not in before.columns:
            lineage.append(
                {
                    "column": SAMPLE_WEIGHT,
                    "origin": "added",
                    "derived_from": [self.validation.target] if self.validation.target else [],
                    "events": [{"step": v.index, "op": v.spec.op, "action": "added"}],
                    "final": "present",
                    "feature": False,  # a per-row training weight, never a model input
                }
            )
        last = result.reports[-1]
        result.reports[-1] = replace(
            last,
            rows_out=len(out),
            columns_added=[c for c in out.columns if c not in before.columns],
            learned=redact(transformer.learned_params(), self.sensitive),
            warnings=list(last.warnings) + list(transformer.resample_warnings_),
            effects=dict(getattr(transformer, "effects_", {})),
            duration_ms=round(last.duration_ms + duration, 3),
        )
        return TransformResult(out, result.reports, lineage)

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """The columns of a transformed or training table that are model inputs: everything
        except the target, protected role columns (identifier, group, time, weight, excluded)
        and internal ``_dm_`` columns (``_dm_row_id``, ``_dm_sample_weight``)."""
        excluded = set(self.validation.protected)
        if self.validation.target is not None:
            excluded.add(self.validation.target)
        return [
            c for c in frame.columns if c not in excluded and not str(c).startswith(INTERNAL_PREFIX)
        ]

    @staticmethod
    def _check_training_table(
        v: ValidatedStep, before: pd.DataFrame, out: pd.DataFrame, target: str | None
    ) -> None:
        """Original rows keep their identifier and label; added rows have negative ids."""

        def fail(message: str) -> EngineError:
            return EngineError(POSTCONDITION_FAILED, message, {"op": v.spec.op, "step": v.index})

        if not out[ROW_ID].is_unique:
            raise fail("Row identifiers of the training table are not unique.")
        expected = list(before.columns) + ([SAMPLE_WEIGHT] if SAMPLE_WEIGHT in out else [])
        if list(out.columns) != expected:
            raise fail("Resampling changed the columns.")
        original = out[out[ROW_ID] >= 0]
        if not original[ROW_ID].isin(before[ROW_ID]).all():
            raise fail("Resampling invented a row identifier.")
        if target is not None:
            labels = before.set_index(ROW_ID)[target]
            kept = original.set_index(ROW_ID)[target]
            if not kept.astype(object).equals(labels.loc[kept.index].astype(object)):
                raise fail("Resampling changed a class label.")

    @staticmethod
    def _check_postconditions(
        v: ValidatedStep,
        before: pd.DataFrame,
        after: pd.DataFrame,
        protected: frozenset[str] = frozenset(),
    ) -> None:
        kind = v.spec.kind
        if kind not in ("row_filter_rule", "row_filter_fitted"):
            if len(after) != len(before) or not after[ROW_ID].equals(before[ROW_ID]):
                raise EngineError(
                    POSTCONDITION_FAILED,
                    "A step that must keep every row changed the rows.",
                    {"op": v.spec.op, "step": v.index},
                )
        if kind == "training_only" and not after.equals(before):
            raise EngineError(
                POSTCONDITION_FAILED,
                "A training-only step changed rows outside the training table.",
                {"op": v.spec.op, "step": v.index},
            )
        removed = [c for c in before.columns if c not in after.columns]
        if any(c not in v.step.columns for c in removed):
            raise EngineError(
                POSTCONDITION_FAILED,
                "A step removed a column it was not given.",
                {"op": v.spec.op, "step": v.index},
            )
        for column in protected:
            if column in before.columns and (
                column not in after.columns or not after[column].equals(before[column])
            ):
                raise EngineError(
                    POSTCONDITION_FAILED,
                    "A step changed a protected column (target, identifier, group, time, "
                    "weight or excluded column).",
                    {"op": v.spec.op, "step": v.index, "column": column},
                )
        if v.spec.op.startswith("impute."):
            for column in v.step.columns:
                if after[column].isna().any():
                    raise EngineError(
                        POSTCONDITION_FAILED,
                        "Missing values remain after imputation.",
                        {"op": v.spec.op, "step": v.index, "column": column},
                    )


def _decisions(transformer: DMTransformer) -> dict[str, str]:
    """Selection decisions per column (feature lineage); empty for other ops."""
    decisions = getattr(transformer, "decisions", None)
    return dict(decisions()) if callable(decisions) else {}
