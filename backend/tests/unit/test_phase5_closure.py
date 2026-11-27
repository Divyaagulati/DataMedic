"""Phase 5 closure corrections: list headlines, serve-time sensitive-column redaction, internal
early stopping under grouped and temporal evaluation, the dummy's ROC-AUC, and the
undeclared-entity audit signal. Synthetic data only."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as hgb_module
from sklearn.ensemble import HistGradientBoostingClassifier

from datamedic.engine import ROW_ID
from datamedic.engine.ml import metrics as mm
from datamedic.engine.ml import spec as ml
from datamedic.engine.ml.models import build_estimator
from datamedic.engine.ml.runner import run_training
from datamedic.engine.preprocess.experiment.models import build_model, feature_kinds, model_matrix
from datamedic.engine.preprocess.split import make_splits
from datamedic.training import privacy
from datamedic.training.schemas import headline_of
from tests.unit.test_ml_audit import _audit
from tests.unit.test_ml_engine import frame, setup

# ---- 1. Headline -------------------------------------------------------------------------------


def _stats(mean: float | None, n: int = 3) -> dict[str, Any]:
    return {"mean": mean, "sd": None if mean is None else 0.01, "n_folds": n, "values": [mean] * 3}


def _eval(model: str, metric: str, mean: float | None, **kw: Any) -> Any:
    return SimpleNamespace(
        model_key=model,
        comparable=kw.get("comparable", True),
        status=kw.get("status", "completed"),
        metrics={"folds": {metric: _stats(mean, kw.get("n", 3))}},
    )


def _run(primary: str = "roc_auc", mode: str = "unbiased_direct", done: bool = True) -> Any:
    return SimpleNamespace(
        summary={"primary_metric": primary} if done else None,
        completed_at=datetime.now(UTC) if done else None,
        evaluation_mode=mode,
        task_type="binary",
        models=["dummy", "linear", "random_forest", "hgb"],
    )


def test_headline_respects_direction_excludes_dummy_and_uncompared_models() -> None:
    evals = [
        _eval("dummy", "roc_auc", 0.99),  # never the headline
        _eval("linear", "roc_auc", 0.80),
        _eval("random_forest", "roc_auc", 0.95, comparable=False, status="partial"),
        _eval("hgb", "roc_auc", 0.85),
    ]
    h = headline_of(_run(), SimpleNamespace(status="succeeded"), evals)
    assert h["state"] == "available" and h["model"] == "hgb" and h["mean"] == 0.85
    assert h["baseline_mean"] == 0.99 and h["higher_is_better"] is True
    assert "not a declared winner" in h["note"] and h["exploratory"] is False
    # Lower is better: the smallest mean is the highest observed performance.
    low = [_eval("linear", "rmse", 2.0), _eval("hgb", "rmse", 1.5)]
    h = headline_of(_run("rmse"), None, low)
    assert h["model"] == "hgb" and h["higher_is_better"] is False
    # Ties go to the earlier model of the run, not to whichever row came first.
    tie = [_eval("hgb", "roc_auc", 0.8), _eval("linear", "roc_auc", 0.8)]
    assert headline_of(_run(), None, tie)["model"] == "linear"
    # A mean over fewer folds than the run has is disclosed.
    part = headline_of(_run(), None, [_eval("linear", "roc_auc", 0.7, n=2)])
    assert part["n_folds"] == 2 and part["total_folds"] == 3


def test_headline_states_are_explicit() -> None:
    pending = headline_of(_run(done=False), SimpleNamespace(status="running"), [])
    assert pending["state"] == "pending" and pending["mean"] is None
    cancelled = headline_of(_run(done=False), SimpleNamespace(status="cancelled"), [])
    assert cancelled["state"] == "cancelled"
    failed = headline_of(_run(done=False), SimpleNamespace(status="failed"), [])
    assert failed["state"] == "failed" and "no results" in failed["reason"]
    only_dummy = headline_of(_run(), None, [_eval("dummy", "roc_auc", 0.5)])
    assert only_dummy["state"] == "unavailable" and "dummy" in only_dummy["reason"]
    undefined = headline_of(_run(), None, [_eval("linear", "roc_auc", None)])
    assert undefined["state"] == "unavailable" and "could not be computed" in undefined["reason"]
    none_compared = headline_of(
        _run(), None, [_eval("linear", "roc_auc", 0.9, comparable=False, status="partial")]
    )
    assert none_compared["state"] == "unavailable" and none_compared["mean"] is None


def test_exploratory_headline_carries_its_qualification() -> None:
    h = headline_of(_run(mode="exploratory"), None, [_eval("linear", "roc_auc", 0.9)])
    assert h["exploratory"] is True and "not a leakage-controlled" in h["exploratory_note"]


# ---- 2. Serve-time sensitivity -----------------------------------------------------------------


def test_sensitivity_propagates_through_derived_columns() -> None:
    edges = [("code", "code__missing"), ("code__missing", "code__missing__x"), ("b", "b__missing")]
    assert privacy.propagate({"code"}, edges) == {"code__missing", "code__missing__x"}
    assert privacy.propagate(set(), edges) == set()


def test_redaction_filters_copies_and_never_mutates_stored_results() -> None:
    details = {
        "error_slices": [
            {"column": "code", "slices": [{"label": "k1", "n": 40, "error": 0.2}]},
            {"column": "a", "slices": [{"label": "0 – 1", "n": 40, "error": 0.1}]},
        ],
        "residuals": {"available": True, "value_range": [0, 9], "mean_residual": 0.1, "n": 9},
    }
    protected = {"code": [privacy.NOW]}
    out, withheld = privacy.redact_details(details, protected, target_protected=True)
    assert [s["column"] for s in out["error_slices"]] == ["a"] and withheld == ["code"]
    assert "value_range" not in out["residuals"] and out["residuals"]["redacted_axes"] is True
    assert len(details["error_slices"]) == 2 and "value_range" in details["residuals"]
    summary = {"classes": ["no", "yes"], "positive_class": "yes"}
    red = privacy.redact_summary(summary, target_protected=True)
    assert red == {"classes": ["class 1", "class 2"], "positive_class": "class 2"}
    assert summary["classes"] == ["no", "yes"]
    assert privacy.redact_summary(summary, target_protected=False) is summary
    d = privacy.disclosure(protected, withheld, "y")
    assert d["withheld_error_slices"] == ["code"] and d["target_values_withheld"] is False


# ---- 4. Internal early stopping -----------------------------------------------------------------

# With 5 folds the training rows exceed scikit-learn's 10,000-row threshold at which
# early_stopping="auto" switches on (grouped: 4/5 of the rows; time: the last fold, ~5/6).
BIG = 14_100
BIG_TIME = 14_100


def _no_internal_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("an internal random validation split was drawn")

    monkeypatch.setattr(hgb_module, "train_test_split", refuse)


def _grouped_and_temporal() -> list[tuple[Any, str, dict[str, Any]]]:
    g = frame("binary", n=BIG, seed=3)
    g["patient"] = np.repeat(np.arange(BIG // 3), 3)
    t = frame("binary", n=BIG_TIME, seed=4)
    t["t"] = np.arange(BIG_TIME)
    return [
        (
            g,
            "stratified_group",
            {"group": "patient", "roles": {"target": ["y"], "group": ["patient"]}},
        ),
        (t, "time", {"time": "t", "roles": {"target": ["y"], "time": ["t"]}}),
    ]


def test_library_default_would_draw_an_ungrouped_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: proves the guard below is effective on large training sets."""
    _no_internal_split(monkeypatch)
    df = frame("binary", n=BIG)
    kinds = feature_kinds(df, ["a", "b"])
    with pytest.raises(AssertionError, match="internal random validation split"):
        HistGradientBoostingClassifier(random_state=0).fit(model_matrix(df, kinds), df["y"])


@pytest.mark.parametrize("which", ["lab", "training"])
def test_gradient_boosting_never_early_stops_on_grouped_or_temporal_training_rows(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_internal_split(monkeypatch)
    for df, method, kw in _grouped_and_temporal():
        splits = make_splits(
            df,
            method=method,
            n_splits=5,
            seed=0,
            target="y" if method != "time" else None,
            group=kw.get("group"),
            time=kw.get("time"),
            duplicate_columns=["a", "b", "c", "y"],
        )
        fold = max(splits.folds, key=lambda f: len(f.train_ids))
        train = df[df[ROW_ID].isin(fold.train_ids)]
        assert len(train) > 10_000  # 'auto' would early-stop on these training rows
        kinds = feature_kinds(train, ["a", "b", "c"])
        est = (
            build_model("hgb", "binary", kinds, 0)
            if which == "lab"
            else build_estimator(
                "hgb", "binary", kinds, dict(ml.MODELS["hgb"].default("binary")), 0
            )
        )
        assert est.named_steps["model"].get_params()["early_stopping"] is False
        est.fit(model_matrix(train, kinds), train["y"])
        model = est.named_steps["model"]
        assert model.do_early_stopping_ is False
        assert model.n_iter_ == model.max_iter == 100  # bounded, fixed iteration count
        assert len(model.validation_score_) == 0  # no rows were held out internally


def test_training_runner_with_gradient_boosting_on_grouped_and_temporal_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Phase 5 path (defaults mode) on > 10,000 rows: no internal split anywhere."""
    _no_internal_split(monkeypatch)
    for df, method, kw in _grouped_and_temporal():
        splits, pre, spec = setup(
            df, "binary", models=("hgb",), method=method, n_splits=5, explain=False, **kw
        )
        out = run_training(df, splits, pre, spec)
        assert out.models["hgb"]["status"] in ("completed", "partial")
        assert not out.models["hgb"]["failures"]


# ---- 6. Dummy ROC-AUC ---------------------------------------------------------------------------


def test_dummy_roc_auc_is_exactly_one_half_when_both_classes_are_present() -> None:
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    constant = np.tile([0.5, 0.5], (len(y), 1))
    m, reasons, _ = mm.classification(y, np.zeros_like(y), constant, 2)
    assert m["roc_auc"] == 0.5 and "roc_auc" not in reasons
    df = frame("binary")
    out = run_training(df, *setup(df, "binary", models=("dummy",), explain=False))
    folds = out.models["dummy"]["metrics"]["roc_auc"]
    assert folds["values"] == [0.5] * len(folds["values"]) and folds["sd"] == 0.0
    # Genuinely undefined (one class among the test rows) stays unavailable, never 0.5.
    one = np.zeros(6, dtype=int)
    m, reasons, _ = mm.classification(one, one, np.tile([0.5, 0.5], (6, 1)), 2)
    assert m["roc_auc"] is None and reasons["roc_auc"]
    assert any("exactly 0.5" in c for c in ml.METRIC_CONVENTIONS)


# ---- 3. Undeclared entities --------------------------------------------------------------------


def test_key_like_feature_without_group_role_is_a_possible_entity_not_a_confirmed_identifier() -> (
    None
):
    df = frame("binary", n=240)
    df["patient_id"] = np.repeat(np.arange(80), 3)  # a model input: no role declared
    f = _audit(df, roles={"target": ["y"]})["by"]["leakage.groups.undeclared"]
    assert f.status == "detected" and f.severity == "medium"
    assert f.title.startswith("Possible") and f.evidence["basis"] == "statistical"
    assert f.evidence["signals"] == [
        {"column": "patient_id", "kind": "key_like_feature_repeats", "rows_per_value": 3.0}
    ]
    assert "not a confirmed identifier" in f.explanation
    # Controls: continuous measurements and low-cardinality categories are not entity keys.
    clean = _audit(frame("binary", n=240), roles={"target": ["y"]})["by"]
    g = clean["leakage.groups.undeclared"]
    assert g.status == "not_detected" and "inspected signals" in g.explanation
    assert clean["leakage.groups.cross_fold"].status == "not_assessed"
