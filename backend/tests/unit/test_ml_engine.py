"""Phase 5 training engine: models, nested selection, split isolation, train-only
preprocessing, replay, metrics, explainability. Synthetic data with known properties only."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from datamedic.domains.bootstrap import build_registry
from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.ml import explain
from datamedic.engine.ml import metrics as mm
from datamedic.engine.ml import spec as ml
from datamedic.engine.ml.models import build_estimator
from datamedic.engine.ml.replay import FoldPreprocessor, ReplayCommit, establish
from datamedic.engine.ml.runner import Cancelled, SplitSpec, TrainingSpec, run_training
from datamedic.engine.preprocess.commit import materialise
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.experiment.models import feature_kinds, model_matrix
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.split import make_splits
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.table_schema import schema_of
from datamedic.engine.preprocess.validate import validate_steps

ROLE_SPECS = {r.key: r for r in build_registry().all_roles()}


def frame(task: str, n: int = 240, seed: int = 0, signal: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    c = rng.choice(["x", "y", "z"], n)
    noise = rng.normal(scale=0.7, size=n)
    if task == "binary":
        y = (signal * a + noise > 0).astype(int)
    elif task == "multiclass":
        y = np.digitize(signal * a + 0.5 * noise, [-0.5, 0.5])
    else:
        y = 2 * signal * a - b + noise
    df = pd.DataFrame({ROW_ID: np.arange(n), "a": a, "b": b, "c": c, "y": y})
    df.loc[rng.random(n) < 0.1, "b"] = np.nan
    return df


def ctx_for(
    df: pd.DataFrame, task: str, roles: dict[str, list[str]] | None = None
) -> ValidationContext:
    return ValidationContext.from_schema(
        schema_of(df, {}), roles or {"target": ["y"]}, ROLE_SPECS, n_rows=len(df), task_type=task
    )


def setup(
    df: pd.DataFrame,
    task: str,
    *,
    steps: list[Step] | None = None,
    models: tuple[str, ...] = ("dummy", "linear"),
    mode: str = "defaults",
    method: str | None = None,
    group: str | None = None,
    time: str | None = None,
    n_splits: int = 4,
    seed: int = 0,
    roles: dict[str, list[str]] | None = None,
    **kw: Any,
) -> tuple[Any, Any, TrainingSpec]:
    method = method or ("kfold" if task == "regression" else "stratified")
    roles = roles or {"target": ["y"]}
    model = validate_steps(steps or [], ctx_for(df, task, roles))
    assert model.valid, model.errors
    feats = [c for c in df.columns if c not in (ROW_ID, "y", group, time)]
    strat = "y" if method in ("stratified", "stratified_group") else None
    splits = make_splits(
        df,
        method=method,
        n_splits=n_splits,
        seed=seed,
        target=strat,
        group=group,
        time=time,
        duplicate_columns=feats + ["y"],
    )
    spec = TrainingSpec(
        task,
        "y",
        models,
        mode,
        seed,
        SplitSpec(method, strat, group, time, tuple(feats + ["y"])),
        bootstrap_resamples=kw.pop("bootstrap_resamples", 100),
        **kw,
    )
    return splits, FoldPreprocessor([], model, seed), spec


# ---- Catalogue and estimators (scikit-learn contract) -------------------------------------------


def test_grids_are_small_ordered_and_contain_the_default() -> None:
    for m in ml.MODELS.values():
        for task in ml.TASKS:
            grid = m.grid(task)
            assert 1 <= len(grid) <= 4
            assert m.default(task) in grid


def test_fit_estimate() -> None:
    # 5 outer folds; linear 4 settings x 3 inner + 1 refit; dummy 1 refit.
    assert ml.estimate_fits(5, ["dummy", "linear"], "binary", "nested") == 5 * (1 + 13)
    assert ml.estimate_fits(5, list(ml.MODEL_KEYS), "binary", "defaults") == 20


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_every_grid_setting_builds_and_fits(task: str) -> None:
    df = frame(task, n=90)
    kinds = feature_kinds(df, ["a", "b", "c"])
    x = model_matrix(df, kinds)
    y = df["y"].to_numpy()
    for key, m in ml.MODELS.items():
        for params in m.grid(task):
            est = build_estimator(key, task, kinds, params, seed=0)
            est.fit(x, y)
            model = est.named_steps["model"]
            for name, value in params.items():  # the parameter really reached the estimator
                assert model.get_params()[name] == value
            assert clone(est).get_params()  # a valid scikit-learn estimator
            if key == "hgb":
                assert model.get_params()["early_stopping"] is False
            if key == "random_forest":
                assert model.get_params()["n_jobs"] == 1


# ---- Training: tasks, baselines, determinism ----------------------------------------------------


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_training_tasks_and_identical_test_rows(task: str) -> None:
    df = frame(task)
    splits, pre, spec = setup(df, task, models=("dummy", "linear", "random_forest", "hgb"))
    out = run_training(df, splits, pre, spec)
    primary = ml.PRIMARY[task]
    scores = {m: r["metrics"][primary]["mean"] for m, r in out.models.items()}
    assert all(r["status"] == "completed" and r["comparable"] for r in out.models.values())
    assert max(v for m, v in scores.items() if m != "dummy") > scores["dummy"] + 0.2
    # Every model predicted exactly the same rows, each once.
    oof = out.oof
    for m in spec.models:
        ids = np.sort(oof[oof["model"] == m][ROW_ID].to_numpy())
        assert np.array_equal(ids, np.sort(df[ROW_ID].to_numpy()))
    assert out.summary["coverage"]["complete"] is True
    assert all(
        b["fit_rows_equal_training_rows"] and b["test_rows_complete"]
        for b in out.evidence["boundaries"]
    )


def test_null_signal_scores_near_chance_and_dummy_is_competitive() -> None:
    # Seeds vary: with a constant-score dummy, a noise model can beat 0.5 in every fold by
    # chance (about 1 in 16 for 4 folds), so the check spans several datasets.
    aucs, verdicts = [], []
    for seed in (1, 2, 3):
        df = frame("binary", n=300, signal=0.0, seed=seed)
        out = run_training(df, *setup(df, "binary"))
        aucs.append(out.models["linear"]["metrics"]["roc_auc"]["mean"])
        verdicts += [c["verdict"] for c in out.summary["comparisons"] if c["metric"] == "roc_auc"]
    assert all(0.35 <= a <= 0.65 for a in aucs) and 0.4 <= float(np.mean(aucs)) <= 0.6
    assert set(verdicts) & {"negligible", "within_fold_variation", "mixed"}


def test_training_is_deterministic() -> None:
    df = frame("binary", n=220)
    a = run_training(df, *setup(df, "binary", mode="nested"))
    b = run_training(df, *setup(df, "binary", mode="nested"))
    assert a.models["linear"]["metrics"] == b.models["linear"]["metrics"]
    assert a.models["linear"]["selection"] == b.models["linear"]["selection"]
    pd.testing.assert_frame_equal(a.oof, b.oof)


# ---- Nested selection ---------------------------------------------------------------------------


def test_nested_selection_uses_outer_training_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    df = frame("binary", n=260)
    splits, pre, spec = setup(df, "binary", mode="nested")
    seen: list[set[int]] = []
    original = FoldPreprocessor.prepare

    def spy(self: FoldPreprocessor, train: pd.DataFrame, test: pd.DataFrame) -> Any:
        seen.append(set(train[ROW_ID].tolist()) | set(test[ROW_ID].tolist()))
        return original(self, train, test)

    monkeypatch.setattr(FoldPreprocessor, "prepare", spy)
    out = run_training(df, splits, pre, spec)
    # Per outer fold: 3 inner prepares (all inside the outer training rows), then the outer one.
    per_fold = 4
    for k, fold in enumerate(splits.folds):
        outer_train = set(fold.train_ids.tolist())
        for rows in seen[k * per_fold : k * per_fold + 3]:
            assert rows <= outer_train and not rows & set(fold.test_ids.tolist())
    records = out.summary["tuning"]["per_fold"]
    assert all(r["mode"] == "nested" and r["inner_test_ids_within_outer_training"] for r in records)
    freq = out.models["linear"]["selection"]["frequency"]
    assert sum(f["folds"] for f in freq) == len(splits.folds)


def test_small_cohort_falls_back_to_defaults_with_reason() -> None:
    df = frame("binary", n=120)
    out = run_training(df, *setup(df, "binary", mode="nested"))
    tuning = out.summary["tuning"]
    assert tuning["used"] == "defaults" and "fewer than" in tuning["fallback_reason"]
    assert out.models["linear"]["selection"]["per_fold"] == [{"C": 1.0}] * 4


def test_rare_class_in_inner_folds_falls_back_per_fold_keeping_groups() -> None:
    df = frame("binary", n=260)
    df.loc[:, "y"] = 0
    df.loc[df.index[:9], "y"] = 1  # 9 positives: inner folds get fewer than 3
    splits, pre, spec = setup(df, "binary", mode="nested", n_splits=3)
    out = run_training(df, splits, pre, spec)
    records = out.summary["tuning"]["per_fold"]
    assert all(r["mode"] == "defaults" and "fewer than" in r["reason"] for r in records)


def test_group_and_time_constraints_hold_in_inner_and_outer_folds() -> None:
    df = frame("binary", n=300)
    df["patient"] = np.repeat(np.arange(100), 3)
    splits, pre, spec = setup(
        df,
        "binary",
        mode="nested",
        method="stratified_group",
        group="patient",
        roles={"target": ["y"], "group": ["patient"]},
    )
    pid = df.set_index(ROW_ID)["patient"]
    for f in splits.folds:
        assert not set(pid[f.train_ids]) & set(pid[f.test_ids])
    out = run_training(df, splits, pre, spec)
    assert all(r["mode"] == "nested" for r in out.summary["tuning"]["per_fold"])
    # Inner folds are built by the same group-aware splitter (checked directly).
    train = df[df[ROW_ID].isin(splits.folds[0].train_ids)]
    inner = make_splits(
        train, method="stratified_group", n_splits=3, seed=1000, target="y", group="patient"
    )
    for f in inner.folds:
        assert not set(pid[f.train_ids]) & set(pid[f.test_ids])

    tdf = frame("regression", n=240)
    tdf["t"] = np.arange(240)
    splits, pre, spec = setup(
        tdf,
        "regression",
        mode="nested",
        method="time",
        time="t",
        roles={"target": ["y"], "time": ["t"]},
    )
    out = run_training(tdf, splits, pre, spec)
    assert (
        out.summary["coverage"]["complete"] is False
        and out.summary["coverage"]["untested_rows"] > 0
    )
    t = tdf.set_index(ROW_ID)["t"]
    for f in splits.folds:
        assert t[f.train_ids].max() < t[f.test_ids].min()
    assert out.models["linear"]["bootstrap"]["available"] is False  # temporal dependence


# ---- Train-only preprocessing, selection and resampling -----------------------------------------


def test_preprocessing_selection_and_resampling_fit_on_training_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = frame("binary", n=240)
    df.loc[:, "y"] = (df["a"] > 0.8).astype(int)  # imbalanced
    steps = [
        Step("impute.median", ("b",), {}),
        Step("select.univariate", ("a", "b"), {"test": "anova_f"}),
        Step("imbalance.random_oversample", (), {}),
    ]
    splits, pre, spec = setup(df, "binary", steps=steps)
    fitted: list[set[int]] = []
    original = PreprocessExecutor.fit

    def spy(self: PreprocessExecutor, frame_: pd.DataFrame) -> Any:
        fitted.append(set(frame_[ROW_ID].tolist()))
        return original(self, frame_)

    monkeypatch.setattr(PreprocessExecutor, "fit", spy)
    out = run_training(df, splits, pre, spec)
    for fold, rows in zip(splits.folds, fitted, strict=True):
        assert rows == set(fold.train_ids.tolist())
    for b in out.evidence["boundaries"]:
        assert b["test_rows_complete"] and not b["synthetic_rows_in_test"]
        assert b["training_rows_after_preprocessing"] > b["n_train"]  # oversampled training only
    assert out.models["linear"]["status"] == "completed"


def test_target_and_roles_are_never_model_inputs() -> None:
    df = frame("binary", n=200)
    df["pid"] = np.arange(200)
    splits, pre, spec = setup(df, "binary", roles={"target": ["y"], "id": ["pid"]})
    out = run_training(df, splits, pre, spec)
    cols = {r["column"] for r in out.models["linear"]["importance"]["permutation"]}
    assert cols <= {"a", "b", "c"} and "y" not in cols and "pid" not in cols


# ---- Failures, cancellation, metrics ------------------------------------------------------------


def test_model_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    import datamedic.engine.ml.runner as runner

    real = runner.build_estimator

    def broken(model: str, *a: Any, **kw: Any) -> Any:
        if model == "linear":
            raise RuntimeError("boom 42")
        return real(model, *a, **kw)

    monkeypatch.setattr(runner, "build_estimator", broken)
    df = frame("binary")
    out = run_training(df, *setup(df, "binary"))
    lin = out.models["linear"]
    assert lin["status"] == "failed" and not lin["comparable"]
    assert "42" not in str(lin["failures"])  # no exception text (could contain values)
    assert out.models["dummy"]["status"] == "completed"
    assert not [c for c in out.summary["comparisons"] if c["model"] == "linear"]


def test_cancellation_stops_before_results() -> None:
    df = frame("binary")
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        run_training(df, *setup(df, "binary"), should_cancel=cancel)


def test_undefined_metrics_are_none_with_reason_never_zero() -> None:
    y = np.array([0, 0, 0, 0])
    m, reasons, _ = mm.classification(y, y, np.column_stack([np.ones(4), np.zeros(4)]), 2)
    assert m["roc_auc"] is None and m["balanced_accuracy"] is None and reasons["roc_auc"]
    r, rr = mm.regression(np.ones(5), np.ones(5))
    assert r["r2"] is None and rr["r2"]
    assert mm.bootstrap("binary", y, y, None, 2, None, 10, 0)["available"] is False


def test_group_bootstrap_resamples_whole_groups() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.2 + rng.random(200) * 0.8, 0, 1)
    proba = np.column_stack([1 - p, p])
    blocks = np.repeat(np.arange(40), 5)
    out = mm.bootstrap("binary", y, (p > 0.5).astype(int), proba, 2, blocks, 200, 0)
    assert out["available"] and out["groups"] == 40
    lo, hi = out["metrics"]["roc_auc"]["low"], out["metrics"]["roc_auc"]["high"]
    assert lo < hi
    assert mm.bootstrap("binary", y, y, proba, 2, np.zeros(200), 50, 0)["available"] is False


def test_curves_calibration_and_residuals_are_bounded_aggregates() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    p = np.clip(0.3 * y + rng.random(5000) * 0.7, 0, 1)
    proba = np.column_stack([1 - p, p])
    c = mm.curves(y, proba, 2)
    assert c["available"] and len(c["roc"]) <= mm.CURVE_POINTS and len(c["pr"]) <= mm.CURVE_POINTS
    cal = mm.calibration(y, proba, 2)
    assert cal["available"] and sum(b["count"] for b in cal["bins"]) == 5000
    r = mm.residuals(rng.normal(size=300), rng.normal(size=300), redact_axes=True)
    assert "value_range" not in r and sum(map(sum, r["actual_vs_predicted"])) == 300


# ---- Explainability -----------------------------------------------------------------------------


def test_permutation_importance_ranks_the_informative_feature_and_maps_derived_columns() -> None:
    df = frame("binary", n=300)
    df["noise"] = np.random.default_rng(5).normal(size=300)
    steps = [Step("missing.indicator", ("b",), {}), Step("impute.median", ("b",), {})]
    out = run_training(df, *setup(df, "binary", steps=steps, models=("dummy", "hgb")))
    ranking = out.models["hgb"]["importance"]["permutation"]
    assert ranking[0]["column"] == "a"
    noise = next(r for r in ranking if r["column"] == "noise")
    assert abs(noise["mean"]) < 0.05
    names = {r["column"] for r in ranking}
    assert "b__missing" not in names and "b" in names  # indicator attributed to its source
    assert out.models["hgb"]["importance"]["coefficients"] is None
    assert out.models["dummy"]["importance"]["permutation"] == []


def test_origin_map_refuses_to_guess() -> None:
    lineage = [{"column": "ab", "origin": "added", "derived_from": ["a", "b"]}]
    origins = explain.origin_map(lineage, ["a", "b"])
    assert origins["ab"] is None
    assert "unresolved:ab" in explain.group_inputs(["a", "ab"], origins)


def test_error_slices_suppress_small_cells_and_sensitive_columns() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "s": rng.choice(["p", "q"], 200),
            "c": ["rare"] * 5 + ["common"] * 195,
        }
    )
    errors = rng.random(200)
    out = explain.error_slices(data, errors, ["a", "s", "c"], frozenset({"s"}), "mae")
    assert [o["column"] for o in out] == ["a", "c"]
    labels = {sl["label"] for sl in out[1]["slices"]}
    assert "rare" not in labels and all(
        sl["n"] >= explain.MIN_SLICE_ROWS for o in out for sl in o["slices"]
    )


# ---- Replay -------------------------------------------------------------------------------------


def test_replay_fits_inside_folds_and_proves_fidelity() -> None:
    df = frame("binary", n=200)
    ctx = ctx_for(df, "binary")
    data_steps = validate_steps([Step("impute.median", ("b",), {})], ctx)
    v1 = materialise(df, data_steps)  # what a commit stores (fitted on all rows)
    commit = ReplayCommit(data_steps, 0)
    proof = establish(df, [commit], v1.content_hash, v1.frame)
    assert proof["status"] == "exact"
    with pytest.raises(EngineError) as exc:
        establish(df, [commit], b"\x00" * 32, v1.frame.assign(b=0.0))
    assert exc.value.code == "REPLAY_NOT_ESTABLISHED"

    empty = validate_steps([], ctx_for(v1.frame.drop(columns=[]), "binary"))
    pre = FoldPreprocessor([commit], empty, 0)
    train, test = df.iloc[:100], df.iloc[100:]
    prepared = pre.prepare(train, test)
    filled = prepared.held_out.frame.set_index(ROW_ID)["b"]
    missing_test = test[test["b"].isna()][ROW_ID]
    assert len(missing_test) > 0
    assert np.allclose(filled[missing_test], train["b"].median())  # training median only
    assert not np.isclose(train["b"].median(), df["b"].median())
    assert all(n == len(train) for n in prepared.fit_rows)
