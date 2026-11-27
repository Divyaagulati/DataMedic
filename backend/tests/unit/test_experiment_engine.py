"""Experiment Lab engine (Phase 4 Part D): identical frozen folds, training-only fitting,
resampling isolation, negative controls, metrics, failure isolation, selection stability,
reproducibility and cancellation. Synthetic data only."""

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import executor as ex_mod
from datamedic.engine.preprocess.executor import MODEL_STAGE_NOT_COMMITTABLE, PreprocessExecutor
from datamedic.engine.preprocess.experiment import (
    Cancelled,
    Candidate,
    ExperimentSpec,
    run_experiment,
)
from datamedic.engine.preprocess.experiment import folds as fold_store
from datamedic.engine.preprocess.experiment import metrics as m
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.split import make_splits
from tests.unit.preprocess_helpers import frame, validated

ROLES = {"target": ["y"]}
MODELS = ("dummy", "linear", "hgb")


def data(n: int = 240, seed: int = 0, minority: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < minority).astype(int)
    a = rng.normal(size=n) + 1.2 * y
    df = frame(
        a=a,
        b=rng.normal(size=n),
        skew=np.exp(rng.normal(1, 0.7, n)),
        c=pd.Series(rng.choice(["u", "v", "w"], n), dtype=object),
        y=y,
    )
    df.loc[rng.random(n) < 0.15, "b"] = np.nan
    return df


def candidates(df: pd.DataFrame, extra: dict[str, list[Step]] | None = None) -> list[Candidate]:
    specs = {
        "baseline": [],
        "median_std": [Step("impute.median", ("b",)), Step("scale.standard", ("a", "b"))],
        "oversample": [Step("impute.median", ("b",)), Step("imbalance.random_oversample", ())],
        "weights": [Step("impute.median", ("b",)), Step("imbalance.class_weight", ())],
        **(extra or {}),
    }
    return [
        Candidate(k, k, validated(df, steps, ROLES, task="binary")) for k, steps in specs.items()
    ]


def splits_for(df: pd.DataFrame, seed: int = 0, n: int = 4) -> Any:
    return make_splits(df, method="stratified", n_splits=n, seed=seed, target="y")


def run(df: pd.DataFrame, cands: list[Candidate], splits: Any = None, **kw: Any) -> dict:  # type: ignore[type-arg]
    return run_experiment(
        df,
        splits or splits_for(df),
        cands,
        ExperimentSpec("binary", "y", kw.pop("models", MODELS), seed=0),
        **kw,
    )


def strip_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_timing(v) for k, v in value.items() if not k.endswith("_ms") and k != "fit_ms"
        }
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


class Spy:
    """Records, per executor call, the row ids fitted on and transformed, and learned state."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fits: list[dict[str, Any]] = []
        self.transforms: list[np.ndarray] = []
        original_fit_resample = PreprocessExecutor.fit_resample
        original_transform = PreprocessExecutor.transform
        spy = self

        def fit_resample(self: PreprocessExecutor, f: pd.DataFrame) -> Any:
            out = original_fit_resample(self, f)
            spy.fits.append(
                {
                    "ids": np.sort(f[ROW_ID].to_numpy()),
                    "learned": copy.deepcopy([t.learned_params() for t in self.transformers()]),
                    "training_ids": np.sort(out.frame[ROW_ID].to_numpy()),
                }
            )
            return out

        def transform(self: PreprocessExecutor, f: pd.DataFrame) -> Any:
            spy.transforms.append(np.sort(f[ROW_ID].to_numpy()))
            return original_transform(self, f)

        monkeypatch.setattr(PreprocessExecutor, "fit_resample", fit_resample)
        monkeypatch.setattr(PreprocessExecutor, "transform", transform)


# ---- Frozen folds -----------------------------------------------------------------------------


def test_every_candidate_uses_identical_folds_and_scores_identical_test_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = data()
    splits = splits_for(df)
    spy = Spy(monkeypatch)
    out = run(df, candidates(df), splits)
    n_folds = len(splits.folds)
    per_candidate = [spy.fits[i : i + n_folds] for i in range(0, len(spy.fits), n_folds)]
    assert len(per_candidate) == 4
    for fits in per_candidate:
        for fold, fit in zip(splits.folds, fits, strict=True):
            np.testing.assert_array_equal(fit["ids"], fold.train_ids)
    tested = [t for t in spy.transforms if len(t) < len(df) * 0.5]  # held-out transforms
    for i, fold in enumerate(splits.folds):
        for c in range(4):
            np.testing.assert_array_equal(tested[c * n_folds + i], fold.test_ids)
    sizes = {
        tuple(f["test_rows"] for f in [fold["quality"] for fold in cand["folds"]])
        for cand in out["candidates"]
    }
    assert len(sizes) == 1  # every candidate scored the same number of test rows per fold
    assert all(c["comparable"] for c in out["candidates"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "kfold"},
        {"method": "stratified", "target": "y"},
        {"method": "group", "group": "g"},
        {"method": "stratified_group", "target": "y", "group": "g"},
        {"method": "time", "time": "t"},
    ],
)
def test_stored_fold_assignment_reproduces_the_folds_exactly(kwargs: dict) -> None:  # type: ignore[type-arg]
    df = data(120).assign(g=np.arange(120) // 3, t=np.arange(120) // 2)
    splits = make_splits(df, n_splits=4, seed=3, **kwargs)
    payload = fold_store.encode(splits)
    rebuilt = fold_store.decode(payload)
    for a, b in zip(rebuilt.folds, splits.folds, strict=True):
        np.testing.assert_array_equal(a.train_ids, b.train_ids)
        np.testing.assert_array_equal(a.test_ids, b.test_ids)
    again = fold_store.encode(make_splits(df, n_splits=4, seed=3, **kwargs))
    assert fold_store.sha256(again) == fold_store.sha256(payload)


# ---- Leakage: negative controls ---------------------------------------------------------------


def _fold_training_view(out: dict, fold: int) -> list[Any]:  # type: ignore[type-arg]
    return [
        {
            "warnings": c["folds"][fold]["step_warnings"],
            "rows": (
                c["folds"][fold]["quality"]["training_rows_before"],
                c["folds"][fold]["quality"]["training_rows_after"],
            ),
        }
        for c in out["candidates"]
    ]


@pytest.mark.parametrize("change", ["values", "labels"])
def test_changing_held_out_rows_changes_no_fitted_parameter_or_decision(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: rewrite the values (or labels) of the test rows of fold 0 only.
    Everything learned in fold 0 (imputation medians, scales, resampled training rows,
    selection decisions) must be identical, because fold 0 never fits on those rows."""
    df = data()
    splits = splits_for(df)
    extra = {
        "select": [
            Step("impute.median", ("b",)),
            Step("select.univariate", ("a", "b", "skew"), {"test": "anova_f", "k": 2}),
        ]
    }
    test0 = df[ROW_ID].isin(splits.folds[0].test_ids)
    changed = df.copy()
    if change == "values":
        changed.loc[test0, "a"] = 1e6
        changed.loc[test0, "b"] = -1e6
        changed.loc[test0, "c"] = "x"
    else:
        changed.loc[test0, "y"] = 1 - changed.loc[test0, "y"]
    results = []
    for frame_ in (df, changed):
        spy = Spy(monkeypatch)
        out = run(frame_, candidates(frame_, extra), splits, models=("linear",))
        monkeypatch.undo()
        results.append((spy, out))
    (spy_a, out_a), (spy_b, out_b) = results
    n = len(splits.folds)
    fold0_a = [f for i, f in enumerate(spy_a.fits) if i % n == 0]
    fold0_b = [f for i, f in enumerate(spy_b.fits) if i % n == 0]
    for a, b in zip(fold0_a, fold0_b, strict=True):
        assert a["learned"] == b["learned"]
        np.testing.assert_array_equal(a["training_ids"], b["training_ids"])
    assert _fold_training_view(out_a, 0) == _fold_training_view(out_b, 0)
    sel_a = out_a["candidates"][-1]["selection_stability"][0]["per_fold"][0]
    sel_b = out_b["candidates"][-1]["selection_stability"][0]["per_fold"][0]
    assert sel_a == sel_b


def test_resampled_rows_never_reach_held_out_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    df = data()
    splits = splits_for(df)
    spy = Spy(monkeypatch)
    out = run(df, candidates(df), splits)
    oversample = next(c for c in out["candidates"] if c["key"] == "oversample")
    for fold, f in zip(splits.folds, oversample["folds"], strict=True):
        q = f["quality"]
        assert q["training_rows_after"] > q["training_rows_before"] == len(fold.train_ids)
        assert q["test_rows"] == len(fold.test_ids)
    assert all((t >= 0).all() for t in spy.transforms)  # no synthetic id is ever transformed
    # Repeated minority rows get new negative ids (Part C): they exist only in training tables.
    added = [f for f in spy.fits if (f["training_ids"] < 0).any()]
    assert len(added) == len(splits.folds)
    assert "ROWSET_DIFFERS" in {w["code"] for w in out["warnings"]}


def test_smote_rows_exist_only_in_training_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    df = data().drop(columns=["c"])
    cands = [
        Candidate("baseline", "b", validated(df, [], ROLES, task="binary")),
        Candidate(
            "smote",
            "s",
            validated(
                df,
                [Step("impute.median", ("b",)), Step("imbalance.smote", ())],
                ROLES,
                task="binary",
            ),
        ),
    ]
    spy = Spy(monkeypatch)
    out = run(df, cands, models=("linear",))
    synthetic = [f for f in spy.fits if (f["training_ids"] < 0).any()]
    assert synthetic  # SMOTE added rows to every training table
    assert all((t >= 0).all() for t in spy.transforms)
    assert out["candidates"][1]["comparable"]


def test_input_frame_is_never_modified() -> None:
    df = data()
    before = df.copy(deep=True)
    run(df, candidates(df), models=("linear",))
    pd.testing.assert_frame_equal(df, before)


# ---- Failures ---------------------------------------------------------------------------------


def test_a_failing_candidate_is_isolated_and_marked_not_comparable() -> None:
    df = data()
    splits = splits_for(df)
    bad = {"bad_log": [Step("transform.log", ("a",))]}  # 'a' has negative values
    out = run(df, candidates(df, bad), splits, models=("linear",))
    alone = run(df, candidates(df), splits, models=("linear",))
    failed = next(c for c in out["candidates"] if c["key"] == "bad_log")
    assert failed["status"] == "failed" and not failed["comparable"]
    assert failed["comparability_reason"]
    assert {f["code"] for f in failed["failures"]} == {"VALUE_OUT_OF_DOMAIN"}
    assert all("message" in f and "e+" not in f["message"] for f in failed["failures"])
    for a, b in zip(out["candidates"][:4], alone["candidates"], strict=True):
        assert strip_timing(a["metrics"]) == strip_timing(b["metrics"])
    assert not any(c["candidate"] == "bad_log" for c in out["comparisons"])
    assert "NOT_DIRECTLY_COMPARABLE" in {w["code"] for w in out["warnings"]}


def test_a_failing_model_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from datamedic.engine.preprocess.experiment import runner

    original = runner.build_model

    def flaky(name: str, *a: Any, **k: Any) -> Any:
        if name == "hgb":
            raise ValueError("secret 1234")
        return original(name, *a, **k)

    monkeypatch.setattr(runner, "build_model", flaky)
    df = data()
    with pytest.raises(ValueError):
        run(df, candidates(df)[:1])  # build errors outside the guarded fit surface
    monkeypatch.setattr(runner, "build_model", original)

    class Boom:
        def fit(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("value 98765")

    def boom(name: str, *a: Any, **k: Any) -> Any:
        return Boom() if name == "hgb" else original(name, *a, **k)

    monkeypatch.setattr(runner, "build_model", boom)
    out = run(df, candidates(df)[:2])
    for cand in out["candidates"]:
        assert cand["status"] == "partial"
        assert cand["metrics"]["linear"]["roc_auc"]["n_folds"] == 4
        assert cand["metrics"]["hgb"]["roc_auc"]["n_folds"] == 0
        fail = [f for f in cand["failures"] if f.get("model") == "hgb"]
        assert len(fail) == 4 and all("98765" not in str(f) for f in fail)


# ---- Metrics ----------------------------------------------------------------------------------


def test_metrics_equal_a_direct_scikit_learn_computation() -> None:
    df = data()
    splits = splits_for(df)
    out = run(df, candidates(df)[:1], splits, models=("linear",))
    fold = splits.folds[0]
    train, test = splits.frames(df, fold)
    cols = ["a", "b", "skew"]
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    (
                                        "i",
                                        SimpleImputer(strategy="median", keep_empty_features=True),
                                    ),
                                    ("s", StandardScaler()),
                                ]
                            ),
                            cols,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("i", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "o",
                                        __import__(
                                            "sklearn.preprocessing", fromlist=["x"]
                                        ).OneHotEncoder(
                                            handle_unknown="infrequent_if_exist",
                                            max_categories=20,
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            ["c"],
                        ),
                    ]
                ),
            ),
            ("m", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    x = lambda f: f[cols + ["c"]].assign(c=f["c"].astype(object))  # noqa: E731
    model.fit(x(train), train["y"])
    proba = model.predict_proba(x(test))[:, 1]
    pred = model.predict(x(test))
    got = out["candidates"][0]["metrics"]["linear"]
    assert got["roc_auc"]["values"][0] == pytest.approx(
        skm.roc_auc_score(test["y"], proba), abs=1e-6
    )
    assert got["accuracy"]["values"][0] == pytest.approx(
        skm.accuracy_score(test["y"], pred), abs=1e-6
    )
    assert got["f1"]["values"][0] == pytest.approx(skm.f1_score(test["y"], pred), abs=1e-6)
    assert got["brier"]["values"][0] == pytest.approx(
        skm.brier_score_loss(test["y"], proba), abs=1e-6
    )
    values = [v for v in got["roc_auc"]["values"]]
    assert got["roc_auc"]["mean"] == pytest.approx(np.mean(values), abs=1e-6)
    assert got["roc_auc"]["sd"] == pytest.approx(np.std(values, ddof=1), abs=1e-6)
    confusion = np.array(out["candidates"][0]["confusion_matrix"]["linear"])
    assert confusion.sum() == len(df)


def test_metric_functions_on_hand_checked_cases() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
    got, reasons, details = m.classification(y, np.array([0, 0, 1, 1]), p, [0, 1])
    assert got["accuracy"] == 1.0 and got["roc_auc"] == 1.0 and got["recall"] == 1.0
    assert details["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m.expected_calibration_error(np.array([1, 0]), np.array([0.5, 0.5])) == 0.0
    one_class, reasons, _ = m.classification(
        np.array([0, 0]), np.array([0, 0]), np.array([[0.8, 0.2], [0.7, 0.3]]), [0, 1]
    )
    assert one_class["roc_auc"] is None and "class" in reasons["roc_auc"]
    reg, _ = m.regression(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert reg["mae"] == pytest.approx(1 / 3) and reg["rmse"] == pytest.approx(np.sqrt(1 / 3))
    flat, why = m.regression(np.array([2.0, 2.0]), np.array([1.0, 3.0]))
    assert flat["r2"] is None and why["r2"]


def test_class_weights_reach_the_model() -> None:
    """The per-row training weights of class weighting are passed to the model: minority
    recall rises compared with the unweighted baseline on imbalanced data."""
    df = data(400, minority=0.12)
    out = run(df, candidates(df), models=("linear",))
    recall = {c["key"]: c["metrics"]["linear"]["recall"]["mean"] for c in out["candidates"]}
    assert recall["weights"] > recall["median_std"]


def test_regression_and_clustering_tasks_run() -> None:
    df = data().assign(y=lambda f: f["a"] * 2 + f["skew"])
    splits = make_splits(df, method="kfold", n_splits=3, seed=0)
    cands = [
        Candidate("baseline", "b", validated(df, [], ROLES, task="regression")),
        Candidate(
            "log", "l", validated(df, [Step("transform.log", ("skew",))], ROLES, task="regression")
        ),
    ]
    out = run_experiment(df, splits, cands, ExperimentSpec("regression", "y", MODELS))
    assert out["primary_metric"] == "r2"
    assert out["candidates"][0]["metrics"]["linear"]["r2"]["mean"] > 0.5
    clust = run_experiment(
        df.drop(columns=["y"]),
        splits,
        [Candidate("baseline", "b", validated(df.drop(columns=["y"]), []))],
        ExperimentSpec("clustering", None, ("kmeans",), n_clusters=3),
    )
    sil = clust["candidates"][0]["metrics"]["kmeans"]["silhouette"]
    assert sil["n_folds"] == 3 and -1 <= sil["mean"] <= 1
    assert any("not accuracy" in text for text in clust["limitations"])


# ---- Comparisons ------------------------------------------------------------------------------


def test_comparisons_describe_fold_differences_without_declaring_a_winner() -> None:
    df = data()
    copy_of_baseline = Candidate("same", "same", validated(df, [], ROLES, task="binary"))
    out = run(df, [candidates(df)[0], copy_of_baseline], models=("linear",))
    auc = next(c for c in out["comparisons"] if c["metric"] == "roc_auc")
    assert auc["per_fold_difference"] == [0.0] * 4 and auc["verdict"] == "negligible"
    assert "not a significance test" in auc["interpretation"]
    text = str(out).lower()
    assert "winner" not in text and "best_pipeline" not in text
    assert any("mean score alone" in lim for lim in out["limitations"])


def test_multiple_comparisons_and_dummy_warnings() -> None:
    rng = np.random.default_rng(1)
    df = frame(a=rng.normal(size=200), b=rng.normal(size=200), y=rng.integers(0, 2, 200))
    specs = [
        [],
        [Step("scale.standard", ("a",))],
        [Step("scale.robust", ("a",))],
        [Step("scale.minmax", ("a",))],
    ]
    cands = [
        Candidate(f"c{i}", f"c{i}", validated(df, s, ROLES, task="binary"))
        for i, s in enumerate(specs)
    ]
    out = run(df, cands)
    codes = {w["code"] for w in out["warnings"]}
    assert {"MULTIPLE_COMPARISONS", "DUMMY_COMPETITIVE"} <= codes


# ---- Feature-selection stability --------------------------------------------------------------


def test_selection_stability_frequency_overlap_and_noise_probes() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    cols = {
        f"f{i}": rng.normal(size=n) + (2.0 if i == 0 else 0.1 if i < 4 else 0) * y
        for i in range(10)
    }
    df = frame(**cols, y=y)
    step = Step("select.univariate", tuple(cols), {"test": "anova_f", "k": 3, "action": "drop"})
    cands = [
        Candidate("baseline", "b", validated(df, [], ROLES, task="binary")),
        Candidate("sel", "s", validated(df, [step], ROLES, task="binary")),
    ]
    out = run(df, cands, splits_for(df, n=5), models=("linear",))
    stab = out["candidates"][1]["selection_stability"][0]
    assert stab["frequency"]["f0"] == 1.0 and "f0" in stab["stable_selected"]
    assert len(stab["per_fold"]) == 5 and all(len(f["selected"]) == 3 for f in stab["per_fold"])
    assert stab["mean_jaccard"] < 1.0  # weak and noise columns change between folds
    assert stab["unstable"]
    assert stab["probe_selection_rate"] is not None
    assert any("not biomarker validation" in text for text in stab["limitations"])
    assert out["candidates"][0]["selection_stability"] == []


def test_noise_selected_as_often_as_real_columns_is_flagged() -> None:
    rng = np.random.default_rng(2)
    n = 200
    df = frame(**{f"n{i}": rng.normal(size=n) for i in range(6)}, y=rng.integers(0, 2, n))
    step = Step("select.univariate", tuple(f"n{i}" for i in range(6)), {"test": "anova_f", "k": 3})
    cands = [
        Candidate("baseline", "b", validated(df, [], ROLES, task="binary")),
        Candidate("sel", "s", validated(df, [step], ROLES, task="binary")),
    ]
    stab = run(df, cands, splits_for(df, n=5), models=("dummy",))["candidates"][1][
        "selection_stability"
    ][0]
    assert stab["probe_selection_rate"] > 0
    assert "NOISE_SELECTED" in {w["code"] for w in stab["warnings"]}
    assert stab["indistinguishable_from_noise"]


# ---- Reproducibility and cancellation ---------------------------------------------------------


def test_results_are_reproducible_from_the_same_configuration() -> None:
    df = data()
    a = run(df, candidates(df), splits_for(df))
    b = run(df.copy(), candidates(df), splits_for(df))
    assert strip_timing(a) == strip_timing(b)


def test_cancellation_stops_before_the_next_fold() -> None:
    df = data()
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    progress: list[tuple[int, int]] = []
    with pytest.raises(Cancelled):
        run(
            df,
            candidates(df),
            on_progress=lambda d, t, _: progress.append((d, t)),
            should_cancel=cancel,
        )
    assert progress == [(1, 16), (2, 16)]


# ---- Commit guard (closes the Part C enforcement gap) -----------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        [Step("scale.standard", ("a",))],
        [Step("impute.median", ("b",)), Step("encode.onehot", ("c",))],
        [Step("select.variance", ("a",), {"action": "drop"})],
        [Step("impute.median", ("b",)), Step("imbalance.random_oversample", ())],
        [Step("impute.median", ("b",)), Step("imbalance.class_weight", ())],
    ],
)
def test_model_stage_steps_can_never_be_committed(steps: list[Step]) -> None:
    df = data()
    executor = PreprocessExecutor(validated(df, steps, ROLES, task="binary"))
    with pytest.raises(EngineError) as exc:
        executor.commit_transform(df)
    assert exc.value.code == MODEL_STAGE_NOT_COMMITTABLE
    assert not executor._fitted  # refused before any data was read


def test_data_and_either_stage_steps_can_be_committed() -> None:
    df = data()
    steps = [
        Step("missing.indicator", ("b",)),
        Step("impute.median", ("b",)),
        Step("transform.log", ("skew",)),
    ]
    out = PreprocessExecutor(validated(df, steps, ROLES, task="binary")).commit_transform(df)
    assert len(out.frame) == len(df) and out.frame["b"].notna().all()
    assert ex_mod.MODEL_STAGE_NOT_COMMITTABLE == "MODEL_STAGE_NOT_COMMITTABLE"
