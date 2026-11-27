"""Feature selection (Part C): variance, correlation proposals, mutual information,
univariate tests and explicit removal. Covers binary, multiclass and regression targets,
categorical and high-cardinality columns, constant and missing values, small samples,
protected and domain-role columns, keep lists, lineage, determinism and limits."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, run, validated

ROLES = {"target": ["y"]}


def data(n: int = 300, seed: int = 0, classes: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    return frame(
        strong=y * 2.0 + rng.normal(0, 0.5, n),
        weak=y * 0.2 + rng.normal(0, 1, n),
        noise=rng.normal(0, 1, n),
        const=np.full(n, 7.0),
        cat=pd.Series(np.where(y == 1, "a", rng.choice(["b", "c"], n)), dtype=object),
        flag=pd.array(rng.random(n) < 0.5, dtype="boolean"),
        y=y,
    )


def errors(
    df: pd.DataFrame, steps: list[Step], roles: dict | None = None, task: str | None = "binary"
) -> list[str]:  # type: ignore[type-arg]
    result = validate_steps(steps, context(df, roles if roles is not None else ROLES, task=task))
    return [e.code for e in result.errors]


# ---- Variance -------------------------------------------------------------------------------------


def test_variance_default_reports_constant_columns_and_removes_nothing() -> None:
    df = data()
    out, _ = run(df, [Step("select.variance", ("strong", "const", "flag"))], roles=ROLES)
    report = out.reports[0]
    assert list(out.frame.columns) == list(df.columns)
    assert report.learned["decisions"] == {
        "strong": "kept",
        "const": "would_remove",
        "flag": "kept",
    }
    assert report.learned["columns"]["const"]["variance"] == 0.0
    flag = df["flag"].astype(float)
    assert report.learned["columns"]["flag"]["variance"] == pytest.approx(np.var(flag))


def test_variance_drop_removes_only_failing_columns_in_order() -> None:
    df = data()
    out, _ = run(
        df,
        [Step("select.variance", ("strong", "const", "noise"), {"action": "drop"})],
        roles=ROLES,
    )
    assert "const" not in out.frame.columns
    assert list(out.frame.columns) == [c for c in df.columns if c != "const"]
    assert out.reports[0].columns_removed == ["const"]


def test_variance_treats_float_noise_free_constants_as_zero() -> None:
    df = frame(x=np.full(37, 0.1), y=np.arange(37) % 2)
    out, _ = run(df, [Step("select.variance", ("x",), {"action": "drop"})], roles=ROLES)
    assert "x" not in out.frame.columns  # np.var would give ~1e-34 here


def test_variance_threshold_above_zero_warns_about_units() -> None:
    df = data()
    result = validate_steps(
        [Step("select.variance", ("strong", "noise"), {"threshold": 0.5})], context(df, ROLES)
    )
    assert "VARIANCE_SCALE_DEPENDENT" in [w.code for w in result.warnings]


def test_variance_uses_observed_values_and_refuses_an_empty_column() -> None:
    df = frame(x=[1.0, np.nan, 1.0, np.nan], z=[np.nan] * 4)
    out, _ = run(df, [Step("select.variance", ("x",))])
    assert out.reports[0].learned["columns"]["x"] == {"variance": 0.0, "n_values": 2}
    with pytest.raises(EngineError) as exc:
        run(df, [Step("select.variance", ("z",))])
    assert exc.value.code == "FIT_COLUMN_EMPTY"


# ---- Keep lists, domain roles and protected columns ----------------------------------------------


def test_keep_and_domain_role_columns_are_never_removed() -> None:
    df = data()
    df["bmi"] = 25.0  # constant biomarker
    roles = {"target": ["y"], "bio.bmi": ["bmi"]}
    steps = [
        Step("select.variance", ("const", "bmi", "noise"), {"action": "drop", "keep": ["const"]})
    ]
    result = validated(df, steps, roles)
    assert "DOMAIN_ROLE_KEPT" in [w.code for w in result.warnings]
    executor = PreprocessExecutor(result).fit(df)
    out = executor.transform(df)
    assert {"const", "bmi"} <= set(out.frame.columns)
    assert out.reports[0].learned["decisions"] == {
        "const": "kept_by_user",
        "bmi": "kept_domain_role",
        "noise": "kept",
    }


def test_keep_must_list_selected_columns() -> None:
    df = data()
    assert "PARAM_INVALID" in errors(
        df, [Step("select.variance", ("noise",), {"keep": ["strong"]})]
    )


@pytest.mark.parametrize("column", ["y", "pid", ROW_ID])
@pytest.mark.parametrize(
    "op", ["select.variance", "select.mutual_information", "select.drop_columns"]
)
def test_target_identifier_and_row_id_can_never_be_selected(op: str, column: str) -> None:
    df = data()
    df["pid"] = np.arange(len(df)).astype(float)
    codes = errors(df, [Step(op, (column,))], {"target": ["y"], "id": ["pid"]})
    assert codes and set(codes) <= {"COLUMN_PROTECTED", "RESERVED_COLUMN"}


def test_later_steps_cannot_use_a_column_selection_may_remove() -> None:
    df = data()
    steps = [
        Step("select.variance", ("noise", "const"), {"action": "drop", "keep": ["noise"]}),
        Step("scale.standard", ("const",)),
    ]
    assert errors(df, steps) == ["COLUMN_NOT_FOUND"]
    steps[1] = Step("scale.standard", ("noise",))  # kept columns stay available
    assert errors(df, steps) == []


# ---- Correlation ------------------------------------------------------------------------------------


def test_pairwise_correlation_matches_pandas_with_missing_values() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(5e3, 10, (500, 4))
    x[:, 1] = x[:, 0] * 3 + rng.normal(0, 1, 500)
    x[rng.random((500, 4)) < 0.2] = np.nan
    ours = rt.pairwise_correlation(x, 10, chunk_rows=37)
    theirs = pd.DataFrame(x).corr(min_periods=10).to_numpy()
    assert np.allclose(ours, theirs, atol=1e-10)


def test_correlation_proposes_but_never_removes() -> None:
    df = data()
    df["strong_copy"] = df["strong"] * 100 + 1
    out, _ = run(df, [Step("select.correlation", ("strong", "strong_copy", "noise"))], roles=ROLES)
    learned = out.reports[0].learned
    assert list(out.frame.columns) == list(df.columns)
    assert len(learned["pairs"]) == 1 and learned["pairs"][0]["correlation"] == pytest.approx(1.0)
    assert learned["proposed_removals"] in (["strong"], ["strong_copy"])
    assert "CORRELATED_PAIRS" in [w["code"] for w in out.reports[0].warnings]


def test_correlation_never_proposes_kept_or_domain_role_columns() -> None:
    df = data()
    df["copy"] = df["strong"] * 2
    steps = [Step("select.correlation", ("strong", "copy"), {"keep": ["copy"]})]
    out, _ = run(df, steps, roles=ROLES)
    assert out.reports[0].learned["proposed_removals"] == ["strong"]
    roles = {"target": ["y"], "bio.heart_rate": ["strong", "copy"]}
    out, _ = run(df, [Step("select.correlation", ("strong", "copy"))], roles=roles)
    assert out.reports[0].learned["proposed_removals"] == []
    assert out.reports[0].learned["pairs"][0]["proposed_removal"] is None


def test_correlation_spearman_detects_monotonic_relations() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    df = frame(a=x, b=np.exp(3 * x))
    pearson = run(df, [Step("select.correlation", ("a", "b"), {"threshold": 0.95})])[0]
    spearman = run(
        df, [Step("select.correlation", ("a", "b"), {"threshold": 0.95, "method": "spearman"})]
    )[0]
    assert pearson.reports[0].learned["pairs"] == []
    assert spearman.reports[0].learned["pairs"][0]["correlation"] == pytest.approx(1.0)


def test_correlation_with_constant_or_sparse_pairs_is_not_assessed() -> None:
    df = frame(a=np.arange(30.0), c=np.full(30, 2.0), s=[1.0] * 5 + [np.nan] * 25)
    out, _ = run(df, [Step("select.correlation", ("a", "c", "s"))])
    assert out.reports[0].learned["pairs_not_assessed"] == 3
    assert "PAIRS_NOT_ASSESSED" in [w["code"] for w in out.reports[0].warnings]


def test_correlation_needs_two_columns_and_respects_the_column_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = data()
    assert errors(df, [Step("select.correlation", ("noise",))]) == ["TOO_FEW_COLUMNS"]
    monkeypatch.setattr(Limits, "CORR_MAX_FIT_ROWS", 10)
    with pytest.raises(EngineError) as exc:
        run(df, [Step("select.correlation", ("noise", "weak"))], roles=ROLES)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


# ---- Mutual information ----------------------------------------------------------------------------


@pytest.mark.parametrize("classes", [2, 4])
def test_mutual_information_ranks_informative_columns_first(classes: int) -> None:
    df = data(classes=classes, seed=3)
    task = "binary" if classes == 2 else "multiclass"
    steps = [
        Step(
            "select.mutual_information",
            ("noise", "weak", "strong", "cat"),
            {"k": 2, "action": "drop"},
        )
    ]
    out, _ = run(df, steps, roles=ROLES, task=task, seed=5)
    columns = out.reports[0].learned["columns"]
    # 'cat' and 'strong' both depend on the class; 'noise' and 'weak' rank below them.
    assert {columns["strong"]["rank"], columns["cat"]["rank"]} == {1, 2}
    assert "noise" not in out.frame.columns and "strong" in out.frame.columns


def test_mutual_information_matches_scikit_learn_and_is_seeded() -> None:
    df = data(seed=4)
    steps = [Step("select.mutual_information", ("strong", "noise"))]
    a = run(df, steps, roles=ROLES, task="binary", seed=7)[0].reports[0].learned["columns"]
    b = run(df, steps, roles=ROLES, task="binary", seed=7)[0].reports[0].learned["columns"]
    assert a == b
    expected = mutual_info_classif(
        df[["strong"]].to_numpy(),
        df["y"].to_numpy(),
        discrete_features=[False],
        n_neighbors=3,
        random_state=7,
    )[0]
    assert a["strong"]["mutual_information"] == pytest.approx(max(expected, 0.0))


def test_mutual_information_regression_and_categorical_columns() -> None:
    rng = np.random.default_rng(5)
    n = 300
    group = rng.choice(["p", "q", "r"], n)
    target = np.where(group == "p", 5.0, 0.0) + rng.normal(0, 0.1, n)
    df = frame(g=pd.Series(group, dtype=object), noise=rng.normal(size=n), t=target)
    result = validated(
        df,
        [Step("select.mutual_information", ("g", "noise"))],
        {"target": ["t"]},
        task="regression",
    )
    assert result.steps[0].step.params["discrete"] == [True, False]
    out = PreprocessExecutor(result).fit(df).transform(df)
    assert out.reports[0].learned["columns"]["g"]["rank"] == 1


def test_missing_values_use_pairwise_rows_and_small_columns_are_kept() -> None:
    df = data(n=60, seed=6)
    df.loc[df.index[5:], "weak"] = np.nan  # 5 observed rows < SELECT_MIN_ROWS
    df.loc[df.index[:20], "strong"] = np.nan
    steps = [
        Step("select.mutual_information", ("weak", "strong", "const"), {"k": 1, "action": "drop"})
    ]
    out, _ = run(df, steps, roles=ROLES, task="binary")
    columns = out.reports[0].learned["columns"]
    assert columns["strong"]["n_rows"] == 40
    assert columns["weak"]["not_assessed"] == "too_few_rows"
    assert columns["const"]["not_assessed"] == "constant"
    assert {"weak", "const", "strong"} <= set(out.frame.columns)  # not assessed -> kept


def test_rows_with_missing_target_are_not_used_for_scores() -> None:
    df = data(n=100, seed=7)
    df["y"] = df["y"].astype("float64")
    df.loc[df.index[:30], "y"] = np.nan
    out, _ = run(df, [Step("select.mutual_information", ("strong",))], roles=ROLES, task="binary")
    assert out.reports[0].learned["columns"]["strong"]["n_rows"] == 70


def test_single_class_fitting_rows_fail_clearly() -> None:
    df = data(n=40)
    df["y"] = 1
    with pytest.raises(EngineError) as exc:
        run(df, [Step("select.mutual_information", ("strong",))], roles=ROLES, task="binary")
    assert exc.value.code == "TARGET_SINGLE_CLASS"


def test_supervised_selection_requires_a_target_and_task_type() -> None:
    df = data()
    step = Step("select.mutual_information", ("strong",))
    assert errors(df, [step], roles={}) == ["TARGET_REQUIRED"]
    assert errors(df, [step], task=None) == ["TASK_TYPE_REQUIRED"]
    df["label"] = "x"
    assert errors(df, [step], roles={"target": ["label"]}, task="regression") == [
        "TARGET_TYPE_UNSUPPORTED"
    ]


def test_target_cannot_be_supplied_by_the_request() -> None:
    df = data()
    step = Step("select.mutual_information", ("strong",), {"target": "noise"})
    assert errors(df, [step]) == ["UNKNOWN_PARAM"]


# ---- Univariate tests -------------------------------------------------------------------------------


def test_anova_matches_scikit_learn_f_classif_and_adjusts_p_values() -> None:
    df = data(classes=3, seed=8)
    out, _ = run(
        df,
        [Step("select.univariate", ("strong", "weak", "noise"), {"test": "anova_f"})],
        roles=ROLES,
        task="multiclass",
    )
    columns = out.reports[0].learned["columns"]
    f, p = f_classif(df[["strong", "weak", "noise"]].to_numpy(), df["y"].to_numpy())
    for j, c in enumerate(["strong", "weak", "noise"]):
        assert columns[c]["statistic"] == pytest.approx(f[j])
        assert columns[c]["p_value"] == pytest.approx(p[j])
    raw = [columns[c]["p_value"] for c in ("strong", "weak", "noise")]
    assert [columns[c]["adjusted_p_value"] for c in ("strong", "weak", "noise")] == pytest.approx(
        stats.false_discovery_control(raw)
    )
    assert columns["strong"]["rank"] == 1


def test_kruskal_chi2_and_regression_tests_match_scipy() -> None:
    df = data(seed=9)
    out, _ = run(
        df,
        [Step("select.univariate", ("strong",), {"test": "kruskal"})],
        roles=ROLES,
        task="binary",
    )
    h = stats.kruskal(df.loc[df.y == 0, "strong"], df.loc[df.y == 1, "strong"])
    assert out.reports[0].learned["columns"]["strong"]["p_value"] == pytest.approx(h.pvalue)
    out, _ = run(
        df,
        [Step("select.univariate", ("cat", "flag"), {"test": "chi2"})],
        roles=ROLES,
        task="binary",
    )
    chi = stats.chi2_contingency(pd.crosstab(df["cat"], df["y"]).to_numpy())
    assert out.reports[0].learned["columns"]["cat"]["p_value"] == pytest.approx(chi.pvalue)
    rng = np.random.default_rng(10)
    reg = frame(x=rng.normal(size=200), t=rng.normal(size=200))
    reg["t"] = reg["t"] + 0.5 * reg["x"]
    out, _ = run(
        reg,
        [Step("select.univariate", ("x",), {"test": "spearman"})],
        roles={"target": ["t"]},
        task="regression",
    )
    rho = stats.spearmanr(reg["x"], reg["t"])
    assert out.reports[0].learned["columns"]["x"]["statistic"] == pytest.approx(abs(rho.statistic))


@pytest.mark.parametrize(
    ("test", "task"),
    [
        ("anova_f", "regression"),
        ("chi2", "regression"),
        ("f_regression", "binary"),
        ("spearman", "multiclass"),
    ],
)
def test_test_and_task_type_must_match(test: str, task: str) -> None:
    df = data()
    df["t"] = df["noise"]
    roles = {"target": ["t" if task == "regression" else "y"]}
    column = "cat" if test == "chi2" else "strong"
    assert "TEST_TASK_MISMATCH" in errors(
        df, [Step("select.univariate", (column,), {"test": test})], roles, task
    )


def test_test_and_column_type_must_match() -> None:
    df = data()
    assert errors(df, [Step("select.univariate", ("cat",), {"test": "anova_f"})]) == [
        "COLUMN_TYPE_UNSUPPORTED"
    ]
    assert errors(df, [Step("select.univariate", ("strong",), {"test": "chi2"})]) == [
        "COLUMN_TYPE_UNSUPPORTED"
    ]
    assert errors(df, [Step("select.univariate", ("strong",))]) == [
        "PARAM_REQUIRED"
    ]  # no default test


def test_doubtful_assumptions_are_reported_with_the_robust_alternative() -> None:
    rng = np.random.default_rng(11)
    n = 400
    y = rng.integers(0, 2, n)
    skewed = rng.lognormal(0, 1.5, n) * np.where(y == 1, 10, 1)
    rare = pd.Series(np.where(rng.random(n) < 0.03, "rare", "common"), dtype=object)
    df = frame(s=skewed, r=rare, y=y)
    out, _ = run(
        df, [Step("select.univariate", ("s",), {"test": "anova_f"})], roles=ROLES, task="binary"
    )
    warnings = [w for w in out.reports[0].warnings if w["code"] == "ASSUMPTION_DOUBTFUL"]
    assert warnings and "kruskal" in warnings[0]["message"]
    assert set(out.reports[0].learned["columns"]["s"]["doubtful_assumptions"]) >= {"normality"}
    df2 = frame(r=pd.Series(["a"] * 12 + ["b"] * 2, dtype=object), y=[0, 1] * 7)
    out, _ = run(
        df2, [Step("select.univariate", ("r",), {"test": "chi2"})], roles=ROLES, task="binary"
    )
    assert "expected_counts" in out.reports[0].learned["columns"]["r"]["doubtful_assumptions"]


def test_high_cardinality_columns_are_not_assessed_by_chi2() -> None:
    rng = np.random.default_rng(12)
    n = 400
    df = frame(
        code=pd.Series([f"c{i}" for i in rng.integers(0, 120, n)], dtype=object),
        y=rng.integers(0, 2, n),
    )
    steps = [Step("select.univariate", ("code",), {"test": "chi2", "action": "drop", "k": 1})]
    result = validated(df, steps, ROLES, {"code": "categorical"}, task="binary")
    out = PreprocessExecutor(result).fit(df).transform(df)
    assert out.reports[0].learned["columns"]["code"]["not_assessed"] == "too_many_levels"
    assert "code" in out.frame.columns


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    p = [0.01, 0.04, 0.03, 0.5, 1e-300]
    adjusted = rt.benjamini_hochberg(p)
    assert adjusted == pytest.approx(list(stats.false_discovery_control(p)))
    assert all(0 <= a <= 1 for a in adjusted)


# ---- Explicit removal and lineage ---------------------------------------------------------------


def test_drop_columns_and_feature_lineage() -> None:
    df = data()
    steps = [
        Step("impute.median", ("weak",)),
        Step("select.variance", ("const", "noise"), {"action": "drop", "keep": ["noise"]}),
        Step("select.drop_columns", ("weak",)),
        Step("missing.indicator", ("noise",)),
    ]
    out, _ = run(df, steps, roles=ROLES)
    lineage = {e["column"]: e for e in out.lineage}
    assert lineage["const"]["final"] == "removed"
    assert lineage["const"]["events"] == [
        {"step": 1, "op": "select.variance", "action": "removed", "decision": "removed"}
    ]
    assert [e["action"] for e in lineage["weak"]["events"]] == ["transformed", "removed"]
    assert lineage["noise"]["final"] == "present"
    assert lineage["noise"]["events"][0]["decision"] == "kept"
    assert lineage["noise__missing"]["origin"] == "added"
    assert lineage["noise__missing"]["derived_from"] == ["noise"]
    assert lineage["y"]["events"] == []  # the target is never touched


def test_selection_runs_on_fitting_rows_and_is_deterministic() -> None:
    df = data(seed=13)
    steps = [
        Step("select.mutual_information", ("strong", "weak", "noise"), {"k": 1, "action": "drop"})
    ]
    first = run(df, steps, roles=ROLES, task="binary", fit_rows=df.index[:150], seed=1)[0]
    second = run(df, steps, roles=ROLES, task="binary", fit_rows=df.index[:150], seed=1)[0]
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.reports[0].learned == second.reports[0].learned
    assert first.frame[ROW_ID].tolist() == df[ROW_ID].tolist()


def test_mutual_information_limits_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    df = data()
    monkeypatch.setattr(Limits, "MI_MAX_FIT_ROWS", 50)
    with pytest.raises(EngineError) as exc:
        run(df, [Step("select.mutual_information", ("strong",))], roles=ROLES, task="binary")
    assert exc.value.code == "OP_LIMIT_EXCEEDED"
