"""Universal diagnostics on synthetic datasets with known, planted issues, plus clean
controls (false-positive checks). PRD §9: clean controls produce no high or critical findings
and at most low severity here."""

import numpy as np
import pandas as pd
import pytest

from tests.unit.diag_helpers import (
    by_check,
    clean_frame,
    detected_ids,
    jsonable,
    make_ctx,
    run_domain,
)

TARGET = {"target": ["y"]}


@pytest.mark.parametrize("seed", range(5))
def test_clean_control_has_no_serious_findings(seed: int) -> None:
    findings = run_domain(make_ctx(clean_frame(seed=seed), TARGET, "binary"))
    jsonable(findings)
    serious = [f for f in findings if f.status == "detected" and f.severity not in ("low", "info")]
    assert serious == []
    assert [f for f in findings if f.status == "error"] == []
    # Only outlier screening may fire on Gaussian data (true tail values), and only as low.
    assert detected_ids(findings) <= {
        "universal.outliers.univariate",
        "universal.outliers.multivariate",
    }


def test_every_finding_has_required_fields() -> None:
    frame = clean_frame()
    frame.loc[:40, "x1"] = None
    for f in run_domain(make_ctx(frame, TARGET, "binary")):
        assert f.check_id and f.category and f.title and f.explanation and f.limitations
        assert f.status in ("detected", "not_detected", "not_assessed", "error")
        assert f.severity in ("critical", "high", "medium", "low", "info")
        if f.status == "detected":
            assert f.evidence and f.recommendation
        if f.status == "not_assessed":
            assert f.not_assessed_reason


def test_missing_values_and_patterns() -> None:
    frame = clean_frame()
    frame.loc[:299, "x1"] = None  # 50% -> high
    frame.loc[:59, "x3"] = None  # 10% -> low, missing together with x1 (phi >= 0.5 ... )
    frame.loc[:299, "x3"] = None  # now identical pattern to x1
    frame["empty"] = None
    findings = run_domain(make_ctx(frame, TARGET, "binary"))
    rate = {f.columns[0]: f for f in by_check(findings, "universal.missingness.column_rate")}
    assert rate["x1"].severity == "high" and rate["x1"].affected_rows == 300
    assert rate["empty"].severity == "high" and "entirely empty" in rate["empty"].title
    pattern = [
        f for f in by_check(findings, "universal.missingness.pattern") if f.status == "detected"
    ]
    pairs = pattern[0].evidence["pairs"]
    assert {pairs[0]["a"], pairs[0]["b"]} == {"x1", "x3"} and pairs[0]["phi"] == 1.0


def test_missingness_associated_with_target() -> None:
    frame = clean_frame()
    frame.loc[frame["y"] == 1, "x3"] = None  # missing only for positives
    frame.loc[frame.index[:5], "x3"] = None
    findings = by_check(
        run_domain(make_ctx(frame, TARGET, "binary")), "universal.missingness.target_association"
    )
    hit = [f for f in findings if f.status == "detected"]
    assert hit and hit[0].columns == ["x3", "y"] and hit[0].severity == "medium"
    assert hit[0].evidence["cramers_v"] >= 0.9


def test_exact_and_identifier_duplicates() -> None:
    frame = clean_frame(n=200)
    frame["rid"] = [f"R{i:04d}" for i in range(200)]
    frame = pd.concat([frame, frame.iloc[:10]], ignore_index=True)  # 10 exact copies
    findings = run_domain(make_ctx(frame, {"target": ["y"], "id": ["rid"]}, "binary"))
    exact = by_check(findings, "universal.duplicates.exact_rows")[0]
    assert exact.status == "detected" and exact.evidence["duplicate_rows"] == 10
    assert exact.evidence["rows_in_duplicate_groups"] == 20
    ident = by_check(findings, "universal.duplicates.identifier")[0]
    assert ident.status == "detected" and ident.severity == "high"
    assert ident.evidence["repeated_values"] == 10


def test_identifier_duplicates_need_a_declared_id() -> None:
    finding = by_check(run_domain(make_ctx(clean_frame())), "universal.duplicates.identifier")[0]
    assert finding.status == "not_assessed" and "Row identifier" in finding.not_assessed_reason


def test_univariate_outliers_found_and_counted_by_each_method() -> None:
    frame = clean_frame()
    frame.loc[:4, "x3"] = 10_000.0  # 5 gross outliers
    finding = [
        f
        for f in by_check(run_domain(make_ctx(frame)), "universal.outliers.univariate")
        if f.columns == ["x3"]
    ][0]
    methods = finding.evidence["counts_by_method"]
    assert methods["iqr_3"] == 5 and methods["modified_z_gt_3_5"] == 5
    assert methods["iqr_1_5"] >= 5 and methods["z_gt_3"] >= 5
    assert finding.affected_rows == 5
    assert "not necessarily an error" in " ".join(finding.limitations)


def test_discrete_columns_are_not_outlier_screened() -> None:
    frame = pd.DataFrame({"count": [0] * 95 + [1, 2, 3, 4, 50]})
    finding = by_check(run_domain(make_ctx(frame)), "universal.outliers.univariate")[0]
    assert finding.status == "not_assessed"  # <= 10 distinct values


def test_multivariate_outliers_find_planted_rows() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"]).round(5)
    # Planted rows: about 3.5 SD in every feature. Not extreme alone (Tukey 3 x IQR is about
    # 4.7 SD), but jointly very unusual.
    frame.loc[:5, ["a", "b", "c", "d"]] = 3.5
    findings = run_domain(make_ctx(frame))
    multi = by_check(findings, "universal.outliers.multivariate")[0]
    assert multi.status == "detected" and multi.severity == "low"
    assert set(multi.evidence["row_ids_example"]) == set(range(6))


def test_relationship_breaking_rows_are_listed_but_not_flagged() -> None:
    """Documented limitation: ECOD/COPOD are per-feature, so only Isolation Forest sees rows
    that break a correlation while each value is ordinary."""
    rng = np.random.default_rng(3)
    n = 1000
    a = rng.normal(size=n)
    frame = pd.DataFrame(
        {"a": a, "b": a + rng.normal(scale=0.1, size=n), "c": rng.normal(size=n)}
    ).round(5)
    frame.loc[:5, "a"] = 2.0
    frame.loc[:5, "b"] = -2.0
    multi = by_check(run_domain(make_ctx(frame)), "universal.outliers.multivariate")[0]
    # Not flagged by the consensus (ECOD/COPOD cannot see a broken relationship), and the
    # limitation is stated on the finding. Identical planted rows also mask each other.
    assert not set(multi.evidence.get("row_ids_example", [])) & set(range(6))
    assert "breaks a relationship" in " ".join(multi.limitations)


def test_multivariate_needs_two_numeric_columns() -> None:
    frame = pd.DataFrame({"a": np.arange(200) * 1.5, "s": ["x"] * 200})
    finding = by_check(run_domain(make_ctx(frame)), "universal.outliers.multivariate")[0]
    assert finding.status == "not_assessed"


def test_skewed_and_zero_inflated_distributions() -> None:
    rng = np.random.default_rng(1)
    frame = clean_frame()
    frame["income"] = rng.lognormal(10, 1.2, len(frame)).round(2)
    frame["visits"] = np.where(rng.random(len(frame)) < 0.7, 0, rng.integers(1, 9, len(frame)))
    findings = {
        f.columns[0]: f
        for f in by_check(run_domain(make_ctx(frame)), "universal.distribution.shape")
        if f.columns
    }
    assert findings["income"].status == "detected" and findings["income"].evidence["skewness"] > 2
    assert findings["visits"].evidence["zero_share_pct"] >= 60
    assert "x1" not in findings  # symmetric


def test_constant_and_near_constant() -> None:
    frame = clean_frame()
    frame["const"] = "same"
    frame["rare"] = ["a"] * 597 + ["b"] * 3
    found = {
        f.columns[0]: f
        for f in by_check(run_domain(make_ctx(frame)), "universal.variance.constant")
        if f.columns
    }
    assert found["const"].severity == "medium" and found["const"].evidence["distinct_values"] == 1
    assert found["rare"].severity == "low" and found["rare"].evidence["top_value_share_pct"] == 99.5


def test_high_cardinality_identifiers() -> None:
    frame = clean_frame()
    frame["customer_code"] = [f"C-{i:05d}" for i in range(len(frame))]
    frame["comment"] = [f"note {i % 400} about visit" for i in range(len(frame))]
    found = {
        f.columns[0]: f
        for f in by_check(run_domain(make_ctx(frame)), "universal.cardinality.high")
        if f.columns
    }
    assert found["customer_code"].severity == "medium"
    assert "comment" in found  # 400 distinct of 600 (> 50%)
    assert "region" not in found


def test_high_cardinality_respects_declared_roles() -> None:
    frame = clean_frame()
    frame["customer_code"] = [f"C-{i:05d}" for i in range(len(frame))]
    finding = by_check(
        run_domain(make_ctx(frame, {"id": ["customer_code"]})), "universal.cardinality.high"
    )[0]
    assert finding.status == "not_detected"  # declared as an identifier: not a feature


def test_correlated_features_and_vif() -> None:
    frame = clean_frame()
    frame["x1_copy"] = (frame["x1"] * 2.2 + 1).round(4)
    frame["x1_noisy"] = (frame["x1"] + np.random.default_rng(0).normal(0, 1, len(frame))).round(3)
    finding = by_check(run_domain(make_ctx(frame)), "universal.correlation.redundancy")[0]
    assert finding.status == "detected" and finding.severity == "medium"
    pairs = {frozenset((p["a"], p["b"])) for p in finding.evidence["pairs"]}
    assert frozenset(("x1", "x1_copy")) in pairs
    assert "x1_copy" in finding.evidence["vif"]
    assert "not say that one feature causes another" in finding.explanation


def test_categorical_association() -> None:
    frame = clean_frame()
    frame["region_code"] = frame["region"].map(
        {"north": "N", "south": "S", "east": "E", "west": "W"}
    )
    finding = by_check(run_domain(make_ctx(frame)), "universal.correlation.categorical")[0]
    assert finding.status == "detected"
    assert finding.evidence["pairs"][0]["cramers_v"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "positives,severity", [(3, "high"), (40, "medium"), (100, "low"), (250, None)]
)
def test_class_imbalance(positives: int, severity: str | None) -> None:
    frame = clean_frame(n=600)
    frame["y"] = [1] * positives + [0] * (600 - positives)
    finding = by_check(run_domain(make_ctx(frame, TARGET, "binary")), "universal.target.imbalance")[
        0
    ]
    if severity is None:
        assert finding.status == "not_detected"
    else:
        assert finding.status == "detected" and finding.severity == severity


def test_imbalance_not_assessed_without_target_or_for_regression() -> None:
    assert (
        by_check(run_domain(make_ctx(clean_frame())), "universal.target.imbalance")[0].status
        == "not_assessed"
    )
    frame = clean_frame()
    findings = run_domain(make_ctx(frame, {"target": ["x1"]}, "regression"))
    assert by_check(findings, "universal.target.imbalance")[0].status == "not_assessed"


def test_target_quality() -> None:
    frame = clean_frame()
    frame.loc[:59, "y"] = None
    frame.loc[60:64, "y"] = 2  # a rare third class in a "binary" target
    found = by_check(run_domain(make_ctx(frame, TARGET, "binary")), "universal.target.quality")
    titles = " | ".join(f.title for f in found)
    assert "60 rows have no value" in titles and "3 classes, not 2" in titles
    assert "fewer than 10 rows" in titles


def test_suspicious_values() -> None:
    frame = clean_frame()
    frame.loc[:19, "x3"] = -999  # placeholder far below 75-165
    frame["status"] = ["ok"] * 580 + ["?"] * 10 + ["unknown"] * 10
    frame["when"] = ["2024-03-01"] * 590 + ["2099-01-01"] * 5 + ["1850-06-01"] * 5
    found = by_check(run_domain(make_ctx(frame)), "universal.values.suspicious")
    titles = {f.title for f in found}
    assert "'x3' may use -999 as a placeholder" in titles
    assert "'status' contains placeholder text" in titles
    dates = [f for f in found if f.columns == ["when"]][0]
    assert dates.evidence["future"] == 5 and dates.evidence["before_1900"] == 5


def test_placeholder_inside_normal_range_is_not_flagged() -> None:
    frame = pd.DataFrame({"temp": list(np.linspace(-1500, 1500, 400)) + [-999] * 10})
    found = by_check(run_domain(make_ctx(frame)), "universal.values.suspicious")
    assert found[0].status == "not_detected"


def test_target_proxy_leakage() -> None:
    frame = clean_frame()
    frame["outcome_code"] = np.where(frame["y"] == 1, "POS", "NEG")
    frame["score_after"] = frame["y"] * 10 + np.random.default_rng(0).normal(0, 0.1, len(frame))
    found = {
        f.columns[0]: f
        for f in by_check(
            run_domain(make_ctx(frame, TARGET, "binary")), "universal.leakage.target_proxy"
        )
        if f.status == "detected"
    }
    assert found["outcome_code"].severity == "critical"
    assert found["score_after"].evidence["roc_auc"] >= 0.999
    assert "x1" not in found  # a genuine but moderate predictor
    assert "not proof" in " ".join(found["outcome_code"].limitations)


def test_regression_target_proxy() -> None:
    frame = clean_frame()
    frame["x1_rescaled"] = frame["x1"] * 3.3
    found = by_check(
        run_domain(make_ctx(frame, {"target": ["x1"]}, "regression")),
        "universal.leakage.target_proxy",
    )
    assert any(f.status == "detected" and f.columns[0] == "x1_rescaled" for f in found)


def test_identifier_order_leakage() -> None:
    frame = clean_frame(n=400).sort_values("y", kind="stable").reset_index(drop=True)
    frame["record_no"] = range(1, 401)  # numbered after sorting by outcome
    finding = by_check(
        run_domain(make_ctx(frame, TARGET, "binary")), "universal.leakage.identifier"
    )[0]
    assert finding.status == "detected" and finding.columns == ["record_no", "y"]


def test_identifier_assigned_randomly_is_not_leakage() -> None:
    frame = clean_frame(n=400)
    frame["record_no"] = np.random.default_rng(5).permutation(400) + 1
    finding = by_check(
        run_domain(make_ctx(frame, TARGET, "binary")), "universal.leakage.identifier"
    )[0]
    assert finding.status == "not_detected"


def test_sensitive_columns_never_quote_values() -> None:
    frame = clean_frame(n=200)
    frame["mrn"] = [f"MRN{i % 190}" for i in range(200)]
    roles = {"id": ["mrn"], "sensitive": ["mrn"]}
    finding = by_check(run_domain(make_ctx(frame, roles)), "universal.duplicates.identifier")[0]
    assert finding.status == "detected" and finding.evidence["examples"] is None


def test_binary_indicators_are_not_called_skewed() -> None:
    """Regression (Phase 3 manual review): a 0/1 column with 10% ones was reported as
    'strongly asymmetric or heavy-tailed'; its skewness only restates the class balance."""
    frame = clean_frame()
    frame["event"] = [1] * 60 + [0] * 540
    found = {
        f.columns[0]
        for f in by_check(run_domain(make_ctx(frame)), "universal.distribution.shape")
        if f.columns
    }
    assert "event" not in found and "y" not in found
