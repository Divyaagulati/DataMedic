"""Domain diagnostics on synthetic datasets with planted issues, plus controls. Domain checks
need declared roles; without them they must report not_assessed, never 'no issue'."""

import numpy as np
import pandas as pd
import pytest

from datamedic.domains.bootstrap import build_registry
from tests.unit.diag_helpers import by_check, make_ctx, run_domain

RNG = np.random.default_rng(42)


def statuses(findings, check_id):  # type: ignore[no-untyped-def]
    return [f.status for f in by_check(findings, check_id)]


@pytest.mark.parametrize("domain", ["biomedical", "financial", "business", "industrial"])
def test_domain_checks_without_roles_are_not_assessed(domain: str) -> None:
    frame = pd.DataFrame({"a": RNG.normal(size=100), "b": RNG.choice(["x", "y"], 100)})
    findings = run_domain(make_ctx(frame, domain=domain))
    module = build_registry().module(domain)
    assert module is not None
    for check in module.checks:
        if check.id == "bio.dimensionality.hdlss":
            continue  # needs data only, no roles
        found = by_check(findings, check.id)
        assert found and all(f.status == "not_assessed" for f in found), check.id
        assert all(f.not_assessed_reason for f in found)


def test_domain_checks_only_run_for_their_domain() -> None:
    frame = pd.DataFrame({"a": RNG.normal(size=50)})
    ids = {f.check_id for f in run_domain(make_ctx(frame, domain="generic"))}
    assert all(i.startswith("universal.") for i in ids)
    ids = {f.check_id for f in run_domain(make_ctx(frame, domain="financial"))}
    assert any(i.startswith("fin.") for i in ids) and not any(i.startswith("bio.") for i in ids)


# ---- Biomedical -----------------------------------------------------------------------------


def clinical(n_patients: int = 200, visits: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid = np.repeat([f"PT{i:04d}" for i in range(n_patients)], visits)
    n = len(pid)
    return pd.DataFrame(
        {
            "patient": pid,
            "site": rng.choice(["A", "B", "C"], n),
            "age": np.repeat(rng.integers(18, 90, n_patients), visits),
            "hr": rng.normal(75, 10, n).round(1),
            "crp": rng.lognormal(1, 0.5, n).round(2),
            "na": rng.normal(140, 3, n).round(1),
            "dx": np.repeat(rng.choice([0, 1], n_patients), visits),
        }
    )


def test_patient_repeats_and_patient_level_leakage() -> None:
    frame = clinical()
    roles = {"bio.patient_id": ["patient"], "target": ["dx"]}
    findings = run_domain(make_ctx(frame, roles, "binary", "biomedical"))
    repeated = by_check(findings, "bio.patients.repeated_rows")[0]
    assert repeated.status == "detected" and repeated.evidence["patients_with_repeats"] == 200
    leak = by_check(findings, "bio.leakage.patient_level")[0]
    assert leak.status == "detected" and leak.severity == "high"
    assert leak.evidence["patients_with_multiple_rows"] == 200
    assert "GroupKFold" in (leak.recommendation or "")


def test_one_row_per_patient_is_not_a_leak() -> None:
    frame = clinical(visits=1)
    roles = {"bio.patient_id": ["patient"], "target": ["dx"]}
    findings = run_domain(make_ctx(frame, roles, "binary", "biomedical"))
    assert statuses(findings, "bio.patients.repeated_rows") == ["not_detected"]
    assert statuses(findings, "bio.leakage.patient_level") == ["not_detected"]


def test_batch_effect_detected_when_planted() -> None:
    frame = clinical(seed=1)
    shift = frame["site"].map({"A": 0.0, "B": 8.0, "C": -8.0})
    frame["hr"] = (frame["hr"] + shift).round(1)
    frame["na"] = (frame["na"] + shift / 2).round(1)
    finding = by_check(
        run_domain(make_ctx(frame, {"bio.batch": ["site"]}, domain="biomedical")),
        "bio.batch.effect",
    )[0]
    assert finding.status == "detected"
    assert {d["feature"] for d in finding.evidence["strongest"]} >= {"hr", "na"}
    assert "may also reflect real differences" in finding.explanation


def test_no_batch_effect_on_random_sites() -> None:
    frame = clinical(seed=2)
    finding = by_check(
        run_domain(make_ctx(frame, {"bio.batch": ["site"]}, domain="biomedical")),
        "bio.batch.effect",
    )[0]
    assert finding.status == "not_detected"


def test_batch_confounded_with_outcome() -> None:
    frame = clinical(visits=1, seed=3)
    frame["site"] = np.where(
        frame["dx"] == 1,
        RNG.choice(["A", "B"], len(frame), p=[0.9, 0.1]),
        RNG.choice(["A", "B"], len(frame), p=[0.1, 0.9]),
    )
    roles = {"bio.batch": ["site"], "target": ["dx"]}
    finding = by_check(
        run_domain(make_ctx(frame, roles, "binary", "biomedical")), "bio.batch.confounding"
    )[0]
    assert finding.status == "detected" and finding.severity == "high"
    assert finding.evidence["cramers_v"] >= 0.5


def test_batch_independent_of_outcome_is_not_confounded() -> None:
    frame = clinical(visits=1, seed=4)
    roles = {"bio.batch": ["site"], "target": ["dx"]}
    finding = by_check(
        run_domain(make_ctx(frame, roles, "binary", "biomedical")), "bio.batch.confounding"
    )[0]
    assert finding.status == "not_detected"


def test_confounding_needs_target_or_condition() -> None:
    frame = clinical(seed=5)
    finding = by_check(
        run_domain(make_ctx(frame, {"bio.batch": ["site"]}, domain="biomedical")),
        "bio.batch.confounding",
    )[0]
    assert finding.status == "not_assessed"


def test_clinical_plausibility() -> None:
    frame = clinical(visits=1)
    frame.loc[:2, "hr"] = [0.5, 400.0, 999.0]
    frame.loc[3, "age"] = 150
    roles = {"bio.heart_rate": ["hr"], "bio.age": ["age"]}
    found = {
        f.columns[0]: f
        for f in by_check(
            run_domain(make_ctx(frame, roles, domain="biomedical")), "bio.clinical.plausibility"
        )
    }
    assert found["hr"].status == "detected" and found["hr"].affected_rows == 3
    assert found["hr"].evidence["below"] == 1 and found["hr"].evidence["above"] == 2
    assert found["age"].affected_rows == 1 and found["age"].evidence["unit_assumed"] == "years"


def test_biomarker_identifier_consistency() -> None:
    frame = pd.DataFrame(
        {
            "gene": ["TP53", "tp53", " TP53", "BRCA1", "ENSG00000141510", "EGFR"] * 10,
            "expr": RNG.normal(size=60),
        }
    )
    finding = by_check(
        run_domain(make_ctx(frame, {"bio.biomarker_id": ["gene"]}, domain="biomedical")),
        "bio.biomarkers.id_consistency",
    )[0]
    assert finding.status == "detected"
    assert finding.evidence["identifiers_with_variants"] == 1
    assert set(finding.evidence["identifier_systems"]) >= {"ensembl_gene", "symbol"}


def test_high_dimensional_low_sample_size() -> None:
    frame = pd.DataFrame(RNG.normal(size=(30, 60)), columns=[f"g{i}" for i in range(60)])
    finding = by_check(
        run_domain(make_ctx(frame, domain="biomedical")), "bio.dimensionality.hdlss"
    )[0]
    assert finding.status == "detected" and finding.severity == "high"
    small = pd.DataFrame(RNG.normal(size=(500, 5)), columns=list("abcde"))
    assert statuses(
        run_domain(make_ctx(small, domain="biomedical")), "bio.dimensionality.hdlss"
    ) == ["not_detected"]


# ---- Financial ------------------------------------------------------------------------------


def transactions(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n, freq="6h")
    return pd.DataFrame(
        {
            "txn": [f"T{i:06d}" for i in range(n)],
            "account": rng.choice([f"A{i:03d}" for i in range(50)], n),
            "ts": times.strftime("%Y-%m-%dT%H:%M:%S"),
            "amount": rng.lognormal(3.5, 0.8, n).round(2),
            "fraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
        }
    )


FIN_ROLES = {
    "fin.transaction_id": ["txn"],
    "fin.account_id": ["account"],
    "fin.timestamp": ["ts"],
    "fin.amount": ["amount"],
    "target": ["fraud"],
}


def test_financial_temporal_inconsistencies() -> None:
    frame = transactions()
    frame.loc[0, "ts"] = "2099-05-01T00:00:00"  # future
    frame.loc[1, "ts"] = "1965-05-01T00:00:00"  # before 1970
    frame = pd.concat([frame, frame.iloc[[10]].assign(ts="2025-02-01T00:00:00")], ignore_index=True)
    findings = run_domain(make_ctx(frame, FIN_ROLES, "binary", "financial"))
    ts = by_check(findings, "fin.timestamps.consistency")[0]
    assert ts.status == "detected" and ts.severity == "high"
    assert ts.evidence["future"] == 1 and ts.evidence["before_1970"] == 1
    assert ts.evidence["transaction_ids_with_several_times"] == 1
    dup = by_check(findings, "fin.transactions.duplicates")[0]
    assert dup.status == "detected" and dup.evidence["repeated_ids"] == 1


def test_clean_transactions() -> None:
    findings = run_domain(make_ctx(transactions(), FIN_ROLES, "binary", "financial"))
    assert statuses(findings, "fin.timestamps.consistency") == ["not_detected"]
    assert statuses(findings, "fin.transactions.duplicates") == ["not_detected"]
    assert statuses(findings, "fin.leakage.temporal") == ["not_detected"]


def test_temporal_leakage_when_target_drifts_over_time() -> None:
    frame = transactions(seed=1)
    frame["fraud"] = (np.arange(len(frame)) > 400).astype(int)  # fraud only late in time
    frame = frame.sample(frac=1, random_state=0).reset_index(drop=True)  # row order is NOT time
    finding = by_check(
        run_domain(make_ctx(frame, FIN_ROLES, "binary", "financial")), "fin.leakage.temporal"
    )[0]
    assert finding.status == "detected" and finding.severity == "high"
    assert "TimeSeriesSplit" in (finding.recommendation or "")


def test_unusual_amounts() -> None:
    frame = transactions()
    frame.loc[:4, "amount"] = [0, -50, -20, 250000, 1e6]
    finding = by_check(
        run_domain(make_ctx(frame, FIN_ROLES, "binary", "financial")), "fin.amounts.unusual"
    )[0]
    assert finding.status == "detected"
    assert finding.evidence["zero"] == 1 and finding.evidence["negative"] == 2
    assert finding.evidence["extreme_by_log_modified_z"] >= 2


# ---- Business -------------------------------------------------------------------------------


def customers(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churn = rng.choice([0, 1], n, p=[0.8, 0.2])
    return pd.DataFrame(
        {
            "cid": [f"CU-{i:05d}" for i in range(n)],
            "plan": rng.choice(["basic", "pro", "team"], n),
            "tenure": rng.integers(1, 60, n),
            "spend": rng.normal(50, 10, n).round(2),
            "churn": churn,
        }
    )


BIZ_ROLES = {"biz.customer_id": ["cid"], "biz.segment": ["plan"], "target": ["churn"]}


def test_business_duplicates_suspicious_ids_and_churn_leakage() -> None:
    frame = customers()
    frame.loc[0, "cid"] = "CU-00001"  # exact repeat of row 1
    frame.loc[2, "cid"] = "cu-00003 "  # collides with CU-00003 after normalising
    frame.loc[4:8, "cid"] = ["test", "0", "unknown", "N/A", "-1"]
    frame["cancel_reason"] = np.where(frame["churn"] == 1, "price", None)
    findings = run_domain(make_ctx(frame, BIZ_ROLES, "binary", "business"))
    dup = by_check(findings, "biz.customers.duplicates")[0]
    assert dup.status == "detected" and dup.evidence["repeated_ids"] == 1
    assert dup.evidence["near_duplicate_id_groups"] >= 1
    ids = by_check(findings, "biz.customers.suspicious_ids")[0]
    assert (
        ids.status == "detected" and ids.evidence["placeholders"] == 4
    )  # "N/A" is parsed as missing
    leak = by_check(findings, "biz.leakage.churn")[0]
    assert leak.status == "detected" and leak.severity == "critical"
    assert leak.columns == ["cancel_reason", "churn"]


def test_clean_customers() -> None:
    findings = run_domain(make_ctx(customers(), BIZ_ROLES, "binary", "business"))
    for check in (
        "biz.customers.duplicates",
        "biz.customers.suspicious_ids",
        "biz.leakage.churn",
        "biz.segments.imbalance",
    ):
        assert statuses(findings, check) == ["not_detected"], check


def test_segment_imbalance() -> None:
    frame = customers(n=1000)
    frame["plan"] = ["basic"] * 950 + ["pro"] * 40 + ["enterprise"] * 10
    finding = by_check(
        run_domain(make_ctx(frame, BIZ_ROLES, "binary", "business")), "biz.segments.imbalance"
    )[0]
    assert finding.status == "detected" and finding.severity == "medium"
    assert set(finding.evidence["small_segments"]) == {"enterprise"}


# ---- Industrial -----------------------------------------------------------------------------


def sensors(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-03-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "t": times.strftime("%Y-%m-%dT%H:%M:%S"),
            "pressure": rng.normal(5, 0.2, n).round(4),
            "flow": rng.normal(20, 1, n).round(3),
            "valve": rng.uniform(10, 90, n).round(2),
        }
    )


IND_ROLES = {
    "ind.timestamp": ["t"],
    "ind.sensor": ["pressure"],
    "ind.sensor_nonneg": ["flow"],
    "ind.sensor_percent": ["valve"],
}


def test_impossible_and_saturated_sensor_values() -> None:
    frame = sensors()
    frame.loc[:2, "flow"] = -3.0  # never-negative sensor
    frame.loc[3, "valve"] = 120.0  # percentage above 100
    frame.loc[100:160, "pressure"] = 6.0  # clipped at the top of the range
    frame.loc[161, "pressure"] = 4.0
    found = {
        f.columns[0]: f
        for f in by_check(
            run_domain(make_ctx(frame, IND_ROLES, domain="industrial")), "ind.sensors.impossible"
        )
    }
    assert (
        found["flow"].status == "detected" and found["flow"].evidence["outside_physical_range"] == 3
    )
    assert found["valve"].evidence["outside_physical_range"] == 1
    assert "at_maximum" in found["pressure"].evidence["saturation"]


def test_irregular_sampling_uses_time_not_row_order() -> None:
    frame = sensors()
    times = pd.to_datetime(frame["t"])
    times = times + pd.to_timedelta(np.where(times.index >= 250, 3, 0), unit="h")  # one gap
    frame["t"] = times.dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame = frame.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle rows
    finding = by_check(
        run_domain(make_ctx(frame, IND_ROLES, domain="industrial")), "ind.sampling.irregular"
    )[0]
    assert finding.status == "detected" and finding.evidence["gaps_over_5x_median"] == 1
    assert finding.evidence["median_interval_seconds"] == 60


def test_regular_shuffled_sensors_are_clean() -> None:
    frame = sensors().sample(frac=1, random_state=2).reset_index(drop=True)
    findings = run_domain(make_ctx(frame, IND_ROLES, domain="industrial"))
    for check in (
        "ind.sampling.irregular",
        "ind.sensors.shift",
        "ind.sensors.drift",
        "ind.sensors.impossible",
    ):
        assert set(statuses(findings, check)) == {"not_detected"}, check


def test_sensor_shift_and_drift() -> None:
    frame = sensors()
    frame["pressure"] = (frame["pressure"] + np.linspace(0, 2, len(frame))).round(4)  # drift
    frame.loc[350:, "flow"] = (frame.loc[350:, "flow"] + 3).round(3)  # step change late
    findings = run_domain(make_ctx(frame, IND_ROLES, domain="industrial"))
    shift = by_check(findings, "ind.sensors.shift")[0]
    assert shift.status == "detected"
    assert {s["sensor"] for s in shift.evidence["shifted"]} >= {"pressure", "flow"}
    drift = by_check(findings, "ind.sensors.drift")[0]
    assert drift.status == "detected"
    # The late step in flow is also a monotonic association with time, so both qualify.
    assert {d["sensor"] for d in drift.evidence["drifting"]} == {"pressure", "flow"}


def test_one_column_with_two_sensor_roles_is_checked_once() -> None:
    """Regression (Phase 3 manual review): a column declared both 'Sensor reading' and
    'Sensor reading (0-100 %)' crashed the drift check with IndexError."""
    frame = sensors()
    noise = np.random.default_rng(9).normal(0, 3, len(frame))
    frame["valve"] = (20 + np.linspace(0, 60, len(frame)) + noise).round(2)  # clear upward trend
    roles = {"ind.timestamp": ["t"], "ind.sensor": ["valve"], "ind.sensor_percent": ["valve"]}
    findings = run_domain(make_ctx(frame, roles, domain="industrial"))
    assert not [f for f in findings if f.status == "error"]
    drift = by_check(findings, "ind.sensors.drift")[0]
    assert drift.status == "detected" and [d["sensor"] for d in drift.evidence["drifting"]] == [
        "valve"
    ]
    shift = by_check(findings, "ind.sensors.shift")[0]
    assert [s["sensor"] for s in shift.evidence["shifted"]] == ["valve"]
