"""Biomedical Intelligence Lite engine, Copilot routing, CSV formula neutralisation and safe
filenames. Synthetic data only."""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.bio.summary import MIN_CELL, summarise
from datamedic.exports.jobs import neutralise
from datamedic.reports import copilot
from datamedic.reports.router import safe_filename


def cohort(n_patients: int = 120, per: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_patients), per)
    y = np.repeat(rng.integers(0, 2, n_patients), per)
    batch = np.where(
        y == 1,
        rng.choice(["B1", "B2"], len(pid), p=[0.85, 0.15]),
        rng.choice(["B1", "B2"], len(pid), p=[0.15, 0.85]),
    )
    marker = rng.normal(size=len(pid)) + 1.2 * y
    noise = rng.normal(size=len(pid))
    shifted = rng.normal(size=len(pid)) + 2.0 * (batch == "B1")
    rare = np.where(np.arange(len(pid)) < 5, "rare", "common")
    return pd.DataFrame(
        {
            ROW_ID: np.arange(len(pid)),
            "patient": pid,
            "y": y,
            "batch": batch,
            "marker": marker,
            "noise": noise,
            "shifted": shifted,
            "sex": rare,
        }
    )


ROLES = {"target": ["y"], "bio.patient_id": ["patient"], "bio.batch": ["batch"], "bio.sex": ["sex"]}
NUMERIC = ["patient", "y", "marker", "noise", "shifted"]


def test_summary_counts_patients_and_compares_at_patient_level() -> None:
    s = summarise(cohort(), ROLES, "binary", {"sex"}, NUMERIC)
    pat = s["cohort"]["patients"]
    assert pat["status"] == "assessed" and pat["unique"] == 120 and pat["repeated_patients"] == 120
    assert pat["observations_per_patient"]["median"] == 3
    assert s["roles"]["biomarkers"] == ["marker", "noise", "shifted"]  # roles never biomarkers
    comp = s["comparisons"]
    assert comp["unit"] == "patient"
    top = comp["items"][0]
    assert top["column"] == "marker" and top["effect"]["name"] == "hedges_g"
    assert top["effect"]["value"] > 0.8 and top["q_value"] is not None
    assert sum(g["n"] for g in top["groups"]) == 120  # one value per patient, not 360 rows
    assert "Research use only" in s["disclaimer"]


def test_batch_confounding_and_batch_effect_are_flagged() -> None:
    s = summarise(cohort(), ROLES, "binary", set(), NUMERIC)
    kinds = {w["kind"] for w in s["confounding"]}
    assert {"batch_outcome_association", "batch_effect"} <= kinds
    shift = s["batches"]["items"][0]["biomarker_shifts"][0]
    assert shift["column"] == "shifted" and shift["name"] == "epsilon_squared"


def test_missing_metadata_is_not_assessed_and_small_groups_are_suppressed() -> None:
    df = cohort()
    s = summarise(df, {"target": ["y"]}, "binary", set(), NUMERIC)
    assert s["cohort"]["patients"]["status"] == "not_assessed"
    assert s["batches"]["status"] == "not_assessed"
    assert s["entities"]["patient_leakage"]["status"] == "not_assessed"
    none = summarise(df, {}, None, set(), NUMERIC)
    assert none["target"]["status"] == "not_assessed"
    assert none["comparisons"]["status"] == "not_assessed"
    # A sensitive categorical batch: labels replaced, groups under MIN_CELL suppressed.
    roles = {"target": ["y"], "bio.batch": ["sex"]}
    s = summarise(df, roles, "binary", {"sex"}, NUMERIC)
    groups = s["batches"]["items"][0]["groups"]
    assert {g["label"] for g in groups} == {"group 1", "group 2"}
    small = [g for g in groups if g["suppressed"]]
    assert small and small[0]["n"] is None and 5 < MIN_CELL


def test_sensitive_target_and_small_cohort() -> None:
    df = cohort(n_patients=30, per=1)
    s = summarise(df, {"target": ["y"]}, "binary", {"y"}, NUMERIC)
    assert {c["label"] for c in s["target"]["classes"]} == {"class 1", "class 2"}
    assert any(w["kind"] == "small_cohort" for w in s["small_cohort"])


def _passport(**over: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "dataset": {"domain": "biomedical", "name": "d"},
        "version": {
            "number": 0,
            "kind": "ingest",
            "content_hash": "ab" * 32,
            "fitted_on_full_data": False,
        },
        "roles": {"roles": {}},
        "protected_columns": ["code"],
        "quality": {"status": "not_available", "reason": "No profile of this version is stored."},
        "diagnostics": {"status": "not_available", "reason": "Diagnostics have not been run."},
        "recommendations": {"status": "not_available", "reason": "None."},
        "experiments": {"status": "not_available", "reason": "None."},
        "models": {"status": "not_available", "reason": "No models were trained."},
        "history": {"status": "available", "versions": [], "events": []},
        "domain_findings": {"status": "not_available", "reason": "x"},
    }
    p.update(over)
    return p


def test_copilot_routes_questions_and_never_invents_results() -> None:
    assert copilot.classify("Can I trust these results, is there leakage?") == "reliability"
    assert copilot.classify("Which features matter most?") == "explain"
    assert copilot.classify("hello") == "overview"
    out = copilot.answer(_passport(), "How well do the models perform?", "/datasets/x")
    assert out["topic"] == "models" and "No models were trained" in " ".join(out["answer"])
    assert out["assistant"]["kind"] == "built-in rule-based assistant"
    assert out["references"][0]["href"] == "/datasets/x/models"
    nxt = copilot.answer(_passport(), "What should I do next?", "/datasets/x")
    text = " ".join(nxt["answer"])
    assert "Declare the target" in text and "biomedical summary" in text
    priv = copilot.answer(_passport(), "Which columns are sensitive?", "/datasets/x")
    assert "code" in " ".join(priv["answer"])


def test_csv_formula_injection_and_filenames() -> None:
    assert neutralise("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert neutralise("+1") == "'+1" and neutralise("@x") == "'@x" and neutralise("\tA") == "'\tA"
    assert neutralise("plain") == "plain" and neutralise(5) == 5
    assert safe_filename('../../etc/pa"ss wd', ".csv") == "etc_pa_ss_wd.csv"
    assert safe_filename("", ".csv") == "dataset.csv"


def test_copilot_routes_insights_and_importance_questions_correctly() -> None:
    assert copilot.classify("What are the most important insights about my dataset?") == "insights"
    assert copilot.classify("Summarise my dataset") == "insights"
    assert copilot.classify("What are the most important data-quality issues?") == "diagnostics"
    assert copilot.classify("Which features matter most?") == "explain"
    assert copilot.classify("Show the feature importance of the model") == "explain"
    assert copilot.classify("What is the most important feature?") == "explain"
    assert copilot.classify("Explain the reliability warnings") == "reliability"
    missing = copilot.answer(
        _passport(), "What are the most important insights about my dataset?", "/d"
    )
    assert missing["topic"] == "insights" and "Importance is computed" not in " ".join(
        missing["answer"]
    )
    assert missing["references"][0]["href"] == "/d/analytics"
    stored = _passport(
        analytics={
            "insights": {
                "status": "available",
                "counts": {"association": 1, "descriptive": 1},
                "top": [
                    {"title": "Strong positive association: 'x' and 'y'", "basis": "association"},
                    {"title": "'z' is strongly skewed to the right", "basis": "descriptive"},
                ],
                "not_assessed": [{"area": "temporal", "reason": "No date column."}],
            }
        }
    )
    text = " ".join(
        copilot.answer(stored, "What are the most important insights about my dataset?", "/d")[
            "answer"
        ]
    )
    assert "[statistical association] Strong positive association" in text
    assert "1 area(s) were not assessed" in text and "not causal" in text
