"""Biomedical check implementations (worker only)."""

import re

import numpy as np
import pandas as pd

from datamedic.domains.biomedical import PLAUSIBLE_RANGES
from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
    pct,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import as_float, bh_adjust, cramers_v, kruskal_by_group, num

ID_SYSTEMS = (
    ("ensembl_gene", re.compile(r"^ENSG\d{11}(\.\d+)?$")),
    ("ensembl_protein", re.compile(r"^ENSP\d{11}(\.\d+)?$")),
    ("entrez_numeric", re.compile(r"^\d+$")),
    ("uniprot", re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]$")),
    ("symbol", re.compile(r"^[A-Za-z][A-Za-z0-9\-\.]{0,19}$")),
)


def repeated_rows(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    patient = ctx.roles.first("bio.patient_id")
    assert patient is not None
    ids = ctx.data[patient]
    counts = ids.dropna().value_counts()
    repeated = counts[counts > 1]
    if repeated.empty:
        return [
            not_detected(
                spec,
                "Each patient has exactly one row.",
                {"patients": int(counts.size)},
                columns=[patient],
            )
        ]
    within_dup = int(ctx.data[ids.isin(repeated.index)].duplicated(keep="first").sum())
    rows = int(repeated.sum())
    return [
        detected(
            spec,
            ctx,
            "medium",
            f"{repeated.size} patients have more than one row",
            f"{rows} rows belong to {repeated.size} patients with repeated rows (up to "
            f"{int(repeated.max())} rows each); {within_dup} rows are exact duplicates within "
            "a patient. Rows of the same patient are not independent.",
            {
                "patients": int(counts.size),
                "patients_with_repeats": int(repeated.size),
                "rows_in_repeated_patients": rows,
                "max_rows_per_patient": int(repeated.max()),
                "mean_rows_per_patient": num(counts.mean(), 3),
                "exact_duplicates_within_patient": within_dup,
            },
            columns=[patient],
            affected_rows=rows,
            recommendation="Treat the patient as the unit of analysis: split and evaluate by "
            "patient (group-aware cross-validation), and check that exact duplicates are not "
            "re-entered records.",
        )
    ]


def patient_leakage(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    patient = ctx.roles.first("bio.patient_id")
    target = ctx.target()
    assert patient is not None and target is not None
    frame = ctx.data[[patient, target]].dropna()
    per_patient = frame.groupby(patient)[target].agg(["size", "nunique"])
    multi = per_patient[per_patient["size"] > 1]
    if multi.empty:
        return [
            not_detected(
                spec,
                "No patient has more than one row with a target, so a "
                "row-level split cannot place one patient on both sides.",
                {"patients": int(len(per_patient))},
                columns=[patient, target],
            )
        ]
    rows = int(multi["size"].sum())
    conflicting = int((multi["nunique"] > 1).sum())
    return [
        detected(
            spec,
            ctx,
            "high",
            "Row-level splits would leak patients between training and test",
            f"{len(multi)} patients have several rows ({rows} rows, "
            f"{pct(rows, len(frame)):g}% of rows with a target). A random row split would put "
            "rows of the same patient in both training and test data, so scores would be "
            f"optimistic. {conflicting} patients have different target values across rows.",
            {
                "patients_with_multiple_rows": int(len(multi)),
                "rows_affected": rows,
                "patients_with_conflicting_targets": conflicting,
            },
            columns=[patient, target],
            affected_rows=rows,
            recommendation="Use group-aware splits by patient (GroupKFold / "
            "StratifiedGroupKFold) for every evaluation. If targets legitimately change over "
            "visits, also consider time-aware evaluation.",
        )
    ]


def biomarker_ids(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for column in ctx.roles.all("bio.biomarker_id"):
        values = ctx.data[column].dropna().astype(str)
        if values.empty:
            continue
        distinct = pd.Series(values.unique())
        normalised = distinct.str.strip().str.upper()
        groups = distinct.groupby(normalised).agg(list)
        variants = groups[groups.map(len) > 1]
        systems: dict[str, int] = {}
        for value in normalised:
            system = next((name for name, rx in ID_SYSTEMS if rx.match(value)), "other")
            systems[system] = systems.get(system, 0) + 1
        mixed = len([s for s, n in systems.items() if s != "other" and n > 0]) > 1
        if variants.empty and not mixed:
            findings.append(
                not_detected(
                    spec,
                    f"Identifiers in '{column}' are consistent.",
                    {"distinct": int(distinct.size), "systems": systems},
                    columns=[column],
                )
            )
            continue
        affected = int(values.str.strip().str.upper().isin(variants.index).sum())
        findings.append(
            detected(
                spec,
                ctx,
                "medium" if not variants.empty else "low",
                f"Identifiers in '{column}' are inconsistent",
                f"{variants.size} identifiers appear in several spellings (case or spacing)"
                + (" and several identifier systems are mixed." if mixed else "."),
                {
                    "identifiers_with_variants": int(variants.size),
                    "examples": [list(v)[:3] for v in variants.head(5)],
                    "identifier_systems": systems,
                },
                columns=[column],
                affected_rows=affected or None,
                recommendation="Normalise identifiers to one system and spelling (e.g. "
                "official HGNC symbols) in a reviewed preprocessing step before joining or "
                "aggregating.",
            )
        )
    return findings


def high_dimensional(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    p = len(ctx.numeric_features())
    if p == 0:
        return [not_assessed(spec, "no numeric features.")]
    patient = ctx.roles.first("bio.patient_id")
    n = int(ctx.data[patient].nunique()) if patient else ctx.n_rows
    unit = "patients" if patient else "rows"
    evidence = {
        "numeric_features": p,
        "samples": n,
        "sample_unit": unit,
        "features_per_sample": num(p / max(n, 1), 3),
    }
    if p < n / 10:
        return [not_detected(spec, f"{p} numeric features for {n} {unit}.", evidence)]
    severity = "high" if p >= n else "medium"
    return [
        detected(
            spec,
            ctx,
            severity,
            f"{p} numeric features for only {n} {unit}",
            "With many features relative to samples, models can fit noise and selection by "
            "p-values produces false discoveries.",
            evidence,
            recommendation="Use strong regularisation, dimensionality reduction or prior "
            "feature selection fitted inside cross-validation; validate on independent data.",
        )
    ]


def _batch_columns(ctx: CheckContext) -> list[str]:
    return ctx.roles.all("bio.batch")


def batch_effect(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    features = ctx.numeric_features()
    if not features:
        return [not_assessed(spec, "no numeric features to compare between batches.")]
    findings = []
    for batch in _batch_columns(ctx):
        sample, info = ctx.sample([batch, *features], ctx.budget.expensive_rows)
        groups = sample[batch].astype(str).where(sample[batch].notna()).to_numpy()
        sizes = pd.Series(groups).value_counts()
        if (sizes >= 3).sum() < 2:
            findings.append(
                not_assessed(spec, f"'{batch}' needs at least 2 batches with 3 or more rows.")
            )
            continue
        tests = {}
        for feature in features:
            res = kruskal_by_group(as_float(sample[feature]), groups)
            if res:
                tests[feature] = res
        if not tests:
            findings.append(not_assessed(spec, "no feature could be compared across batches."))
            continue
        qvalues = dict(zip(tests, bh_adjust([t["p"] for t in tests.values()]), strict=True))
        differing = {
            f: t for f, t in tests.items() if qvalues[f] < 0.05 and t["epsilon_sq"] >= 0.06
        }
        share = len(differing) / len(tests)
        evidence = {
            "batch_column": batch,
            "batches": int((sizes >= 3).sum()),
            "features_tested": len(tests),
            "features_differing": len(differing),
            "share_differing_pct": num(share * 100, 2),
            "strongest": sorted(
                (
                    {
                        "feature": f,
                        "epsilon_sq": num(t["epsilon_sq"], 4),
                        "q_value": num(qvalues[f], 6),
                    }
                    for f, t in differing.items()
                ),
                key=lambda d: -float(d["epsilon_sq"] or 0),
            )[:10],
        }
        if share < 0.2:
            findings.append(
                not_detected(
                    spec,
                    f"Few features differ between batches of "
                    f"'{batch}' ({len(differing)} of {len(tests)}).",
                    evidence,
                    columns=[batch],
                    sample_info=info,
                )
            )
            continue
        findings.append(
            detected(
                spec,
                ctx,
                "high" if share >= 0.5 else "medium",
                f"{len(differing)} of {len(tests)} features differ between batches of '{batch}'",
                "Feature distributions differ systematically between batches. This is "
                "consistent with a batch effect but may also reflect real differences between "
                "the samples in each batch.",
                evidence,
                columns=[batch, *list(differing)[:20]],
                recommendation="Check whether batches differ in sample composition (see the "
                "confounding check). Evaluate models with batch-aware splits; consider batch "
                "correction only with a balanced design and the outcome as a covariate.",
                sample_info=info,
            )
        )
    return findings


def batch_confounding(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    others = list(dict.fromkeys(c for c in [ctx.target(), *ctx.roles.all("bio.condition")] if c))
    if not others:
        return [
            not_assessed(spec, "declare a target or a condition column to compare with the batch.")
        ]
    findings = []
    for batch in _batch_columns(ctx):
        for other in others:
            is_regression_target = other == ctx.target() and ctx.roles.task_type == "regression"
            if is_regression_target:
                res = kruskal_by_group(
                    as_float(ctx.data[other]),
                    ctx.data[batch].astype(str).where(ctx.data[batch].notna()).to_numpy(),
                )
                strength, p = (res.get("epsilon_sq"), res.get("p")) if res else (None, None)
                measure = "epsilon_squared"
            else:
                strength, p, _ = cramers_v(ctx.data[batch], ctx.data[other])
                measure = "cramers_v"
            evidence = {
                "batch_column": batch,
                "compared_with": other,
                measure: num(strength, 4),
                "p_value": num(p, 8),
            }
            if strength is None or p is None:
                findings.append(
                    not_assessed(
                        spec, f"'{batch}' and '{other}' need at least two levels each with values."
                    )
                )
                continue
            if strength < 0.1 or p >= 0.01:
                findings.append(
                    not_detected(
                        spec,
                        f"'{batch}' is not meaningfully associated with '{other}'.",
                        evidence,
                        columns=[batch, other],
                    )
                )
                continue
            severity = "high" if strength >= 0.5 else "medium" if strength >= 0.3 else "low"
            findings.append(
                detected(
                    spec,
                    ctx,
                    severity,
                    f"Batch '{batch}' is associated with '{other}'",
                    "The distribution of the outcome or condition differs between batches. Any "
                    "batch-related artefact can then masquerade as a real effect, and batch "
                    "correction could remove real signal.",
                    evidence,
                    columns=[batch, other],
                    recommendation="Report results stratified by batch, evaluate with "
                    "batch-aware splits, and collect design information. Do not apply batch "
                    "correction without including the outcome or condition as a covariate.",
                )
            )
    return findings


def clinical_plausibility(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for role, (low, high, unit) in PLAUSIBLE_RANGES.items():
        for column in ctx.roles.all(role):
            values = as_float(ctx.data[column])
            present = values[~np.isnan(values)]
            if present.size == 0:
                continue
            outside = (present < low) | (present > high)
            n_out = int(outside.sum())
            evidence = {
                "role": role,
                "plausible_range": [low, high],
                "unit_assumed": unit,
                "below": int((present < low).sum()),
                "above": int((present > high).sum()),
                "observed_min": num(present.min()),
                "observed_max": num(present.max()),
            }
            if n_out == 0:
                findings.append(
                    not_detected(
                        spec,
                        f"All values of '{column}' are within {low:g}-{high:g} {unit}.",
                        evidence,
                        columns=[column],
                    )
                )
                continue
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"'{column}' has physiologically implausible values",
                    f"{n_out} values lie outside {low:g}-{high:g} {unit}. Values this extreme "
                    "usually indicate a unit mix-up (e.g. Fahrenheit, months) or an entry error.",
                    evidence,
                    columns=[column],
                    affected_rows=n_out,
                    recommendation="Check units and data entry with the source. Flag or set "
                    "the values to missing only in a reviewed preprocessing step.",
                )
            )
    return findings
