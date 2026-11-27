"""Analytics Studio service (API side): column catalogue, queries, version comparison,
automated insights and saved dashboards. Everything is owner-scoped (and RLS-protected); no
endpoint writes dataset versions."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from datamedic.analyses.service import current_semantics_hash
from datamedic.analytics import data
from datamedic.config import Settings
from datamedic.db.models import (
    AnalysisRun,
    Dashboard,
    DashboardWidget,
    Dataset,
    DatasetVersion,
    Job,
    TransformationRun,
)
from datamedic.errors import Conflict, DatasetNotReady, NotFound, QuotaExceeded, ValidationFailed
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import roles_snapshot
from datamedic.storage import ObjectStorage
from datamedic.versions.service import get_owned_version

CHART_TYPES = (
    "bar",
    "hbar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "area",
    "scatter",
    "bubble",
    "histogram",
    "box",
    "pie",
    "donut",
    "correlation_heatmap",
    "missingness_heatmap",
    "table",
)
CHART_KINDS = {
    "bar": "aggregate",
    "hbar": "aggregate",
    "grouped_bar": "aggregate",
    "stacked_bar": "aggregate",
    "line": "aggregate",
    "area": "aggregate",
    "pie": "aggregate",
    "donut": "aggregate",
    "table": "aggregate",
    "scatter": "scatter",
    "bubble": "scatter",
    "histogram": "histogram",
    "box": "box",
    "correlation_heatmap": "correlation",
    "missingness_heatmap": "missingness",
}
MAX_TEXT = 200


# ---- Columns and queries -------------------------------------------------------------------------


def catalogue(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
    version, ds = get_owned_version(db, owner_id, version_id)
    pol = data.policy(db, owner_id, ds)
    snap = roles_snapshot(db, owner_id, version.id)
    roles_by_col: dict[str, list[str]] = {}
    for role, cols in snap.get("roles", {}).items():
        for c in cols:
            roles_by_col.setdefault(c, []).append(role)
    cols = []
    for c in version.schema:
        name = str(c["name"])
        if name.startswith("_dm_"):
            continue
        n_missing = c.get("n_missing")
        cols.append(
            {
                "name": name,
                "inferred_type": c.get("inferred_type"),
                "dtype": c.get("dtype"),
                "n_unique": c.get("n_unique"),
                "n_missing": n_missing,
                "missing_pct": round(100 * n_missing / version.n_rows, 3)
                if n_missing is not None and version.n_rows
                else None,
                "roles": roles_by_col.get(name, []),
                "protected": name in pol.protected,
            }
        )
    return {
        "version_id": str(version.id),
        "version_number": version.version_number,
        "dataset_id": str(ds.id),
        "domain": ds.domain,
        "rows": version.n_rows,
        "columns": cols,
        "privacy_mode": pol.privacy_mode,
        "min_cell": pol.min_cell,
        "task_type": snap.get("task_type"),
        "roles": snap.get("roles", {}),
    }


def run_query(
    db: Session,
    settings: Settings,
    storage: ObjectStorage,
    owner_id: uuid.UUID,
    version_id: uuid.UUID,
    spec: dict[str, Any],
) -> dict[str, Any]:
    from datamedic.engine.analytics.query import QueryError, execute, validate

    version, ds = get_owned_version(db, owner_id, version_id)
    pol = data.policy(db, owner_id, ds)
    data.raise_errors(validate(spec, version.schema, pol))
    with data.Slot(settings):
        frame = data.frame(settings, storage, version)
        try:
            result = execute(frame, spec, version.schema, pol)
        except QueryError as exc:
            data.raise_errors(exc.errors)
            raise
    return {"version_id": str(version.id), "spec": spec, "result": result}


# ---- Version comparison ---------------------------------------------------------------


def compare(
    db: Session,
    settings: Settings,
    storage: ObjectStorage,
    owner_id: uuid.UUID,
    version_id: uuid.UUID,
    other_id: uuid.UUID,
) -> dict[str, Any]:
    """Side-by-side comparison of two versions of one dataset on shared bins."""
    import numpy as np
    import pandas as pd

    a_ver, ds = get_owned_version(db, owner_id, version_id)
    b_ver, ds_b = get_owned_version(db, owner_id, other_id)
    if ds.id != ds_b.id:
        raise ValidationFailed("Both versions must belong to the same dataset.")
    pol = data.policy(db, owner_id, ds)
    with data.Slot(settings):
        a = data.frame(settings, storage, a_ver)
        b = data.frame(settings, storage, b_ver)
        ca = [c for c in a.columns if not str(c).startswith("_dm_")]
        cb = [c for c in b.columns if not str(c).startswith("_dm_")]
        common = [c for c in ca if c in cb]
        ids_a = set(a["_dm_row_id"]) if "_dm_row_id" in a else set()
        ids_b = set(b["_dm_row_id"]) if "_dm_row_id" in b else set()
        columns = []
        for col in common[:40]:
            item: dict[str, Any] = {
                "column": col,
                "protected": col in pol.protected,
                "missing_pct": [
                    round(100 * float(a[col].isna().mean()), 3),
                    round(100 * float(b[col].isna().mean()), 3),
                ],
            }
            if col in pol.protected:
                columns.append(item)
                continue
            na_, nb_ = pd.api.types.is_numeric_dtype(a[col]), pd.api.types.is_numeric_dtype(b[col])
            if na_ and nb_ and not pd.api.types.is_bool_dtype(a[col]):
                xa = a[col].dropna().astype(float)
                xb = b[col].dropna().astype(float)
                both = pd.concat([xa, xb])
                if len(both):
                    lo, hi = float(both.min()), float(both.max())
                    edges = np.linspace(lo, hi if hi > lo else lo + 1, 21)
                    ha, _ = np.histogram(xa, bins=edges)
                    hb, _ = np.histogram(xb, bins=edges)
                    if pol.privacy_mode:
                        ha = [None if 0 < c < pol.min_cell else int(c) for c in ha]
                        hb = [None if 0 < c < pol.min_cell else int(c) for c in hb]
                    else:
                        ha, hb = [int(c) for c in ha], [int(c) for c in hb]

                    def stats(x: Any) -> dict[str, Any]:
                        if not len(x):
                            return {"n": 0}
                        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
                        iqr = q3 - q1
                        return {
                            "n": int(len(x)),
                            "mean": round(float(x.mean()), 6),
                            "sd": round(float(x.std(ddof=1)), 6) if len(x) > 1 else None,
                            "median": round(float(x.median()), 6),
                            "q1": round(q1, 6),
                            "q3": round(q3, 6),
                            "outliers_iqr": int(
                                ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
                            ),
                        }

                    item.update(
                        type="numeric",
                        edges=[round(float(e), 6) for e in edges],
                        histogram=[ha, hb],
                        stats=[stats(xa), stats(xb)],
                    )
            else:
                fa = a[col].dropna().astype(str).value_counts()
                fb = b[col].dropna().astype(str).value_counts()
                cats = list(dict.fromkeys(list(fa.index[:10]) + list(fb.index[:10])))[:15]

                def cnt(f: Any, k: str) -> int | None:
                    v = int(f.get(k, 0))
                    return None if pol.privacy_mode and 0 < v < pol.min_cell else v

                item.update(
                    type="categorical",
                    categories=[
                        {
                            "label": k
                            if not pol.privacy_mode
                            else f"category {i + 1}"
                            if col in pol.protected
                            else k,
                            "counts": [cnt(fa, k), cnt(fb, k)],
                        }
                        for i, k in enumerate(cats)
                    ],
                )
            columns.append(item)
        num = [
            c
            for c in common
            if c not in pol.protected
            and pd.api.types.is_numeric_dtype(a[c])
            and pd.api.types.is_numeric_dtype(b[c])
        ][:15]
        corr_changes = []
        if len(num) >= 2:
            ra = a[num].astype(float).corr(method="spearman", min_periods=10)
            rb = b[num].astype(float).corr(method="spearman", min_periods=10)
            for i, x in enumerate(num):
                for y in num[i + 1 :]:
                    va, vb = ra.loc[x, y], rb.loc[x, y]
                    if pd.notna(va) and pd.notna(vb) and abs(vb - va) >= 0.05:
                        corr_changes.append(
                            {"pair": [x, y], "rho": [round(float(va), 4), round(float(vb), 4)]}
                        )
            corr_changes.sort(key=lambda t: -abs(t["rho"][1] - t["rho"][0]))
    effects = None
    run = (
        db.get(TransformationRun, b_ver.transformation_run_id)
        if b_ver.transformation_run_id
        else None
    )
    if run is not None and run.source_version_id == a_ver.id:
        effects = []
        for step in run.step_reports or []:
            effects.append(
                {
                    "op": step.get("op"),
                    "rows_removed": step.get("rows_removed", 0),
                    "columns_added": step.get("columns_added", []),
                    "columns_removed": step.get("columns_removed", []),
                    "columns": [
                        {
                            k: c.get(k)
                            for k in (
                                "column",
                                "cells_filled",
                                "observed_changed",
                                "missing_before",
                                "missing_after",
                            )
                        }
                        for c in step.get("columns", [])
                    ],
                }
            )
    return {
        "versions": [
            {
                "id": str(v.id),
                "number": v.version_number,
                "rows": int(v.n_rows),
                "columns": int(v.n_cols),
                "kind": v.kind,
                "fitted_on_full_data": v.fitted_on_full_data,
            }
            for v in (a_ver, b_ver)
        ],
        "columns_added": [c for c in cb if c not in ca],
        "columns_removed": [c for c in ca if c not in cb],
        "rows": {
            "in_both": len(ids_a & ids_b),
            "only_first": len(ids_a - ids_b),
            "only_second": len(ids_b - ids_a),
        },
        "columns": columns,
        "correlation_changes": corr_changes[:10],
        "transformation_effects": effects,
        "privacy_mode": pol.privacy_mode,
        "notes": [
            "Histograms use the same 20 bins for both versions.",
            "cells_filled are estimated (imputed) values; observed_changed are re-expressed or "
            "capped observed values. Value changes alone do not show better predictive "
            "performance.",
        ],
    }


# ---- Insights ----------------------------------------------------------------------------------


def start_insights(
    db: Session, settings: Settings, owner_id: uuid.UUID, version_id: uuid.UUID
) -> Job:
    version, ds = get_owned_version(db, owner_id, version_id)
    if ds.status != "ready":
        raise DatasetNotReady("The dataset is not ready.")
    active = db.scalar(
        select(Job).where(
            Job.owner_id == owner_id,
            Job.type == "insights",
            Job.params["version_id"].astext == str(version.id),
            Job.status.in_(("queued", "running", "cancel_requested")),
        )
    )
    if active is not None:
        raise Conflict("Insights for this version are already being generated.")
    job = create_job(db, settings, owner_id, "insights", ds.id, {"version_id": str(version.id)})
    db.commit()
    return job


def latest_insights(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[AnalysisRun | None, Job | None, bool]:
    version, ds = get_owned_version(db, owner_id, version_id)
    run = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.dataset_version_id == version.id,
            AnalysisRun.kind == "insights",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )
    job = db.scalar(
        select(Job)
        .where(
            Job.owner_id == owner_id,
            Job.type == "insights",
            Job.params["version_id"].astext == str(version.id),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )
    stale = run is not None and run.semantics_hash != current_semantics_hash(
        db, owner_id, version.id, ds.domain
    )
    return run, job, stale


# ---- Dashboards -----------------------------------------------------------------------


def _text(value: Any, pointer: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValidationFailed(
            f"Text longer than {limit} characters.",
            errors=[
                {"pointer": pointer, "code": "INVALID", "message": f"At most {limit} characters."}
            ],
        )
    return value


def validate_widget(
    kind: str, spec: dict[str, Any], version: DatasetVersion, pol: Any
) -> list[dict[str, str]]:
    from datamedic.engine.analytics.query import validate

    errors: list[dict[str, str]] = []
    query = spec.get("query")
    display = spec.get("display") or {}
    if not isinstance(query, dict):
        return [
            {
                "pointer": "/spec/query",
                "code": "VALUE_REQUIRED",
                "message": "A widget needs a query.",
            }
        ]
    if not isinstance(display, dict):
        return [
            {"pointer": "/spec/display", "code": "INVALID", "message": "display must be an object."}
        ]
    qkind = query.get("kind", "aggregate")
    if kind == "kpi" and (
        qkind != "aggregate" or query.get("group_by") or len(query.get("measures", [])) != 1
    ):
        errors.append(
            {
                "pointer": "/spec/query",
                "code": "INVALID",
                "message": "A KPI card is one measure without grouping.",
            }
        )
    if kind == "pivot" and (qkind != "aggregate" or not 1 <= len(query.get("group_by", [])) <= 2):
        errors.append(
            {
                "pointer": "/spec/query",
                "code": "INVALID",
                "message": "A pivot table needs one or two groupings.",
            }
        )
    if kind == "chart":
        ct = display.get("chart_type")
        if ct not in CHART_TYPES:
            errors.append(
                {
                    "pointer": "/spec/display/chart_type",
                    "code": "INVALID",
                    "message": f"Chart types: {', '.join(CHART_TYPES)}.",
                }
            )
        elif CHART_KINDS[ct] != qkind:
            errors.append(
                {
                    "pointer": "/spec/display/chart_type",
                    "code": "INVALID",
                    "message": f"A {ct} chart needs a {CHART_KINDS[ct]} query.",
                }
            )
    for k, v in display.items():
        if isinstance(v, str) and len(v) > MAX_TEXT:
            errors.append(
                {"pointer": f"/spec/display/{k}", "code": "INVALID", "message": "Text too long."}
            )
        elif not isinstance(v, (str, bool, int, float)) and v is not None:
            errors.append(
                {
                    "pointer": f"/spec/display/{k}",
                    "code": "INVALID",
                    "message": "Display options are simple values.",
                }
            )
    errors += [
        {**e, "pointer": "/spec/query" + e["pointer"]} for e in validate(query, version.schema, pol)
    ]
    return errors


def get_dashboard(db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID) -> Dashboard:
    d = db.scalar(
        select(Dashboard).where(Dashboard.id == dashboard_id, Dashboard.owner_id == owner_id)
    )
    if d is None:
        raise NotFound()
    get_owned_version(db, owner_id, d.dataset_version_id)  # dataset not deleted
    return d


def widgets(db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID) -> list[DashboardWidget]:
    return list(
        db.scalars(
            select(DashboardWidget)
            .where(
                DashboardWidget.owner_id == owner_id, DashboardWidget.dashboard_id == dashboard_id
            )
            .order_by(DashboardWidget.position, DashboardWidget.created_at)
        )
    )


def list_dashboards(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID
) -> list[tuple[Dashboard, int]]:
    from datamedic.datasets.service import get_owned as get_dataset

    ds = get_dataset(db, owner_id, dataset_id)
    rows = db.execute(
        select(Dashboard, func.count(DashboardWidget.id))
        .join(DashboardWidget, DashboardWidget.dashboard_id == Dashboard.id, isouter=True)
        .where(Dashboard.owner_id == owner_id, Dashboard.dataset_id == ds.id)
        .group_by(Dashboard.id)
        .order_by(Dashboard.updated_at.desc())
    ).all()
    return [(d, int(n)) for d, n in rows]


def _check_filters(filters: Any, version: DatasetVersion, pol: Any) -> list[dict[str, Any]]:
    from datamedic.engine.analytics.query import validate

    if not isinstance(filters, list):
        raise ValidationFailed("filters must be a list.")
    errors = validate(
        {"kind": "aggregate", "filters": filters, "measures": [{"agg": "count"}]},
        version.schema,
        pol,
    )
    data.raise_errors(errors)
    return filters


def create_dashboard(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    body: dict[str, Any],
) -> Dashboard:
    version, ds = get_owned_version(db, owner_id, uuid.UUID(str(body["version_id"])))
    if ds.id != dataset_id:
        raise NotFound()
    count = db.scalar(
        select(func.count())
        .select_from(Dashboard)
        .where(Dashboard.owner_id == owner_id, Dashboard.dataset_id == ds.id)
    )
    if int(count or 0) >= settings.analytics_max_dashboards_per_dataset:
        raise QuotaExceeded(
            f"At most {settings.analytics_max_dashboards_per_dataset} dashboards per dataset."
        )
    pol = data.policy(db, owner_id, ds)
    d = Dashboard(
        owner_id=owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        name=_text(body["name"], "/name").strip() or "Dashboard",
        description=_text(body.get("description") or "", "/description", 2000),
        filters=_check_filters(body.get("filters") or [], version, pol),
    )
    db.add(d)
    db.commit()
    return d


def update_dashboard(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, body: dict[str, Any]
) -> Dashboard:
    d = get_dashboard(db, owner_id, dashboard_id)
    version, ds = get_owned_version(db, owner_id, d.dataset_version_id)
    if body.get("name") is not None:
        d.name = _text(body["name"], "/name").strip() or d.name
    if body.get("description") is not None:
        d.description = _text(body["description"], "/description", 2000)
    if body.get("filters") is not None:
        d.filters = _check_filters(body["filters"], version, data.policy(db, owner_id, ds))
    d.updated_at = datetime.now(UTC)
    db.commit()
    return d


def delete_dashboard(db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID) -> None:
    d = get_dashboard(db, owner_id, dashboard_id)
    db.delete(d)
    db.commit()


def compatibility(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, version_id: uuid.UUID
) -> dict[str, Any]:
    d = get_dashboard(db, owner_id, dashboard_id)
    version, ds = get_owned_version(db, owner_id, version_id)
    if ds.id != d.dataset_id:
        raise ValidationFailed("The version belongs to another dataset.")
    pol = data.policy(db, owner_id, ds)
    items = []
    for w in widgets(db, owner_id, d.id):
        errs = validate_widget(w.kind, w.spec, version, pol)
        items.append(
            {
                "widget_id": str(w.id),
                "title": w.title,
                "compatible": not errs,
                "problems": [e["message"] for e in errs],
            }
        )
    ferrs: list[str] = []
    try:
        _check_filters(d.filters, version, pol)
    except ValidationFailed as exc:
        ferrs = [e["message"] for e in (exc.errors or [])]
    return {
        "dashboard_id": str(d.id),
        "current_version_id": str(d.dataset_version_id),
        "version_id": str(version.id),
        "version_number": version.version_number,
        "widgets": items,
        "filter_problems": ferrs,
        "compatible": all(i["compatible"] for i in items) and not ferrs,
    }


def switch_version(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, version_id: uuid.UUID
) -> Dashboard:
    """Explicit, user-confirmed rebind; refused unless every widget and filter is compatible."""
    report = compatibility(db, owner_id, dashboard_id, version_id)
    if not report["compatible"]:
        raise ValidationFailed(
            "Some widgets or filters are not compatible with that version; duplicate the "
            "dashboard onto it instead, or fix them first.",
            errors=[
                {"pointer": "/version_id", "code": "INCOMPATIBLE", "message": m}
                for w in report["widgets"]
                for m in w["problems"]
            ]
            + [
                {"pointer": "/filters", "code": "INCOMPATIBLE", "message": m}
                for m in report["filter_problems"]
            ],
        )
    d = get_dashboard(db, owner_id, dashboard_id)
    d.dataset_version_id = version_id
    d.updated_at = datetime.now(UTC)
    db.commit()
    return d


def duplicate_dashboard(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    dashboard_id: uuid.UUID,
    body: dict[str, Any],
) -> tuple[Dashboard, list[str]]:
    """Copy onto the same or another version; incompatible widgets are skipped and named."""
    src = get_dashboard(db, owner_id, dashboard_id)
    target_id = uuid.UUID(str(body.get("version_id") or src.dataset_version_id))
    report = compatibility(db, owner_id, src.id, target_id)
    count = db.scalar(
        select(func.count())
        .select_from(Dashboard)
        .where(Dashboard.owner_id == owner_id, Dashboard.dataset_id == src.dataset_id)
    )
    if int(count or 0) >= settings.analytics_max_dashboards_per_dataset:
        raise QuotaExceeded(
            f"At most {settings.analytics_max_dashboards_per_dataset} dashboards per dataset."
        )
    name = _text(body.get("name") or f"Copy of {src.name}"[:200], "/name")
    d = Dashboard(
        owner_id=owner_id,
        dataset_id=src.dataset_id,
        dataset_version_id=target_id,
        name=name,
        description=src.description,
        filters=src.filters if not report["filter_problems"] else [],
        source_dashboard_id=src.id,
    )
    db.add(d)
    db.flush()
    ok = {i["widget_id"] for i in report["widgets"] if i["compatible"]}
    skipped = []
    for w in widgets(db, owner_id, src.id):
        if str(w.id) not in ok:
            skipped.append(w.title)
            continue
        db.add(
            DashboardWidget(
                owner_id=owner_id,
                dataset_id=src.dataset_id,
                dashboard_id=d.id,
                position=w.position,
                kind=w.kind,
                title=w.title,
                size=w.size,
                spec=w.spec,
            )
        )
    db.commit()
    return d, skipped


def add_widget(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    dashboard_id: uuid.UUID,
    body: dict[str, Any],
) -> DashboardWidget:
    d = get_dashboard(db, owner_id, dashboard_id)
    version, ds = get_owned_version(db, owner_id, d.dataset_version_id)
    existing = widgets(db, owner_id, d.id)
    if len(existing) >= settings.analytics_max_widgets_per_dashboard:
        raise QuotaExceeded(
            f"At most {settings.analytics_max_widgets_per_dashboard} widgets per dashboard."
        )
    data.raise_errors(
        validate_widget(body["kind"], body["spec"], version, data.policy(db, owner_id, ds))
    )
    w = DashboardWidget(
        owner_id=owner_id,
        dataset_id=d.dataset_id,
        dashboard_id=d.id,
        position=(max((x.position for x in existing), default=-1) + 1),
        kind=body["kind"],
        title=_text(body["title"], "/title").strip() or "Widget",
        size=body.get("size") or "medium",
        spec=body["spec"],
    )
    db.add(w)
    d.updated_at = datetime.now(UTC)
    db.commit()
    return w


def _widget(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, widget_id: uuid.UUID
) -> tuple[Dashboard, DashboardWidget]:
    d = get_dashboard(db, owner_id, dashboard_id)
    w = db.scalar(
        select(DashboardWidget).where(
            DashboardWidget.id == widget_id,
            DashboardWidget.owner_id == owner_id,
            DashboardWidget.dashboard_id == d.id,
        )
    )
    if w is None:
        raise NotFound()
    return d, w


def update_widget(
    db: Session,
    owner_id: uuid.UUID,
    dashboard_id: uuid.UUID,
    widget_id: uuid.UUID,
    body: dict[str, Any],
) -> DashboardWidget:
    d, w = _widget(db, owner_id, dashboard_id, widget_id)
    if body.get("spec") is not None or body.get("kind") is not None:
        version, ds = get_owned_version(db, owner_id, d.dataset_version_id)
        kind = body.get("kind") or w.kind
        spec = body.get("spec") or w.spec
        data.raise_errors(validate_widget(kind, spec, version, data.policy(db, owner_id, ds)))
        w.kind, w.spec = kind, spec
    if body.get("title") is not None:
        w.title = _text(body["title"], "/title").strip() or w.title
    if body.get("size") is not None:
        w.size = body["size"]
    w.updated_at = d.updated_at = datetime.now(UTC)
    db.commit()
    return w


def delete_widget(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, widget_id: uuid.UUID
) -> None:
    d, w = _widget(db, owner_id, dashboard_id, widget_id)
    db.delete(w)
    d.updated_at = datetime.now(UTC)
    db.commit()


def duplicate_widget(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    dashboard_id: uuid.UUID,
    widget_id: uuid.UUID,
) -> DashboardWidget:
    _, w = _widget(db, owner_id, dashboard_id, widget_id)
    return add_widget(
        db,
        settings,
        owner_id,
        dashboard_id,
        {"kind": w.kind, "title": f"{w.title} (copy)"[:200], "size": w.size, "spec": w.spec},
    )


def reorder(
    db: Session, owner_id: uuid.UUID, dashboard_id: uuid.UUID, widget_ids: list[uuid.UUID]
) -> list[DashboardWidget]:
    d = get_dashboard(db, owner_id, dashboard_id)
    current = widgets(db, owner_id, d.id)
    if sorted(str(w.id) for w in current) != sorted(str(i) for i in widget_ids):
        raise ValidationFailed("Give every widget of the dashboard exactly once.")
    pos = {str(i): n for n, i in enumerate(widget_ids)}
    for w in current:
        w.position = pos[str(w.id)]
    d.updated_at = datetime.now(UTC)
    db.commit()
    return widgets(db, owner_id, d.id)


def summaries_for_version(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> list[dict[str, Any]]:
    """Saved dashboard configurations bound to a version (for the Data Passport; no queries)."""
    rows = db.execute(
        select(Dashboard, func.count(DashboardWidget.id))
        .join(DashboardWidget, DashboardWidget.dashboard_id == Dashboard.id, isouter=True)
        .where(Dashboard.owner_id == owner_id, Dashboard.dataset_version_id == version_id)
        .group_by(Dashboard.id)
        .order_by(Dashboard.updated_at.desc())
        .limit(20)
    ).all()
    return [
        {
            "name": d.name,
            "description": d.description,
            "widgets": int(n),
            "updated_at": d.updated_at.isoformat(),
        }
        for d, n in rows
    ]


__all__ = ["Dataset"]
