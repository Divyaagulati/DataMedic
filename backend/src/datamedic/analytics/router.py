"""Analytics Studio API (owner only; responses never cached; no endpoint writes versions)."""

import csv
import io
import json
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Response, status
from pydantic import BaseModel, Field

from datamedic.analytics import service
from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.db.models import Dashboard, DashboardWidget
from datamedic.errors import ValidationFailed
from datamedic.exports.jobs import neutralise
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.reports.router import safe_filename
from datamedic.versions.service import get_owned_version

router = APIRouter(prefix="/api/v1", tags=["analytics"])


class QueryIn(BaseModel):
    """A structured analytical query (see docs/API_SPECIFICATION.md §18). No SQL or code."""

    spec: dict[str, Any]


class DashboardIn(BaseModel):
    version_id: uuid.UUID
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    filters: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class DashboardPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    filters: list[dict[str, Any]] | None = Field(default=None, max_length=20)


class DuplicateIn(BaseModel):
    version_id: uuid.UUID | None = None
    name: str | None = Field(default=None, min_length=1, max_length=200)


class SwitchIn(BaseModel):
    version_id: uuid.UUID
    confirm: bool = False


class WidgetIn(BaseModel):
    kind: Literal["chart", "kpi", "pivot"]
    title: str = Field(min_length=1, max_length=200)
    size: Literal["small", "medium", "large"] = "medium"
    spec: dict[str, Any]


class WidgetPatch(BaseModel):
    kind: Literal["chart", "kpi", "pivot"] | None = None
    title: str | None = Field(default=None, min_length=1, max_length=200)
    size: Literal["small", "medium", "large"] | None = None
    spec: dict[str, Any] | None = None


class ReorderIn(BaseModel):
    widget_ids: list[uuid.UUID] = Field(max_length=100)


class WidgetOut(BaseModel):
    id: uuid.UUID
    kind: str
    title: str
    size: str
    position: int
    spec: dict[str, Any]

    @classmethod
    def build(cls, w: DashboardWidget) -> "WidgetOut":
        return cls(
            id=w.id, kind=w.kind, title=w.title, size=w.size, position=w.position, spec=w.spec
        )


class DashboardOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_id: uuid.UUID
    name: str
    description: str
    filters: list[dict[str, Any]]
    source_dashboard_id: uuid.UUID | None
    created_at: str
    updated_at: str
    widget_count: int
    widgets: list[WidgetOut] | None = None

    @classmethod
    def build(
        cls, d: Dashboard, widgets: list[DashboardWidget] | None = None, count: int | None = None
    ) -> "DashboardOut":
        return cls(
            id=d.id,
            dataset_id=d.dataset_id,
            version_id=d.dataset_version_id,
            name=d.name,
            description=d.description,
            filters=d.filters,
            source_dashboard_id=d.source_dashboard_id,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat(),
            widget_count=count if count is not None else len(widgets or []),
            widgets=[WidgetOut.build(w) for w in widgets] if widgets is not None else None,
        )


# ---- Columns, queries, exports ------------------------------------------------------------------


@router.get("/versions/{version_id}/analytics/columns")
def get_columns(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    """Column browser: names, types, roles, missingness and which columns are protected."""
    return service.catalogue(db, principal.user.id, version_id)


@router.post("/versions/{version_id}/analytics/query")
def post_query(
    version_id: uuid.UUID, body: QueryIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> dict[str, Any]:
    """Run a validated structured query (read-only; bounded; privacy rules applied)."""
    return service.run_query(
        db, ctx.settings, ctx.storage, principal.user.id, version_id, body.spec
    )


def _csv_rows(result: dict[str, Any]) -> list[list[Any]]:
    if result.get("suppressed"):
        return [["note"], [result.get("note")]]
    kind = result.get("kind")
    if kind == "aggregate":
        header = (
            list(result["group_columns"]) + ["n"] + list(result["measure_labels"]) + ["suppressed"]
        )
        rows = [header]
        for r in result["rows"]:
            rows.append(
                list(r["groups"]) + [r["n"]] + list(r["values"]) + [bool(r.get("suppressed"))]
            )
        rows.append(
            ["TOTAL"] * len(result["group_columns"])
            + [result["totals"]["n"]]
            + list(result["totals"]["values"])
            + [False]
        )
        return rows
    if kind == "histogram":
        e, c = result["edges"], result["counts"]
        return [["bin_low", "bin_high", "count"]] + [[e[i], e[i + 1], c[i]] for i in range(len(c))]
    if kind == "box":
        keys = [
            "label",
            "n",
            "mean",
            "q1",
            "median",
            "q3",
            "whisker_low",
            "whisker_high",
            "outliers",
        ]
        return [keys] + [[i.get(k) for k in keys] for i in result["items"]]
    if kind == "correlation":
        cols = result["columns"]
        return [[""] + cols] + [[cols[i]] + row for i, row in enumerate(result["matrix"])]
    if kind == "missingness":
        return [["column", "missing_pct"]] + [
            list(t) for t in zip(result["columns"], result["missing_pct"], strict=True)
        ]
    raise ValidationFailed(
        "Row-level results (scatter points) are not exportable; export an aggregate instead."
    )


@router.post(
    "/versions/{version_id}/analytics/export",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
def export_query(
    version_id: uuid.UUID, body: QueryIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> Response:
    """Aggregated query or pivot result as CSV (formula-neutralised; never row-level data)."""
    out = service.run_query(db, ctx.settings, ctx.storage, principal.user.id, version_id, body.spec)
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in _csv_rows(out["result"]):
        writer.writerow([neutralise(v) if isinstance(v, str) else v for v in row])
    version, ds = get_owned_version(db, principal.user.id, version_id)
    name = safe_filename(
        ds.name, f"_v{version.version_number}_{out['result'].get('kind', 'query')}.csv"
    )
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/versions/{version_id}/compare/{other_version_id}")
def get_compare(
    version_id: uuid.UUID,
    other_version_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
) -> dict[str, Any]:
    """Visual comparison of two versions of one dataset (shared bins; read-only)."""
    return service.compare(
        db, ctx.settings, ctx.storage, principal.user.id, version_id, other_version_id
    )


# ---- Insights -----------------------------------------------------------------------------------


@router.post("/versions/{version_id}/insights", status_code=status.HTTP_202_ACCEPTED)
def start_insights(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> dict[str, Any]:
    job = service.start_insights(db, ctx.settings, principal.user.id, version_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return {"job": JobOut.from_job(job).model_dump(mode="json")}


@router.get("/versions/{version_id}/insights")
def get_insights(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    run, job, stale = service.latest_insights(db, principal.user.id, version_id)
    return {
        "summary": run.summary if run else None,
        "created_at": run.created_at.isoformat() if run else None,
        "stale": stale,
        "job": JobOut.from_job(job).model_dump(mode="json") if job else None,
    }


# ---- Dashboards ---------------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/dashboards")
def list_dashboards(dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    return {
        "items": [
            DashboardOut.build(d, count=n).model_dump(mode="json")
            for d, n in service.list_dashboards(db, principal.user.id, dataset_id)
        ]
    }


@router.post("/datasets/{dataset_id}/dashboards", status_code=status.HTTP_201_CREATED)
def create_dashboard(
    dataset_id: uuid.UUID, body: DashboardIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> DashboardOut:
    d = service.create_dashboard(
        db, ctx.settings, principal.user.id, dataset_id, body.model_dump(mode="json")
    )
    return DashboardOut.build(d, [])


@router.get("/dashboards/{dashboard_id}")
def get_dashboard(dashboard_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> DashboardOut:
    d = service.get_dashboard(db, principal.user.id, dashboard_id)
    return DashboardOut.build(d, service.widgets(db, principal.user.id, d.id))


@router.patch("/dashboards/{dashboard_id}")
def patch_dashboard(
    dashboard_id: uuid.UUID, body: DashboardPatch, principal: PrincipalDep, db: DbDep
) -> DashboardOut:
    d = service.update_dashboard(
        db, principal.user.id, dashboard_id, body.model_dump(exclude_none=True)
    )
    return DashboardOut.build(d, service.widgets(db, principal.user.id, d.id))


@router.delete("/dashboards/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dashboard(dashboard_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> Response:
    service.delete_dashboard(db, principal.user.id, dashboard_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/dashboards/{dashboard_id}/duplicate", status_code=status.HTTP_201_CREATED)
def duplicate_dashboard(
    dashboard_id: uuid.UUID, body: DuplicateIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> dict[str, Any]:
    d, skipped = service.duplicate_dashboard(
        db,
        ctx.settings,
        principal.user.id,
        dashboard_id,
        body.model_dump(mode="json", exclude_none=True),
    )
    return {
        "dashboard": DashboardOut.build(d, service.widgets(db, principal.user.id, d.id)).model_dump(
            mode="json"
        ),
        "skipped_widgets": skipped,
    }


@router.get("/dashboards/{dashboard_id}/compatibility")
def get_compatibility(
    dashboard_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    version_id: Annotated[uuid.UUID | None, Query()] = None,
) -> dict[str, Any]:
    d = service.get_dashboard(db, principal.user.id, dashboard_id)  # 404 before validation
    return service.compatibility(db, principal.user.id, d.id, version_id or d.dataset_version_id)


@router.post("/dashboards/{dashboard_id}/switch-version")
def switch_version(
    dashboard_id: uuid.UUID, body: SwitchIn, principal: PrincipalDep, db: DbDep
) -> DashboardOut:
    """Rebind the dashboard to another version of its dataset: only on explicit confirmation and
    only when every widget and filter is compatible (never silent)."""
    if not body.confirm:
        raise ValidationFailed(
            "Confirm the version switch explicitly.",
            errors=[
                {
                    "pointer": "/confirm",
                    "code": "CONFIRM_REQUIRED",
                    "message": "Set confirm to true.",
                }
            ],
        )
    d = service.switch_version(db, principal.user.id, dashboard_id, body.version_id)
    return DashboardOut.build(d, service.widgets(db, principal.user.id, d.id))


@router.get(
    "/dashboards/{dashboard_id}/export",
    response_class=Response,
    responses={200: {"content": {"application/json": {}}}},
)
def export_dashboard(dashboard_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> Response:
    """The dashboard specification as JSON (configuration only; no data)."""
    d = service.get_dashboard(db, principal.user.id, dashboard_id)
    payload = DashboardOut.build(d, service.widgets(db, principal.user.id, d.id)).model_dump(
        mode="json"
    )
    payload["format"] = "datamedic.dashboard/1"
    name = safe_filename(d.name, "_dashboard.json")
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/dashboards/{dashboard_id}/widgets", status_code=status.HTTP_201_CREATED)
def add_widget(
    dashboard_id: uuid.UUID, body: WidgetIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> WidgetOut:
    return WidgetOut.build(
        service.add_widget(
            db, ctx.settings, principal.user.id, dashboard_id, body.model_dump(mode="json")
        )
    )


@router.patch("/dashboards/{dashboard_id}/widgets/{widget_id}")
def patch_widget(
    dashboard_id: uuid.UUID,
    widget_id: uuid.UUID,
    body: WidgetPatch,
    principal: PrincipalDep,
    db: DbDep,
) -> WidgetOut:
    return WidgetOut.build(
        service.update_widget(
            db,
            principal.user.id,
            dashboard_id,
            widget_id,
            body.model_dump(mode="json", exclude_none=True),
        )
    )


@router.delete(
    "/dashboards/{dashboard_id}/widgets/{widget_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_widget(
    dashboard_id: uuid.UUID, widget_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> Response:
    service.delete_widget(db, principal.user.id, dashboard_id, widget_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/dashboards/{dashboard_id}/widgets/{widget_id}/duplicate", status_code=status.HTTP_201_CREATED
)
def duplicate_widget(
    dashboard_id: uuid.UUID, widget_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> WidgetOut:
    return WidgetOut.build(
        service.duplicate_widget(db, ctx.settings, principal.user.id, dashboard_id, widget_id)
    )


@router.post("/dashboards/{dashboard_id}/reorder")
def reorder(
    dashboard_id: uuid.UUID, body: ReorderIn, principal: PrincipalDep, db: DbDep
) -> dict[str, Any]:
    ws = service.reorder(db, principal.user.id, dashboard_id, body.widget_ids)
    return {"widgets": [WidgetOut.build(w).model_dump(mode="json") for w in ws]}
