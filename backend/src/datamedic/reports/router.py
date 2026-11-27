"""Data Passport, Copilot and exports (owner only; responses are never cached)."""

import json
import re
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.db.models import Artifact
from datamedic.errors import NotFound, ValidationFailed
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.jobs.service import create_job, get_owned_job
from datamedic.reports import copilot, passport
from datamedic.training import privacy
from datamedic.versions.service import get_owned_version

router = APIRouter(prefix="/api/v1", tags=["reports"])
NO_STORE: dict[str, str] = {}  # the security middleware adds no-store and nosniff


def safe_filename(name: str, suffix: str) -> str:
    """ASCII letters, digits, dot, dash and underscore only; never a path."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:60] or "dataset"
    return f"{stem}{suffix}"


class CopilotIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ExportIn(BaseModel):
    format: Literal["csv"] = "csv"
    # Row-level data leaves the platform: always an explicit acknowledgement.
    acknowledged: bool = False
    include_sensitive: bool = False
    acknowledge_sensitive: bool = False


class ExportAcceptedOut(BaseModel):
    job: JobOut
    excluded_sensitive_columns: list[str]


@router.get("/versions/{version_id}/passport")
def get_passport(
    version_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    download: bool = Query(default=False),
) -> Response:
    """The Data Passport as structured JSON (``download=true``: as an attachment)."""
    data = passport.build(db, principal.user.id, version_id)
    headers = dict(NO_STORE)
    if download:
        name = safe_filename(
            data["dataset"]["name"], f"_v{data['version']['number']}_passport.json"
        )
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return Response(
        content=json.dumps(data, indent=2, default=str),
        media_type="application/json",
        headers=headers,
    )


@router.post("/versions/{version_id}/copilot")
def ask_copilot(
    version_id: uuid.UUID, body: CopilotIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> dict[str, Any]:
    """Built-in, rule-based assistant over this version's stored results. Read-only."""
    version, ds = get_owned_version(db, principal.user.id, version_id)
    analysis = _analytical_answer(db, ctx, principal.user.id, version, body.question)
    if analysis is not None:
        return analysis
    data = passport.build(db, principal.user.id, version.id)
    return copilot.answer(data, body.question, f"/datasets/{ds.id}")


def _analytical_answer(
    db: Any, ctx: Any, owner_id: uuid.UUID, version: Any, question: str
) -> dict[str, Any] | None:
    """Natural-language analytics: question -> validated spec -> engine -> explanation. Every
    number comes from the query engine; nothing is sent to an external service."""
    from datamedic.analytics import nl
    from datamedic.analytics import service as analytics

    cat = analytics.catalogue(db, owner_id, version.id)
    parsed = nl.parse(question, cat["columns"], cat["roles"])
    if parsed is None:
        return None
    base = f"/datasets/{cat['dataset_id']}"
    reply: dict[str, Any] = {
        "assistant": copilot.ASSISTANT,
        "topic": "analytics",
        "answer": [],
        "references": [
            {"section": "analytics", "label": "Analytics Studio", "href": f"{base}/analytics"}
        ],
        "suggestions": [s for s in nl_suggestions(cat) if s.lower() != question.lower()][:4],
    }
    if "status" in parsed:
        reply["answer"] = [parsed["message"]]
        reply["analysis"] = {"status": parsed["status"], "options": parsed.get("options", [])}
        return reply
    try:
        if parsed["intent"] == "compare_versions":
            if version.parent_version_id is None:
                reply["answer"] = [
                    "This version has no parent version to compare with (it is the original "
                    "upload)."
                ]
                return reply
            cmp = analytics.compare(
                db, ctx.settings, ctx.storage, owner_id, version.parent_version_id, version.id
            )
            col = next((c for c in cmp["columns"] if c["column"] == parsed["column"]), None)
            if col is None or col.get("type") != "numeric":
                reply["answer"] = [
                    f"'{parsed['column']}' is not a numerical column present in both versions."
                ]
                return reply
            a, b = col["stats"]
            reply["answer"] = [
                f"Before (v{cmp['versions'][0]['number']}): mean {a.get('mean')}, median "
                f"{a.get('median')}, "
                f"{col['missing_pct'][0]}% missing, {a.get('outliers_iqr')} value(s) beyond 1.5 × "
                f"IQR.",
                f"After (v{cmp['versions'][1]['number']}): mean {b.get('mean')}, median "
                f"{b.get('median')}, "
                f"{col['missing_pct'][1]}% missing, {b.get('outliers_iqr')} value(s) beyond 1.5 × "
                f"IQR.",
                "Both histograms use the same bins. Changed values alone do not show better "
                "predictive performance.",
            ]
            reply["analysis"] = {
                "intent": "compare_versions",
                "interpretation": parsed["interpretation"],
                "comparison": col,
                "versions": cmp["versions"],
                "chart": {"type": "compare_histogram"},
            }
            return reply
        out = analytics.run_query(
            db, ctx.settings, ctx.storage, owner_id, version.id, parsed["spec"]
        )
        extra: dict[str, Any] = {}
        if parsed.get("box_spec"):
            extra["box"] = analytics.run_query(
                db, ctx.settings, ctx.storage, owner_id, version.id, parsed["box_spec"]
            )["result"]
        if parsed.get("correlation_spec"):
            extra["correlation"] = analytics.run_query(
                db, ctx.settings, ctx.storage, owner_id, version.id, parsed["correlation_spec"]
            )["result"]
    except ValidationFailed as exc:
        reply["answer"] = [
            "I cannot run that analysis: " + "; ".join(e["message"] for e in (exc.errors or []))
            or str(exc)
        ]
        reply["analysis"] = {"status": "invalid"}
        return reply
    reply["answer"] = [f"Interpreted as: {parsed['interpretation']}"] + nl.explain(
        parsed, out["result"], extra
    )
    reply["analysis"] = {
        "intent": parsed["intent"],
        "interpretation": parsed["interpretation"],
        "spec": parsed["spec"],
        "result": out["result"],
        "chart": parsed.get("chart"),
        "extra": extra,
    }
    return reply


def nl_suggestions(cat: dict[str, Any]) -> list[str]:
    cols = [c for c in cat["columns"] if not c["protected"]]
    num = [c["name"] for c in cols if c["inferred_type"] == "numeric"]
    catg = [c["name"] for c in cols if c["inferred_type"] in ("categorical", "boolean")]
    out = ["Which numerical column has the most missing values?"]
    if num and catg:
        out.append(f"Show average {num[0]} by {catg[0]}")
    if len(num) >= 2:
        out.append(f"Show the relationship between {num[0]} and {num[1]}")
    if num:
        out.append(f"Create a histogram of {num[0]}")
    if any(c["inferred_type"] == "datetime" for c in cols):
        out.append("Generate a monthly trend chart")
    out += [
        "Explain the strongest correlations",
        "Compare model performance",
        "Explain the reliability warnings",
    ]
    return out


@router.post("/versions/{version_id}/exports", status_code=status.HTTP_202_ACCEPTED)
def start_export(
    version_id: uuid.UUID, body: ExportIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ExportAcceptedOut:
    """Queue a CSV export of the version. Sensitive columns are dropped unless explicitly
    included with a second acknowledgement. The source version is never modified."""
    version, ds = get_owned_version(db, principal.user.id, version_id)
    if not body.acknowledged:
        raise ValidationFailed(
            "Confirm that the export contains row-level data.",
            errors=[
                {
                    "pointer": "/acknowledged",
                    "code": "ACK_REQUIRED",
                    "message": "Acknowledge that the file contains row-level data.",
                }
            ],
        )
    sensitive = sorted(
        set(privacy.protected_columns(db, principal.user.id, ds.id))
        & {c["name"] for c in version.schema}
    )
    if body.include_sensitive and sensitive and not body.acknowledge_sensitive:
        raise ValidationFailed(
            "Including sensitive columns needs a separate acknowledgement.",
            errors=[
                {
                    "pointer": "/acknowledge_sensitive",
                    "code": "SENSITIVE_ACK_REQUIRED",
                    "message": "Acknowledge that sensitive columns will be exported.",
                }
            ],
        )
    job = create_job(
        db,
        ctx.settings,
        principal.user.id,
        "export",
        ds.id,
        {
            "version_id": str(version.id),
            "format": "csv",
            "acknowledged": True,
            "include_sensitive": bool(body.include_sensitive),
        },
    )
    db.commit()
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return ExportAcceptedOut(
        job=JobOut.from_job(job),
        excluded_sensitive_columns=[] if body.include_sensitive else sensitive,
    )


@router.get(
    "/exports/{job_id}/download",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/csv": {}}}},
)
def download_export(
    job_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> StreamingResponse:
    """Stream a finished export (owner only, never cached; no storage URL is exposed)."""
    job = get_owned_job(db, principal.user.id, job_id)
    result = job.result or {}
    if job.type != "export" or job.status != "succeeded" or not result.get("artifact_id"):
        raise NotFound()
    artifact_id = uuid.UUID(str(result["artifact_id"]))
    artifact = db.scalar(
        select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.owner_id == principal.user.id,
            Artifact.kind == "export",
        )
    )
    if artifact is None:
        raise NotFound()
    version, ds = get_owned_version(db, principal.user.id, artifact.owner_ref_id)
    name = safe_filename(ds.name, f"_v{version.version_number}.csv")
    reader = ctx.storage.open_reader(artifact.object_key)

    def chunks() -> Any:
        with reader:
            while block := reader.read(1 << 16):
                yield block

    return StreamingResponse(
        chunks(),
        media_type="text/csv; charset=utf-8",
        headers={**NO_STORE, "Content-Disposition": f'attachment; filename="{name}"'},
    )
