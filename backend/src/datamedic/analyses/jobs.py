"""Analysis job handler (worker). Runs the registered checks for the dataset's domain over a
version and stores the run and its findings. Never writes versions (invariant 9)."""

import hashlib
import io
import uuid

from sqlalchemy import select

from datamedic.analyses.semantics import check_set_hash, roles_snapshot, semantics_hash
from datamedic.db.models import AnalysisRun, ColumnRole, Dataset, DatasetVersion, FindingRow
from datamedic.errors import INTEGRITY_FAILED
from datamedic.storage import ObjectNotFound
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

ANALYSIS_SEED = 0


@register_handler("analysis")
def analysis(hctx: HandlerContext) -> JobOutcome:
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    try:
        version_id = uuid.UUID(str(job.params.get("version_id")))
    except ValueError:
        return JobOutcome(False, error_code="VALIDATION_FAILED", error_message="Bad job params.")
    version = db.get(DatasetVersion, version_id)
    ds = db.get(Dataset, version.dataset_id) if version else None
    if version is None or ds is None or ds.deleted_at is not None or version.deleted_at:
        return JobOutcome(True, result={"skipped": "version deleted"})
    try:
        with storage.open_reader(version.object_key) as fh:
            raw = fh.read()
    except ObjectNotFound:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="Version data missing.")
    if hashlib.sha256(raw).digest() != version.file_sha256:
        return JobOutcome(
            False,
            error_code=INTEGRITY_FAILED,
            error_message="Stored version does not match its recorded hash.",
        )

    import pandas as pd

    from datamedic.domains.bootstrap import build_registry
    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.checks.framework import CheckContext, RoleMap, run_checks, summarise
    from datamedic.engine.checks.spec import GENERIC
    from datamedic.engine.versions import library_versions

    roles = list(db.scalars(select(ColumnRole).where(ColumnRole.dataset_version_id == version.id)))
    snapshot = roles_snapshot(roles)
    role_map = RoleMap()
    for role in roles:
        role_map.columns.setdefault(role.role, []).append(role.column_name)
        if role.role == "target":
            role_map.task_type = role.task_type
    registry = build_registry()
    domain = ds.domain if ds.domain in registry.domain_keys() else GENERIC
    specs = registry.checks_for(domain)

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    ctx = CheckContext(
        frame=frame,
        schema=version.schema,
        roles=role_map,
        domain=domain,
        role_specs={r.key: r for r in registry.all_roles()},
        seed=ANALYSIS_SEED,
    )
    findings = run_checks(specs, ctx)
    summary = summarise(findings)
    run = AnalysisRun(
        owner_id=version.owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        job_id=job.id,
        domain=domain,
        check_set_hash=check_set_hash(specs),
        semantics_hash=semantics_hash(ds.domain, snapshot),
        roles_snapshot=snapshot,
        config={
            "seed": ANALYSIS_SEED,
            "expensive_rows": ctx.budget.expensive_rows,
            "moderate_rows": ctx.budget.moderate_rows,
            "checks": [f"{s.id}@{s.version}" for s in specs],
        },
        summary=summary,
        engine_version=ENGINE_VERSION,
        library_versions={**library_versions(), **_analysis_libraries()},
    )
    db.add(run)
    db.flush()
    for position, f in enumerate(findings):
        db.add(
            FindingRow(
                owner_id=version.owner_id,
                dataset_id=ds.id,
                analysis_run_id=run.id,
                position=position,
                **f.as_dict(),
            )
        )
    db.flush()
    return JobOutcome(
        True,
        result={
            "analysis_run_id": str(run.id),
            "version_id": str(version.id),
            "detected": summary["by_status"]["detected"],
            "check_errors": summary["check_errors"],
        },
    )


def _analysis_libraries() -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in ("scipy", "scikit-learn")}
