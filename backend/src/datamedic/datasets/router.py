import hashlib
import uuid

from fastapi import APIRouter, Request, status
from starlette.concurrency import run_in_threadpool

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep, VerifiedPrincipalDep
from datamedic.datasets import service
from datamedic.datasets.schemas import (
    DatasetCreatedOut,
    DatasetCreateIn,
    DatasetListOut,
    DatasetOut,
    DeletionAcceptedOut,
    UploadAcceptedOut,
)
from datamedic.db.models import Dataset
from datamedic.errors import LengthRequired, UploadTooLarge, ValidationFailed
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.storage import raw_object_key

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


def _out(ds: Dataset, detected_format: str | None) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        domain=ds.domain,
        status=ds.status,
        filename=ds.declared_filename,
        size_bytes=ds.declared_size_bytes,
        detected_format=detected_format,
        created_at=ds.created_at,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def create_dataset(
    body: DatasetCreateIn, principal: VerifiedPrincipalDep, db: DbDep, ctx: CtxDep
) -> DatasetCreatedOut:
    ds = service.create_dataset(
        db, ctx.settings, principal.user.id, body.name, body.filename, body.size_bytes
    )
    return DatasetCreatedOut(dataset=_out(ds, None), max_upload_bytes=ctx.settings.max_upload_bytes)


@router.get("")
def list_datasets(principal: PrincipalDep, db: DbDep) -> DatasetListOut:
    rows = service.list_datasets(db, principal.user.id)
    return DatasetListOut(items=[_out(ds, fmt) for ds, fmt in rows])


@router.get("/{dataset_id}")
def get_dataset(dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> DatasetOut:
    ds = service.get_owned(db, principal.user.id, dataset_id)
    return _out(ds, service.detected_format(db, ds.id))


@router.put(
    "/{dataset_id}/content",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload_content(
    dataset_id: uuid.UUID, request: Request, principal: VerifiedPrincipalDep, db: DbDep, ctx: CtxDep
) -> UploadAcceptedOut:
    """Stream the raw file body to private storage, hashing while streaming."""
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        raise LengthRequired()
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValidationFailed("Invalid Content-Length.") from exc
    if length > ctx.settings.max_upload_bytes:
        raise UploadTooLarge(f"Maximum upload size is {ctx.settings.max_upload_bytes} bytes.")

    owner_id = principal.user.id
    peek = await run_in_threadpool(service.get_owned, db, owner_id, dataset_id)
    if length != peek.declared_size_bytes:
        raise ValidationFailed("Content-Length does not match the declared file size.")
    ds = await run_in_threadpool(service.begin_upload, db, ctx.settings, owner_id, dataset_id)

    writer = ctx.storage.open_writer(raw_object_key(owner_id, ds.id))
    hasher = hashlib.sha256()
    received = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            received += len(chunk)
            if received > length:
                raise UploadTooLarge("Body is larger than Content-Length.")
            hasher.update(chunk)
            await run_in_threadpool(writer.write, chunk)
        if received != length:
            raise ValidationFailed("Body is shorter than Content-Length.")
        stored = await run_in_threadpool(writer.commit)
        if stored.sha256 != hasher.digest():  # storage and API must agree on the bytes
            raise ValidationFailed("Upload integrity check failed.")
        job = await run_in_threadpool(service.register_upload, db, ctx.settings, ds, stored)
    except BaseException:
        writer.abort()
        await run_in_threadpool(service.abort_upload, db, ds)
        raise

    await run_in_threadpool(publish_committed, db, ctx.require_publisher(), job.id, job.type)
    await run_in_threadpool(db.refresh, ds)
    await run_in_threadpool(db.refresh, job)
    return UploadAcceptedOut(dataset=_out(ds, None), job=JobOut.from_job(job))


@router.delete("/{dataset_id}", status_code=status.HTTP_202_ACCEPTED)
def delete_dataset(
    dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> DeletionAcceptedOut:
    job = service.soft_delete(db, ctx.settings, principal.user.id, dataset_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return DeletionAcceptedOut(dataset_id=dataset_id, job=JobOut.from_job(job))
