from typing import Annotated

from fastapi import APIRouter, Query

from datamedic.api.deps import PrincipalDep
from datamedic.engine.preprocess.catalog import all_ops
from datamedic.preprocessing.schemas import OpListOut, OpOut

router = APIRouter(prefix="/api/v1", tags=["preprocessing"])


@router.get("/ops")
def list_ops(
    principal: PrincipalDep, family: Annotated[str | None, Query(max_length=64)] = None
) -> OpListOut:
    """The preprocessing op catalogue. Risk, destructive flag, kind and stage shown here are
    the only ones the platform uses; they are never taken from a request."""
    return OpListOut(
        items=[OpOut.build(s) for s in all_ops() if family is None or s.family == family]
    )
