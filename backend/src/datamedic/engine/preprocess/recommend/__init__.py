"""Deterministic preprocessing recommendations (Phase 4 Part D). Standard library only, so
the API can generate them synchronously without loading pandas."""

from datamedic.engine.preprocess.recommend.engine import recommend
from datamedic.engine.preprocess.recommend.facts import FindingRef, RecommendInput
from datamedic.engine.preprocess.recommend.rules import RULES_VERSION

__all__ = ["RULES_VERSION", "FindingRef", "RecommendInput", "recommend"]
