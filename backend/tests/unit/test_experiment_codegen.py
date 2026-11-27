"""Generated-code compatibility of Experiment Lab candidates: every pipeline the
recommendation engine suggests renders to code that reproduces the platform's fold
behaviour (held-out rows and training table) in an isolated ``python -I`` subprocess."""

from pathlib import Path

import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.codegen import render_steps
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.recommend.facts import Facts
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.recommend_helpers import clinical_frame, make_input, recommend
from tests.unit.test_preprocess_codegen_c import assert_equivalent, run_generated

DF = clinical_frame(240)
INPUT = make_input(DF, {"target": ["y"]}, task="binary")
SUGGESTED = [c for c in recommend(INPUT)["candidates"] if c["steps"]]


def test_there_are_suggested_candidates() -> None:
    assert len(SUGGESTED) >= 2


@pytest.mark.parametrize("key", [c["key"] for c in SUGGESTED])
def test_suggested_candidate_code_matches_the_platform(key: str, tmp_path: Path) -> None:
    candidate = next(c for c in SUGGESTED if c["key"] == key)
    steps = [Step(s["op"], tuple(s["columns"]), s["params"]) for s in candidate["steps"]]
    validation = validate_steps(steps, Facts(INPUT).context)
    assert validation.valid
    fit_df = DF[DF[ROW_ID] % 4 != 0]
    ex = PreprocessExecutor(validation, seed=3)
    training = ex.fit_resample(fit_df).frame
    platform = ex.transform(DF).frame
    generated, generated_training = run_generated(
        tmp_path, render_steps(validation.steps, seed=3), fit_df, DF
    )
    assert_equivalent(platform.reset_index(drop=True), generated.reset_index(drop=True))
    assert_equivalent(training.reset_index(drop=True), generated_training.reset_index(drop=True))
