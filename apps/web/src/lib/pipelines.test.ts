import { describe, expect, it } from "vitest";

import { approvalState, canCommit, cell, moveStep, nextActions, type Pipeline, splitByStage } from "./pipelines";

function pipeline(overrides: Partial<Pipeline> = {}): Pipeline {
  return {
    id: "p",
    dataset_id: "d",
    name: "P",
    state: "validated",
    source_version_id: "v0",
    committed_version_id: null,
    created_at: "",
    updated_at: "",
    source_is_head: true,
    spec_hash: "h",
    validation: { valid: true, errors: [], warnings: [], data_steps: 1, model_steps: 1 },
    steps: [
      { id: "s1", position: 0, op: "outliers.cap", op_version: "1", op_title: "Cap", catalog_stage: "either", stage: "data", columns: ["a"], params: {}, risk: "high", destructive: true, kind: "fitted", step_hash: "x", approved: false, from_recommendation_id: null },
      { id: "s2", position: 1, op: "scale.standard", op_version: "1", op_title: "Scale", catalog_stage: "model", stage: "model", columns: ["a"], params: {}, risk: "medium", destructive: false, kind: "fitted", step_hash: "y", approved: false, from_recommendation_id: null },
    ],
    approvals: [],
    readiness: {
      ready: false,
      preview_current: false,
      high_risk_steps: [{ step_id: "s1", approved: false }],
      confirmation_required: true,
      confirmation_valid: false,
      stale_approvals: 0,
      problems: [],
    },
    preview_job: null,
    from_experiment_candidate_id: null,
    ...overrides,
  } as Pipeline;
}

describe("pipeline helpers", () => {
  it("separates data-stage from modelling-only steps", () => {
    const { data, model } = splitByStage(pipeline().steps);
    expect(data.map((s) => s.op)).toEqual(["outliers.cap"]);
    expect(model.map((s) => s.op)).toEqual(["scale.standard"]);
  });

  it("lists what is missing before a commit and never allows a stale one", () => {
    const p = pipeline();
    expect(nextActions(p)).toEqual([
      "Generate the impact preview for the current pipeline.",
      "Approve 1 high-risk step(s) one by one.",
      "Confirm the pipeline (medium-risk steps).",
    ]);
    expect(canCommit(p)).toBe(false);
    const ready = pipeline({ readiness: { ...p.readiness, ready: true, preview_current: true } });
    expect(canCommit(ready)).toBe(true);
    expect(canCommit({ ...ready, source_is_head: false })).toBe(false);
    expect(nextActions({ ...ready, source_is_head: false })[0]).toMatch(/newer version/);
  });

  it("shows an approval as valid only with a current preview", () => {
    const approvedStep = { ...pipeline().steps[0], approved: true };
    const stale = pipeline({ steps: [approvedStep] });
    expect(approvalState(stale, approvedStep)).toBe("required");
    const current = pipeline({ steps: [approvedStep], readiness: { ...stale.readiness, preview_current: true } });
    expect(approvalState(current, approvedStep)).toBe("approved");
    expect(approvalState(current, pipeline().steps[1])).toBe("not_required");
  });

  it("bounds moves and formats cells", () => {
    expect(moveStep(pipeline().steps, 0, -1)).toBeNull();
    expect(moveStep(pipeline().steps, 0, 1)).toBe(1);
    expect(cell(null)).toBe("missing");
    expect(cell(2)).toBe("2");
    expect(cell(1.23456789)).toBe("1.23457");
  });
});
