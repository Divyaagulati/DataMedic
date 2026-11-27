import { describe, expect, it } from "vitest";

import {
  addStep,
  type CandidateDraft,
  draftsFromSuggestions,
  experimentPayload,
  formatStats,
  type RecommendationSet,
  removeStep,
  stabilityClass,
} from "./experiments";

const empty: CandidateDraft = { id: "c", label: "A", steps: [], recommendationIds: [] };

describe("addStep", () => {
  it("keeps pipeline order: indicators, imputation, transforms, resampling last", () => {
    let c = addStep(empty, { op: "imbalance.class_weight", columns: [], params: {} });
    c = addStep(c, { op: "transform.log", columns: ["lab"], params: {} });
    c = addStep(c, { op: "impute.median", columns: ["b"], params: {} });
    c = addStep(c, { op: "missing.indicator", columns: ["b"], params: {} }, "rec-1");
    expect(c.steps.map((s) => s.op)).toEqual([
      "missing.indicator",
      "impute.median",
      "transform.log",
      "imbalance.class_weight",
    ]);
    expect(c.recommendationIds).toEqual(["rec-1"]);
  });

  it("allows only one training-only step and never duplicates a step", () => {
    let c = addStep(empty, { op: "imbalance.class_weight", columns: [], params: {} });
    c = addStep(c, { op: "imbalance.smote", columns: [], params: {} });
    expect(c.steps.map((s) => s.op)).toEqual(["imbalance.smote"]);
    c = addStep(c, { op: "impute.median", columns: ["b"], params: {} });
    c = addStep(c, { op: "impute.median", columns: ["b"], params: {} });
    expect(c.steps).toHaveLength(2);
    expect(removeStep(c, 0).steps.map((s) => s.op)).toEqual(["imbalance.smote"]);
  });
});

describe("suggestions and payload", () => {
  const set = {
    id: "set-1",
    recommendations: [{ id: "db-1", key: "rec_a" }],
    candidates: [
      { key: "baseline", label: "Baseline", steps: [], recommendation_ids: [] },
      {
        key: "core",
        label: "Recommended core",
        steps: [{ op: "impute.median", columns: ["b"], params: {} }],
        recommendation_ids: ["rec_a"],
      },
    ],
  } as unknown as RecommendationSet;

  it("skips the baseline and maps engine keys to stored ids", () => {
    const drafts = draftsFromSuggestions(set);
    expect(drafts).toHaveLength(1);
    expect(drafts[0].label).toBe("Recommended core");
    expect(drafts[0].recommendationIds).toEqual(["db-1"]);
  });

  it("builds the API request", () => {
    const body = experimentPayload(
      draftsFromSuggestions(set),
      {
        name: " Lab ",
        models: ["dummy", "linear"],
        method: null,
        nSplits: 5,
        seed: 0,
        clustering: false,
        nClusters: 3,
      },
      "set-1",
    );
    expect(body.name).toBe("Lab");
    expect(body.task).toBeNull();
    expect(body.split).toEqual({ method: null, n_splits: 5, seed: 0 });
    expect(body.candidates[0].steps[0].op).toBe("impute.median");
    expect(body.recommendation_set_id).toBe("set-1");
  });
});

describe("formatting", () => {
  it("shows mean and fold standard deviation, or a dash", () => {
    expect(formatStats({ mean: 0.71234, sd: 0.0345, n_folds: 5, values: [] })).toBe(
      "0.712 ± 0.035",
    );
    expect(formatStats({ mean: null, sd: null, n_folds: 0, values: [] })).toBe("—");
    expect(formatStats(undefined)).toBe("—");
  });

  it("classifies selection frequency", () => {
    expect(stabilityClass(1)).toBe("stable");
    expect(stabilityClass(0.6)).toBe("unstable");
    expect(stabilityClass(0)).toBe("rarely");
  });
});
