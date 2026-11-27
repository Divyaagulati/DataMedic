import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CandidateDraft, ExperimentDetail, LabSettings, RecommendationItem } from "@/lib/experiments";
import type { Op } from "@/lib/preprocess";

import { ExperimentResults } from "./experiment-results";
import { PipelineBuilder } from "./pipeline-builder";
import { RecommendationCard } from "./recommendation-card";

afterEach(cleanup);

const detail = {
  id: "rec_1",
  title: "Fill missing values in 1 numeric column with the training-fold median",
  op: "impute.median",
  op_title: "Median imputation",
  what_it_does: "Replaces missing values with the median of the fitting rows.",
  applies_to: "Selected columns: b.",
  reason: "1 numeric feature has missing values (up to 20% of rows).",
  evidence: [{ source: "schema", metric: "missing_pct", value: 20, column: "b" }],
  expected_benefit: "Keeps every row usable.",
  risks: ["Imputed values are estimates."],
  information_loss: { level: "low", detail: "Observed values are unchanged." },
  domain_compatibility: { status: "caution", detail: "Missing clinical measurements are often not random." },
  task_compatibility: { status: "relevant", detail: "The linear baseline needs complete inputs." },
  required_decisions: ["Confirm why values are missing."],
  evidence_strength: { level: "strong", explanation: "Exact counts." },
  limitations: ["Assumes missing at random."],
  not_a_guarantee: "A recommendation is a hypothesis to test, not a guaranteed improvement.",
  conflict_group: "impute.numeric",
  alternatives: [],
  ranking: {
    score: 3,
    factors: [
      { factor: "evidence", points: 3, detail: "Strong evidence: exact counts." },
      { factor: "risk", points: -1, detail: "Medium-risk operation." },
    ],
  },
  kind: "fitted",
};

const item = {
  id: "db-1",
  key: "rec_1",
  rank: 1,
  status: "open",
  op: "impute.median",
  stage: "either",
  risk: "medium",
  destructive: false,
  columns: ["b"],
  params: {},
  detail,
} as unknown as RecommendationItem;

describe("RecommendationCard", () => {
  it("explains stage, risk, evidence, ranking factors and limitations", () => {
    render(
      <ul>
        <RecommendationCard item={item} candidates={[]} onAdd={vi.fn()} onStatus={vi.fn()} />
      </ul>,
    );
    expect(screen.getByText("Cleaning or model preparation")).toBeTruthy();
    expect(screen.getByText("Medium risk")).toBeTruthy();
    expect(screen.getByText("Domain caution")).toBeTruthy();
    expect(screen.getByText(/missing values \(up to 20% of rows\)/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Why and what" }));
    expect(screen.getByText(/missing pct = 20/)).toBeTruthy();
    expect(screen.getByText("Strong evidence: exact counts.")).toBeTruthy();
    expect(screen.getByText(/not a guaranteed improvement/)).toBeTruthy();
    expect(screen.getByText("Assumes missing at random.")).toBeTruthy();
    expect(screen.getByText("Confirm why values are missing.")).toBeTruthy();
  });

  it("adds the step to the chosen candidate and can be dismissed", () => {
    const onAdd = vi.fn();
    const onStatus = vi.fn();
    render(
      <ul>
        <RecommendationCard
          item={item}
          candidates={[{ id: "c1", label: "Core" }, { id: "c2", label: "Alt" }]}
          onAdd={onAdd}
          onStatus={onStatus}
        />
      </ul>,
    );
    fireEvent.change(screen.getByLabelText("Add to"), { target: { value: "c2" } });
    fireEvent.click(screen.getByRole("button", { name: "Add step" }));
    expect(onAdd).toHaveBeenCalledWith("c2", item);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onStatus).toHaveBeenCalledWith(item, "dismissed");
  });
});

const ops = [
  {
    op: "impute.median",
    title: "Median imputation",
    description: "Median of the fitting rows.",
    stage: "either",
    kind: "fitted",
    risk: "medium",
    destructive: false,
    uses_columns: true,
    params: [],
    family: "missing",
    column_types: ["numeric"],
  },
  {
    op: "outliers.cap",
    title: "Cap outliers (winsorize)",
    description: "Clips values.",
    stage: "either",
    kind: "fitted",
    risk: "high",
    destructive: true,
    uses_columns: true,
    family: "outliers",
    column_types: ["numeric"],
    params: [
      { name: "method", type: "choice", description: "Rule", default: "percentile", required: false, choices: ["iqr", "percentile"] },
      { name: "iqr_multiplier", type: "float", description: "k", default: 1.5, required: false, choices: null },
    ],
  },
] as unknown as Op[];

const settings: LabSettings = {
  name: "Lab",
  models: ["dummy", "linear"],
  method: null,
  nSplits: 5,
  seed: 0,
  clustering: false,
  nClusters: 3,
};

describe("PipelineBuilder", () => {
  it("shows stage and risk per step, edits parameters and removes steps", () => {
    const drafts: CandidateDraft[] = [
      {
        id: "c1",
        label: "Core",
        steps: [{ op: "outliers.cap", columns: ["a"], params: { method: "percentile" } }],
        recommendationIds: [],
      },
    ];
    const onDrafts = vi.fn();
    render(
      <PipelineBuilder
        drafts={drafts}
        onDrafts={onDrafts}
        ops={ops}
        columns={["a", "b"]}
        settings={settings}
        onSettings={vi.fn()}
        hasTarget
        onRun={vi.fn()}
        running={false}
      />,
    );
    expect(screen.getByText("Destructive")).toBeTruthy();
    expect(screen.getByText("High risk")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("method"), { target: { value: "iqr" } });
    expect(onDrafts.mock.calls[0][0][0].steps[0].params.method).toBe("iqr");
    fireEvent.click(screen.getByRole("button", { name: "Remove outliers.cap" }));
    expect(onDrafts.mock.calls[1][0][0].steps).toEqual([]);
    expect(screen.getByRole("button", { name: /Run experiment \(2 pipelines incl. baseline\)/ })).toBeTruthy();
  });

  it("will not run a supervised experiment without a target", () => {
    render(
      <PipelineBuilder
        drafts={[{ id: "c1", label: "A", steps: [], recommendationIds: [] }]}
        onDrafts={vi.fn()}
        ops={ops}
        columns={["a"]}
        settings={settings}
        onSettings={vi.fn()}
        hasTarget={false}
        onRun={vi.fn()}
        running={false}
      />,
    );
    expect(screen.getByText(/No target is declared/)).toBeTruthy();
    expect((screen.getByRole("button", { name: /Run experiment/ }) as HTMLButtonElement).disabled).toBe(true);
  });
});

function stats(values: (number | null)[]) {
  const present = values.filter((v): v is number => v !== null);
  const mean = present.length ? present.reduce((a, b) => a + b, 0) / present.length : null;
  return { mean, sd: present.length > 1 ? 0.01 : null, n_folds: present.length, values, min: null, max: null };
}

const experiment = {
  id: "e1",
  version_id: "v1",
  name: "Lab run",
  task_type: "binary",
  target_column: "y",
  models: ["dummy", "linear"],
  seed: 0,
  config_hash: "ab",
  rerun_of_id: null,
  created_at: "2026-11-24T00:00:00Z",
  completed_at: "2026-11-24T00:01:00Z",
  job: { id: "j1", type: "experiment", status: "succeeded", progress: null, error: null },
  config: {},
  recommendation_set_id: null,
  library_versions: {},
  split: {
    id: "s1",
    method: "stratified",
    n_splits: 2,
    seed: 0,
    target_column: "y",
    group_column: null,
    time_column: null,
    assignment_sha256: "0123456789abcdef",
    cohort: {},
    fold_stats: [],
    warnings: [],
    created_at: "",
  },
  summary: {
    metrics: ["roc_auc", "accuracy"],
    primary_metric: "roc_auc",
    metric_info: { roc_auc: { label: "ROC-AUC", higher_is_better: true }, accuracy: { label: "Accuracy", higher_is_better: true } },
    cohort: { rows: 100, excluded_missing_target: 3 },
    warnings: [{ code: "MULTIPLE_COMPARISONS", message: "Several comparisons." }],
    split_warnings: [],
    limitations: ["A higher mean score alone does not make a pipeline better."],
    model_notes: { linear: "Logistic regression." },
    comparisons: [
      {
        candidate: "c1",
        baseline: "baseline",
        model: "linear",
        metric: "roc_auc",
        per_fold_difference: [0.01, -0.02],
        mean_difference: -0.005,
        sd_difference: 0.02,
        folds_better: 1,
        folds_worse: 1,
        verdict: "negligible",
        interpretation: "Describes these folds only; not a significance test.",
      },
    ],
  },
  candidates: [
    {
      id: "k0",
      key: "baseline",
      label: "Baseline: no preprocessing",
      is_baseline: true,
      steps: [],
      max_risk: "low",
      has_destructive: false,
      stages: [],
      recommendation_ids: [],
      spec_hash: "x",
      result: {
        status: "completed",
        comparable: true,
        comparability_reason: null,
        metrics: { dummy: { roc_auc: stats([0.5, 0.5]) }, linear: { roc_auc: stats([0.7, 0.72]) } },
        folds: [
          { fold: 0, status: "ok", error: null, step_warnings: [] },
          { fold: 1, status: "ok", error: null, step_warnings: [] },
        ],
        data_quality: {
          before: { missing_cells: 10 },
          after: { missing_cells: 10 },
          cells_imputed: 0,
          information_loss_indicators: [],
        },
        selection_stability: [],
        failures: [],
      },
    },
    {
      id: "k1",
      key: "c1",
      label: "Core",
      is_baseline: false,
      steps: [],
      max_risk: "high",
      has_destructive: true,
      stages: ["model"],
      recommendation_ids: [],
      spec_hash: "y",
      result: {
        status: "completed",
        comparable: true,
        comparability_reason: null,
        metrics: { dummy: { roc_auc: stats([0.5, 0.5]) }, linear: { roc_auc: stats([0.71, 0.70]) } },
        folds: [
          { fold: 0, status: "ok", error: null, step_warnings: [] },
          { fold: 1, status: "ok", error: null, step_warnings: [] },
        ],
        data_quality: {
          before: { missing_cells: 10 },
          after: { missing_cells: 0 },
          cells_imputed: 10,
          information_loss_indicators: ["Missing values were replaced by estimates."],
        },
        selection_stability: [
          {
            step: 1,
            op: "select.univariate",
            folds: 2,
            frequency: { a: 1, noise: 0.5 },
            mean_jaccard: 0.5,
            probe_selection_rate: 0.25,
            indistinguishable_from_noise: ["noise"],
            forced: [],
            warnings: [{ code: "NOISE_SELECTED", message: "Random noise columns were selected." }],
            limitations: ["Predictive selection is not biomarker validation."],
          },
        ],
        failures: [],
      },
    },
    {
      id: "k2",
      key: "c2",
      label: "Broken",
      is_baseline: false,
      steps: [],
      max_risk: "medium",
      has_destructive: false,
      stages: ["either"],
      recommendation_ids: [],
      spec_hash: "z",
      result: {
        status: "failed",
        comparable: false,
        comparability_reason: "2 of 2 folds failed; this candidate is excluded from comparisons.",
        metrics: { dummy: { roc_auc: stats([null, null]) }, linear: { roc_auc: stats([null, null]) } },
        folds: [
          { fold: 0, status: "failed", error: { code: "VALUE_OUT_OF_DOMAIN", message: "Values must be positive." }, step_warnings: [] },
          { fold: 1, status: "failed", error: { code: "VALUE_OUT_OF_DOMAIN", message: "Values must be positive." }, step_warnings: [] },
        ],
        data_quality: null,
        selection_stability: [],
        failures: [{ fold: 0, code: "VALUE_OUT_OF_DOMAIN", message: "Values must be positive." }],
      },
    },
  ],
} as unknown as ExperimentDetail;

describe("ExperimentResults", () => {
  it("compares side by side without declaring a winner and separates non-comparable runs", () => {
    render(<ExperimentResults experiment={experiment} onCancel={vi.fn()} onRerun={vi.fn()} />);
    expect(screen.getByText(/No pipeline is\s+declared a winner/)).toBeTruthy();
    expect(screen.getByText("0.710 ± 0.010")).toBeTruthy();
    expect(screen.getByText(/2 of 2 folds failed/)).toBeTruthy();
    expect(screen.getByText("No practical difference.")).toBeTruthy();
    expect(screen.getByText(/not a significance test/)).toBeTruthy();
    expect(screen.getByText(/3 without a target excluded/)).toBeTruthy();
    expect(screen.getByText("MULTIPLE_COMPARISONS")).toBeTruthy();
    expect(screen.queryByText(/best/i)).toBeNull();
  });

  it("shows fold-level scores, data-quality changes and selection stability", () => {
    render(<ExperimentResults experiment={experiment} onCancel={vi.fn()} onRerun={vi.fn()} />);
    const folds = screen.getByText(/Fold-level scores/).closest("section")!;
    expect(within(folds).getAllByText("failed")).toHaveLength(2);
    expect(within(folds).getByText("0.720")).toBeTruthy();
    expect(screen.getByText("10.0 → 0.0")).toBeTruthy();
    expect(screen.getByText("Missing values were replaced by estimates.")).toBeTruthy();
    expect(screen.getByText(/noise columns\s+selected in 25% of draws/)).toBeTruthy();
    expect(screen.getByText(/1\/2 · unstable · noise-like/)).toBeTruthy();
    expect(screen.getByText("Predictive selection is not biomarker validation.")).toBeTruthy();
  });

  it("offers cancel while running and re-run when complete", () => {
    const onCancel = vi.fn();
    const running = { ...experiment, completed_at: null, summary: null, job: { ...experiment.job!, status: "running", progress: { pct: 40 } } } as unknown as ExperimentDetail;
    render(<ExperimentResults experiment={running} onCancel={onCancel} onRerun={vi.fn()} />);
    expect(screen.getByRole("status").textContent).toContain("40%");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
  });
});
