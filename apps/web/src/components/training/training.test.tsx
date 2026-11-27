import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "@/lib/api/client";
import type { Audit, TrainingPlan, TrainingRun } from "@/lib/training";

import { NewTrainingRun } from "./run-config";
import { Comparison, Explainability, Performance, Reliability, Reproducibility } from "./run-results";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const stats = (mean: number | null, values: (number | null)[] = [mean, mean]) => ({
  mean, sd: mean === null ? null : 0.01, min: mean, max: mean, n_folds: values.filter((v) => v !== null).length, values,
});

function plan(overrides: Partial<TrainingPlan> = {}): TrainingPlan {
  return {
    valid: true, errors: [], warnings: [], task: "binary", target: "y", features: ["a", "b"],
    excluded_columns: [{ column: "pid", roles: ["id"] }], datetime_columns: [], sensitive_columns: [],
    models: ["dummy", "linear"], tuning: { requested: "nested", planned: "nested" }, estimated_fits: 70, fit_limit: 400,
    seed: 0, split: { method: "stratified", group: null, time: null }, evaluation_mode: "unbiased_direct",
    eligibility: { mode: "unbiased_direct", evaluation_version_id: "v0", reasons: [], replay: null },
    evaluation_version_id: "v0", evaluation_version_number: 0, preprocess_steps: [], preprocess_source: "version_model_pipeline_spec",
    spec_hash: "h", allow_exploratory: false,
    ...overrides,
  } as TrainingPlan;
}

const versions = [{ id: "v1", version_number: 1, kind: "transform", fitted_on_full_data: true }];

describe("New training run", () => {
  it("shows eligible predictors and starts a run", async () => {
    const post = vi.spyOn(api, "POST").mockImplementation((async (path: string) => {
      if (path.endsWith("training-plan")) return { data: plan(), response: new Response() };
      return { data: { training_run: { id: "r1" } }, response: new Response() };
    }) as never);
    const started = vi.fn();
    render(<NewTrainingRun versions={versions} headId="v1" ops={[]} onStarted={started} />);
    await waitFor(() => expect(screen.getByText(/2 feature column/)).toBeTruthy());
    expect(screen.getByText(/pid \(id\)/)).toBeTruthy();
    const badge = screen.getByText("Leakage-controlled evaluation");
    expect(badge.getAttribute("title")).toMatch(/does not guarantee external validity/);
    expect(screen.queryByText(/unbiased/i)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(started).toHaveBeenCalledWith("r1"));
    expect(post.mock.calls.some((c) => String(c[0]).endsWith("training-runs"))).toBe(true);
  });

  it("blocks unsafe evaluation until exploratory is chosen explicitly", async () => {
    const blocked = plan({
      valid: false, evaluation_mode: "blocked",
      eligibility: { mode: "blocked", evaluation_version_id: "v1",
        reasons: [{ code: "TARGET_TRANSFORMED_IN_LINEAGE", message: "A committed step changed the target." }], replay: null },
    });
    vi.spyOn(api, "POST").mockResolvedValue({ data: blocked, response: new Response() } as never);
    render(<NewTrainingRun versions={versions} headId="v1" ops={[]} onStarted={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("A committed step changed the target.")).toBeTruthy());
    const start = screen.getByRole("button", { name: "Start training" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText(/Run as exploratory anyway/));
    await waitFor(() => expect(screen.getByText("Exploratory only")).toBeTruthy());
    expect(screen.getByText(/never\s+reported as leakage-controlled/)).toBeTruthy();
  });
});

function evaluation(model: string, auc: number | null, extra: Record<string, unknown> = {}) {
  return {
    model, status: "completed", comparable: true, comparability_reason: null,
    metrics: {
      folds: { roc_auc: stats(auc), accuracy: stats(0.7) },
      reasons: auc === null ? { roc_auc: [{ fold: 0, reason: "Not every class occurs among the test rows." }] } : {},
      pooled: {}, bootstrap: { available: true, metrics: { roc_auc: auc === null ? null : { low: auc - 0.05, high: auc + 0.05 } } },
    },
    details: {
      curves: { available: true, per_class: null, roc: [[0, 0], [0.2, 0.7], [1, 1]], pr: [[0, 1], [1, 0.4]], prevalence: 0.4 },
      calibration: { available: false, reason: "Fewer than 100 rows with predictions." },
      residuals: { available: false }, confusion: [[40, 10], [8, 42]],
      per_class: [{ support: 50, precision: 0.83, recall: 0.8 }, { support: 50, precision: 0.8, recall: 0.84 }],
      error_slices: [{ column: "a", metric: "error_rate", slices: [{ label: "-1 – 0", n: 40, error: 0.2 }] }],
    },
    fold_results: [], selection: { per_fold: [{ C: 1 }], frequency: [] },
    importance: {
      metric: "roc_auc", permutation: [{ column: "a", mean: 0.2, sd: 0.02, folds_present: 2, n_folds: 2 }],
      coefficients: [{ column: "a", mean: 1.5, sd: 0.1, folds_present: 2, n_folds: 2 }], impurity: null,
    },
    failures: [],
    ...extra,
  };
}

const run = {
  id: "r1", task_type: "binary", evaluation_mode: "unbiased_direct", tuning: {}, replay: null,
  split_config: { method: "stratified_group", group: "patient_id", time: null },
  config: { roles: { roles: { group: ["patient_id"], sensitive: ["code"], target: ["y"] } } },
  privacy: { policy: "Stored results are not changed; they are filtered when shown.", protected_columns: [],
    withheld_error_slices: [], target_values_withheld: false },
  library_versions: {}, evaluation_version_id: "v0", version_id: "v0",
  summary: {
    metrics: ["roc_auc", "accuracy"], primary_metric: "roc_auc", classes: ["no", "yes"],
    metric_info: { roc_auc: { label: "ROC-AUC", higher_is_better: true, probability: true },
      accuracy: { label: "Accuracy", higher_is_better: true, probability: false } },
    comparisons: [{ model: "linear", reference: "dummy", metric: "roc_auc", mean_difference: 0.3, verdict: "consistently_better" }],
    warnings: [{ code: "MULTIPLE_COMPARISONS", message: "Several models are compared." }],
    coverage: { complete: true }, explain_limitations: ["Importance is not causal."],
  },
  evaluations: [evaluation("dummy", null), evaluation("linear", 0.85)],
} as unknown as TrainingRun;

describe("Run results", () => {
  it("compares models without a winner and shows unavailable metrics as unavailable", () => {
    render(<Comparison run={run} />);
    expect(screen.getByText(/No\s+model is declared best/)).toBeTruthy();
    expect(screen.getByText("0.850 ± 0.010")).toBeTruthy();
    expect(screen.getByText("—")).toBeTruthy(); // dummy ROC-AUC unavailable, not 0
    expect(screen.getByText("unavailable in some folds")).toBeTruthy();
    expect(screen.queryByText(/best model/i)).toBeNull();
  });

  it("renders curves, confusion matrix and the reason when calibration is unavailable", () => {
    render(<Performance run={run} />);
    expect(screen.getByRole("img", { name: "ROC curve" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Confusion matrix" })).toBeTruthy();
    expect(screen.getByText(/Unavailable: Fewer than 100 rows/)).toBeTruthy();
  });

  it("shows importance with its limitations and error slices", () => {
    render(<Explainability run={run} />);
    expect(screen.getByRole("list", { name: "Permutation importance per original column" })).toBeTruthy();
    expect(screen.getByText("Importance is not causal.")).toBeTruthy();
    expect(screen.getByText(/sensitive columns are never sliced/)).toBeTruthy();
  });

  it("groups audit findings by basis and states that the status is not a certification", () => {
    const f = (id: string, status: string, basis: string) => ({
      id, check_id: id, check_version: "1", domain: "reliability", category: "leakage", status, severity: "high",
      title: `title ${id}`, explanation: "e", evidence: { basis }, columns: [], affected_rows: null, affected_pct: null,
      recommendation: "r", limitations: ["l"], not_assessed_reason: status === "not_assessed" ? "No time role." : null,
      error_code: null, sampled: false, sample_info: null,
    });
    const audit = {
      training_run_id: "r1", evaluation_mode: "unbiased_direct", stale: false, job: null,
      run: { summary: { evaluation_status: { status: "concerns", rules: ["rule"], limitation: "It is not a certification of model validity." } } },
      findings: [f("a", "detected", "confirmed"), f("b", "detected", "statistical"), f("c", "not_assessed", "configuration")],
    } as unknown as Audit;
    render(<Reliability audit={audit} onRerun={vi.fn()} busy={false} />);
    expect(screen.getByText("Concerns found")).toBeTruthy();
    expect(screen.getByText(/not a certification/)).toBeTruthy();
    expect(screen.getByText("Confirmed and configuration violations (1)")).toBeTruthy();
    expect(screen.getByText("Statistical suspicions (1)")).toBeTruthy();
    expect(screen.getByText("Could not be assessed (1)")).toBeTruthy();
  });

  it("says that roles changed since training and that re-runs keep the training-time roles", () => {
    const audit = {
      training_run_id: "r1", evaluation_mode: "exploratory", stale: true, job: null,
      run: { summary: { evaluation_status: { status: "exploratory", rules: [], limitation: "l" } } }, findings: [],
    } as unknown as Audit;
    render(<Reliability audit={audit} onRerun={vi.fn()} busy={false} />);
    expect(screen.getByText("Exploratory only")).toBeTruthy();
    const badge = screen.getByText("roles changed since training");
    expect(badge.getAttribute("title")).toMatch(/training-time roles; they are not reinterpreted/);
  });

  it("distinguishes fold standard deviations from bootstrap intervals and discloses dependence", () => {
    render(<Comparison run={run} />);
    expect(screen.getByText(/mean ± standard deviation of the outer test-fold scores/)).toBeTruthy();
    expect(screen.getByText(/separate 95% bootstrap interval/)).toBeTruthy();
    expect(screen.getByTestId("dependence-note").textContent).toMatch(/resample whole groups \(patient_id\)/);
    cleanup();
    render(<Comparison run={{ ...run, split_config: { method: "kfold", group: null, time: null } } as TrainingRun} />);
    expect(screen.getByTestId("dependence-note").textContent).toMatch(/intervals are too narrow/);
    cleanup();
    render(<Comparison run={{ ...run, split_config: { method: "time", group: null, time: "t" } } as TrainingRun} />);
    expect(screen.getByTestId("dependence-note").textContent).toMatch(/bootstrap intervals are not shown/);
  });

  it("discloses withheld error slices of protected columns", () => {
    const withheld = {
      ...run,
      privacy: { policy: "Stored results are not changed.", protected_columns: [{ column: "code", reasons: ["sensitive_now"] }],
        withheld_error_slices: ["code"], target_values_withheld: false },
    } as unknown as TrainingRun;
    render(<Explainability run={withheld} />);
    expect(screen.getByTestId("privacy-note").textContent).toMatch(/withheld for 1 protected column\(s\): code/);
    cleanup();
    render(<Explainability run={run} />);
    expect(screen.queryByTestId("privacy-note")).toBeNull();
  });

  it("scopes reproducibility to the stored configuration and shows the training-time roles", () => {
    render(<Reproducibility run={{ ...run, split: null } as unknown as TrainingRun} />);
    expect(screen.getByText(/same library and\s+engine versions/)).toBeTruthy();
    expect(screen.getByText(/Current role changes are not applied to a re-run/)).toBeTruthy();
    expect(screen.getByText("group: patient_id · sensitive: code · target: y")).toBeTruthy();
  });
});
