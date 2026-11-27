import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { StepEditor } from "@/app/datasets/[datasetId]/pipelines/[pipelineId]/pipeline-view";
import { downloadCode, VersionBadges } from "@/app/datasets/[datasetId]/history/history-view";
import { ExperimentResults } from "@/components/lab/experiment-results";
import { api } from "@/lib/api/client";
import type { ExperimentDetail } from "@/lib/experiments";
import type { Pipeline, PipelineStep, PreviewStep, PreviewSummary, VersionSummary } from "@/lib/pipelines";

import { ConfirmDialog, StepApprovalDialog } from "./approval-dialog";
import { AffectedRows, ImpactPreviewPanel } from "./impact-preview";
import { type ComparisonResult, VersionComparison } from "./version-comparison";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const capStep: PipelineStep = {
  id: "s1", position: 0, op: "outliers.cap", op_version: "1", op_title: "Cap outliers", catalog_stage: "either",
  stage: "data", columns: ["a"], params: {}, risk: "high", destructive: true, kind: "fitted", step_hash: "x",
  approved: false, from_recommendation_id: null,
};

const previewStep: PreviewStep = {
  index: 0, op: "outliers.cap", op_title: "Cap outliers", risk: "high", destructive: true, fitted: true,
  rows_before: 100, rows_after: 100, columns_added: [], columns_removed: [], rows_affected: 1500,
  rows_listed: 1000, rows_truncated: true, information_loss: ["Observed values overwritten (cannot be recovered from this version)."],
  warnings: [],
  columns: [{ column: "a", rows_changed: 1500, missing_before: 0, missing_after: 0, estimates_filled: 0,
    observed_overwritten: 1500, observed_re_expressed: 0, before: { median: 10 }, after: { median: 9.5 } },
  { column: "code", rows_changed: 3, missing_before: 3, missing_after: 0, estimates_filled: 3,
    observed_overwritten: 0, observed_re_expressed: 0, before: { redacted: true }, after: { redacted: true } }],
};

const summary: PreviewSummary = {
  spec_hash: "h", expected_content_hash: "c", fitted_on_full_data: true,
  totals: { rows: 100, columns_before: 4, columns_after: 5, estimates_filled: 3, observed_overwritten: 1500,
    observed_re_expressed: 0, columns_added: ["b__missing"], columns_removed: [] },
  steps: [previewStep],
  model_steps: [{ op: "scale.standard", columns: ["a"], note: "Stored as the version's modelling specification; does not change the saved data." }],
  max_rows_per_step: 1000,
  limitations: ["Fewer missing values or extreme values is not evidence of better predictive performance."],
};

function withClient(node: React.ReactNode) {
  return <QueryClientProvider client={new QueryClient()}>{node}</QueryClientProvider>;
}

describe("Impact preview", () => {
  it("shows totals, full-data fitting, information loss and the modelling specification", () => {
    render(<ImpactPreviewPanel pipelineId="p" summary={summary} current={false} />);
    expect(screen.getByRole("alert").textContent).toMatch(/out of date/);
    expect(screen.getByText("Fitted on all rows")).toBeTruthy();
    expect(screen.getByText(/Observed values overwritten \(cannot be recovered/)).toBeTruthy();
    expect(screen.getByText("hidden (sensitive)")).toBeTruthy();
    expect(screen.getAllByText(/does not change the saved data/).length).toBeGreaterThan(0);
    expect(screen.getByText(/not evidence of better predictive performance/)).toBeTruthy();
  });

  it("loads affected rows and states truncation and redaction", async () => {
    vi.spyOn(api, "GET").mockResolvedValue({
      data: {
        step_index: 0, columns: ["a", "code"], redacted_columns: ["code"], rows_affected: 1500, rows_listed: 1000,
        rows_truncated: true, next_offset: 50,
        rows: [{ row_id: 3, before: { a: 80, code: "[redacted]" }, after: { a: 18, code: "[redacted]" } }],
      },
      response: new Response(null, { status: 200 }),
    } as never);
    render(<AffectedRows pipelineId="p" step={previewStep} />);
    fireEvent.click(screen.getByRole("button", { name: "Show affected rows" }));
    await waitFor(() => expect(screen.getByRole("note")).toBeTruthy());
    const note = screen.getByRole("note").textContent ?? "";
    expect(note).toMatch(/1500 rows affected/);
    expect(note).toMatch(/limited to the first 1000 rows/);
    expect(note).toMatch(/sensitive columns are hidden: code/);
    expect(screen.getByText("80 → 18")).toBeTruthy();
    expect(screen.getByText("[redacted] → [redacted]")).toBeTruthy();
  });
});

describe("Approvals", () => {
  it("requires the destructive acknowledgement before approving", () => {
    const onApprove = vi.fn();
    render(<StepApprovalDialog step={capStep} impact={previewStep} onApprove={onApprove} onClose={vi.fn()} />);
    const button = screen.getByRole("button", { name: "Approve destructive step" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(screen.getByText("1500")).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(button);
    expect(onApprove).toHaveBeenCalledWith(true);
  });

  it("does not approve on opening and can be cancelled", () => {
    const onConfirm = vi.fn();
    const onClose = vi.fn();
    render(
      <ConfirmDialog title="Revert to version 0" confirmLabel="Create revert version" onConfirm={onConfirm} onClose={onClose}>
        <p>No version is deleted.</p>
      </ConfirmDialog>,
    );
    expect(onConfirm).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Create revert version" }));
    expect(onConfirm).toHaveBeenCalledWith(false);
  });

  it("offers approval only with a current preview and lets 'either' steps choose a stage", () => {
    const base = {
      steps: [capStep],
      readiness: { ready: false, preview_current: false, high_risk_steps: [], confirmation_required: false,
        confirmation_valid: false, stale_approvals: 1, problems: [] },
    } as unknown as Pipeline;
    const onPatch = vi.fn();
    const onApprove = vi.fn();
    render(
      <ul>
        <StepEditor pipeline={base} step={capStep} index={0} op={undefined} editable onPatch={onPatch} onRemove={vi.fn()} onApprove={onApprove} />
      </ul>,
    );
    expect((screen.getByRole("button", { name: "Approve…" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Destructive")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Stage of outliers.cap"), { target: { value: "model" } });
    expect(onPatch).toHaveBeenCalledWith({ stage: "model" });
    expect(screen.queryByText("Approved")).toBeNull();
  });
});

describe("History", () => {
  const versions = [
    { id: "v0", version_number: 0, kind: "ingest", n_rows: 10, n_cols: 3, created_at: "", parent_version_id: null,
      restored_from_version_id: null, transformation_run_id: null, fitted_on_full_data: false, has_model_spec: false },
    { id: "v2", version_number: 2, kind: "revert", n_rows: 10, n_cols: 3, created_at: "", parent_version_id: "v1",
      restored_from_version_id: "v0", transformation_run_id: null, fitted_on_full_data: false, has_model_spec: false },
    { id: "v1", version_number: 1, kind: "transform", n_rows: 10, n_cols: 4, created_at: "", parent_version_id: "v0",
      restored_from_version_id: null, transformation_run_id: "r", fitted_on_full_data: true, has_model_spec: true },
  ] as VersionSummary[];

  it("labels kinds, the current version, full-data fitting and reverts", () => {
    render(
      <div>
        <VersionBadges v={versions[1]} isHead versions={versions} />
        <VersionBadges v={versions[2]} isHead={false} versions={versions} />
      </div>,
    );
    expect(screen.getByText("Revert")).toBeTruthy();
    expect(screen.getByText("Current")).toBeTruthy();
    expect(screen.getByText("Restores v0")).toBeTruthy();
    expect(screen.getByText("Fitted on all rows")).toBeTruthy();
    expect(screen.getByText("Modelling specification")).toBeTruthy();
  });

  it("downloads generated code and reports failures", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("print(1)", { status: 200 }));
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
    expect(await downloadCode("v1", 1)).toBe(true);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/v1\/versions\/v1\/code$/);
    fetchMock.mockResolvedValue(new Response("no", { status: 404 }));
    expect(await downloadCode("v1", 1)).toBe(false);
  });

  it("shows comparisons with honest unassessed tests", () => {
    const result: ComparisonResult = {
      rows_before: 10, rows_after: 10, rows_affected: 4, columns_added: ["b__missing"], columns_removed: [],
      columns_type_changed: [], information_loss: ["Missing values were filled (estimates)."],
      limitations: ["Both versions describe the same rows, so they are not independent samples: p-values are descriptive only."],
      version_numbers: [0, 1], lineage: [{ version_number: 0, kind: "ingest" }, { version_number: 1, kind: "transform" }],
      columns: [
        { column: "b", type_before: "numeric", type_after: "numeric", values_changed: 0, filled: 4, emptied: 0,
          before: { missing_pct: 40, mean: 1 }, after: { missing_pct: 0, mean: 1.1 },
          test: { status: "assessed", test: "kolmogorov_smirnov", p_value: 0.5, effect_size: { ks_d: 0.1 } } },
        { column: "k", type_before: "numeric", type_after: "numeric", values_changed: 0, filled: 0, emptied: 0,
          before: { missing_pct: 0 }, after: { missing_pct: 0 },
          test: { status: "not_assessed", reason: "The column is constant in both versions." } },
      ],
    };
    render(<VersionComparison result={result} />);
    expect(screen.getByText(/v0 \(ingest\) → v1 \(transform\)/)).toBeTruthy();
    expect(screen.getByText("not assessed: The column is constant in both versions.")).toBeTruthy();
    expect(screen.getByText(/KS p=0.5/)).toBeTruthy();
    expect(screen.getByText(/not independent samples/)).toBeTruthy();
  });
});

describe("Experiment Lab bridge", () => {
  it("offers 'To pipeline' for every non-baseline candidate", () => {
    const onToPipeline = vi.fn();
    const experiment = {
      id: "e", name: "Lab", task_type: "binary", models: ["linear"], job: { status: "succeeded" }, completed_at: "x",
      split: null,
      summary: { metrics: ["roc_auc"], primary_metric: "roc_auc", metric_info: {}, cohort: { rows: 1 }, warnings: [],
        split_warnings: [], limitations: [], model_notes: {}, comparisons: [] },
      candidates: [
        { id: "c0", key: "baseline", label: "Baseline", is_baseline: true, max_risk: "low", has_destructive: false, result: null },
        { id: "c1", key: "c1", label: "Core", is_baseline: false, max_risk: "medium", has_destructive: false, result: null },
      ],
    } as unknown as ExperimentDetail;
    render(withClient(<ExperimentResults experiment={experiment} onCancel={vi.fn()} onRerun={vi.fn()} onToPipeline={onToPipeline} />));
    expect(screen.queryByRole("button", { name: "To pipeline: Baseline" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "To pipeline: Core" }));
    expect(onToPipeline).toHaveBeenCalledWith("c1");
    expect(screen.getByText(/evidence, not approval/)).toBeTruthy();
  });
});
