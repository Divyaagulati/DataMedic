import type { components } from "@/lib/api/schema";

export type Pipeline = components["schemas"]["PipelineOut"];
export type PipelineStep = components["schemas"]["PipelineStepOut"];
export type Approval = components["schemas"]["ApprovalOut"];
export type ImpactPreview = components["schemas"]["ImpactPreviewOut"];
export type PreviewRows = components["schemas"]["ImpactPreviewRowsOut"];
export type VersionSummary = components["schemas"]["VersionSummaryOut"];
export type TransformationRun = components["schemas"]["TransformationRunOut"];

/** One data-stage step of the impact preview (engine/preprocess/commit.py). */
export interface PreviewStep {
  index: number;
  op: string;
  op_title: string;
  risk: "low" | "medium" | "high";
  destructive: boolean;
  fitted: boolean;
  rows_before: number;
  rows_after: number;
  columns_added: string[];
  columns_removed: string[];
  rows_affected: number;
  rows_listed: number;
  rows_truncated: boolean;
  information_loss: string[];
  warnings: { code: string; message: string }[];
  columns: {
    column: string;
    rows_changed: number;
    missing_before: number;
    missing_after: number;
    estimates_filled: number;
    observed_overwritten: number;
    observed_re_expressed: number;
    before: Record<string, unknown>;
    after: Record<string, unknown>;
  }[];
}

export interface PreviewSummary {
  spec_hash: string;
  expected_content_hash: string;
  fitted_on_full_data: boolean;
  totals: {
    rows: number;
    columns_before: number;
    columns_after: number;
    estimates_filled: number;
    observed_overwritten: number;
    observed_re_expressed: number;
    columns_added: string[];
    columns_removed: string[];
  };
  steps: PreviewStep[];
  model_steps: { op: string; columns: string[]; note: string }[];
  max_rows_per_step: number;
  limitations: string[];
}

export const STAGE_CHOICE: Record<string, { label: string; help: string }> = {
  data: {
    label: "Changes the saved data",
    help: "Applied to every row when committed: the new version stores the result.",
  },
  model: {
    label: "Modelling only",
    help:
      "Stored with the new version as its modelling specification and run only inside model " +
      "training folds. It does not change the saved data.",
  },
};

export const VERSION_KIND: Record<string, string> = {
  ingest: "Original upload",
  transform: "Committed transformation",
  revert: "Revert",
  reparse: "Re-parse",
};

/** Steps that modify the saved dataset vs. steps stored only as a modelling specification. */
export function splitByStage(steps: PipelineStep[]): { data: PipelineStep[]; model: PipelineStep[] } {
  return {
    data: steps.filter((s) => s.stage === "data"),
    model: steps.filter((s) => s.stage === "model"),
  };
}

/** What the user still has to do before committing, in order. Server readiness is final. */
export function nextActions(p: Pipeline): string[] {
  const out: string[] = [];
  if (p.state === "committed") return out;
  if (!p.validation?.valid) out.push("Fix the validation errors.");
  if (!p.source_is_head) out.push("The dataset has a newer version: create a new pipeline from it.");
  if (!p.readiness.preview_current) out.push("Generate the impact preview for the current pipeline.");
  const pending = p.readiness.high_risk_steps.filter((s) => !s.approved);
  if (pending.length) out.push(`Approve ${pending.length} high-risk step(s) one by one.`);
  if (p.readiness.confirmation_required && !p.readiness.confirmation_valid)
    out.push("Confirm the pipeline (medium-risk steps).");
  return out;
}

export function canCommit(p: Pipeline): boolean {
  return p.state !== "committed" && p.readiness.ready && p.source_is_head;
}

/** The step's approval state for display: approval needed, approved, or not required. */
export function approvalState(p: Pipeline, step: PipelineStep): "required" | "approved" | "not_required" {
  if (step.risk !== "high") return "not_required";
  return step.approved && p.readiness.preview_current ? "approved" : "required";
}

export function moveStep(steps: PipelineStep[], index: number, delta: number): number | null {
  const target = index + delta;
  return target < 0 || target >= steps.length ? null : target;
}

export function cell(value: unknown): string {
  if (value === null || value === undefined) return "missing";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(6);
  return String(value);
}
