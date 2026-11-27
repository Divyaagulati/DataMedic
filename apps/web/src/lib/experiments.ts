import type { components } from "@/lib/api/schema";

export type RecommendationSet = components["schemas"]["RecommendationSetOut"];
export type RecommendationItem = components["schemas"]["RecommendationOut"];
export type ExperimentDetail = components["schemas"]["ExperimentOut"];
export type ExperimentSummary = components["schemas"]["ExperimentSummaryOut"];
export type ExperimentIn = components["schemas"]["ExperimentIn"];

/** The full recommendation produced by the engine (``RecommendationOut.detail``). */
export interface RecommendationDetail {
  id: string;
  title: string;
  op: string;
  op_title: string;
  what_it_does: string;
  applies_to: string;
  reason: string;
  evidence: { source: string; metric: string; value: unknown; column?: string | null }[];
  expected_benefit: string;
  risks: string[];
  information_loss: { level: "none" | "low" | "moderate" | "high"; detail: string };
  domain_compatibility: { status: "compatible" | "caution"; detail: string };
  task_compatibility: { status: "relevant" | "indirect"; detail: string };
  required_decisions: string[];
  evidence_strength: { level: "strong" | "moderate" | "weak"; explanation: string };
  limitations: string[];
  not_a_guarantee: string;
  conflict_group: string | null;
  alternatives: string[];
  ranking: { score: number; factors: { factor: string; points: number; detail: string }[] };
  kind: string;
}

export interface StepDraft {
  op: string;
  columns: string[];
  params: Record<string, unknown>;
}

export interface CandidateDraft {
  id: string;
  label: string;
  steps: StepDraft[];
  recommendationIds: string[];
}

export const MAX_CANDIDATES = 6;

export const EVIDENCE_LABELS: Record<RecommendationDetail["evidence_strength"]["level"], string> = {
  strong: "Strong evidence",
  moderate: "Moderate evidence",
  weak: "Weak evidence",
};

export const LOSS_LABELS: Record<RecommendationDetail["information_loss"]["level"], string> = {
  none: "No information lost",
  low: "Little information lost",
  moderate: "Some information lost",
  high: "Substantial information lost",
};

export const VERDICT_LABELS: Record<string, string> = {
  negligible: "No practical difference",
  consistently_better: "Better in every fold",
  consistently_worse: "Worse in every fold",
  within_fold_variation: "Within fold-to-fold variation",
  mixed: "Mixed across folds",
  insufficient_folds: "Too few folds to compare",
};

/** Pipeline position of an op family (indicators before imputation, resampling last). */
const FAMILY_ORDER: Record<string, number> = {
  missing: 0,
  impute: 1,
  outliers: 2,
  transform: 3,
  encode: 4,
  scale: 5,
  select: 6,
  imbalance: 7,
};

export function stepPosition(op: string): number {
  return FAMILY_ORDER[op.split(".")[0]] ?? 99;
}

export const isTrainingOnly = (op: string): boolean => op.startsWith("imbalance.");

let counter = 0;
export function newCandidateId(): string {
  counter += 1;
  return `cand-${Date.now().toString(36)}-${counter}`;
}

/**
 * Add a step in pipeline order. A pipeline holds at most one training-only step
 * (resampling or class weights), and it must be last: adding another replaces it.
 */
export function addStep(candidate: CandidateDraft, step: StepDraft, recId?: string): CandidateDraft {
  let steps = candidate.steps.filter(
    (s) => !(isTrainingOnly(step.op) && isTrainingOnly(s.op)) && !sameStep(s, step),
  );
  const index = steps.findIndex((s) => stepPosition(s.op) > stepPosition(step.op));
  steps = index === -1 ? [...steps, step] : [...steps.slice(0, index), step, ...steps.slice(index)];
  const recommendationIds =
    recId && !candidate.recommendationIds.includes(recId)
      ? [...candidate.recommendationIds, recId]
      : candidate.recommendationIds;
  return { ...candidate, steps, recommendationIds };
}

function sameStep(a: StepDraft, b: StepDraft): boolean {
  return a.op === b.op && a.columns.join("\u0000") === b.columns.join("\u0000");
}

export function removeStep(candidate: CandidateDraft, index: number): CandidateDraft {
  return { ...candidate, steps: candidate.steps.filter((_, i) => i !== index) };
}

/** Candidate drafts from the engine's suggestions (the baseline is always added by the server). */
export function draftsFromSuggestions(set: RecommendationSet): CandidateDraft[] {
  const keyToId = new Map(set.recommendations.map((r) => [r.key, r.id]));
  return set.candidates
    .filter((c) => Array.isArray(c.steps) && (c.steps as unknown[]).length > 0)
    .slice(0, MAX_CANDIDATES)
    .map((c) => ({
      id: newCandidateId(),
      label: String(c.label),
      steps: (c.steps as StepDraft[]).map((s) => ({
        op: s.op,
        columns: [...s.columns],
        params: { ...s.params },
      })),
      recommendationIds: ((c.recommendation_ids as string[]) ?? [])
        .map((k) => keyToId.get(k))
        .filter((v): v is string => Boolean(v)),
    }));
}

export interface LabSettings {
  name: string;
  models: string[];
  method: string | null;
  nSplits: number;
  seed: number;
  clustering: boolean;
  nClusters: number;
}

export function experimentPayload(
  drafts: CandidateDraft[],
  settings: LabSettings,
  recommendationSetId: string | null,
): ExperimentIn {
  return {
    name: settings.name.trim() || "Experiment",
    models: settings.models as ExperimentIn["models"],
    task: settings.clustering ? "clustering" : null,
    n_clusters: settings.clustering ? settings.nClusters : null,
    seed: settings.seed,
    split: {
      method: (settings.method || null) as components["schemas"]["SplitIn"]["method"],
      n_splits: settings.nSplits,
      seed: settings.seed,
    },
    recommendation_set_id: recommendationSetId,
    candidates: drafts.map((d) => ({
      label: d.label.trim(),
      steps: d.steps,
      recommendation_ids: d.recommendationIds,
    })),
  };
}

export interface MetricStats {
  mean: number | null;
  sd: number | null;
  n_folds: number;
  values: (number | null)[];
}

/** "0.712 ± 0.034 (5 folds)", or a dash when nothing was scored. */
export function formatStats(stats: MetricStats | undefined, digits = 3): string {
  if (!stats || stats.mean === null) return "—";
  const sd = stats.sd === null ? "" : ` ± ${stats.sd.toFixed(digits)}`;
  return `${stats.mean.toFixed(digits)}${sd}`;
}

export function formatValue(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

/** Selection frequency class (stability.py thresholds). */
export function stabilityClass(frequency: number): "stable" | "unstable" | "rarely" {
  if (frequency >= 0.8) return "stable";
  if (frequency <= 0.2) return "rarely";
  return "unstable";
}

export const ACTIVE_JOB_STATUSES = new Set(["queued", "running", "cancel_requested"]);
