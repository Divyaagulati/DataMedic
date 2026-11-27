import type { components } from "@/lib/api/schema";
import type { FindingView } from "@/lib/diagnostics";

export type TrainingRun = components["schemas"]["TrainingRunOut"];
export type TrainingSummary = components["schemas"]["TrainingRunSummaryOut"];
export type TrainingPlan = components["schemas"]["TrainingPlanOut"];
export type TrainingIn = components["schemas"]["TrainingRunIn"];
export type ModelEvaluation = components["schemas"]["ModelEvaluationOut"];
export type Audit = components["schemas"]["AuditOut"];
export type ModelKey = "dummy" | "linear" | "random_forest" | "hgb";

export const MODEL_KEYS: ModelKey[] = ["dummy", "linear", "random_forest", "hgb"];
export const MODEL_LABELS: Record<string, string> = {
  dummy: "Dummy baseline",
  linear: "Linear",
  random_forest: "Random forest",
  hgb: "Gradient boosting",
};

export type TrainingSettings = {
  name: string;
  models: ModelKey[];
  tuning: "nested" | "defaults";
  method: "kfold" | "stratified" | "group" | "stratified_group" | "time" | null;
  nSplits: number;
  seed: number;
  allowExploratory: boolean;
};

export const DEFAULT_TRAINING: TrainingSettings = {
  name: "",
  models: [...MODEL_KEYS],
  tuning: "nested",
  method: null,
  nSplits: 5,
  seed: 0,
  allowExploratory: false,
};

export type StepDraft = { op: string; columns: string[]; params: Record<string, unknown> };

/** Request body. ``steps === null`` keeps the version's stored modelling specification. */
export function trainingPayload(s: TrainingSettings, steps: StepDraft[] | null): TrainingIn {
  return {
    name: s.name.trim() || null,
    models: s.models.length ? s.models : null,
    tuning: { mode: s.tuning },
    split: { method: s.method, n_splits: s.nSplits, seed: s.seed },
    preprocess_steps: steps,
    seed: s.seed,
    allow_exploratory: s.allowExploratory,
  };
}

/** What "leakage-controlled" means; shown with every leakage-controlled label. */
export const LEAKAGE_CONTROL_SCOPE =
  "Leakage-controlled refers to the implemented safeguards: frozen folds that keep declared groups, duplicates and time order apart, and every learned step, setting and model fitted on training rows only. It does not guarantee external validity, the absence of bias or clinical reliability.";

// Internal identifiers (unbiased_direct, unbiased_replay) are kept for compatibility; only the
// user-facing wording changed.
export const MODE_INFO: Record<string, { label: string; tone: "ok" | "warn" | "bad"; help: string }> = {
  unbiased_direct: {
    label: "Leakage-controlled evaluation",
    tone: "ok",
    help: `The version was not fitted on all of its rows; it is evaluated as stored. ${LEAKAGE_CONTROL_SCOPE}`,
  },
  unbiased_replay: {
    label: "Leakage-controlled evaluation (replayed)",
    tone: "ok",
    help: `The version's committed steps learned from all rows. They are re-run inside every training fold from an earlier version, after proving the replay reproduces the version. ${LEAKAGE_CONTROL_SCOPE}`,
  },
  exploratory: {
    label: "Exploratory only",
    tone: "bad",
    help:
      "Evaluated on a version whose committed steps learned from every row, including test rows. Scores are likely optimistic and are not a leakage-controlled estimate.",
  },
  blocked: {
    label: "Blocked",
    tone: "bad",
    help: "Leakage-controlled evaluation is not possible for this version.",
  },
};

export const STATUS_INFO: Record<string, { label: string; tone: "ok" | "warn" | "bad" }> = {
  blocked: { label: "Blocked: violation detected", tone: "bad" },
  exploratory: { label: "Exploratory only", tone: "bad" },
  concerns: { label: "Concerns found", tone: "warn" },
  no_issues_detected_among_assessed: { label: "No issues among the assessed checks", tone: "ok" },
};

export const BASIS_LABELS: Record<string, string> = {
  confirmed: "Confirmed fact",
  configuration: "Configuration fact",
  statistical: "Statistical suspicion",
};

export type FindingGroups = {
  violations: FindingView[];
  suspicions: FindingView[];
  passed: FindingView[];
  notAssessed: FindingView[];
  errors: FindingView[];
};

export function basisOf(f: FindingView): string {
  const b = (f.evidence as Record<string, unknown> | null)?.basis;
  return typeof b === "string" ? b : "statistical";
}

/** Detected facts vs suspicions; checks that passed within their scope; not assessed. */
export function groupFindings(findings: FindingView[]): FindingGroups {
  const g: FindingGroups = { violations: [], suspicions: [], passed: [], notAssessed: [], errors: [] };
  for (const f of findings) {
    if (f.status === "detected") (basisOf(f) === "statistical" ? g.suspicions : g.violations).push(f);
    else if (f.status === "not_detected") g.passed.push(f);
    else if (f.status === "not_assessed") g.notAssessed.push(f);
    else g.errors.push(f);
  }
  return g;
}

export type Stats = {
  mean: number | null;
  sd: number | null;
  min: number | null;
  max: number | null;
  n_folds: number;
  values: (number | null)[];
};

export function fmt(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function fmtStats(s: Stats | undefined, digits = 3): string {
  if (!s || s.mean === null) return "—";
  return s.sd === null ? fmt(s.mean, digits) : `${fmt(s.mean, digits)} ± ${fmt(s.sd, digits)}`;
}

/** A metric is unavailable in some folds: the reasons, deduplicated. */
export function metricReasons(evaluation: ModelEvaluation, metric: string): string[] {
  const reasons = (evaluation.metrics.reasons as Record<string, { fold: number; reason: string }[]> | undefined)?.[
    metric
  ];
  return [...new Set((reasons ?? []).map((r) => `fold ${r.fold + 1}: ${r.reason}`))];
}

export function foldStats(evaluation: ModelEvaluation, metric: string): Stats | undefined {
  return (evaluation.metrics.folds as Record<string, Stats> | undefined)?.[metric];
}

export function interval(evaluation: ModelEvaluation, metric: string): { low: number; high: number } | null {
  const b = evaluation.metrics.bootstrap as { available?: boolean; metrics?: Record<string, { low: number; high: number } | null> };
  return b?.available ? (b.metrics?.[metric] ?? null) : null;
}

export function isActive(status: string | undefined): boolean {
  return status === "queued" || status === "running" || status === "cancel_requested";
}

export function paramsLabel(params: Record<string, unknown> | null | undefined): string {
  const entries = Object.entries(params ?? {});
  return entries.length ? entries.map(([k, v]) => `${k}=${String(v)}`).join(", ") : "no settings";
}

/** SVG path of a polyline in a unit square drawn into a ``size`` box (y up). */
export function linePath(points: number[][], size: number): string {
  return points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${(x * size).toFixed(1)},${((1 - y) * size).toFixed(1)}`)
    .join(" ");
}

// ---- Headline (list of runs) --------------------------------------------------------------------

export type Headline = {
  state: "available" | "unavailable" | "pending" | "failed" | "cancelled";
  metric: string;
  label: string;
  higher_is_better: boolean;
  model: string | null;
  mean: number | null;
  sd: number | null;
  baseline_mean: number | null;
  n_folds: number | null;
  total_folds: number | null;
  exploratory: boolean;
  note: string;
  exploratory_note: string | null;
  reason: string | null;
};

/** Text, tooltip and qualifier of a run's headline. Never a declared winner: "highest observed". */
export function headlineView(raw: unknown): { text: string; title: string; qualifier: string | null } {
  const h = raw as Headline | null | undefined;
  if (!h || !h.state) return { text: "—", title: "No headline available.", qualifier: null };
  if (h.state !== "available" || h.mean === null) {
    const text = { pending: "Pending", cancelled: "Cancelled", failed: "No results" }[h.state as string] ?? "Unavailable";
    return { text, title: h.reason ?? "", qualifier: null };
  }
  const arrow = h.higher_is_better ? "↑" : "↓";
  const sd = h.sd === null ? "" : ` ± ${fmt(h.sd)}`;
  const folds =
    h.n_folds !== null && h.total_folds !== null && h.n_folds < h.total_folds ? ` (${h.n_folds} of ${h.total_folds} folds)` : "";
  const text = `${h.label} ${arrow} ${fmt(h.mean)}${sd}${folds} · highest observed: ${MODEL_LABELS[h.model ?? ""] ?? h.model}`;
  const baseline = h.baseline_mean === null ? "" : ` Dummy baseline: ${fmt(h.baseline_mean)}.`;
  const direction = h.higher_is_better ? "Higher is better." : "Lower is better.";
  const title = [h.note, direction + baseline, h.exploratory_note].filter(Boolean).join(" ");
  return { text, title, qualifier: h.exploratory ? "Exploratory: likely optimistic" : null };
}
