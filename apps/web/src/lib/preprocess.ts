import type { components } from "@/lib/api/schema";

export type Op = components["schemas"]["OpOut"];
export type OpParam = components["schemas"]["OpParamOut"];

/** Plain-language explanations used as labels and tooltips in the Preprocessing workspace. */
export const RISK_INFO: Record<Op["risk"], { label: string; help: string }> = {
  low: {
    label: "Low risk",
    help: "Adds information without changing existing values (for example a flag column).",
  },
  medium: {
    label: "Medium risk",
    help: "Introduces estimated values. Needs one confirmation before it is saved as a new version.",
  },
  high: {
    label: "High risk",
    help: "Removes rows or columns or overwrites observed values. Each step needs its own approval.",
  },
};

export const STAGE_INFO: Record<Op["stage"], { label: string; help: string }> = {
  data: {
    label: "Data cleaning",
    help: "Can be saved into a new dataset version after approval.",
  },
  model: {
    label: "Model preparation",
    help: "Runs only inside model training, learned from training folds. Never saved into a version.",
  },
  either: {
    label: "Cleaning or model preparation",
    help:
      "Can run inside model training (recommended for evaluation) or be saved into a new version. " +
      "A saved version that learned from all rows must not be used directly for leakage-controlled model evaluation.",
  },
};

export const KIND_INFO: Record<Op["kind"], string> = {
  fitted: "Learns values from data. In experiments it learns from the training rows only.",
  stateless_column: "Learns nothing from data; applies the same rule to every row.",
  stateless_row: "Learns nothing from data; applies the same rule to every row.",
  row_filter_rule: "Removes rows matching a fixed rule. In experiments only training rows are removed.",
  row_filter_fitted:
    "Removes rows using a threshold learned from data. In experiments only training rows are removed.",
  training_only:
    "Changes only the rows a model is trained on (resampling or class weights), inside each training fold. Validation and test rows and saved versions are never changed.",
};

/** Ops that accept a column of the given semantic type (from the version schema). */
export function opsForColumnType(ops: Op[], inferredType: string): Op[] {
  return ops.filter((op) => op.column_types.includes(inferredType));
}

/** Group ops by family, keeping catalogue order. */
export function groupByFamily(ops: Op[]): Map<string, Op[]> {
  const groups = new Map<string, Op[]>();
  for (const op of ops) {
    const list = groups.get(op.family) ?? [];
    list.push(op);
    groups.set(op.family, list);
  }
  return groups;
}

/** Initial parameter values for a step: catalogue defaults; required params left unset. */
export function defaultParams(op: Op): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const p of op.params) {
    if (p.default !== null && p.default !== undefined) params[p.name] = p.default;
  }
  return params;
}

export function missingRequiredParams(op: Op, params: Record<string, unknown>): string[] {
  return op.params
    .filter((p) => p.required && (params[p.name] === undefined || params[p.name] === null))
    .map((p) => p.name);
}

/** Name of the column a missing-value indicator adds (mirrors the backend). */
export function indicatorColumn(column: string): string {
  return `${column}__missing`;
}
