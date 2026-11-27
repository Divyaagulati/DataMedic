import type { Schemas } from "@/lib/api/client";

export type FindingView = Schemas["FindingOut"];
export type Severity = FindingView["severity"];
export type FindingStatus = FindingView["status"];

export const SEVERITIES: Severity[] = ["critical", "high", "medium", "low", "info"];

export const SEVERITY_LABELS: Record<Severity, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  info: "Informational",
};

/** Status palette (reserved for state, never for data series). Always paired with a label. */
export const SEVERITY_COLORS: Record<Severity, string> = {
  critical: "var(--sev-critical)",
  high: "var(--sev-high)",
  medium: "var(--sev-medium)",
  low: "var(--sev-low)",
  info: "var(--sev-info)",
};

export const STATUS_LABELS: Record<FindingStatus, string> = {
  detected: "Detected",
  not_detected: "Not detected",
  not_assessed: "Not assessed",
  error: "Check error",
};

export const STATUS_ORDER: FindingStatus[] = ["detected", "not_detected", "not_assessed", "error"];

export type Filters = {
  status: FindingStatus;
  severity: Severity | "all";
  category: string | "all";
  domain: string | "all";
};

export function filterFindings(findings: FindingView[], f: Filters): FindingView[] {
  return findings
    .filter(
      (x) =>
        x.status === f.status &&
        (f.severity === "all" || x.severity === f.severity) &&
        (f.category === "all" || x.category === f.category) &&
        (f.domain === "all" || x.domain === f.domain),
    )
    .sort(
      (a, b) =>
        SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) ||
        a.check_id.localeCompare(b.check_id),
    );
}

export function countBy<T extends string>(items: FindingView[], key: (f: FindingView) => T): Map<T, number> {
  const out = new Map<T, number>();
  for (const item of items) out.set(key(item), (out.get(key(item)) ?? 0) + 1);
  return out;
}

const LABEL_OVERRIDES: Record<string, string> = {
  pct: "%",
  iqr: "IQR",
  vif: "VIF",
  roc_auc: "ROC AUC",
  p_value: "p-value",
  q_value: "q-value (FDR)",
};

/** "rows_in_duplicate_groups" -> "Rows in duplicate groups"; "missing_pct" -> "Missing %". */
export function evidenceLabel(key: string): string {
  if (LABEL_OVERRIDES[key]) return LABEL_OVERRIDES[key];
  const words = key.split("_").map((w) => LABEL_OVERRIDES[w] ?? w);
  const text = words.join(" ").replace(" %", " %");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Compact, readable rendering of a scalar evidence value. */
export function evidenceScalar(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString("en");
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) return value.toExponential(2);
    return new Intl.NumberFormat("en", { maximumSignificantDigits: 4 }).format(value);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

export function severityRank(s: Severity): number {
  return SEVERITIES.indexOf(s);
}
