import type { Schemas } from "@/lib/api/client";

export type Profile = Schemas["ProfileOut"];
export type ColumnProfile = Schemas["ColumnProfileOut"];
export type Frequencies = Schemas["FrequenciesOut"];
export type SemanticType = ColumnProfile["inferred_type"];

/** Fixed categorical order for column types (never cycled; see the validated palette). */
export const TYPE_ORDER: SemanticType[] = [
  "numeric",
  "categorical",
  "boolean",
  "datetime",
  "text",
  "identifier",
  "empty",
];

export const TYPE_LABELS: Record<SemanticType, string> = {
  numeric: "Numeric",
  categorical: "Categorical",
  boolean: "Boolean",
  datetime: "Date/time",
  text: "Free text",
  identifier: "Identifier-like",
  empty: "Empty",
};

export function typeColor(type: SemanticType): string {
  return `var(--viz-cat-${TYPE_ORDER.indexOf(type) + 1})`;
}

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const precise = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });
const whole = new Intl.NumberFormat("en");

/** Readable number: integers in full below 10,000, compact above; decimals to 4 significant. */
export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Number.isInteger(value)) return Math.abs(value) < 10_000 ? whole.format(value) : compact.format(value);
  if (Math.abs(value) >= 10_000) return compact.format(value);
  return precise.format(value);
}

const stat = new Intl.NumberFormat("en", { maximumSignificantDigits: 6 });

/** Statistics tables: up to 6 significant digits, never compacted (123,456.8, not 123.5K). */
export function formatStat(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return stat.format(value);
}

/** One decimal. Never rounds a partial share up to 100% or down to 0%. */
export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value === 0) return "0%";
  if (value === 100) return "100%";
  if (value < 0.1) return "<0.1%";
  if (value > 99.9) return ">99.9%";
  return `${value.toFixed(1)}%`;
}

/** Plain-language skewness (descriptive thresholds 0.5 and 1, as commonly used). */
export function skewLabel(skew: number | null | undefined): string {
  if (skew === null || skew === undefined) return "not available";
  const side = skew > 0 ? "right" : "left";
  const size = Math.abs(skew);
  if (size < 0.5) return "approximately symmetric";
  if (size < 1) return `moderately skewed (${side})`;
  return `highly skewed (${side})`;
}

export type ClassBalance =
  | {
      ok: true;
      classes: { label: string; count: number; pct: number }[];
      majority: { label: string; count: number };
      minority: { label: string; count: number };
      /** majority count / minority count (1 = perfectly balanced) */
      ratio: number;
    }
  | { ok: false; reason: string };

export function cellLabel(value: unknown): string {
  if (value === null || value === undefined) return "(missing)";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

/** Class distribution for a candidate target column, or an honest reason it cannot be shown. */
export function classBalance(column: ColumnProfile): ClassBalance {
  const freq = column.frequencies;
  if (column.is_empty || !freq) return { ok: false, reason: "The column has no values." };
  if (column.inferred_type === "identifier") {
    return { ok: false, reason: "This column looks like an identifier, not a target." };
  }
  if (!freq.complete) {
    return {
      ok: false,
      reason:
        `The column has ${formatNumber(column.n_unique)} distinct values, too many for a class ` +
        "distribution (more than 50). It may be a regression target: see its histogram.",
    };
  }
  if (freq.values.length < 2) {
    return { ok: false, reason: "Only one class is present, so there is nothing to compare." };
  }
  const classes = freq.values.map((v) => ({ label: cellLabel(v.value), count: v.count, pct: v.pct }));
  const sorted = [...classes].sort((a, b) => b.count - a.count);
  const majority = sorted[0];
  const minority = sorted[sorted.length - 1];
  return { ok: true, classes, majority, minority, ratio: majority.count / minority.count };
}

/** Diverging fill for a correlation in [-1, 1]: blue (negative) <- grey -> red (positive). */
export function correlationColor(r: number | null): string {
  if (r === null) return "var(--viz-surface)";
  const strength = Math.round(Math.min(1, Math.abs(r)) * 100);
  const pole = r < 0 ? "var(--viz-div-neg)" : "var(--viz-div-pos)";
  return `color-mix(in oklab, ${pole} ${strength}%, var(--viz-div-mid))`;
}

export function missingRanking(columns: ColumnProfile[]): ColumnProfile[] {
  return columns.filter((c) => c.n_missing > 0).sort((a, b) => b.missing_pct - a.missing_pct);
}

/** Text for a histogram bin edge pair. */
export function binLabel(kind: "numeric" | "datetime", lo: number | string | null, hi: number | string | null): string {
  if (kind === "datetime") {
    const fmt = (v: number | string | null) => (typeof v === "string" ? v.slice(0, 10) : "—");
    return `${fmt(lo)} – ${fmt(hi)}`;
  }
  const fmt = (v: number | string | null) => (typeof v === "number" ? formatNumber(v) : "—");
  return `${fmt(lo)} – ${fmt(hi)}`;
}
