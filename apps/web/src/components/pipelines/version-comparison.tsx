"use client";

import { cell } from "@/lib/pipelines";

interface ColumnComparison {
  column: string;
  type_before: string;
  type_after: string;
  values_changed: number;
  filled: number;
  emptied: number;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
  test: {
    status: string;
    reason?: string;
    test?: string;
    statistic?: number | null;
    p_value?: number | null;
    effect_size?: Record<string, number | null>;
    note?: string | null;
  };
}

export interface ComparisonResult {
  rows_before: number;
  rows_after: number;
  rows_affected: number;
  columns_added: string[];
  columns_removed: string[];
  columns_type_changed: string[];
  columns: ColumnComparison[];
  information_loss: string[];
  limitations: string[];
  version_numbers: number[];
  lineage: { version_number: number; kind: string }[];
}

export function VersionComparison({ result }: { result: ComparisonResult }) {
  const [a, b] = result.version_numbers;
  return (
    <div className="space-y-3 text-sm">
      <p>
        Version {a} → version {b}: rows {result.rows_before} → {result.rows_after}; {result.rows_affected} rows have at
        least one changed value. Lineage: {result.lineage.map((l) => `v${l.version_number} (${l.kind})`).join(" → ")}.
      </p>
      <p>
        Columns added: {result.columns_added.join(", ") || "none"}. Removed: {result.columns_removed.join(", ") || "none"}.
        {result.columns_type_changed.length ? ` Type changed: ${result.columns_type_changed.join(", ")}.` : ""}
      </p>
      {result.information_loss.length ? (
        <ul className="list-disc pl-5">
          {result.information_loss.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      ) : null}
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr className="text-left">
              <th className="pr-2">Column</th>
              <th className="pr-2">Changed</th>
              <th className="pr-2">Filled</th>
              <th className="pr-2">Missing % before → after</th>
              <th className="pr-2">Mean before → after</th>
              <th className="pr-2">Outliers (IQR×1.5)</th>
              <th className="pr-2">Test</th>
              <th>Effect size</th>
            </tr>
          </thead>
          <tbody>
            {result.columns.map((c) => (
              <tr key={c.column} className="border-t align-top">
                <td className="pr-2">{c.column}</td>
                <td className="pr-2">{c.values_changed}</td>
                <td className="pr-2">{c.filled}</td>
                <td className="pr-2">
                  {cell(c.before.missing_pct)} → {cell(c.after.missing_pct)}
                </td>
                <td className="pr-2 font-mono">
                  {"mean" in c.before ? `${cell(c.before.mean)} → ${cell(c.after.mean)}` : "—"}
                </td>
                <td className="pr-2">
                  {"outliers_iqr_1_5" in c.before ? `${cell(c.before.outliers_iqr_1_5)} → ${cell(c.after.outliers_iqr_1_5)}` : "—"}
                </td>
                <td className="pr-2">
                  {c.test.status === "assessed"
                    ? `${c.test.test === "chi_square" ? "χ²" : "KS"} p=${cell(c.test.p_value)}${c.test.note ? " (approximation unreliable)" : ""}`
                    : `not assessed: ${c.test.reason}`}
                </td>
                <td>
                  {c.test.effect_size
                    ? Object.entries(c.test.effect_size)
                        .map(([k, v]) => `${k.replaceAll("_", " ")} ${cell(v)}`)
                        .join(", ")
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="list-disc pl-5 text-xs text-muted-foreground">
        {result.limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </div>
  );
}
