"use client";

import { useState } from "react";

import { RiskBadge } from "@/components/lab/recommendation-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api, unwrap } from "@/lib/api/client";
import { cell, type PreviewRows, type PreviewStep, type PreviewSummary } from "@/lib/pipelines";

function Totals({ summary }: { summary: PreviewSummary }) {
  const t = summary.totals;
  return (
    <ul className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
      <li>
        <span className="block text-xs text-muted-foreground">Rows (unchanged)</span>
        {t.rows}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Columns</span>
        {t.columns_before} → {t.columns_after}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Missing values replaced by estimates</span>
        {t.estimates_filled}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Observed values overwritten</span>
        {t.observed_overwritten}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Values re-expressed (invertible)</span>
        {t.observed_re_expressed}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Columns created</span>
        {t.columns_added.join(", ") || "none"}
      </li>
      <li>
        <span className="block text-xs text-muted-foreground">Columns removed</span>
        {t.columns_removed.join(", ") || "none"}
      </li>
    </ul>
  );
}

export function AffectedRows({ pipelineId, step }: { pipelineId: string; step: PreviewStep }) {
  const [rows, setRows] = useState<PreviewRows | null>(null);
  const [error, setError] = useState<string | null>(null);
  async function load() {
    try {
      setRows(
        unwrap(
          await api.GET("/api/v1/pipelines/{pipeline_id}/impact-preview/rows", {
            params: { path: { pipeline_id: pipelineId }, query: { step: step.index, limit: 50 } },
          }),
        ),
      );
    } catch {
      setError("The affected rows could not be loaded.");
    }
  }
  if (step.rows_affected === 0) return <p className="text-xs text-muted-foreground">No row is affected.</p>;
  if (!rows)
    return (
      <div>
        <Button size="sm" variant="outline" onClick={load}>
          Show affected rows
        </Button>
        {error ? <p className="text-xs text-destructive">{error}</p> : null}
      </div>
    );
  return (
    <div className="space-y-1 overflow-x-auto">
      <p className="text-xs text-muted-foreground" role="note">
        {rows.rows_affected} rows affected; showing {rows.rows.length} of {rows.rows_listed} listed
        {rows.rows_truncated ? ` (the list is limited to the first ${rows.rows_listed} rows by row id; counts cover every row)` : ""}.
        {rows.redacted_columns.length ? ` Values of sensitive columns are hidden: ${rows.redacted_columns.join(", ")}.` : ""}
      </p>
      <table className="text-xs">
        <thead>
          <tr className="text-left">
            <th className="pr-2">Row</th>
            {rows.columns.map((c) => (
              <th key={c} className="pr-2">
                {c} (before → after)
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.rows.map((r) => (
            <tr key={String(r.row_id)} className="border-t">
              <td className="pr-2 font-mono">{String(r.row_id)}</td>
              {rows.columns.map((c) => (
                <td key={c} className="pr-2 font-mono">
                  {c in (r.before as object) ? cell((r.before as Record<string, unknown>)[c]) : "—"} →{" "}
                  {cell((r.after as Record<string, unknown>)[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ImpactPreviewPanel({
  pipelineId,
  summary,
  current,
}: {
  pipelineId: string;
  summary: PreviewSummary;
  current: boolean;
}) {
  return (
    <div className="space-y-4">
      {!current ? (
        <p className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-sm dark:bg-amber-950/30" role="alert">
          This preview is out of date: the pipeline or the dataset changed after it was computed. Generate it again; it
          cannot back an approval.
        </p>
      ) : null}
      <Totals summary={summary} />
      {summary.fitted_on_full_data ? (
        <p className="text-sm">
          <Badge variant="outline">Fitted on all rows</Badge> Steps that learn from data will be fitted on every row of the
          current version. Do not use the new version directly for leakage-controlled model evaluation; evaluate in the Experiment
          Lab instead, where steps are fitted inside training folds.
        </p>
      ) : null}
      <ol className="space-y-3">
        {summary.steps.map((s) => (
          <li key={s.index} className="space-y-2 rounded-md border p-3 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">
                {s.index + 1}. {s.op_title}
              </span>
              <RiskBadge risk={s.risk} destructive={s.destructive} />
              <span className="text-muted-foreground">{s.rows_affected} rows affected</span>
            </div>
            {s.information_loss.length ? (
              <ul className="list-disc pl-5 text-xs">
                {s.information_loss.map((l) => (
                  <li key={l}>{l}</li>
                ))}
              </ul>
            ) : null}
            <table className="text-xs">
              <thead>
                <tr className="text-left">
                  <th className="pr-2">Column</th>
                  <th className="pr-2">Missing before → after</th>
                  <th className="pr-2">Estimates</th>
                  <th className="pr-2">Overwritten</th>
                  <th className="pr-2">Re-expressed</th>
                  <th className="pr-2">Median before → after</th>
                </tr>
              </thead>
              <tbody>
                {s.columns.map((c) => (
                  <tr key={c.column} className="border-t">
                    <td className="pr-2">{c.column}</td>
                    <td className="pr-2">
                      {c.missing_before} → {c.missing_after}
                    </td>
                    <td className="pr-2">{c.estimates_filled}</td>
                    <td className="pr-2">{c.observed_overwritten}</td>
                    <td className="pr-2">{c.observed_re_expressed}</td>
                    <td className="pr-2 font-mono">
                      {"redacted" in c.before ? "hidden (sensitive)" : `${cell(c.before.median)} → ${cell(c.after.median)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <AffectedRows pipelineId={pipelineId} step={s} />
          </li>
        ))}
      </ol>
      {summary.model_steps.length ? (
        <section className="space-y-1 text-sm">
          <h4 className="font-medium">Modelling specification (does not change the saved data)</h4>
          <ul className="list-disc pl-5">
            {summary.model_steps.map((m, i) => (
              <li key={i}>
                {m.op} {m.columns.length ? `(${m.columns.join(", ")})` : ""} — {m.note}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <ul className="list-disc pl-5 text-xs text-muted-foreground">
        {summary.limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </div>
  );
}
