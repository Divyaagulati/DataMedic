"use client";

import { useState } from "react";

import { RiskBadge } from "@/components/lab/recommendation-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ACTIVE_JOB_STATUSES,
  type ExperimentDetail,
  formatStats,
  formatValue,
  type MetricStats,
  stabilityClass,
  VERDICT_LABELS,
} from "@/lib/experiments";

type Dict = Record<string, unknown>;
const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";

interface CandidateResult {
  status: string;
  comparable: boolean;
  comparability_reason: string | null;
  metrics: Record<string, Record<string, MetricStats>>;
  folds: { fold: number; status: string; error: { code: string; message: string } | null; step_warnings: string[] }[];
  data_quality: Dict | null;
  selection_stability: Stability[];
  failures: { fold: number; model?: string; code: string; message: string }[];
  per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }[]> | null;
}

interface Stability {
  step: number;
  op: string;
  folds: number;
  frequency: Record<string, number>;
  mean_jaccard: number | null;
  probe_selection_rate: number | null;
  indistinguishable_from_noise: string[];
  forced: string[];
  warnings: { code: string; message: string }[];
  limitations: string[];
}

interface Comparison {
  candidate: string;
  model: string;
  metric: string;
  per_fold_difference: number[];
  mean_difference: number | null;
  sd_difference: number | null;
  folds_better: number | null;
  folds_worse: number | null;
  verdict: string;
  interpretation: string;
}

const QUALITY_ROWS: [string, string][] = [
  ["missing_cells", "Missing cells (test rows)"],
  ["outlier_cells_iqr_1_5", "Values beyond IQR × 1.5"],
  ["mean_abs_skewness", "Mean |skewness|"],
  ["features", "Model input columns"],
  ["constant_columns", "Constant columns"],
  ["category_levels", "Category levels"],
  ["duplicate_rows", "Duplicate rows"],
];

function Warnings({ items }: { items: { code: string; message: string }[] }) {
  if (items.length === 0) return null;
  return (
    <ul className="space-y-1 text-sm">
      {items.map((w, i) => (
        <li key={`${w.code}-${i}`} className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1 dark:bg-amber-950/30">
          <span className="font-mono text-xs">{w.code}</span> {w.message}
        </li>
      ))}
    </ul>
  );
}

export function ExperimentResults({
  experiment,
  onCancel,
  onRerun,
  onToPipeline,
}: {
  experiment: ExperimentDetail;
  onCancel: () => void;
  onRerun: () => void;
  /** Create an editable pipeline from a candidate (nothing is committed or approved). */
  onToPipeline?: (candidateId: string) => void;
}) {
  const job = experiment.job;
  const summary = experiment.summary as Dict | null;
  const active = job ? ACTIVE_JOB_STATUSES.has(job.status) : false;
  const metricNames = (summary?.metrics as string[] | undefined) ?? [];
  const info = (summary?.metric_info ?? {}) as Record<string, { label: string; higher_is_better: boolean }>;
  const [metric, setMetric] = useState<string>((summary?.primary_metric as string) ?? "");
  const [model, setModel] = useState<string>(experiment.models.find((m) => m !== "dummy") ?? experiment.models[0]);
  const chosen = metric || (summary?.primary_metric as string) || metricNames[0];
  const candidates = experiment.candidates;
  const results = candidates.map((c) => ({ c, r: c.result as unknown as CandidateResult | null }));
  const comparable = results.filter((x) => x.r?.comparable);
  const notComparable = results.filter((x) => x.r && !x.r.comparable);
  const comparisons = ((summary?.comparisons as Comparison[] | undefined) ?? []).filter(
    (c) => c.metric === chosen && c.model === model,
  );
  const labelOf = new Map(candidates.map((c) => [c.key, c.label]));
  const warnings = [
    ...(((summary?.warnings as { code: string; message: string }[]) ?? [])),
    ...(((summary?.split_warnings as { code: string; message: string }[]) ?? [])),
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-lg font-semibold">{experiment.name}</h3>
        <Badge variant="outline">{experiment.task_type}</Badge>
        {job ? <Badge variant={job.status === "failed" ? "destructive" : "secondary"}>{job.status}</Badge> : null}
        {active && job?.progress ? (
          <span className="text-sm text-muted-foreground" role="status">
            {String((job.progress as Dict).pct ?? 0)}% evaluated
          </span>
        ) : null}
        {active ? (
          <Button size="sm" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        ) : null}
        {experiment.completed_at ? (
          <Button size="sm" variant="ghost" onClick={onRerun} title="Runs the stored configuration again on the same folds">
            Re-run identically
          </Button>
        ) : null}
      </div>
      {job?.error ? (
        <p className="text-sm text-destructive" role="alert">
          {job.error.code}: {job.error.message}
        </p>
      ) : null}
      {!summary ? (
        <p className="text-sm text-muted-foreground">{active ? "Results appear when every fold has been evaluated." : "No results were stored for this experiment."}</p>
      ) : (
        <>
          <section className="space-y-1 text-sm">
            <h4 className="font-medium">Frozen folds</h4>
            {experiment.split ? (
              <p>
                {experiment.split.method} · {experiment.split.n_splits} folds · seed {experiment.split.seed} ·
                assignment <span className="font-mono">{experiment.split.assignment_sha256.slice(0, 12)}</span>{" "}
                · {String((summary.cohort as Dict).rows)} rows evaluated
                {Number((summary.cohort as Dict).excluded_missing_target) > 0
                  ? ` (${String((summary.cohort as Dict).excluded_missing_target)} without a target excluded)`
                  : ""}
              </p>
            ) : null}
            <p className="text-muted-foreground">
              Every pipeline below was fitted and scored on exactly these training and test rows.
            </p>
          </section>
          <Warnings items={warnings} />

          <section className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="font-medium">Model performance (held-out rows, mean ± fold standard deviation)</h4>
              <label className="text-xs" htmlFor="metric-select">
                Metric
              </label>
              <select id="metric-select" className={SELECT} value={chosen} onChange={(e) => setMetric(e.target.value)}>
                {metricNames.map((m) => (
                  <option key={m} value={m}>
                    {info[m]?.label ?? m}
                  </option>
                ))}
              </select>
              <label className="text-xs" htmlFor="model-select">
                Model for details
              </label>
              <select id="model-select" className={SELECT} value={model} onChange={(e) => setModel(e.target.value)}>
                {experiment.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-xs text-muted-foreground">
              {info[chosen]?.higher_is_better === false ? "Lower is better." : "Higher is better."} No pipeline is
              declared a winner: compare fold-to-fold variation, the size of differences, data-quality changes
              and information loss.
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  <th className="py-1 pr-2">Pipeline</th>
                  <th className="pr-2">Risk</th>
                  {experiment.models.map((m) => (
                    <th key={m} className="pr-2">
                      {m}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparable.map(({ c, r }) => (
                  <tr key={c.id} className="border-t">
                    <td className="py-1 pr-2">{c.label}</td>
                    <td className="pr-2">
                      <RiskBadge risk={c.max_risk as "low" | "medium" | "high"} destructive={c.has_destructive} />
                    </td>
                    {experiment.models.map((m) => (
                      <td key={m} className="pr-2 font-mono">
                        {formatStats(r?.metrics[m]?.[chosen])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {onToPipeline ? (
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="text-xs text-muted-foreground">
                  Turn a candidate into an editable pipeline (experiment results are evidence, not approval):
                </span>
                {candidates
                  .filter((c) => !c.is_baseline)
                  .map((c) => (
                    <Button key={c.id} size="sm" variant="outline" onClick={() => onToPipeline(c.id)}>
                      To pipeline: {c.label}
                    </Button>
                  ))}
              </div>
            ) : null}
            {notComparable.length > 0 ? (
              <div className="space-y-1 text-sm">
                <h5 className="font-medium">Not directly comparable (excluded from comparisons)</h5>
                <ul className="list-disc pl-5">
                  {notComparable.map(({ c, r }) => (
                    <li key={c.id}>
                      {c.label}: {r?.comparability_reason}{" "}
                      {r?.failures.slice(0, 3).map((f) => `${f.code} (${f.message})`).join("; ")}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>

          <section className="space-y-2">
            <h4 className="font-medium">Compared with the baseline ({model}, {info[chosen]?.label ?? chosen})</h4>
            {comparisons.length === 0 ? (
              <p className="text-sm text-muted-foreground">No comparable candidates.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left">
                    <th className="py-1 pr-2">Pipeline</th>
                    <th className="pr-2">Per-fold difference</th>
                    <th className="pr-2">Mean ± sd</th>
                    <th className="pr-2">Folds better / worse</th>
                    <th>Reading</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map((c) => (
                    <tr key={c.candidate} className="border-t align-top">
                      <td className="py-1 pr-2">{labelOf.get(c.candidate)}</td>
                      <td className="pr-2 font-mono text-xs">{c.per_fold_difference.map((d) => d.toFixed(3)).join("  ")}</td>
                      <td className="pr-2 font-mono">
                        {formatValue(c.mean_difference)} ± {formatValue(c.sd_difference)}
                      </td>
                      <td className="pr-2">
                        {c.folds_better ?? "—"} / {c.folds_worse ?? "—"}
                      </td>
                      <td>
                        <span className="font-medium">{VERDICT_LABELS[c.verdict] ?? c.verdict}.</span>{" "}
                        <span className="text-muted-foreground">{c.interpretation}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="space-y-2">
            <h4 className="font-medium">Fold-level scores ({model}, {info[chosen]?.label ?? chosen})</h4>
            <table className="text-sm">
              <thead>
                <tr className="text-left">
                  <th className="py-1 pr-2">Pipeline</th>
                  {experiment.split
                    ? Array.from({ length: experiment.split.n_splits }, (_, i) => (
                        <th key={i} className="pr-2">
                          Fold {i + 1}
                        </th>
                      ))
                    : null}
                </tr>
              </thead>
              <tbody>
                {results.map(({ c, r }) => (
                  <tr key={c.id} className="border-t">
                    <td className="py-1 pr-2">{c.label}</td>
                    {(r?.metrics[model]?.[chosen]?.values ?? []).map((v, i) => (
                      <td key={i} className="pr-2 font-mono">
                        {r?.folds[i]?.status === "failed" ? <span title={r.folds[i].error?.message}>failed</span> : formatValue(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="space-y-2">
            <h4 className="font-medium">Data-quality changes (mean per fold, held-out rows before → after)</h4>
            <p className="text-xs text-muted-foreground">
              Fewer missing values or extreme values is not automatically better: imputed values are estimates
              and capped values are overwritten observations.
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  <th className="py-1 pr-2">Measure</th>
                  {results.map(({ c }) => (
                    <th key={c.id} className="pr-2">
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {QUALITY_ROWS.map(([key, label]) => (
                  <tr key={key} className="border-t">
                    <td className="py-1 pr-2">{label}</td>
                    {results.map(({ c, r }) => {
                      const q = r?.data_quality as { before: Dict; after: Dict } | null | undefined;
                      return (
                        <td key={c.id} className="pr-2 font-mono text-xs">
                          {q ? `${formatValue(q.before[key] as number, 1)} → ${formatValue(q.after[key] as number, 1)}` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
                {[
                  ["cells_imputed", "Cells imputed"],
                  ["observed_values_overwritten", "Observed values overwritten"],
                  ["observed_values_transformed", "Values re-expressed (invertible)"],
                  ["rows_affected", "Test rows changed"],
                  ["training_rows_after", "Training rows after resampling"],
                ].map(([key, label]) => (
                  <tr key={key} className="border-t">
                    <td className="py-1 pr-2">{label}</td>
                    {results.map(({ c, r }) => (
                      <td key={c.id} className="pr-2 font-mono text-xs">
                        {r?.data_quality ? formatValue((r.data_quality as Dict)[key] as number, 1) : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t align-top">
                  <td className="py-1 pr-2">Information loss</td>
                  {results.map(({ c, r }) => (
                    <td key={c.id} className="pr-2 text-xs">
                      {((r?.data_quality as Dict | null)?.information_loss_indicators as string[] | undefined)?.join(" ") || "None indicated"}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </section>

          {results.some(({ r }) => (r?.selection_stability ?? []).length > 0) ? (
            <section className="space-y-3">
              <h4 className="font-medium">Feature-selection stability across folds</h4>
              {results.flatMap(({ c, r }) =>
                (r?.selection_stability ?? []).map((s) => (
                  <div key={`${c.id}-${s.step}`} className="space-y-1 rounded-md border p-3 text-sm">
                    <p className="font-medium">
                      {c.label} · step {s.step + 1} ({s.op})
                    </p>
                    <p>
                      Overlap between folds (mean Jaccard): {formatValue(s.mean_jaccard, 2)} · random noise columns
                      selected in {s.probe_selection_rate === null ? "— (not measured)" : `${Math.round(s.probe_selection_rate * 100)}%`} of draws
                    </p>
                    <ul className="grid grid-cols-1 gap-1 sm:grid-cols-2">
                      {Object.entries(s.frequency)
                        .sort((a, b) => b[1] - a[1])
                        .map(([col, f]) => (
                          <li key={col} className="flex items-center gap-2">
                            <span className="w-32 truncate">{col}</span>
                            <span className="h-2 flex-1 rounded bg-muted">
                              <span className="block h-2 rounded bg-foreground/70" style={{ width: `${f * 100}%` }} />
                            </span>
                            <span className="w-20 text-xs">
                              {Math.round(f * s.folds)}/{s.folds} · {stabilityClass(f)}
                              {s.indistinguishable_from_noise.includes(col) ? " · noise-like" : ""}
                            </span>
                          </li>
                        ))}
                    </ul>
                    {s.forced.length > 0 ? <p className="text-xs">Always kept (your list or a domain role): {s.forced.join(", ")}</p> : null}
                    <Warnings items={s.warnings} />
                    <ul className="list-disc pl-5 text-xs text-muted-foreground">
                      {s.limitations.map((l) => (
                        <li key={l}>{l}</li>
                      ))}
                    </ul>
                  </div>
                )),
              )}
            </section>
          ) : null}

          <section className="space-y-1">
            <h4 className="font-medium">Limitations</h4>
            <ul className="list-disc pl-5 text-sm text-muted-foreground">
              {((summary.limitations as string[]) ?? []).map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
            <p className="text-xs text-muted-foreground">
              Models: {Object.entries((summary.model_notes as Record<string, string>) ?? {}).map(([k, v]) => `${k}: ${v}`).join(" ")}
            </p>
          </section>
        </>
      )}
    </div>
  );
}
