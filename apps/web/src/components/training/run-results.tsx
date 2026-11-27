"use client";

import { useState } from "react";

import { BarList } from "@/components/charts/charts";
import { FoldDots, HeatGrid, UnitCurve } from "@/components/charts/ml-charts";
import { FindingCard } from "@/components/diagnostics/finding-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { FindingView } from "@/lib/diagnostics";
import {
  type Audit,
  BASIS_LABELS,
  basisOf,
  fmt,
  fmtStats,
  foldStats,
  groupFindings,
  interval,
  metricReasons,
  MODEL_LABELS,
  type ModelEvaluation,
  paramsLabel,
  STATUS_INFO,
  type TrainingRun,
} from "@/lib/training";

type Dict = Record<string, unknown>;
type MetricInfo = Record<string, { label: string; higher_is_better: boolean; probability: boolean }>;

function List({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <ul className="list-disc space-y-1 pl-5 text-xs text-muted-foreground">
      {items.map((t, i) => (
        <li key={i}>{t}</li>
      ))}
    </ul>
  );
}

// ---- Comparison ---------------------------------------------------------------------------------

/** How the uncertainty shown respects dependence between rows (or why it cannot). */
export function dependenceNote(run: TrainingRun): string {
  const split = (run.split_config ?? {}) as Dict;
  if (split.method === "time" || split.time) {
    return "Folds are time-ordered: bootstrap intervals are not shown, because resampling rows would ignore temporal dependence. Fold standard deviations come from few, non-independent folds.";
  }
  if (split.group) {
    return `Bootstrap intervals resample whole groups (${String(split.group)}) and exact duplicate clusters, so rows of one entity stay together. Fold scores are not independent (training rows overlap), so no p-values are given.`;
  }
  return "No group role was declared: the bootstrap treats rows as independent (only exact duplicate rows are resampled together). If several rows belong to one entity, the intervals are too narrow. Fold scores are not independent, so no p-values are given.";
}

export function Comparison({ run }: { run: TrainingRun }) {
  const summary = (run.summary ?? {}) as Dict;
  const metrics = (summary.metrics as string[]) ?? [];
  const info = (summary.metric_info ?? {}) as MetricInfo;
  const primary = String(summary.primary_metric ?? "");
  const comparisons = (summary.comparisons as Dict[]) ?? [];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model comparison</CardTitle>
        <CardDescription>
          Each cell is the mean ± standard deviation of the outer test-fold scores (spread between folds). The bracketed
          range below it is a separate 95% bootstrap interval of the pooled out-of-fold predictions (test-sample
          variability only; it ignores training, tuning and model-selection variability). No model is declared best:
          picking the highest score on these same folds is optimistic, and no independent holdout exists.
        </CardDescription>
        <p className="text-xs text-muted-foreground" data-testid="dependence-note">
          {dependenceNote(run)}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                {metrics.map((m) => (
                  <TableHead key={m} title={info[m]?.probability ? "Uses predicted probabilities" : "Uses hard labels"}>
                    {info[m]?.label ?? m}
                    {info[m]?.higher_is_better === false ? " ↓" : ""}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {run.evaluations.map((e) => (
                <TableRow key={e.model}>
                  <TableCell className="font-medium">{MODEL_LABELS[e.model] ?? e.model}</TableCell>
                  <TableCell>
                    {e.comparable ? (
                      <Badge variant="secondary">{e.status}</Badge>
                    ) : (
                      <Badge variant="destructive" title={e.comparability_reason ?? ""}>
                        {e.status}: not compared
                      </Badge>
                    )}
                  </TableCell>
                  {metrics.map((m) => {
                    const ci = interval(e, m);
                    const reasons = metricReasons(e, m);
                    return (
                      <TableCell key={m} className="tabular-nums" title={reasons.join("\n") || undefined}>
                        {fmtStats(foldStats(e, m))}
                        {ci ? <div className="text-[10px] text-muted-foreground">[{fmt(ci.low)}, {fmt(ci.high)}]</div> : null}
                        {reasons.length ? <div className="text-[10px] text-muted-foreground">unavailable in some folds</div> : null}
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <section className="space-y-1">
          <h3 className="text-sm font-medium">{info[primary]?.label ?? primary} per fold</h3>
          <FoldDots
            ariaLabel={`${primary} per outer fold and model`}
            rows={run.evaluations.map((e) => ({
              key: e.model,
              label: MODEL_LABELS[e.model] ?? e.model,
              values: foldStats(e, primary)?.values ?? [],
            }))}
          />
        </section>
        {comparisons.length ? (
          <details>
            <summary className="cursor-pointer text-sm">Paired per-fold differences (identical test rows)</summary>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>vs</TableHead>
                  <TableHead>Metric</TableHead>
                  <TableHead>Mean difference</TableHead>
                  <TableHead>Verdict</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {comparisons.map((c, i) => (
                  <TableRow key={i}>
                    <TableCell>{MODEL_LABELS[String(c.model)] ?? String(c.model)}</TableCell>
                    <TableCell>{MODEL_LABELS[String(c.reference)] ?? String(c.reference)}</TableCell>
                    <TableCell>{info[String(c.metric)]?.label ?? String(c.metric)}</TableCell>
                    <TableCell className="tabular-nums">{fmt(c.mean_difference as number | null)}</TableCell>
                    <TableCell title={String(c.interpretation ?? "")}>{String(c.verdict).replaceAll("_", " ")}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </details>
        ) : null}
        <List items={((summary.warnings as Dict[]) ?? []).map((w) => String(w.message))} />
      </CardContent>
    </Card>
  );
}

// ---- Performance --------------------------------------------------------------------------------

function curveSeries(curves: Dict, kind: "roc" | "pr", classes: string[] | null) {
  if (curves.per_class) {
    return (curves.per_class as Dict[])
      .filter((c) => c.available)
      .map((c) => ({
        key: String(c.class_index),
        label: classes?.[Number(c.class_index)] ?? `class ${Number(c.class_index) + 1}`,
        points: c[kind] as number[][],
      }));
  }
  return curves.available ? [{ key: kind, label: kind.toUpperCase(), points: curves[kind] as number[][] }] : [];
}

export function Performance({ run }: { run: TrainingRun }) {
  const usable = run.evaluations.filter((e) => e.status !== "failed");
  const [model, setModel] = useState(usable.find((e) => e.model !== "dummy")?.model ?? usable[0]?.model);
  const e = usable.find((x) => x.model === model);
  const summary = (run.summary ?? {}) as Dict;
  const classes = (summary.classes as string[] | null) ?? null;
  if (!e) return null;
  const d = e.details as Dict;
  const curves = (d.curves ?? {}) as Dict;
  const calibration = (d.calibration ?? {}) as Dict;
  const residuals = (d.residuals ?? {}) as Dict;
  const confusion = d.confusion as number[][] | null;
  const coverage = (summary.coverage ?? {}) as Dict;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance</CardTitle>
        <CardDescription>
          Charts use the pooled out-of-fold predictions: each tested row is predicted once, by the model of the fold in
          which it was a test row. {coverage.complete ? "" : String(coverage.note ?? "")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2 text-sm">
          <label htmlFor="perf-model">Model</label>
          <select id="perf-model" className="h-8 rounded-md border bg-background px-2" value={model} onChange={(x) => setModel(x.target.value)}>
            {usable.map((x) => (
              <option key={x.model} value={x.model}>
                {MODEL_LABELS[x.model] ?? x.model}
              </option>
            ))}
          </select>
          {!e.comparable ? <Badge variant="destructive">partial: not every fold completed</Badge> : null}
        </div>
        {run.task_type !== "regression" ? (
          <div className="grid gap-6 md:grid-cols-3">
            <section className="space-y-1">
              <h3 className="text-sm font-medium">ROC curve</h3>
              {curveSeries(curves, "roc", classes).length ? (
                <UnitCurve series={curveSeries(curves, "roc", classes)} reference="diagonal" xLabel="False positive rate" yLabel="True positive rate" ariaLabel="ROC curve" />
              ) : (
                <p className="text-xs text-muted-foreground">Unavailable: {String(curves.reason ?? "")}</p>
              )}
            </section>
            <section className="space-y-1">
              <h3 className="text-sm font-medium">Precision-recall curve</h3>
              {curveSeries(curves, "pr", classes).length ? (
                <UnitCurve
                  series={curveSeries(curves, "pr", classes)}
                  reference={curves.prevalence !== undefined ? { y: Number(curves.prevalence) } : null}
                  xLabel="Recall"
                  yLabel="Precision"
                  ariaLabel="Precision-recall curve"
                />
              ) : (
                <p className="text-xs text-muted-foreground">Unavailable: {String(curves.reason ?? "")}</p>
              )}
            </section>
            <section className="space-y-1">
              <h3 className="text-sm font-medium">Calibration</h3>
              {calibration.available ? (
                <>
                  <UnitCurve
                    series={[
                      {
                        key: "cal",
                        label: "Observed vs predicted",
                        points: (calibration.bins as Dict[]).map((b) => [Number(b.predicted), Number(b.observed)]),
                      },
                    ]}
                    reference="diagonal"
                    xLabel="Mean predicted probability"
                    yLabel="Observed frequency"
                    ariaLabel="Calibration reliability curve"
                  />
                  <p className="text-xs text-muted-foreground">
                    {calibration.kind === "top_label" ? "Top-label confidence vs accuracy. " : ""}Quantile bins;
                    ECE {fmt(calibration.ece_quantile_bins as number)}.
                  </p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">Unavailable: {String(calibration.reason ?? "")}</p>
              )}
            </section>
          </div>
        ) : null}
        {confusion ? (
          <section className="space-y-1">
            <h3 className="text-sm font-medium">Confusion matrix (summed over folds; rows: actual, columns: predicted)</h3>
            <HeatGrid
              rows={confusion}
              rowLabels={classes ?? undefined}
              colLabels={classes ?? undefined}
              showValues={confusion.length <= 10}
              ariaLabel="Confusion matrix"
              cellTip={(r, c, v) => `actual ${classes?.[r] ?? r} · predicted ${classes?.[c] ?? c}: ${v} rows`}
            />
            {Array.isArray(d.per_class) ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Class</TableHead>
                    <TableHead>Support</TableHead>
                    <TableHead>Precision</TableHead>
                    <TableHead>Recall</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(d.per_class as Dict[]).map((p, i) => (
                    <TableRow key={i}>
                      <TableCell>{classes?.[i] ?? i}</TableCell>
                      <TableCell>{String(p.support)}</TableCell>
                      <TableCell>{fmt(p.precision as number | null)}</TableCell>
                      <TableCell>{fmt(p.recall as number | null)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : null}
          </section>
        ) : null}
        {run.task_type === "regression" ? (
          residuals.available ? (
            <div className="grid gap-6 md:grid-cols-2">
              <section className="space-y-1">
                <h3 className="text-sm font-medium">Actual vs predicted (row counts per cell)</h3>
                <HeatGrid
                  rows={[...(residuals.actual_vs_predicted as number[][])].reverse()}
                  cell={6}
                  ariaLabel="Binned actual versus predicted values"
                  cellTip={(_, __, v) => `${v} rows`}
                />
                <p className="text-xs text-muted-foreground">
                  x: predicted, y: actual
                  {residuals.redacted_axes
                    ? " (axis values withheld: sensitive target)"
                    : `, range ${(residuals.value_range as number[]).map((v) => fmt(v)).join(" to ")}`}
                  ; perfect predictions lie on the diagonal.
                </p>
              </section>
              <section className="space-y-1">
                <h3 className="text-sm font-medium">Residuals</h3>
                <BarList
                  ariaLabel="Residual histogram"
                  items={(residuals.residual_histogram as number[]).map((n, i) => ({
                    key: String(i),
                    label: residuals.redacted_axes
                      ? `bin ${i + 1}`
                      : `${fmt((residuals.residual_histogram_edges as number[])[i], 2)}`,
                    value: n,
                    display: String(n),
                  }))}
                  labelWidth="5rem"
                />
                {!residuals.redacted_axes ? (
                  <p className="text-xs text-muted-foreground">
                    Mean residual {fmt(residuals.mean_residual as number)}; median{" "}
                    {fmt((residuals.residual_quantiles as Dict).p50 as number)}.
                  </p>
                ) : null}
              </section>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Residuals unavailable: {String(residuals.reason ?? "")}</p>
          )
        ) : null}
      </CardContent>
    </Card>
  );
}

// ---- Reliability --------------------------------------------------------------------------------

function Group({ title, items }: { title: string; items: FindingView[] }) {
  if (!items.length) return null;
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-medium">
        {title} ({items.length})
      </h3>
      {items.map((f) => (
        <div key={f.id} className="space-y-1">
          <Badge variant="outline">{BASIS_LABELS[basisOf(f)] ?? basisOf(f)}</Badge>
          <FindingCard finding={f} />
        </div>
      ))}
    </section>
  );
}

export function Reliability({ audit, onRerun, busy }: { audit: Audit | undefined; onRerun: () => void; busy: boolean }) {
  if (!audit?.run) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Reliability audit</CardTitle>
          <CardDescription>
            {audit?.job?.status === "failed" ? "The audit failed; run it again." : "The audit is running…"}
          </CardDescription>
        </CardHeader>
        {audit?.job?.status === "failed" ? (
          <CardContent>
            <Button size="sm" onClick={onRerun} disabled={busy}>
              Run audit again
            </Button>
          </CardContent>
        ) : null}
      </Card>
    );
  }
  const status = (audit.run.summary as Dict).evaluation_status as Dict;
  const info = STATUS_INFO[String(status.status)] ?? { label: String(status.status), tone: "warn" };
  const g = groupFindings(audit.findings);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          Reliability audit
          <Badge variant={info.tone === "bad" ? "destructive" : info.tone === "ok" ? "secondary" : "outline"}>
            {info.label}
          </Badge>
          {audit.stale ? (
            <Badge
              variant="outline"
              title="The version's current roles differ from the roles recorded when this run was trained. The run, its audit and audit re-runs use the training-time roles; they are not reinterpreted. Start a new training run to evaluate with the current roles."
            >
              roles changed since training
            </Badge>
          ) : null}
        </CardTitle>
        <CardDescription>{String(status.limitation)}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">How the status is decided</summary>
          <List items={(status.rules as string[]) ?? []} />
        </details>
        <Group title="Confirmed and configuration violations" items={g.violations} />
        <Group title="Statistical suspicions" items={g.suspicions} />
        <Group title="Could not be assessed" items={g.notAssessed} />
        <Group title="Check errors" items={g.errors} />
        <details>
          <summary className="cursor-pointer text-sm">Passed within their assessed scope ({g.passed.length})</summary>
          <div className="mt-2 space-y-2">
            {g.passed.map((f) => (
              <FindingCard key={f.id} finding={f} />
            ))}
          </div>
        </details>
        <Button size="sm" variant="outline" onClick={onRerun} disabled={busy}>
          Run audit again
        </Button>
      </CardContent>
    </Card>
  );
}

// ---- Explainability -----------------------------------------------------------------------------

export function Explainability({ run }: { run: TrainingRun }) {
  const usable = run.evaluations.filter((e) => e.model !== "dummy" && e.status !== "failed");
  const [model, setModel] = useState(usable[0]?.model);
  const e: ModelEvaluation | undefined = usable.find((x) => x.model === model);
  const limits = ((run.summary ?? {}) as Dict).explain_limitations as string[] | undefined;
  if (!e) return null;
  const imp = e.importance as Dict;
  const perm = (imp.permutation as Dict[]) ?? [];
  const coef = imp.coefficients as Dict[] | null;
  const impurity = imp.impurity as Dict[] | null;
  const slices = ((e.details as Dict).error_slices as Dict[]) ?? [];
  const privacy = (run.privacy ?? {}) as Dict;
  const withheld = (privacy.withheld_error_slices as string[] | undefined) ?? [];
  const bars = (rows: Dict[], digits = 4) =>
    rows.slice(0, 15).map((r) => ({
      key: String(r.column),
      label: String(r.column),
      value: Math.max(0, Number(r.mean)),
      display: `${fmt(r.mean as number, digits)}${r.sd !== null && r.sd !== undefined ? ` ± ${fmt(r.sd as number, digits)}` : ""}`,
      tooltip: `${String(r.column)}: ${fmt(r.mean as number, digits)} (model input in ${String(r.folds_present)} of ${String(r.n_folds)} folds)`,
    }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Explainability</CardTitle>
        <CardDescription>
          How these models use their inputs on held-out rows. Importance is not causal and never influenced model
          selection.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <select aria-label="Model" className="h-8 rounded-md border bg-background px-2 text-sm" value={model} onChange={(x) => setModel(x.target.value)}>
          {usable.map((x) => (
            <option key={x.model} value={x.model}>
              {MODEL_LABELS[x.model] ?? x.model}
            </option>
          ))}
        </select>
        <section className="space-y-1">
          <h3 className="text-sm font-medium">Permutation importance (drop in {String(imp.metric)})</h3>
          {perm.length ? (
            <BarList ariaLabel="Permutation importance per original column" items={bars(perm)} />
          ) : (
            <p className="text-xs text-muted-foreground">Unavailable for this model.</p>
          )}
          <p className="text-xs text-muted-foreground">
            Derived columns (missing indicators, encodings) are permuted together with their source column. Negative
            values are shown as 0 bars; see the value.
            {imp.permutation_sampled ? " Test rows were sampled (at most 5,000 per fold)." : ""}
          </p>
        </section>
        {coef ? (
          <section className="space-y-1">
            <h3 className="text-sm font-medium">Linear coefficients (size of standardised effect, per column)</h3>
            <BarList ariaLabel="Coefficient size per original column" items={bars(coef)} />
          </section>
        ) : null}
        {impurity ? (
          <section className="space-y-1">
            <h3 className="text-sm font-medium">Random-forest impurity importance</h3>
            <BarList ariaLabel="Impurity importance per original column" items={bars(impurity)} />
            <p className="text-xs text-muted-foreground">Biased towards continuous and high-cardinality columns.</p>
          </section>
        ) : null}
        {slices.length ? (
          <section className="space-y-2">
            <h3 className="text-sm font-medium">Error by slice of the most important columns</h3>
            {slices.map((s) => (
              <div key={String(s.column)}>
                <p className="text-xs font-medium">
                  {String(s.column)} ({String(s.metric).replaceAll("_", " ")})
                </p>
                <BarList
                  ariaLabel={`Error by slice of ${String(s.column)}`}
                  labelWidth="9rem"
                  items={(s.slices as Dict[]).map((sl, i) => ({
                    key: String(i),
                    label: String(sl.label),
                    value: Number(sl.error),
                    display: `${fmt(sl.error as number)} (n=${String(sl.n)})`,
                  }))}
                />
              </div>
            ))}
            <p className="text-xs text-muted-foreground">
              Slices with fewer than 30 rows are merged or withheld; sensitive columns are never sliced.
            </p>
          </section>
        ) : null}
        {withheld.length || privacy.target_values_withheld ? (
          <p className="text-xs text-muted-foreground" data-testid="privacy-note">
            {withheld.length
              ? `Error slices withheld for ${withheld.length} protected column(s): ${withheld.join(", ")}. `
              : ""}
            {privacy.target_values_withheld ? "Target class names and residual axis values are withheld. " : ""}
            {String(privacy.policy ?? "")}
          </p>
        ) : null}
        <List items={limits ?? []} />
      </CardContent>
    </Card>
  );
}

// ---- Reproducibility ----------------------------------------------------------------------------

export function rolesLabel(run: TrainingRun): string {
  const roles = (((run.config ?? {}) as Dict).roles as Dict | undefined)?.roles as Record<string, string[]> | undefined;
  const parts = Object.entries(roles ?? {}).map(([role, cols]) => `${role}: ${cols.join(", ")}`);
  return parts.length ? parts.join(" · ") : "none";
}

export function Reproducibility({ run }: { run: TrainingRun }) {
  const summary = (run.summary ?? {}) as Dict;
  const tuning = (summary.tuning ?? run.tuning) as Dict;
  const equivalence = (run.replay as Dict | null)?.equivalence as Dict | undefined;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Reproducibility</CardTitle>
        <CardDescription>
          Re-running this stored configuration (the same data version, the roles recorded at training time, folds,
          seeds and settings) is expected to reproduce the same folds, settings and scores with the same library and
          engine versions. Other library versions or execution environments may change results; each re-run records
          its own versions. Current role changes are not applied to a re-run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[12rem_1fr]">
          <dt className="text-muted-foreground">Configuration hash</dt>
          <dd className="break-all font-mono">{run.config_hash}</dd>
          <dt className="text-muted-foreground">Evaluated version content hash</dt>
          <dd className="break-all font-mono">{run.source_content_hash}</dd>
          <dt className="text-muted-foreground">Fold assignment</dt>
          <dd className="break-all font-mono">
            {run.split ? `${run.split.method}, ${run.split.n_splits} folds, seed ${run.split.seed} · ${run.split.assignment_sha256}` : "—"}
          </dd>
          <dt className="text-muted-foreground">Roles at training time</dt>
          <dd>{rolesLabel(run)}</dd>
          <dt className="text-muted-foreground">Seed</dt>
          <dd>{run.seed}</dd>
          <dt className="text-muted-foreground">Settings</dt>
          <dd>
            requested {String(tuning.requested)}, used {String(tuning.used ?? tuning.planned)}
            {tuning.fallback_reason ? ` (${String(tuning.fallback_reason)})` : ""}; ties go to the more regularised
            setting
          </dd>
          {equivalence ? (
            <>
              <dt className="text-muted-foreground">Replay fidelity</dt>
              <dd>{String(equivalence.status).replaceAll("_", " ")}</dd>
            </>
          ) : null}
          <dt className="text-muted-foreground">Out-of-fold predictions</dt>
          <dd className="break-all font-mono">{run.oof_artifact_sha256 ?? "—"} (stored, owner only, not downloadable)</dd>
          <dt className="text-muted-foreground">Libraries</dt>
          <dd>
            {Object.entries(run.library_versions ?? {})
              .map(([k, v]) => `${k} ${String(v)}`)
              .join(" · ")}
          </dd>
        </dl>
        <details>
          <summary className="cursor-pointer text-xs">Selected settings per fold</summary>
          <ul className="mt-1 space-y-1 text-xs">
            {run.evaluations.map((e) => (
              <li key={e.model}>
                <strong>{MODEL_LABELS[e.model]}</strong>:{" "}
                {((e.selection.per_fold as Dict[]) ?? []).map((p, i) => `fold ${i + 1}: ${paramsLabel(p)}`).join("; ")}
              </li>
            ))}
          </ul>
        </details>
        <details>
          <summary className="cursor-pointer text-xs">Metric conventions and limitations</summary>
          <List items={[...((summary.metric_conventions as string[]) ?? []), ...((summary.limitations as string[]) ?? [])]} />
        </details>
      </CardContent>
    </Card>
  );
}
