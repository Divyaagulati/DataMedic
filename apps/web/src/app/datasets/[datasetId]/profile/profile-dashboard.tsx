"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";

import {
  BarList,
  CorrelationHeatmap,
  Histogram,
  StatTile,
  TypeBadge,
  TypeMixBar,
} from "@/components/charts/charts";
import { DatasetNav } from "@/components/dataset-nav";
import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import {
  cellLabel,
  classBalance,
  formatNumber,
  formatPct,
  formatStat,
  missingRanking,
  skewLabel,
  type ColumnProfile,
  type Profile,
} from "@/lib/profile";
import { isActiveJob, useDataset, useProfileState } from "@/lib/queries";

export function ProfileDashboard() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const queryClient = useQueryClient();
  const dataset = useDataset(datasetId);
  const versionId = dataset.data?.head_version?.id;
  const state = useProfileState(versionId);
  const [error, setError] = useState<string | null>(null);

  async function rerun() {
    if (!versionId) return;
    setError(null);
    const { error: problem, response } = await api.POST("/api/v1/versions/{version_id}/profile", {
      params: { path: { version_id: versionId } },
    });
    if (response.status !== 202) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "profile"] });
  }

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const job = state.data?.job;
  const profile = state.data?.profile;
  const running = isActiveJob(job?.status);

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="profile" />
      <FormError message={error ?? (state.error ? problemMessage(state.error) : null)} />
      {!ds.head_version ? (
        <FormNotice message="The dataset has no parsed version yet, so there is nothing to profile." />
      ) : null}
      {versionId ? (
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
          <span>
            {profile
              ? `Version v${ds.head_version?.version_number} · profiled ${new Date(profile.created_at).toLocaleString()} · ${profile.profiler} profiler ${profile.profiler_version}`
              : running
                ? "Profiling is running in the background. Results appear here automatically."
                : job?.status === "failed"
                  ? `The last profiling run failed: ${job.error?.code} ${job.error?.message ?? ""}`
                  : "No profile yet."}
          </span>
          <Button variant="outline" size="sm" onClick={rerun} disabled={running}>
            {running ? "Profiling…" : "Run profiling again"}
          </Button>
        </div>
      ) : null}
      {profile ? <ProfileBody profile={profile} /> : null}
    </div>
  );
}

function ProfileBody({ profile }: { profile: Profile }) {
  const s = profile.summary;
  const missing = missingRanking(profile.columns);
  return (
    <>
      <section aria-label="Dataset summary" className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Rows" value={formatNumber(s.n_rows)} />
        <StatTile label="Columns" value={formatNumber(s.n_cols)} />
        <StatTile
          label="Missing cells"
          value={formatPct(s.missing_cells_pct)}
          detail={`${formatNumber(s.missing_cells)} of ${formatNumber(s.n_cells)}`}
        />
        <StatTile
          label="Rows with a missing value"
          value={formatPct(s.rows_with_missing_pct)}
          detail={`${formatNumber(s.rows_with_missing)} rows`}
        />
        <StatTile
          label="Duplicate rows"
          value={formatNumber(s.duplicate_rows)}
          detail={`${formatPct(s.duplicate_rows_pct)} · ${formatNumber(s.rows_in_duplicate_groups)} rows involved`}
        />
        <StatTile
          label="Constant / empty columns"
          value={`${s.constant_columns.length} / ${s.empty_columns.length}`}
        />
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Column types</CardTitle>
            <CardDescription>Detected automatically; see limitations below.</CardDescription>
          </CardHeader>
          <CardContent>
            <TypeMixBar counts={s.type_counts} />
            {s.constant_columns.length || s.empty_columns.length ? (
              <p className="mt-4 text-xs text-muted-foreground">
                {s.constant_columns.length ? `Constant: ${s.constant_columns.join(", ")}. ` : ""}
                {s.empty_columns.length ? `Entirely empty: ${s.empty_columns.join(", ")}.` : ""}
              </p>
            ) : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Missing values by column</CardTitle>
            <CardDescription>Share of rows with no value, worst first.</CardDescription>
          </CardHeader>
          <CardContent>
            {missing.length === 0 ? (
              <p className="text-sm text-muted-foreground">No column has missing values.</p>
            ) : (
              <>
                <BarList
                  ariaLabel="Missing values by column"
                  max={100}
                  items={missing.slice(0, 25).map((c) => ({
                    key: c.name,
                    label: c.name,
                    value: c.missing_pct,
                    display: formatPct(c.missing_pct),
                    tooltip: `${c.name}: ${formatNumber(c.n_missing)} of ${formatNumber(s.n_rows)} rows missing (${formatPct(c.missing_pct)})`,
                  }))}
                />
                {missing.length > 25 ? (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Showing 25 of {missing.length} columns with missing values; the table below lists all.
                  </p>
                ) : null}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <ColumnsExplorer profile={profile} />
      <TargetPanel profile={profile} />
      <CorrelationPanel profile={profile} />

      <Card>
        <CardHeader>
          <CardTitle>Limitations and method</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <ul className="list-disc space-y-1 pl-5">
            {profile.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p className="text-xs">
            Statistics: {profile.sampling.statistics}. Correlations: {profile.sampling.correlations}.
            Engine {profile.engine_version}; pandas {String(profile.library_versions["pandas"] ?? "?")}.
          </p>
        </CardContent>
      </Card>
    </>
  );
}

function ColumnsExplorer({ profile }: { profile: Profile }) {
  const [selected, setSelected] = useState(profile.columns[0]?.name);
  const column = profile.columns.find((c) => c.name === selected) ?? profile.columns[0];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Columns</CardTitle>
        <CardDescription>Select a column to see its distribution and statistics.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="max-h-[36rem] overflow-auto rounded-md border">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted text-xs">
              <tr>
                <th scope="col" className="px-3 py-2 text-left font-medium">Column</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Type</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Missing</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Unique</th>
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr
                  key={c.name}
                  aria-selected={c.name === column?.name}
                  className={"border-t " + (c.name === column?.name ? "bg-accent" : "hover:bg-muted/50")}
                >
                  <td className="max-w-[14rem] px-3 py-1.5">
                    <button
                      type="button"
                      className="max-w-full truncate text-left underline-offset-2 hover:underline"
                      title={c.name}
                      onClick={() => setSelected(c.name)}
                    >
                      {c.name}
                    </button>
                    {c.is_constant ? <Flag>constant</Flag> : null}
                    {c.is_empty ? <Flag>empty</Flag> : null}
                  </td>
                  <td className="px-3 py-1.5">
                    <TypeBadge type={c.inferred_type} />
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{formatPct(c.missing_pct)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{formatNumber(c.n_unique)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {column ? <ColumnDetail column={column} /> : null}
      </CardContent>
    </Card>
  );
}

function Flag({ children }: { children: string }) {
  return (
    <span className="ml-2 rounded border px-1 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
      {children}
    </span>
  );
}

function ColumnDetail({ column }: { column: ColumnProfile }) {
  const n = column.numeric;
  const facts: [string, string][] = [
    ["Storage type", column.dtype],
    ["Values present", `${formatNumber(column.n_present)} (${formatPct(100 - column.missing_pct)})`],
    ["Missing", `${formatNumber(column.n_missing)} (${formatPct(column.missing_pct)})`],
    ["Distinct values", `${formatNumber(column.n_unique)} (${formatPct(column.unique_pct)} of present)`],
  ];
  if (column.top_value_pct !== null) facts.push(["Most common value share", formatPct(column.top_value_pct)]);
  if (n) {
    facts.push(
      ["Mean", formatStat(n.mean)],
      ["Median", formatStat(n.median)],
      ["Standard deviation", formatStat(n.std)],
      ["Minimum", formatStat(n.min)],
      ["25th percentile (Q1)", formatStat(n.q1)],
      ["75th percentile (Q3)", formatStat(n.q3)],
      ["Maximum", formatStat(n.max)],
      ["Interquartile range", formatStat(n.iqr)],
      ["Skewness", n.skewness === null ? (n.skewness_note ?? "—") : `${formatStat(n.skewness)} · ${skewLabel(n.skewness)}`],
      ["Kurtosis (excess)", formatStat(n.kurtosis)],
      ["Zeros / negatives", `${formatNumber(n.n_zeros)} / ${formatNumber(n.n_negative)}`],
    );
    if (n.n_infinite) facts.push(["Infinite values", formatNumber(n.n_infinite)]);
  }
  if (column.datetime) {
    facts.push(["Earliest", column.datetime.min ?? "—"], ["Latest", column.datetime.max ?? "—"]);
  }
  if (column.text) {
    facts.push([
      "Text length (min / mean / max)",
      `${formatNumber(column.text.min_length)} / ${formatNumber(column.text.mean_length)} / ${formatNumber(column.text.max_length)}`,
    ]);
  }
  const freq = column.frequencies;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="truncate font-medium" title={column.name}>
          {column.name}
        </h3>
        <TypeBadge type={column.inferred_type} />
      </div>
      {column.histogram ? (
        <Histogram
          kind={column.histogram.kind}
          edges={column.histogram.edges}
          counts={column.histogram.counts}
          ariaLabel={`Distribution of ${column.name}`}
        />
      ) : null}
      {freq && freq.values.length ? (
        <div className="space-y-2">
          <BarList
            ariaLabel={`Most common values of ${column.name}`}
            items={freq.values.map((v) => ({
              key: cellLabel(v.value),
              label: cellLabel(v.value),
              value: v.count,
              display: `${formatNumber(v.count)} · ${formatPct(v.pct)}`,
            }))}
          />
          {!freq.complete ? (
            <p className="text-xs text-muted-foreground">
              Top {freq.values.length} values shown; {formatNumber(freq.other_count)} other values not listed.
            </p>
          ) : null}
        </div>
      ) : freq?.reason ? (
        <p className="text-xs text-muted-foreground">{freq.reason}</p>
      ) : null}
      {column.inferred_type === "identifier" ? (
        <p className="text-xs text-muted-foreground">
          Every value is distinct, so this looks like an identifier (a label, not a measurement).
          Averages, spread and histograms are not shown for it.
        </p>
      ) : null}
      <dl className="grid grid-cols-[minmax(0,12rem)_1fr] gap-x-3 gap-y-1.5 text-sm">
        {facts.map(([k, v]) => (
          <Fact key={k} term={k} value={v} />
        ))}
      </dl>
    </div>
  );
}

function Fact({ term, value }: { term: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="tabular-nums">{value}</dd>
    </>
  );
}

function TargetPanel({ profile }: { profile: Profile }) {
  const [target, setTarget] = useState("");
  const column = profile.columns.find((c) => c.name === target);
  const balance = column ? classBalance(column) : null;
  const corrIndex = profile.correlations.columns.indexOf(target);
  const related = useMemo(() => {
    if (corrIndex < 0) return [];
    const row = profile.correlations.pearson[corrIndex] ?? [];
    return profile.correlations.columns
      .map((name, j) => ({ name, r: row[j] ?? null, rho: profile.correlations.spearman[corrIndex]?.[j] ?? null }))
      .filter((x) => x.name !== target && x.r !== null)
      .sort((a, b) => Math.abs(b.r ?? 0) - Math.abs(a.r ?? 0))
      .slice(0, 10);
  }, [corrIndex, profile.correlations, target]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Target column</CardTitle>
        <CardDescription>
          Choose the column you want to predict to see its class distribution. This choice is not
          saved; declaring roles comes in a later phase.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="flex max-w-md flex-col gap-1 text-sm">
          <span className="text-muted-foreground">Target column</span>
          <select
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">Select a column…</option>
            {profile.columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {column && balance ? (
          balance.ok ? (
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_16rem]">
              <BarList
                ariaLabel={`Class distribution of ${column.name}`}
                items={balance.classes.map((c) => ({
                  key: c.label,
                  label: c.label,
                  value: c.count,
                  display: `${formatNumber(c.count)} · ${formatPct(c.pct)}`,
                }))}
              />
              <div className="space-y-3">
                <StatTile label="Classes" value={formatNumber(balance.classes.length)} />
                <StatTile
                  label="Imbalance ratio (largest ÷ smallest)"
                  value={`${balance.ratio.toFixed(balance.ratio < 10 ? 2 : 0)} : 1`}
                  detail={`Smallest class “${balance.minority.label}” has ${formatNumber(balance.minority.count)} rows`}
                />
                {column.n_missing ? (
                  <p className="text-xs text-muted-foreground">
                    {formatNumber(column.n_missing)} rows have no value in this column and are not counted.
                  </p>
                ) : null}
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">{balance.reason}</p>
          )
        ) : null}
        {related.length ? (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Strongest correlations with {target}</h3>
            <table className="w-full max-w-xl text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr>
                  <th scope="col" className="py-1 text-left font-normal">Column</th>
                  <th scope="col" className="py-1 text-right font-normal">Pearson r</th>
                  <th scope="col" className="py-1 text-right font-normal">Spearman ρ</th>
                </tr>
              </thead>
              <tbody>
                {related.map((x) => (
                  <tr key={x.name} className="border-t">
                    <td className="max-w-[16rem] truncate py-1" title={x.name}>{x.name}</td>
                    <td className="py-1 text-right tabular-nums">{x.r?.toFixed(3)}</td>
                    <td className="py-1 text-right tabular-nums">{x.rho === null ? "—" : x.rho.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-muted-foreground">
              Association only, not causation. A very strong correlation with the target can be a sign of leakage.
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CorrelationPanel({ profile }: { profile: Profile }) {
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  const corr = profile.correlations;
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <CardTitle>Correlations</CardTitle>
          <CardDescription>
            Numeric and boolean columns. Pearson measures linear association; Spearman measures
            monotonic (rank) association.
          </CardDescription>
        </div>
        {corr.status === "ok" ? (
          <div role="group" aria-label="Correlation method" className="flex gap-1">
            {(["pearson", "spearman"] as const).map((m) => (
              <Button
                key={m}
                size="sm"
                variant={method === m ? "default" : "outline"}
                aria-pressed={method === m}
                onClick={() => setMethod(m)}
              >
                {m === "pearson" ? "Pearson" : "Spearman"}
              </Button>
            ))}
          </div>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-4">
        {corr.status !== "ok" ? (
          <p className="text-sm text-muted-foreground">Not computed: {corr.reason}</p>
        ) : (
          <>
            <CorrelationHeatmap
              columns={corr.columns}
              matrix={method === "pearson" ? corr.pearson : corr.spearman}
              method={method === "pearson" ? "Pearson" : "Spearman"}
            />
            {corr.sampled || corr.truncated_columns ? (
              <p className="text-xs text-muted-foreground">
                {corr.sampled ? `Computed on a random sample of ${formatNumber(corr.n_rows_used)} rows (seed ${corr.seed}). ` : ""}
                {corr.truncated_columns ? `${corr.truncated_columns} eligible columns were left out (limit 50).` : ""}
              </p>
            ) : null}
            <div>
              <h3 className="mb-2 text-sm font-medium">Strongest pairs</h3>
              <table className="w-full max-w-2xl text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr>
                    <th scope="col" className="py-1 text-left font-normal">Pair</th>
                    <th scope="col" className="py-1 text-right font-normal">Pearson r</th>
                    <th scope="col" className="py-1 text-right font-normal">Spearman ρ</th>
                  </tr>
                </thead>
                <tbody>
                  {corr.top_pairs.slice(0, 10).map((p) => (
                    <tr key={`${p.a}|${p.b}`} className="border-t">
                      <td className="max-w-[24rem] truncate py-1" title={`${p.a} × ${p.b}`}>
                        {p.a} × {p.b}
                      </td>
                      <td className="py-1 text-right tabular-nums">{p.pearson.toFixed(3)}</td>
                      <td className="py-1 text-right tabular-nums">{p.spearman === null ? "—" : p.spearman.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
