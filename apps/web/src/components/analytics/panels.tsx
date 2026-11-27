"use client";

import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { FilterEditor } from "@/components/analytics/builder";
import { ChartView } from "@/components/analytics/chart-view";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  AGG_LABELS,
  analyticsApi,
  type Catalogue,
  type Dict,
  exportQueryCsv,
  type Filter,
  fmtValue,
  useAnalyticsQuery,
} from "@/lib/analytics";

const select = "h-8 rounded-md border bg-background px-2 text-sm";

// ---- KPI cards -------------------------------------------------------------------------------------

export function KpiBuilder({ catalogue, dashboards, onAdd }: { catalogue: Catalogue; dashboards: Dict[]; onAdd: (id: string, w: Dict) => Promise<void> }) {
  const [column, setColumn] = useState("");
  const [agg, setAgg] = useState("count");
  const [title, setTitle] = useState("");
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const cols = catalogue.columns;
  const numeric = new Set(cols.filter((c) => c.inferred_type === "numeric" || c.inferred_type === "boolean").map((c) => c.name));
  const valid = agg === "count" || agg === "count_distinct" || (column && numeric.has(column));
  const measure: Dict = { agg, ...(column ? { column } : {}) };
  const query = valid ? { kind: "aggregate", measures: [measure] } : null;
  const res = useAnalyticsQuery(catalogue.version_id, query);
  const label = title || `${AGG_LABELS[agg]} of ${column || "rows"}`;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">KPI card</CardTitle>
        <CardDescription>One aggregated number. Meanings (revenue, patients…) are only what the column contains.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <select aria-label="KPI aggregation" className={select} value={agg} onChange={(e) => setAgg(e.target.value)}>
            {Object.entries(AGG_LABELS).filter(([k]) => k !== "pct_total").map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select aria-label="KPI column" className={select} value={column} onChange={(e) => setColumn(e.target.value)}>
            <option value="">rows</option>
            {cols.filter((c) => !c.protected || agg === "count").map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
          <Input aria-label="KPI title" className="h-8 w-48" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        {!valid ? <p className="text-xs text-destructive">{AGG_LABELS[agg]} needs a numerical column.</p> : null}
        <div className="rounded-xl border bg-gradient-to-br from-card to-secondary/40 p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
          {res.data ? <ChartView result={res.data.result} chartType="kpi" /> : res.error ? <p className="text-sm text-destructive">{(res.error as Error).message}</p> : <p className="text-sm text-muted-foreground">…</p>}
        </div>
        <div className="flex gap-2">
          <select aria-label="KPI dashboard" className={select} value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">Add to dashboard…</option>
            {dashboards.map((d) => <option key={String(d.id)} value={String(d.id)}>{String(d.name)}</option>)}
          </select>
          <Button size="sm" disabled={!target || !query} onClick={async () => {
            try { await onAdd(target, { kind: "kpi", title: label, size: "small", spec: { query, display: {} } }); setMsg("Added."); } catch (e) { setMsg((e as Error).message); }
          }}>Add KPI</Button>
          {msg ? <span className="text-xs text-muted-foreground" role="status">{msg}</span> : null}
        </div>
      </CardContent>
    </Card>
  );
}

// ---- Pivot table -----------------------------------------------------------------------------------

export function PivotTable({ result }: { result: Dict }) {
  if (result.suppressed) return <p className="text-sm text-muted-foreground">{String(result.note)}</p>;
  const groups = (result.group_columns as string[]) ?? [];
  const rows = (result.rows as Dict[]) ?? [];
  const margins = result.margins as { rows: Record<string, Dict>; columns: Record<string, Dict> } | undefined;
  if (groups.length === 1) {
    return (
      <table className="w-full text-sm">
        <thead><tr className="border-b"><th className="p-2 text-left">{groups[0]}</th><th className="p-2 text-right">n</th><th className="p-2 text-right">{(result.measure_labels as string[])[0]}</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b">
              <td className="p-2">{(r.groups as string[])[0]}</td>
              <td className="p-2 text-right tabular-nums">{r.suppressed ? "suppressed" : String(r.n)}</td>
              <td className="p-2 text-right tabular-nums">{fmtValue((r.values as unknown[])[0])}</td>
            </tr>
          ))}
          <tr className="font-medium"><td className="p-2">Total</td><td className="p-2 text-right">{String((result.totals as Dict).n)}</td><td className="p-2 text-right">{fmtValue(((result.totals as Dict).values as unknown[])[0])}</td></tr>
        </tbody>
      </table>
    );
  }
  const rowKeys = [...new Set(rows.map((r) => (r.groups as string[])[0]))];
  const colKeys = [...new Set(rows.map((r) => (r.groups as string[])[1]))];
  const cell = (a: string, b: string) => rows.find((r) => (r.groups as string[])[0] === a && (r.groups as string[])[1] === b);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" aria-label="Pivot table">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="p-2 text-left">{groups[0]} \ {groups[1]}</th>
            {colKeys.map((c) => <th key={c} className="p-2 text-right">{c}</th>)}
            <th className="p-2 text-right">Total</th>
          </tr>
        </thead>
        <tbody>
          {rowKeys.map((rk) => (
            <tr key={rk} className="border-b">
              <th className="p-2 text-left font-medium">{rk}</th>
              {colKeys.map((ck) => {
                const c = cell(rk, ck);
                return <td key={ck} className="p-2 text-right tabular-nums" title={c ? `n = ${String(c.n ?? "suppressed")}` : "no rows"}>{!c ? "·" : c.suppressed ? "—s" : fmtValue((c.values as unknown[])[0])}</td>;
              })}
              <td className="p-2 text-right font-medium tabular-nums">{margins?.rows[rk]?.suppressed ? "—s" : fmtValue((margins?.rows[rk]?.values as unknown[] | undefined)?.[0])}</td>
            </tr>
          ))}
          <tr className="font-medium">
            <th className="p-2 text-left">Total</th>
            {colKeys.map((ck) => <td key={ck} className="p-2 text-right tabular-nums">{margins?.columns[ck]?.suppressed ? "—s" : fmtValue((margins?.columns[ck]?.values as unknown[] | undefined)?.[0])}</td>)}
            <td className="p-2 text-right tabular-nums">{fmtValue(((result.totals as Dict).values as unknown[])[0])}</td>
          </tr>
        </tbody>
      </table>
      <p className="mt-1 text-xs text-muted-foreground">· no rows · —s suppressed (fewer than {String(result.min_cell)} rows in privacy mode). Missing values appear as “(missing)”.</p>
    </div>
  );
}

export function PivotBuilder({ catalogue, dashboards, onAdd }: { catalogue: Catalogue; dashboards: Dict[]; onAdd: (id: string, w: Dict) => Promise<void> }) {
  const cols = catalogue.columns.filter((c) => !c.protected);
  const dims = cols.filter((c) => c.inferred_type !== "text");
  const [rowDim, setRowDim] = useState("");
  const [colDim, setColDim] = useState("");
  const [agg, setAgg] = useState("count");
  const [value, setValue] = useState("");
  const [filters, setFilters] = useState<Filter[]>([]);
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const group = (n: string): Dict => {
    const c = cols.find((x) => x.name === n);
    if (c?.inferred_type === "datetime") return { column: n, date_part: "month" };
    if (c?.inferred_type === "numeric" && (c.n_unique ?? 0) > 50) return { column: n, bins: 8 };
    return { column: n };
  };
  const query = rowDim
    ? { kind: "aggregate", group_by: [group(rowDim), ...(colDim ? [group(colDim)] : [])], measures: [{ agg, ...(value ? { column: value } : {}) }], filters, margins: true, sort: { by: "group", dir: "asc" }, limit: 500 }
    : null;
  const res = useAnalyticsQuery(catalogue.version_id, query);
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Pivot table</CardTitle>
        <CardDescription>Rows × columns with totals. Limited to 1,000 groups and 500 shown cells.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <select aria-label="Pivot rows" className={select} value={rowDim} onChange={(e) => setRowDim(e.target.value)}>
            <option value="">Rows…</option>
            {dims.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
          <select aria-label="Pivot columns" className={select} value={colDim} onChange={(e) => setColDim(e.target.value)}>
            <option value="">Columns (optional)…</option>
            {dims.filter((c) => c.name !== rowDim).map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
          <select aria-label="Pivot aggregation" className={select} value={agg} onChange={(e) => setAgg(e.target.value)}>
            {Object.entries(AGG_LABELS).filter(([k]) => k !== "pct_total").map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select aria-label="Pivot value" className={select} value={value} onChange={(e) => setValue(e.target.value)}>
            <option value="">rows</option>
            {cols.filter((c) => c.inferred_type === "numeric").map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
        </div>
        <FilterEditor filters={filters} onChange={setFilters} columns={catalogue.columns} />
        {!query ? <p className="text-sm text-muted-foreground">Choose a row dimension.</p> : res.error ? <p role="alert" className="text-sm text-destructive">{(res.error as Error).message}</p> : res.data ? <PivotTable result={res.data.result} /> : <p className="text-sm text-muted-foreground">Calculating…</p>}
        {query ? (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => exportQueryCsv(catalogue.version_id, query).catch((e) => setMsg(e.message))}>Export CSV</Button>
            <select aria-label="Pivot dashboard" className={select} value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Add to dashboard…</option>
              {dashboards.map((d) => <option key={String(d.id)} value={String(d.id)}>{String(d.name)}</option>)}
            </select>
            <Button size="sm" disabled={!target} onClick={async () => {
              try { await onAdd(target, { kind: "pivot", title: `${rowDim}${colDim ? ` × ${colDim}` : ""}`, size: "large", spec: { query, display: {} } }); setMsg("Added."); } catch (e) { setMsg((e as Error).message); }
            }}>Add pivot</Button>
            {msg ? <span className="text-xs text-muted-foreground" role="status">{msg}</span> : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

// ---- Insights --------------------------------------------------------------------------------------

const BASIS: Record<string, { label: string; variant: "secondary" | "outline" | "destructive" }> = {
  descriptive: { label: "Observed", variant: "secondary" },
  association: { label: "Statistical association", variant: "outline" },
  suspicion: { label: "Suspicion", variant: "destructive" },
  configuration: { label: "Configuration fact", variant: "secondary" },
};

export function InsightsPanel({ versionId, catalogue }: { versionId: string; catalogue: Catalogue }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["analytics", versionId, "insights"],
    queryFn: () => analyticsApi.insights(versionId),
    refetchInterval: (query) => {
      const job = (query.state.data as Dict | undefined)?.job as Dict | undefined;
      return job && ["queued", "running"].includes(String(job.status)) ? 2000 : false;
    },
  });
  const [error, setError] = useState<string | null>(null);
  const [chart, setChart] = useState<Dict | null>(null);
  const preview = useAnalyticsQuery(versionId, chart);
  const data = q.data as Dict | undefined;
  const summary = data?.summary as Dict | null | undefined;
  const running = ["queued", "running"].includes(String((data?.job as Dict | undefined)?.status));
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          Evidence-based observations with their method, sample size and limitations. Associations use
          multiplicity-adjusted q-values; nothing here is causal.
        </p>
        <div className="flex items-center gap-2">
          {data?.stale ? <Badge variant="outline">roles changed since generation</Badge> : null}
          <Button size="sm" disabled={running} onClick={async () => {
            setError(null);
            try { await analyticsApi.startInsights(versionId); await qc.invalidateQueries({ queryKey: ["analytics", versionId, "insights"] }); } catch (e) { setError((e as Error).message); }
          }}>{running ? "Generating…" : summary ? "Regenerate insights" : "Generate insights"}</Button>
        </div>
      </div>
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
      {!summary && !running ? <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">No insights yet. Generate them to scan distributions, missingness, outliers, correlations, groups, trends and target associations.</p> : null}
      {summary ? (
        <>
          <p className="text-xs text-muted-foreground">
            {Number(summary.rows).toLocaleString()} rows{summary.sampled ? ` (analysed on a deterministic sample of ${Number(summary.sample_rows).toLocaleString()})` : ""} · unit: {String(summary.unit)}
            {summary.entity_column ? ` (${String(summary.entity_column)})` : ""}
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            {((summary.insights as Dict[]) ?? []).map((i) => {
              const b = BASIS[String(i.basis)] ?? BASIS.descriptive;
              return (
                <Card key={String(i.id)} className="flex flex-col">
                  <CardHeader className="pb-2">
                    <div className="flex flex-wrap gap-1"><Badge variant={b.variant}>{b.label}</Badge><Badge variant="outline">{String(i.category).replaceAll("_", " ")}</Badge></div>
                    <CardTitle className="text-sm">{String(i.title)}</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 space-y-2 text-sm">
                    <p>{String(i.interpretation)}</p>
                    <details className="text-xs text-muted-foreground">
                      <summary className="cursor-pointer">Evidence and method</summary>
                      <p className="mt-1">{String(i.method)}{i.n ? ` · n = ${String(i.n)}` : ""}</p>
                      <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-muted p-2">{JSON.stringify(i.evidence, null, 1)}</pre>
                      {((i.limitations as string[]) ?? []).map((l) => <p key={l}>• {l}</p>)}
                    </details>
                    {i.next_action ? <p className="text-xs"><strong>Next:</strong> {String(i.next_action)}</p> : null}
                    {i.chart ? <Button size="sm" variant="outline" onClick={() => setChart(i.chart as Dict)}>Show chart</Button> : null}
                  </CardContent>
                </Card>
              );
            })}
          </div>
          {chart ? (
            <Card>
              <CardContent className="pt-6">
                {preview.data ? <ChartView result={preview.data.result} chartType={chartTypeFor(chart)} /> : preview.error ? <p className="text-sm text-destructive">{(preview.error as Error).message}</p> : <p className="text-sm text-muted-foreground">Calculating…</p>}
              </CardContent>
            </Card>
          ) : null}
          {((summary.not_assessed as Dict[]) ?? []).length ? (
            <div className="text-xs text-muted-foreground">
              <p className="font-medium">Not assessed</p>
              {((summary.not_assessed as Dict[]) ?? []).map((n, k) => <p key={k}>• {String(n.area)}: {String(n.reason)}</p>)}
            </div>
          ) : null}
          <ul className="list-disc pl-5 text-xs text-muted-foreground">{((summary.limitations as string[]) ?? []).map((l) => <li key={l}>{l}</li>)}</ul>
        </>
      ) : null}
      {catalogue.privacy_mode ? <p className="text-xs text-muted-foreground">Privacy mode is on for this dataset: small groups are suppressed.</p> : null}
    </div>
  );
}

export function chartTypeFor(query: Dict): "bar" | "line" | "histogram" | "box" | "scatter" | "correlation_heatmap" | "missingness_heatmap" {
  switch (query.kind) {
    case "histogram": return "histogram";
    case "box": return "box";
    case "scatter": return "scatter";
    case "correlation": return "correlation_heatmap";
    case "missingness": return "missingness_heatmap";
    default: return ((query.group_by as Dict[]) ?? []).some((g) => g.date_part) ? "line" : "bar";
  }
}

// ---- Version comparison ------------------------------------------------------------------------------

export function ComparePanel({ versions, defaultA, defaultB }: { versions: { id: string; version_number: number }[]; defaultA?: string; defaultB?: string }) {
  const [a, setA] = useState(defaultA ?? versions[0]?.id ?? "");
  const [b, setB] = useState(defaultB ?? versions[versions.length - 1]?.id ?? "");
  const q = useQuery({ queryKey: ["analytics", "compare", a, b], queryFn: () => analyticsApi.compare(a, b), enabled: Boolean(a && b && a !== b), retry: false });
  const c = q.data as Dict | undefined;
  if (versions.length < 2) return <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">This dataset has one version. Commit a pipeline to compare versions.</p>;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        Compare
        <select aria-label="First version" className={select} value={a} onChange={(e) => setA(e.target.value)}>{versions.map((v) => <option key={v.id} value={v.id}>v{v.version_number}</option>)}</select>
        with
        <select aria-label="Second version" className={select} value={b} onChange={(e) => setB(e.target.value)}>{versions.map((v) => <option key={v.id} value={v.id}>v{v.version_number}</option>)}</select>
      </div>
      {a === b ? <p className="text-sm text-muted-foreground">Choose two different versions.</p> : q.error ? <p className="text-sm text-destructive">{(q.error as Error).message}</p> : !c ? <p className="text-sm text-muted-foreground">Comparing…</p> : (
        <>
          <div className="grid gap-3 sm:grid-cols-4">
            {((c.versions as Dict[]) ?? []).map((v) => (
              <div key={String(v.id)} className="rounded-lg border bg-card p-3 text-sm"><p className="text-xs text-muted-foreground">v{String(v.number)} · {String(v.kind)}</p><p className="font-semibold tabular-nums">{String(v.rows)} × {String(v.columns)}</p></div>
            ))}
            <div className="rounded-lg border bg-card p-3 text-sm"><p className="text-xs text-muted-foreground">Rows in both</p><p className="font-semibold">{String((c.rows as Dict).in_both)}</p><p className="text-xs">only first {String((c.rows as Dict).only_first)} · only second {String((c.rows as Dict).only_second)}</p></div>
            <div className="rounded-lg border bg-card p-3 text-sm"><p className="text-xs text-muted-foreground">Columns</p><p className="text-xs">added: {((c.columns_added as string[]) ?? []).join(", ") || "none"}</p><p className="text-xs">removed: {((c.columns_removed as string[]) ?? []).join(", ") || "none"}</p></div>
          </div>
          {Number((c.rows as Dict).only_first) || Number((c.rows as Dict).only_second) ? <p role="note" className="text-xs text-amber-700">Row populations differ: distributions are not computed on the same rows.</p> : null}
          <div className="grid gap-3 md:grid-cols-2">
            {((c.columns as Dict[]) ?? []).map((col) => (
              <Card key={String(col.column)}>
                <CardHeader className="pb-1"><CardTitle className="text-sm">{String(col.column)}</CardTitle><CardDescription className="text-xs">missing {String((col.missing_pct as number[])[0])}% → {String((col.missing_pct as number[])[1])}%{col.protected ? " · sensitive: values not shown" : ""}</CardDescription></CardHeader>
                <CardContent>
                  {col.type === "numeric" ? <CompareBars col={col} /> : col.type === "categorical" ? (
                    <table className="w-full text-xs"><tbody>{((col.categories as Dict[]) ?? []).map((k) => <tr key={String(k.label)} className="border-b"><td>{String(k.label)}</td><td className="text-right tabular-nums">{fmtValue((k.counts as unknown[])[0])}</td><td className="text-right tabular-nums">{fmtValue((k.counts as unknown[])[1])}</td></tr>)}</tbody></table>
                  ) : null}
                </CardContent>
              </Card>
            ))}
          </div>
          {((c.transformation_effects as Dict[]) ?? []).length ? (
            <Card><CardHeader className="pb-1"><CardTitle className="text-sm">Transformation effects</CardTitle></CardHeader><CardContent className="space-y-1 text-xs">
              {((c.transformation_effects as Dict[]) ?? []).map((e, i) => (
                <p key={i}><strong>{String(e.op)}</strong>: {((e.columns as Dict[]) ?? []).map((x) => `${String(x.column)} — ${String(x.cells_filled ?? 0)} estimated, ${String(x.observed_changed ?? 0)} changed observed values`).join("; ")}{Number(e.rows_removed) ? ` · ${String(e.rows_removed)} rows removed` : ""}</p>
              ))}
            </CardContent></Card>
          ) : null}
          <ul className="list-disc pl-5 text-xs text-muted-foreground">{((c.notes as string[]) ?? []).map((n) => <li key={n}>{n}</li>)}</ul>
        </>
      )}
    </div>
  );
}

function CompareBars({ col }: { col: Dict }) {
  const edges = col.edges as number[];
  const [ha, hb] = col.histogram as (number | null)[][];
  const max = Math.max(1, ...ha.map((v) => v ?? 0), ...hb.map((v) => v ?? 0));
  const [sa, sb] = col.stats as Dict[];
  const W = 300;
  const bw = W / ha.length;
  return (
    <div className="space-y-1">
      <svg data-chart role="img" aria-label={`Distribution of ${String(col.column)} in both versions`} width={W} height={90}>
        {ha.map((v, i) => <rect key={`a${i}`} x={i * bw} y={80 - (80 * (v ?? 0)) / max} width={bw - 1} height={(80 * (v ?? 0)) / max} fill="var(--viz-cat-1)" fillOpacity={0.45}><title>{`${fmtValue(edges[i])}–${fmtValue(edges[i + 1])}: first ${v ?? "suppressed"}`}</title></rect>)}
        {hb.map((v, i) => <rect key={`b${i}`} x={i * bw + bw / 4} y={80 - (80 * (v ?? 0)) / max} width={bw / 2} height={(80 * (v ?? 0)) / max} fill="var(--viz-cat-2)"><title>{`${fmtValue(edges[i])}–${fmtValue(edges[i + 1])}: second ${v ?? "suppressed"}`}</title></rect>)}
      </svg>
      <p className="text-xs text-muted-foreground">mean {fmtValue(sa.mean)} → {fmtValue(sb.mean)} · median {fmtValue(sa.median)} → {fmtValue(sb.median)} · outliers {String(sa.outliers_iqr ?? "—")} → {String(sb.outliers_iqr ?? "—")}</p>
    </div>
  );
}

// ---- Dashboards list ---------------------------------------------------------------------------------

export function DashboardsPanel({ datasetId, versionId, dashboards, onChange }: { datasetId: string; versionId: string; dashboards: Dict[]; onChange: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2"><CardTitle className="text-base">New dashboard</CardTitle><CardDescription>Bound to the selected version; it never switches version on its own.</CardDescription></CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Input aria-label="Dashboard name" className="h-8 w-56" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
          <Input aria-label="Dashboard description" className="h-8 w-72" placeholder="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />
          <Button size="sm" disabled={!name.trim()} onClick={async () => {
            setError(null);
            try { await analyticsApi.createDashboard(datasetId, { version_id: versionId, name, description }); setName(""); setDescription(""); onChange(); } catch (e) { setError((e as Error).message); }
          }}>Create</Button>
          {error ? <p role="alert" className="w-full text-sm text-destructive">{error}</p> : null}
        </CardContent>
      </Card>
      {!dashboards.length ? <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">No dashboards yet.</p> : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {dashboards.map((d) => (
            <Card key={String(d.id)}>
              <CardHeader className="pb-2"><CardTitle className="text-sm"><Link className="text-primary hover:underline" href={`/datasets/${datasetId}/analytics/dashboards/${String(d.id)}`}>{String(d.name)}</Link></CardTitle><CardDescription className="text-xs">{String(d.widget_count)} widget(s) · updated {new Date(String(d.updated_at)).toLocaleString()}</CardDescription></CardHeader>
              <CardContent className="flex gap-2">
                <Button size="sm" variant="outline" onClick={async () => { await analyticsApi.duplicateDashboard(String(d.id), {}); onChange(); }}>Duplicate</Button>
                <Button size="sm" variant="outline" onClick={async () => { if (window.confirm(`Delete dashboard "${String(d.name)}"?`)) { await analyticsApi.deleteDashboard(String(d.id)); onChange(); } }}>Delete</Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
