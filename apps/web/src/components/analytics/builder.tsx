"use client";

import { useMemo, useState } from "react";

import { ChartView } from "@/components/analytics/chart-view";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  AGG_LABELS,
  type BuilderState,
  type Catalogue,
  CHART_LABELS,
  type ChartType,
  chartWarnings,
  type Column,
  DEFAULT_STATE,
  type Dict,
  exportQueryCsv,
  type Filter,
  isBlocking,
  OP_LABELS,
  recommend,
  toSpec,
  useAnalyticsQuery,
} from "@/lib/analytics";

const select = "h-8 w-full rounded-md border bg-background px-2 text-sm";

function ColumnSelect({ label, value, onChange, columns, allowNone = true, filter }: {
  label: string; value: string | null; onChange: (v: string | null) => void; columns: Column[]; allowNone?: boolean;
  filter?: (c: Column) => boolean;
}) {
  return (
    <label className="block space-y-1 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <select aria-label={label} className={select} value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
        {allowNone ? <option value="">—</option> : null}
        {columns.filter((c) => !c.protected && (!filter || filter(c))).map((c) => (
          <option key={c.name} value={c.name}>{c.name} ({c.inferred_type})</option>
        ))}
      </select>
    </label>
  );
}

export function FilterEditor({ filters, onChange, columns }: { filters: Filter[]; onChange: (f: Filter[]) => void; columns: Column[] }) {
  const [col, setCol] = useState<string>("");
  const [op, setOp] = useState("eq");
  const [val, setVal] = useState("");
  const [val2, setVal2] = useState("");
  const c = columns.find((x) => x.name === col);
  const ops = !c ? [] : c.protected ? ["is_missing", "not_missing"]
    : c.inferred_type === "numeric" ? ["eq", "ne", "gt", "gte", "lt", "lte", "between", "is_missing", "not_missing"]
    : c.inferred_type === "datetime" ? ["date_between", "is_missing", "not_missing"]
    : c.inferred_type === "boolean" ? ["eq", "is_missing", "not_missing"]
    : ["eq", "ne", "in", "not_in", "is_missing", "not_missing"];
  const parse = (v: string): unknown => {
    if (c?.inferred_type === "numeric") return v.trim() === "" ? null : Number(v);
    if (c?.inferred_type === "boolean") return v === "true";
    return v;
  };
  function add() {
    if (!c) return;
    const f: Filter = { column: c.name, op };
    if (op === "between" || op === "date_between") {
      if (val) f.min = parse(val);
      if (val2) f.max = parse(val2);
    } else if (op === "in" || op === "not_in") f.values = val.split(",").map((s) => parse(s.trim())).filter((v) => v !== "");
    else if (!["is_missing", "not_missing"].includes(op)) f.value = parse(val);
    if (c.inferred_type === "numeric" && [f.value, f.min, f.max].some((v) => v !== undefined && v !== null && Number.isNaN(v as number))) return;
    onChange([...filters, f]);
    setVal("");
    setVal2("");
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {filters.map((f, i) => (
          <Badge key={i} variant="secondary" className="gap-1">
            {f.column} {OP_LABELS[f.op]} {f.values ? (f.values as unknown[]).join(", ") : f.min !== undefined || f.max !== undefined ? `${String(f.min ?? "…")}–${String(f.max ?? "…")}` : f.value !== undefined ? String(f.value) : ""}
            <button aria-label={`Remove filter ${i + 1}`} className="ml-1" onClick={() => onChange(filters.filter((_, j) => j !== i))}>×</button>
          </Badge>
        ))}
        {!filters.length ? <span className="text-xs text-muted-foreground">No filters</span> : null}
      </div>
      <div className="grid grid-cols-2 gap-1">
        <select aria-label="Filter column" className={select} value={col} onChange={(e) => { setCol(e.target.value); setOp("eq"); }}>
          <option value="">Column…</option>
          {columns.map((x) => <option key={x.name} value={x.name}>{x.name}{x.protected ? " (sensitive)" : ""}</option>)}
        </select>
        <select aria-label="Filter operator" className={select} value={op} onChange={(e) => setOp(e.target.value)} disabled={!c}>
          {ops.map((o) => <option key={o} value={o}>{OP_LABELS[o]}</option>)}
        </select>
        {!["is_missing", "not_missing"].includes(op) && c ? (
          <>
            <Input aria-label="Filter value" className="h-8" placeholder={op.includes("between") ? "from" : op.includes("in") ? "a, b, c" : "value"} value={val} onChange={(e) => setVal(e.target.value)} type={c.inferred_type === "datetime" ? "date" : "text"} />
            {op.includes("between") ? <Input aria-label="Filter value to" className="h-8" placeholder="to" value={val2} onChange={(e) => setVal2(e.target.value)} type={c.inferred_type === "datetime" ? "date" : "text"} /> : <span />}
          </>
        ) : null}
      </div>
      <Button size="sm" variant="outline" onClick={add} disabled={!c}>Add filter</Button>
    </div>
  );
}

export function stateFromSpec(spec: Dict): BuilderState {
  const q = (spec.query ?? {}) as Dict;
  const d = (spec.display ?? {}) as Dict;
  const g = (q.group_by as Dict[]) ?? [];
  const m = ((q.measures as Dict[]) ?? [])[0] ?? {};
  const cols = (q.columns as string[]) ?? [];
  const kind = q.kind;
  return {
    ...DEFAULT_STATE,
    chartType: (d.chart_type as ChartType) ?? "bar",
    x: kind === "scatter" || kind === "histogram" ? cols[0] ?? null : (g[0]?.column as string) ?? null,
    y: kind === "scatter" ? cols[1] ?? null : kind === "box" ? cols[0] ?? null : (m.column as string) ?? null,
    agg: (m.agg as string) ?? "count",
    color: kind === "scatter" ? ((q.color as string) ?? null) : ((g[1]?.column as string) ?? null),
    size: (q.size as string) ?? null,
    filters: (q.filters as Filter[]) ?? [],
    title: (d.title as string) ?? "",
    xLabel: (d.x_label as string) ?? "",
    yLabel: (d.y_label as string) ?? "",
    datePart: (g[0]?.date_part as string) ?? "month",
    bins: (q.bins as number) ?? (g[0]?.bins as number) ?? 20,
    q: (m.q as number) ?? DEFAULT_STATE.q,
    sortBy: ((q.sort as Dict | undefined)?.by as "group" | "measure") ?? DEFAULT_STATE.sortBy,
    sortDir: ((q.sort as Dict | undefined)?.dir as "asc" | "desc") ?? DEFAULT_STATE.sortDir,
    limit: (q.limit as number) ?? DEFAULT_STATE.limit,
  };
}

export function ChartBuilder({
  catalogue,
  initial,
  dashboards,
  onAdd,
  onSave,
  extraFilters = [],
}: {
  catalogue: Catalogue;
  initial?: BuilderState;
  dashboards?: Dict[];
  onAdd?: (dashboardId: string, widget: Dict) => Promise<void>;
  onSave?: (widget: Dict) => Promise<void>;
  extraFilters?: Filter[];
}) {
  const [s, setS] = useState<BuilderState>(initial ?? DEFAULT_STATE);
  const [target, setTarget] = useState<string>("");
  const [message, setMessage] = useState<string | null>(null);
  const set = <K extends keyof BuilderState>(k: K, v: BuilderState[K]) => setS((p) => ({ ...p, [k]: v }));
  const cols = catalogue.columns;
  const built = useMemo(() => toSpec(s, cols), [s, cols]);
  const warnings = chartWarnings(s, cols);
  const blocked = isBlocking(warnings);
  const query: Dict | null = "query" in built && !blocked ? { ...built.query, filters: [...((built.query.filters as Filter[]) ?? []), ...extraFilters] } : null;
  const res = useAnalyticsQuery(catalogue.version_id, query);
  const recs = recommend(cols.find((c) => c.name === s.x), cols.find((c) => c.name === s.y));
  const needsAgg = !["scatter", "bubble", "histogram", "box", "correlation_heatmap", "missingness_heatmap"].includes(s.chartType);
  const widget = "query" in built ? { kind: "chart", title: s.title || CHART_LABELS[s.chartType], spec: { query: built.query, display: built.display } } : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
      <Card className="h-fit">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Chart</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="block space-y-1 text-xs">
            <span className="text-muted-foreground">Chart type</span>
            <select aria-label="Chart type" className={select} value={s.chartType} onChange={(e) => set("chartType", e.target.value as ChartType)}>
              {(Object.keys(CHART_LABELS) as ChartType[]).map((t) => <option key={t} value={t}>{CHART_LABELS[t]}</option>)}
            </select>
          </label>
          {!["correlation_heatmap"].includes(s.chartType) ? (
            <ColumnSelect label={s.chartType === "box" ? "Group by (optional)" : s.chartType === "histogram" ? "Column" : "X axis"} value={s.x} onChange={(v) => set("x", v)} columns={cols}
              filter={["scatter", "bubble", "histogram"].includes(s.chartType) ? (c) => c.inferred_type === "numeric" : undefined} />
          ) : null}
          {!["histogram", "correlation_heatmap", "missingness_heatmap"].includes(s.chartType) ? (
            <ColumnSelect label={["scatter", "bubble", "box"].includes(s.chartType) ? "Y axis (numerical)" : "Value (Y)"} value={s.y} onChange={(v) => set("y", v)} columns={cols}
              filter={["scatter", "bubble", "box"].includes(s.chartType) || (needsAgg && !["count", "count_distinct"].includes(s.agg)) ? (c) => c.inferred_type === "numeric" || c.inferred_type === "boolean" : undefined} />
          ) : null}
          {needsAgg ? (
            <label className="block space-y-1 text-xs">
              <span className="text-muted-foreground">Aggregation</span>
              <select aria-label="Aggregation" className={select} value={s.agg} onChange={(e) => set("agg", e.target.value)}>
                {Object.entries(AGG_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
          ) : null}
          {s.agg === "quantile" && needsAgg ? (
            <label className="block text-xs">Quantile <Input aria-label="Quantile" type="number" step="0.05" min={0} max={1} value={s.q} onChange={(e) => set("q", Number(e.target.value))} /></label>
          ) : null}
          {["grouped_bar", "stacked_bar", "scatter", "bubble"].includes(s.chartType) ? (
            <ColumnSelect label="Colour / group" value={s.color} onChange={(v) => set("color", v)} columns={cols} filter={(c) => c.inferred_type !== "numeric" || (c.n_unique ?? 99) <= 12} />
          ) : null}
          {s.chartType === "bubble" ? <ColumnSelect label="Bubble size" value={s.size} onChange={(v) => set("size", v)} columns={cols} filter={(c) => c.inferred_type === "numeric"} /> : null}
          {cols.find((c) => c.name === s.x)?.inferred_type === "datetime" ? (
            <label className="block space-y-1 text-xs">
              <span className="text-muted-foreground">Date grouping</span>
              <select aria-label="Date grouping" className={select} value={s.datePart} onChange={(e) => set("datePart", e.target.value)}>
                {["day", "week", "month", "quarter", "year"].map((d) => <option key={d}>{d}</option>)}
              </select>
            </label>
          ) : null}
          {s.chartType === "histogram" || (cols.find((c) => c.name === s.x)?.inferred_type === "numeric" && needsAgg) ? (
            <label className="block text-xs">Bins <Input aria-label="Bins" type="number" min={2} max={60} value={s.bins} onChange={(e) => set("bins", Number(e.target.value))} /></label>
          ) : null}
          {needsAgg ? (
            <div className="grid grid-cols-2 gap-1 text-xs">
              <select aria-label="Sort by" className={select} value={s.sortBy} onChange={(e) => set("sortBy", e.target.value as "group" | "measure")}>
                <option value="measure">Sort by value</option>
                <option value="group">Sort by label</option>
              </select>
              <select aria-label="Sort direction" className={select} value={s.sortDir} onChange={(e) => set("sortDir", e.target.value as "asc" | "desc")}>
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
          ) : null}
          <Input aria-label="Chart title" className="h-8" placeholder="Title" value={s.title} onChange={(e) => set("title", e.target.value)} />
          <div className="grid grid-cols-2 gap-1">
            <Input aria-label="X axis label" className="h-8" placeholder="X label" value={s.xLabel} onChange={(e) => set("xLabel", e.target.value)} />
            <Input aria-label="Y axis label" className="h-8" placeholder="Y label" value={s.yLabel} onChange={(e) => set("yLabel", e.target.value)} />
          </div>
          <div className="space-y-1 border-t pt-3">
            <p className="text-xs font-medium">Filters</p>
            <FilterEditor filters={s.filters} onChange={(f) => set("filters", f)} columns={cols} />
          </div>
        </CardContent>
      </Card>
      <div className="space-y-3">
        {recs.length ? (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">Suggested:</span>
            {recs.map((r) => (
              <Button key={r.type + r.agg} size="sm" variant="outline" title={r.why} onClick={() => setS((p) => ({ ...p, chartType: r.type, agg: r.agg === "count" && p.agg !== "count" && ["bar", "line", "area"].includes(r.type) ? p.agg : r.agg }))}>
                {CHART_LABELS[r.type]}
              </Button>
            ))}
          </div>
        ) : null}
        {warnings.map((w) => (
          <p key={w} role="alert" className="rounded-md border border-amber-500/50 bg-amber-50 p-2 text-xs text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">{w}</p>
        ))}
        <Card>
          <CardContent className="pt-6">
            {"error" in built ? (
              <p className="text-sm text-muted-foreground">{built.error}</p>
            ) : blocked ? null : res.isPending ? (
              <p className="text-sm text-muted-foreground">Calculating…</p>
            ) : res.error ? (
              <p role="alert" className="text-sm text-destructive">{(res.error as Error).message}</p>
            ) : res.data ? (
              <ChartView result={res.data.result} chartType={s.chartType} title={s.title} xLabel={s.xLabel} yLabel={s.yLabel}
                onSelect={(cat) => s.x && set("filters", [...s.filters, { column: s.x, op: "eq", value: cat }])} />
            ) : null}
          </CardContent>
        </Card>
        {widget && !blocked ? (
          <div className="no-print flex flex-wrap items-center gap-2">
            {onSave ? <Button size="sm" onClick={() => onSave(widget)}>Save widget</Button> : null}
            {onAdd && dashboards ? (
              <>
                <select aria-label="Dashboard" className="h-8 rounded-md border bg-background px-2 text-sm" value={target} onChange={(e) => setTarget(e.target.value)}>
                  <option value="">Add to dashboard…</option>
                  {dashboards.map((d) => <option key={String(d.id)} value={String(d.id)}>{String(d.name)}</option>)}
                </select>
                <Button size="sm" disabled={!target} onClick={async () => {
                  try { await onAdd(target, widget); setMessage("Added to the dashboard."); } catch (e) { setMessage((e as Error).message); }
                }}>Add</Button>
              </>
            ) : null}
            {query && ["aggregate", "histogram", "box", "correlation", "missingness"].includes(String(query.kind)) ? (
              <Button size="sm" variant="outline" onClick={() => exportQueryCsv(catalogue.version_id, query).catch((e) => setMessage(e.message))}>Export CSV</Button>
            ) : null}
            {message ? <span className="text-xs text-muted-foreground" role="status">{message}</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
