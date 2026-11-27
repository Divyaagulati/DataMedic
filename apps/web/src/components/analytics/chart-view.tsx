"use client";

import { useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { aggregateRows, type ChartType, type Dict, exportChart, fmtValue, wideRows } from "@/lib/analytics";

const COLORS = ["var(--viz-cat-1)", "var(--viz-cat-3)", "var(--viz-cat-2)", "var(--viz-cat-4)", "var(--viz-cat-7)", "var(--viz-cat-5)", "var(--viz-cat-6)"];
const H = 300;

function Notes({ result }: { result: Dict }) {
  const notes: string[] = [];
  if (result.rows_filtered !== undefined && result.rows_total !== undefined && result.rows_filtered !== null) {
    notes.push(`${Number(result.rows_filtered).toLocaleString()} of ${Number(result.rows_total).toLocaleString()} rows after filters`);
  }
  if (result.suppressed_groups) notes.push(`${String(result.suppressed_groups)} small group(s) suppressed for privacy`);
  if (result.suppressed_bins) notes.push(`${String(result.suppressed_bins)} small bin(s) suppressed`);
  if (result.truncated) notes.push(`showing the first ${((result.rows as Dict[]) ?? []).length} of ${String(result.n_groups)} groups`);
  if (result.note) notes.push(String(result.note));
  if (result.privacy_mode) notes.push(`privacy mode: groups under ${String(result.min_cell)} rows hidden`);
  if (!notes.length) return null;
  return <p className="text-xs text-muted-foreground" data-testid="chart-notes">{notes.join(" · ")}</p>;
}

function DataTable({ result }: { result: Dict }) {
  const kind = result.kind;
  let head: string[] = [];
  let body: (string | number | null)[][] = [];
  if (kind === "aggregate") {
    head = [...((result.group_columns as string[]) ?? []), "n", ...((result.measure_labels as string[]) ?? [])];
    body = ((result.rows as Dict[]) ?? []).map((r) => [...(r.groups as string[]), r.suppressed ? "<min" : (r.n as number), ...((r.values as (number | null)[]) ?? [])]);
  } else if (kind === "histogram") {
    head = ["from", "to", "count"];
    const e = result.edges as number[];
    body = ((result.counts as (number | null)[]) ?? []).map((c, i) => [e[i], e[i + 1], c]);
  } else if (kind === "box") {
    head = ["group", "n", "q1", "median", "q3", "mean", "outliers"];
    body = ((result.items as Dict[]) ?? []).map((i) => [String(i.label), i.n as number, i.q1 as number, i.median as number, i.q3 as number, i.mean as number, i.outliers as number]);
  } else if (kind === "missingness") {
    head = ["column", "missing %"];
    body = ((result.columns as string[]) ?? []).map((c, i) => [c, (result.missing_pct as number[])[i]]);
  } else if (kind === "correlation") {
    const cols = (result.columns as string[]) ?? [];
    head = ["", ...cols];
    body = ((result.matrix as (number | null)[][]) ?? []).map((row, i) => [cols[i], ...row]);
  } else if (kind === "scatter") {
    head = ["x", "y"];
    body = ((result.points as Dict[]) ?? []).slice(0, 200).map((p) => [p.x as number, p.y as number]);
  }
  return (
    <div className="max-h-72 overflow-auto rounded border">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-muted">
          <tr>{head.map((h, i) => <th key={i} className="px-2 py-1 text-left font-medium">{h}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i} className="border-t">{r.map((v, j) => <td key={j} className="px-2 py-1 tabular-nums">{fmtValue(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Heat({ labels, matrix, rowLabels, diverging, aria }: { labels: string[]; rowLabels: string[]; matrix: (number | null)[][]; diverging: boolean; aria: string }) {
  const cell = 26;
  const left = 110;
  const top = 90;
  const width = left + labels.length * cell + 10;
  const height = top + rowLabels.length * cell + 10;
  const color = (v: number | null) => {
    if (v === null) return "var(--muted)";
    if (diverging) {
      const a = Math.min(Math.abs(v), 1);
      return v >= 0 ? `color-mix(in oklch, var(--viz-div-pos) ${a * 100}%, var(--viz-div-mid))` : `color-mix(in oklch, var(--viz-div-neg) ${a * 100}%, var(--viz-div-mid))`;
    }
    return `color-mix(in oklch, var(--viz-seq) ${Math.min(v, 100)}%, var(--viz-surface))`;
  };
  return (
    <div className="overflow-x-auto">
      <svg data-chart role="img" aria-label={aria} width={width} height={height} className="font-sans text-[10px]">
        {labels.map((l, j) => (
          <text key={l} transform={`translate(${left + j * cell + cell / 2},${top - 6}) rotate(-50)`} fill="currentColor">{l.slice(0, 14)}</text>
        ))}
        {rowLabels.map((r, i) => (
          <g key={r + i}>
            <text x={left - 6} y={top + i * cell + cell / 2 + 3} textAnchor="end" fill="currentColor">{r.slice(0, 16)}</text>
            {labels.map((l, j) => {
              const v = matrix[i]?.[j] ?? null;
              return (
                <rect key={l} x={left + j * cell} y={top + i * cell} width={cell - 2} height={cell - 2} rx={3} fill={color(v)}>
                  <title>{`${r} × ${l}: ${fmtValue(v)}`}</title>
                </rect>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

function BoxPlot({ items }: { items: Dict[] }) {
  const shown = items.filter((i) => !i.suppressed && i.n);
  if (!shown.length) return <p className="text-sm text-muted-foreground">No group has enough values.</p>;
  const lo = Math.min(...shown.map((i) => Number(i.whisker_low)));
  const hi = Math.max(...shown.map((i) => Number(i.whisker_high)));
  const W = 620;
  const left = 110;
  const row = 34;
  const scale = (v: number) => left + ((v - lo) / (hi - lo || 1)) * (W - left - 20);
  return (
    <div className="overflow-x-auto">
      <svg data-chart role="img" aria-label="Box plot" width={W} height={items.length * row + 30} className="text-[10px]">
        {items.map((it, i) => {
          const y = 10 + i * row;
          if (it.suppressed || !it.n) {
            return <text key={i} x={left - 6} y={y + 14} textAnchor="end" fill="currentColor">{String(it.label)} (suppressed)</text>;
          }
          const [wl, q1, md, q3, wh] = [it.whisker_low, it.q1, it.median, it.q3, it.whisker_high].map(Number);
          return (
            <g key={i}>
              <title>{`${String(it.label)}: median ${fmtValue(md)}, IQR ${fmtValue(q1)}–${fmtValue(q3)}, n ${String(it.n)}, ${String(it.outliers)} outliers`}</title>
              <text x={left - 6} y={y + 14} textAnchor="end" fill="currentColor">{String(it.label).slice(0, 16)}</text>
              <line x1={scale(wl)} x2={scale(wh)} y1={y + 10} y2={y + 10} stroke="var(--viz-axis)" />
              <rect x={scale(q1)} y={y} width={Math.max(scale(q3) - scale(q1), 1)} height={20} fill="var(--viz-seq-muted)" stroke="var(--viz-seq)" rx={3} />
              <line x1={scale(md)} x2={scale(md)} y1={y} y2={y + 20} stroke="var(--viz-seq)" strokeWidth={2} />
            </g>
          );
        })}
        <text x={left} y={items.length * row + 24} fill="currentColor">{fmtValue(lo)}</text>
        <text x={W - 20} y={items.length * row + 24} textAnchor="end" fill="currentColor">{fmtValue(hi)}</text>
      </svg>
    </div>
  );
}

export function ChartView({
  result,
  chartType,
  title,
  xLabel,
  yLabel,
  onSelect,
}: {
  result: Dict;
  chartType: ChartType | "kpi";
  title?: string;
  xLabel?: string;
  yLabel?: string;
  onSelect?: (category: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [table, setTable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (result.suppressed) {
    return <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">{String(result.note)}</p>;
  }
  const kind = result.kind;
  let chart: React.ReactNode = null;
  const axis = { stroke: "var(--viz-axis)", fontSize: 11 };
  if (chartType === "kpi") {
    const v = ((result.totals as Dict)?.values as (number | null)[])?.[0];
    return (
      <div className="py-2">
        <p className="text-3xl font-semibold tabular-nums" data-testid="kpi-value">{fmtValue(v)}</p>
        <p className="text-xs text-muted-foreground">
          {((result.measure_labels as string[]) ?? [])[0]} · {Number((result.totals as Dict)?.n ?? 0).toLocaleString()} rows
        </p>
      </div>
    );
  }
  if (kind === "aggregate") {
    const rows = aggregateRows(result).filter((r) => !r.suppressed);
    const label = ((result.measure_labels as string[]) ?? [])[0];
    if (!rows.length && chartType !== "table") chart = <p className="text-sm text-muted-foreground">No data for this selection.</p>;
    else if (chartType === "table") chart = <DataTable result={result} />;
    else if (chartType === "pie" || chartType === "donut") {
      chart = (
        <ResponsiveContainer width="100%" height={H}>
          <PieChart>
            <Pie data={rows} dataKey="value" nameKey="name" innerRadius={chartType === "donut" ? 60 : 0} outerRadius={110} onClick={(d) => onSelect?.(String((d as unknown as Dict).name))}>
              {rows.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip formatter={(v) => fmtValue(v)} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      );
    } else if (chartType === "grouped_bar" || chartType === "stacked_bar") {
      const { data, series } = wideRows(result);
      chart = (
        <ResponsiveContainer width="100%" height={H}>
          <BarChart data={data}>
            <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
            <XAxis dataKey="name" {...axis} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2 } : undefined} />
            <YAxis {...axis} tickFormatter={(v) => fmtValue(v)} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined} />
            <Tooltip formatter={(v) => fmtValue(v)} />
            <Legend />
            {series.map((s, i) => <Bar key={s} dataKey={s} stackId={chartType === "stacked_bar" ? "a" : undefined} fill={COLORS[i % COLORS.length]} radius={chartType === "stacked_bar" ? 0 : [3, 3, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      );
    } else if (chartType === "line" || chartType === "area") {
      const C = chartType === "line" ? LineChart : AreaChart;
      chart = (
        <ResponsiveContainer width="100%" height={H}>
          <C data={rows}>
            <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
            <XAxis dataKey="name" {...axis} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2 } : undefined} />
            <YAxis {...axis} tickFormatter={(v) => fmtValue(v)} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined} />
            <Tooltip formatter={(v) => [fmtValue(v), label]} />
            {chartType === "line" ? (
              <Line type="monotone" dataKey="value" stroke="var(--viz-cat-1)" strokeWidth={2} dot={rows.length < 40} name={label} />
            ) : (
              <Area type="monotone" dataKey="value" stroke="var(--viz-cat-1)" fill="var(--viz-seq-muted)" name={label} />
            )}
            {rows.length > 20 ? <Brush dataKey="name" height={18} stroke="var(--viz-axis)" /> : null}
          </C>
        </ResponsiveContainer>
      );
    } else {
      const horizontal = chartType === "hbar";
      chart = (
        <ResponsiveContainer width="100%" height={horizontal ? Math.max(H, rows.length * 22) : H}>
          <BarChart data={rows} layout={horizontal ? "vertical" : "horizontal"} onClick={(s) => s?.activeLabel && onSelect?.(String(s.activeLabel))}>
            <CartesianGrid stroke="var(--viz-grid)" vertical={horizontal} horizontal={!horizontal} />
            {horizontal ? (
              <>
                <XAxis type="number" {...axis} tickFormatter={(v) => fmtValue(v)} />
                <YAxis type="category" dataKey="name" width={120} {...axis} />
              </>
            ) : (
              <>
                <XAxis dataKey="name" {...axis} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2 } : undefined} />
                <YAxis {...axis} tickFormatter={(v) => fmtValue(v)} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined} />
              </>
            )}
            <Tooltip formatter={(v) => [fmtValue(v), label]} labelFormatter={(l) => `${String(l)}`} />
            <Bar dataKey="value" name={label} fill="var(--viz-cat-1)" radius={horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]} />
            {rows.length > 30 && !horizontal ? <Brush dataKey="name" height={18} stroke="var(--viz-axis)" /> : null}
          </BarChart>
        </ResponsiveContainer>
      );
    }
  } else if (kind === "histogram") {
    const e = result.edges as number[];
    const data = ((result.counts as (number | null)[]) ?? []).map((c, i) => ({ name: `${fmtValue(e[i])}–${fmtValue(e[i + 1])}`, value: c }));
    chart = (
      <ResponsiveContainer width="100%" height={H}>
        <BarChart data={data} barCategoryGap={1}>
          <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
          <XAxis dataKey="name" {...axis} interval="preserveStartEnd" label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2 } : undefined} />
          <YAxis {...axis} label={{ value: yLabel || "count", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Bar dataKey="value" name="count" fill="var(--viz-cat-1)" />
        </BarChart>
      </ResponsiveContainer>
    );
  } else if (kind === "box") {
    chart = <BoxPlot items={(result.items as Dict[]) ?? []} />;
  } else if (kind === "scatter") {
    if (result.mode === "binned") {
      const cells = (result.cells as Dict[]) ?? [];
      const xs = [...new Set(cells.map((c) => Number(c.x0)))].sort((a, b) => a - b);
      const ys = [...new Set(cells.map((c) => Number(c.y0)))].sort((a, b) => b - a);
      const max = Math.max(1, ...cells.map((c) => Number(c.count ?? 0)));
      const matrix = ys.map((y) => xs.map((x) => {
        const c = cells.find((k) => Number(k.x0) === x && Number(k.y0) === y);
        return c && c.count !== null ? (100 * Number(c.count)) / max : null;
      }));
      chart = <Heat labels={xs.map((x) => fmtValue(x))} rowLabels={ys.map((y) => fmtValue(y))} matrix={matrix} diverging={false} aria="Binned scatter (counts)" />;
    } else {
      const pts = (result.points as Dict[]) ?? [];
      const groups = [...new Set(pts.map((p) => String(p.color ?? "")))];
      chart = (
        <ResponsiveContainer width="100%" height={H}>
          <ScatterChart>
            <CartesianGrid stroke="var(--viz-grid)" />
            <XAxis type="number" dataKey="x" {...axis} name={xLabel || "x"} tickFormatter={(v) => fmtValue(v)} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2 } : undefined} />
            <YAxis type="number" dataKey="y" {...axis} name={yLabel || "y"} tickFormatter={(v) => fmtValue(v)} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined} />
            {pts.some((p) => p.size !== undefined) ? <ZAxis type="number" dataKey="size" range={[20, 300]} /> : null}
            <Tooltip formatter={(v) => fmtValue(v)} />
            {groups.length > 1 ? <Legend /> : null}
            {groups.map((g, i) => (
              <Scatter key={g} name={g || "rows"} data={pts.filter((p) => String(p.color ?? "") === g)} fill={COLORS[i % COLORS.length]} fillOpacity={0.6} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      );
    }
  } else if (kind === "correlation") {
    const cols = (result.columns as string[]) ?? [];
    chart = <Heat labels={cols} rowLabels={cols} matrix={(result.matrix as (number | null)[][]) ?? []} diverging aria="Correlation heatmap" />;
  } else if (kind === "missingness") {
    const cols = (result.columns as string[]) ?? [];
    const groups = result.groups as Dict[] | undefined;
    chart = groups?.length ? (
      <Heat labels={cols} rowLabels={groups.map((g) => String(g.label))} matrix={groups.map((g) => (g.missing_pct as (number | null)[]) ?? [])} diverging={false} aria="Missingness heatmap" />
    ) : (
      <ResponsiveContainer width="100%" height={H}>
        <BarChart data={cols.map((c, i) => ({ name: c, value: (result.missing_pct as number[])[i] }))}>
          <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
          <XAxis dataKey="name" {...axis} interval={0} angle={-35} textAnchor="end" height={70} />
          <YAxis {...axis} unit="%" />
          <Tooltip formatter={(v) => [`${fmtValue(v)}%`, "missing"]} />
          <Bar dataKey="value" name="missing %" fill="var(--viz-cat-2)" />
        </BarChart>
      </ResponsiveContainer>
    );
  }
  return (
    <figure className="space-y-2" aria-label={title || "Chart"}>
      {title ? <figcaption className="text-sm font-medium">{title}</figcaption> : null}
      <div ref={ref} className="min-h-[2rem]">{table ? <DataTable result={result} /> : chart}</div>
      <div className="no-print flex flex-wrap items-center gap-2">
        <Button size="sm" variant="ghost" onClick={() => setTable((t) => !t)}>
          {table ? "Show chart" : "Show data table"}
        </Button>
        {!table ? (
          <>
            <Button size="sm" variant="ghost" onClick={() => exportChart(ref.current, title || "chart", "svg").catch((e) => setError(e.message))}>
              SVG
            </Button>
            <Button size="sm" variant="ghost" onClick={() => exportChart(ref.current, title || "chart", "png").catch((e) => setError(e.message))}>
              PNG
            </Button>
          </>
        ) : null}
        {error ? <span className="text-xs text-destructive">{error}</span> : null}
      </div>
      <Notes result={result} />
    </figure>
  );
}
