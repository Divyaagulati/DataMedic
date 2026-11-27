"use client";

import { useQuery } from "@tanstack/react-query";

import { API_BASE_URL, getCsrfToken } from "@/lib/api/client";

export type Dict = Record<string, unknown>;

export type Column = {
  name: string;
  inferred_type: string;
  n_unique: number | null;
  missing_pct: number | null;
  roles: string[];
  protected: boolean;
};

export type Catalogue = {
  version_id: string;
  version_number: number;
  dataset_id: string;
  domain: string;
  rows: number;
  columns: Column[];
  privacy_mode: boolean;
  min_cell: number;
  roles: Record<string, string[]>;
};

export type Filter = {
  column: string;
  op: string;
  value?: unknown;
  values?: unknown[];
  min?: unknown;
  max?: unknown;
};

export type ChartType =
  | "bar"
  | "hbar"
  | "grouped_bar"
  | "stacked_bar"
  | "line"
  | "area"
  | "scatter"
  | "bubble"
  | "histogram"
  | "box"
  | "pie"
  | "donut"
  | "correlation_heatmap"
  | "missingness_heatmap"
  | "table";

export const CHART_LABELS: Record<ChartType, string> = {
  bar: "Bar",
  hbar: "Horizontal bar",
  grouped_bar: "Grouped bar",
  stacked_bar: "Stacked bar",
  line: "Line",
  area: "Area",
  scatter: "Scatter",
  bubble: "Bubble",
  histogram: "Histogram",
  box: "Box plot",
  pie: "Pie",
  donut: "Donut",
  correlation_heatmap: "Correlation heatmap",
  missingness_heatmap: "Missingness heatmap",
  table: "Summary table",
};

export const AGG_LABELS: Record<string, string> = {
  count: "Count",
  count_distinct: "Distinct count",
  sum: "Sum",
  mean: "Mean",
  median: "Median",
  min: "Minimum",
  max: "Maximum",
  std: "Std. deviation",
  var: "Variance",
  quantile: "Quantile",
  pct_total: "% of total",
};

export const OP_LABELS: Record<string, string> = {
  eq: "=",
  ne: "≠",
  gt: ">",
  gte: "≥",
  lt: "<",
  lte: "≤",
  between: "between",
  in: "is one of",
  not_in: "is not one of",
  is_missing: "is missing",
  not_missing: "is not missing",
  date_between: "date between",
};

export type BuilderState = {
  chartType: ChartType;
  x: string | null;
  y: string | null;
  agg: string;
  q: number;
  color: string | null;
  size: string | null;
  bins: number;
  datePart: string;
  filters: Filter[];
  sortBy: "group" | "measure";
  sortDir: "asc" | "desc";
  limit: number;
  title: string;
  xLabel: string;
  yLabel: string;
};

export const DEFAULT_STATE: BuilderState = {
  chartType: "bar",
  x: null,
  y: null,
  agg: "count",
  q: 0.5,
  color: null,
  size: null,
  bins: 20,
  datePart: "month",
  filters: [],
  sortBy: "measure",
  sortDir: "desc",
  limit: 50,
  title: "",
  xLabel: "",
  yLabel: "",
};

const NUMERIC_AGGS = new Set(["sum", "mean", "median", "min", "max", "std", "var", "quantile", "pct_total"]);

function groupOf(col: Column | undefined, s: BuilderState): Dict | null {
  if (!col) return null;
  if (col.inferred_type === "datetime") return { column: col.name, date_part: s.datePart };
  if (col.inferred_type === "numeric" && (col.n_unique ?? 0) > 50) return { column: col.name, bins: Math.min(Math.max(s.bins, 2), 60) };
  return { column: col.name };
}

/** Builder state → validated-on-the-server query specification and display options. */
export function toSpec(s: BuilderState, columns: Column[]): { query: Dict; display: Dict } | { error: string } {
  const find = (n: string | null) => columns.find((c) => c.name === n);
  const x = find(s.x);
  const y = find(s.y);
  const display: Dict = { chart_type: s.chartType };
  if (s.title) display.title = s.title;
  if (s.xLabel) display.x_label = s.xLabel;
  if (s.yLabel) display.y_label = s.yLabel;
  const filters = s.filters;
  switch (s.chartType) {
    case "scatter":
    case "bubble": {
      if (!x || !y) return { error: "Choose numerical X and Y columns." };
      const query: Dict = { kind: "scatter", columns: [x.name, y.name], filters };
      if (s.color) query.color = s.color;
      if (s.chartType === "bubble") {
        if (!s.size) return { error: "Choose a numerical size column for a bubble chart." };
        query.size = s.size;
      }
      return { query, display };
    }
    case "histogram":
      if (!x) return { error: "Choose a numerical column." };
      return { query: { kind: "histogram", columns: [x.name], bins: s.bins, filters }, display };
    case "box": {
      if (!y) return { error: "Choose the numerical column (Y)." };
      const query: Dict = { kind: "box", columns: [y.name], filters };
      const g = groupOf(x, s);
      if (g) query.group_by = [g];
      return { query, display };
    }
    case "correlation_heatmap": {
      const cols = columns.filter((c) => c.inferred_type === "numeric" && !c.protected).slice(0, 20).map((c) => c.name);
      return { query: { kind: "correlation", columns: cols, method: "spearman", filters }, display };
    }
    case "missingness_heatmap": {
      const query: Dict = { kind: "missingness", columns: columns.slice(0, 60).map((c) => c.name), filters };
      const g = groupOf(x, s);
      if (g && x?.inferred_type !== "numeric") query.group_by = [g];
      return { query, display };
    }
    default: {
      const measure: Dict = { agg: s.agg };
      if (s.agg !== "count" || s.y) {
        if (!y && s.agg !== "count") return { error: `${AGG_LABELS[s.agg]} needs a column (Y).` };
        if (y) measure.column = y.name;
      }
      if (s.agg === "quantile") measure.q = s.q;
      const groups = [groupOf(x, s), ["grouped_bar", "stacked_bar"].includes(s.chartType) ? groupOf(find(s.color), s) : null].filter(Boolean);
      if (["grouped_bar", "stacked_bar"].includes(s.chartType) && groups.length < 2) {
        return { error: "Grouped and stacked bars need an X column and a colour column." };
      }
      const timeLike = x?.inferred_type === "datetime" || (x?.inferred_type === "numeric");
      return {
        query: {
          kind: "aggregate",
          group_by: groups,
          measures: [measure],
          filters,
          sort: timeLike || s.sortBy === "group" ? { by: "group", dir: timeLike ? "asc" : s.sortDir } : { by: "measure", index: 0, dir: s.sortDir },
          limit: s.limit,
        },
        display,
      };
    }
  }
}

/** Scientific guards: warnings for misleading configurations (shown, and enforced where needed). */
export function chartWarnings(s: BuilderState, columns: Column[]): string[] {
  const x = columns.find((c) => c.name === s.x);
  const y = columns.find((c) => c.name === s.y);
  const out: string[] = [];
  if ((s.chartType === "line" || s.chartType === "area") && x && x.inferred_type !== "datetime" && x.inferred_type !== "numeric") {
    out.push("A line connects ordered values; for unordered categories use a bar chart.");
  }
  if ((s.chartType === "pie" || s.chartType === "donut") && !["count", "sum", "pct_total"].includes(s.agg)) {
    out.push("Pie and donut charts show parts of a whole: use count, sum or % of total, not averages.");
  }
  if ((s.chartType === "pie" || s.chartType === "donut") && x && x.inferred_type === "numeric") {
    out.push("A pie chart of a continuous measurement is misleading; use a histogram.");
  }
  if ((s.chartType === "pie" || s.chartType === "donut") && x && (x.n_unique ?? 0) > 8) {
    out.push("More than 8 slices are hard to compare; use a bar chart.");
  }
  if (NUMERIC_AGGS.has(s.agg) && y && y.inferred_type !== "numeric" && y.inferred_type !== "boolean") {
    out.push(`${AGG_LABELS[s.agg]} needs a numerical column; '${y.name}' is ${y.inferred_type}.`);
  }
  if ((s.agg === "sum" || s.agg === "pct_total") && y && y.roles.some((r) => r === "target") === false && /pct|percent|rate|ratio/i.test(y.name)) {
    out.push(`Summing '${y.name}' (a rate or percentage) is rarely meaningful; consider the mean.`);
  }
  return out;
}

export function isBlocking(warnings: string[]): boolean {
  return warnings.some((w) => /needs a numerical|misleading/.test(w));
}

/** Chart recommendations from column types and intent. */
export function recommend(x: Column | undefined, y: Column | undefined): { type: ChartType; agg: string; why: string }[] {
  const t = (c?: Column) => c?.inferred_type;
  const out: { type: ChartType; agg: string; why: string }[] = [];
  if (x && !y) {
    if (t(x) === "numeric") out.push({ type: "histogram", agg: "count", why: "Distribution of a numerical column" }, { type: "box", agg: "count", why: "Quartiles and outliers" });
    if (t(x) === "categorical" || t(x) === "boolean") {
      out.push({ type: "bar", agg: "count", why: "Counts per category" });
      if ((x.n_unique ?? 99) <= 6) out.push({ type: "donut", agg: "count", why: "Share of a few categories" });
    }
    if (t(x) === "datetime") out.push({ type: "line", agg: "count", why: "Rows over time" });
  }
  if (x && y) {
    if (t(x) === "numeric" && t(y) === "numeric") out.push({ type: "scatter", agg: "count", why: "Relationship between two measurements" });
    if ((t(x) === "categorical" || t(x) === "boolean") && t(y) === "numeric") {
      out.push({ type: "bar", agg: "mean", why: "Compare averages across groups" }, { type: "box", agg: "count", why: "Compare distributions across groups" });
    }
    if (t(x) === "datetime" && t(y) === "numeric") out.push({ type: "line", agg: "mean", why: "Trend over time" }, { type: "area", agg: "sum", why: "Cumulative volume over time" });
  }
  if (!x && !y) out.push({ type: "correlation_heatmap", agg: "count", why: "All numerical relationships" }, { type: "missingness_heatmap", agg: "count", why: "Where values are missing" });
  return out;
}

// ---- Data shaping ---------------------------------------------------------------------------------

export type Row = { name: string; value: number | null; n: number | null; suppressed: boolean; groups: string[] };

export function aggregateRows(result: Dict): Row[] {
  return ((result.rows as Dict[]) ?? []).map((r) => ({
    name: (r.groups as string[]).join(" / "),
    groups: r.groups as string[],
    value: (r.values as (number | null)[])[0],
    n: r.n as number | null,
    suppressed: Boolean(r.suppressed),
  }));
}

/** Two groupings → one row per first group with a key per second group (for grouped/stacked bars). */
export function wideRows(result: Dict): { data: Dict[]; series: string[] } {
  const series: string[] = [];
  const byX = new Map<string, Dict>();
  for (const r of (result.rows as Dict[]) ?? []) {
    const [gx, gs] = r.groups as string[];
    if (!series.includes(gs)) series.push(gs);
    const row = byX.get(gx) ?? { name: gx };
    row[gs] = (r.values as (number | null)[])[0];
    byX.set(gx, row);
  }
  return { data: [...byX.values()], series };
}

export function fmtValue(v: unknown, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : Number(v.toFixed(digits)).toString();
  }
  return String(v);
}

// ---- API -----------------------------------------------------------------------------------------

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (init?.method && init.method !== "GET") headers["X-CSRF-Token"] = getCsrfToken() ?? "";
  const res = await fetch(`${API_BASE_URL}${path}`, { credentials: "include", ...init, headers });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const errors = (body.errors as { message: string }[] | undefined)?.map((e) => e.message).join(" ");
    throw new Error(errors || body.detail || body.title || `Request failed (${res.status}).`);
  }
  return body as T;
}

export const analyticsApi = {
  columns: (vid: string) => call<Catalogue>(`/api/v1/versions/${vid}/analytics/columns`),
  query: (vid: string, spec: Dict) =>
    call<{ result: Dict; spec: Dict }>(`/api/v1/versions/${vid}/analytics/query`, { method: "POST", body: JSON.stringify({ spec }) }),
  compare: (a: string, b: string) => call<Dict>(`/api/v1/versions/${a}/compare/${b}`),
  insights: (vid: string) => call<Dict>(`/api/v1/versions/${vid}/insights`),
  startInsights: (vid: string) => call<Dict>(`/api/v1/versions/${vid}/insights`, { method: "POST" }),
  dashboards: (dsId: string) => call<{ items: Dict[] }>(`/api/v1/datasets/${dsId}/dashboards`),
  dashboard: (id: string) => call<Dict>(`/api/v1/dashboards/${id}`),
  createDashboard: (dsId: string, body: Dict) => call<Dict>(`/api/v1/datasets/${dsId}/dashboards`, { method: "POST", body: JSON.stringify(body) }),
  patchDashboard: (id: string, body: Dict) => call<Dict>(`/api/v1/dashboards/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteDashboard: (id: string) => call<void>(`/api/v1/dashboards/${id}`, { method: "DELETE" }),
  duplicateDashboard: (id: string, body: Dict) => call<Dict>(`/api/v1/dashboards/${id}/duplicate`, { method: "POST", body: JSON.stringify(body) }),
  compatibility: (id: string, vid: string) => call<Dict>(`/api/v1/dashboards/${id}/compatibility?version_id=${vid}`),
  switchVersion: (id: string, vid: string) =>
    call<Dict>(`/api/v1/dashboards/${id}/switch-version`, { method: "POST", body: JSON.stringify({ version_id: vid, confirm: true }) }),
  addWidget: (id: string, body: Dict) => call<Dict>(`/api/v1/dashboards/${id}/widgets`, { method: "POST", body: JSON.stringify(body) }),
  patchWidget: (id: string, wid: string, body: Dict) =>
    call<Dict>(`/api/v1/dashboards/${id}/widgets/${wid}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteWidget: (id: string, wid: string) => call<void>(`/api/v1/dashboards/${id}/widgets/${wid}`, { method: "DELETE" }),
  duplicateWidget: (id: string, wid: string) => call<Dict>(`/api/v1/dashboards/${id}/widgets/${wid}/duplicate`, { method: "POST" }),
  reorder: (id: string, ids: string[]) => call<Dict>(`/api/v1/dashboards/${id}/reorder`, { method: "POST", body: JSON.stringify({ widget_ids: ids }) }),
};

export function useCatalogue(versionId: string | undefined) {
  return useQuery({ queryKey: ["analytics", versionId, "columns"], queryFn: () => analyticsApi.columns(versionId!), enabled: Boolean(versionId) });
}

export function useAnalyticsQuery(versionId: string | undefined, spec: Dict | null) {
  return useQuery({
    queryKey: ["analytics", versionId, "query", JSON.stringify(spec)],
    queryFn: () => analyticsApi.query(versionId!, spec!),
    enabled: Boolean(versionId && spec),
    retry: false,
    staleTime: 60_000,
  });
}

/** Merge dashboard-level filters into a widget query. */
export function withFilters(query: Dict, filters: Filter[]): Dict {
  return { ...query, filters: [...((query.filters as Filter[]) ?? []), ...filters] };
}

/** Save an SVG chart (as SVG or PNG). Only the rendered aggregate chart is exported. */
export async function exportChart(container: HTMLElement | null, name: string, format: "svg" | "png"): Promise<void> {
  const svg = container?.querySelector("svg.recharts-surface, svg[data-chart]") as SVGSVGElement | null;
  if (!svg) throw new Error("No chart to export.");
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const box = svg.getBoundingClientRect();
  const w = Math.max(1, Math.round(box.width || Number(svg.getAttribute("width")) || 800));
  const h = Math.max(1, Math.round(box.height || Number(svg.getAttribute("height")) || 400));
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  const text = new XMLSerializer().serializeToString(clone);
  const safe = name.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 60) || "chart";
  const save = (url: string, ext: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safe}.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const svgUrl = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
  if (format === "svg") {
    save(svgUrl, "svg");
    URL.revokeObjectURL(svgUrl);
    return;
  }
  await new Promise<void>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = w * 2;
      canvas.height = h * 2;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Canvas unavailable."));
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) return reject(new Error("PNG export failed."));
        const url = URL.createObjectURL(blob);
        save(url, "png");
        URL.revokeObjectURL(url);
        resolve();
      }, "image/png");
    };
    img.onerror = () => reject(new Error("PNG export failed."));
    img.src = svgUrl;
  });
  URL.revokeObjectURL(svgUrl);
}

/** Aggregated result CSV through the API (formula-neutralised there). */
export async function exportQueryCsv(versionId: string, spec: Dict): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/versions/${versionId}/analytics/export`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": getCsrfToken() ?? "" },
    body: JSON.stringify({ spec }),
  });
  if (!res.ok) throw new Error(`Export failed (${res.status}).`);
  const match = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "");
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = match?.[1] ?? "query.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
