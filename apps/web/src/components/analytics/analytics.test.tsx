import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { analyticsApi, type Catalogue, chartWarnings, DEFAULT_STATE, exportQueryCsv, recommend, toSpec } from "@/lib/analytics";

import { AnalysisResult } from "./analysis-result";
import { ChartBuilder, stateFromSpec } from "./builder";
import { ChartView } from "./chart-view";
import { KpiBuilder } from "./panels";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const cat: Catalogue = {
  version_id: "v1",
  version_number: 0,
  dataset_id: "d1",
  domain: "business",
  rows: 300,
  privacy_mode: false,
  min_cell: 10,
  roles: {},
  columns: [
    { name: "region", inferred_type: "categorical", n_unique: 3, missing_pct: 0, roles: [], protected: false },
    { name: "amount", inferred_type: "numeric", n_unique: 290, missing_pct: 1, roles: [], protected: false },
    { name: "units", inferred_type: "numeric", n_unique: 7, missing_pct: 0, roles: [], protected: false },
    { name: "date", inferred_type: "datetime", n_unique: 300, missing_pct: 0, roles: [], protected: false },
    { name: "code", inferred_type: "categorical", n_unique: 3, missing_pct: 0, roles: ["sensitive"], protected: true },
  ],
};

const aggResult = {
  kind: "aggregate",
  group_columns: ["region"],
  measure_labels: ["mean(amount)"],
  rows: [
    { groups: ["North"], n: 100, values: [120.5] },
    { groups: ["South"], n: 90, values: [80.25] },
  ],
  totals: { n: 190, values: [101.2] },
  rows_total: 300,
  rows_filtered: 190,
};

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

function mockFetch(handler: (url: string, init?: RequestInit) => unknown) {
  return vi.spyOn(globalThis, "fetch").mockImplementation((async (url: string, init?: RequestInit) =>
    new Response(JSON.stringify(handler(String(url), init)), { headers: { "content-type": "application/json" } })) as never);
}

describe("query specification mapping and guards", () => {
  it("maps builder state to structured specs", () => {
    const bar = toSpec({ ...DEFAULT_STATE, chartType: "bar", x: "region", y: "amount", agg: "mean" }, cat.columns);
    expect("query" in bar && bar.query).toMatchObject({ kind: "aggregate", group_by: [{ column: "region" }], measures: [{ agg: "mean", column: "amount" }] });
    const line = toSpec({ ...DEFAULT_STATE, chartType: "line", x: "date", agg: "count", datePart: "month" }, cat.columns);
    expect("query" in line && line.query.group_by).toEqual([{ column: "date", date_part: "month" }]);
    const binned = toSpec({ ...DEFAULT_STATE, chartType: "bar", x: "amount", agg: "count", bins: 8 }, cat.columns);
    expect("query" in binned && binned.query.group_by).toEqual([{ column: "amount", bins: 8 }]);
    expect(toSpec({ ...DEFAULT_STATE, chartType: "grouped_bar", x: "region" }, cat.columns)).toHaveProperty("error");
    const sc = toSpec({ ...DEFAULT_STATE, chartType: "scatter", x: "amount", y: "units" }, cat.columns);
    expect("query" in sc && sc.query).toMatchObject({ kind: "scatter", columns: ["amount", "units"] });
    const corr = toSpec({ ...DEFAULT_STATE, chartType: "correlation_heatmap" }, cat.columns);
    expect("query" in corr && corr.query.columns).toEqual(["amount", "units"]);
  });

  it("warns about scientifically misleading charts", () => {
    expect(chartWarnings({ ...DEFAULT_STATE, chartType: "pie", x: "region", agg: "mean", y: "amount" }, cat.columns).join(" ")).toMatch(/parts of a whole/);
    expect(chartWarnings({ ...DEFAULT_STATE, chartType: "line", x: "region" }, cat.columns).join(" ")).toMatch(/bar chart/);
    expect(chartWarnings({ ...DEFAULT_STATE, chartType: "pie", x: "amount" }, cat.columns).join(" ")).toMatch(/misleading/);
    expect(recommend(cat.columns[1], undefined).map((r) => r.type)).toContain("histogram");
    expect(recommend(cat.columns[0], cat.columns[1]).map((r) => r.type)).toContain("bar");
    expect(recommend(cat.columns[1], cat.columns[2]).map((r) => r.type)).toEqual(["scatter"]);
  });
});

describe("saved widget configuration", () => {
  it("round-trips aggregation, columns, filters, title, labels and sorting", () => {
    const state = {
      ...DEFAULT_STATE,
      chartType: "hbar" as const,
      x: "region",
      y: "amount",
      agg: "sum",
      filters: [{ column: "units", op: "gte", value: 2 }, { column: "region", op: "in", values: ["North", "South"] }],
      sortBy: "measure" as const,
      sortDir: "asc" as const,
      limit: 20,
      title: "Total amount by region",
      xLabel: "Region",
      yLabel: "Amount (sum)",
    };
    const first = toSpec(state, cat.columns);
    if (!("query" in first)) throw new Error("spec expected");
    expect(first.query).toMatchObject({ measures: [{ agg: "sum", column: "amount" }], sort: { by: "measure", index: 0, dir: "asc" }, limit: 20 });
    expect(first.display).toEqual({ chart_type: "hbar", title: "Total amount by region", x_label: "Region", y_label: "Amount (sum)" });
    const reopened = stateFromSpec({ query: first.query, display: first.display });
    expect(toSpec(reopened, cat.columns)).toEqual(first);
  });
});

describe("chart builder", () => {
  it("builds a grouped chart, applies filters and adds it to a dashboard", async () => {
    const fetchMock = mockFetch(() => ({ result: aggResult, spec: {} }));
    const onAdd = vi.fn().mockResolvedValue(undefined);
    wrap(<ChartBuilder catalogue={cat} dashboards={[{ id: "dash1", name: "Sales" }]} onAdd={onAdd} />);
    fireEvent.change(screen.getByLabelText("X axis"), { target: { value: "region" } });
    fireEvent.change(screen.getByLabelText("Aggregation"), { target: { value: "mean" } });
    fireEvent.change(screen.getByLabelText("Value (Y)"), { target: { value: "amount" } });
    await waitFor(() => expect(screen.getByText(/190 of 300 rows after filters/)).toBeTruthy());
    // Sensitive columns are not offered as dimensions.
    expect(screen.getByLabelText("X axis").innerHTML).not.toContain("code");
    fireEvent.click(screen.getByText("Show data table"));
    expect(screen.getByText("120.5")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Filter column"), { target: { value: "units" } });
    fireEvent.change(screen.getByLabelText("Filter operator"), { target: { value: "gte" } });
    fireEvent.change(screen.getByLabelText("Filter value"), { target: { value: "3" } });
    fireEvent.click(screen.getByText("Add filter"));
    await waitFor(() => {
      const bodies = fetchMock.mock.calls.map((c) => String((c[1] as RequestInit | undefined)?.body ?? ""));
      expect(bodies.some((b) => b.includes('"op":"gte"') && b.includes('"value":3'))).toBe(true);
    });
    fireEvent.change(screen.getByLabelText("Dashboard"), { target: { value: "dash1" } });
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(onAdd).toHaveBeenCalled());
    const [id, widget] = onAdd.mock.calls[0];
    expect(id).toBe("dash1");
    expect(widget.spec.query).toMatchObject({ kind: "aggregate", measures: [{ agg: "mean", column: "amount" }] });
    expect(widget.spec.display.chart_type).toBe("bar");
  });

  it("shows validation errors and empty states honestly", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((async () =>
      new Response(JSON.stringify({ errors: [{ message: "Too many groups." }] }), { status: 422 })) as never);
    wrap(<ChartBuilder catalogue={cat} />);
    fireEvent.change(screen.getByLabelText("Aggregation"), { target: { value: "mean" } });
    expect(screen.getByText("Mean needs a column (Y).")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Aggregation"), { target: { value: "count" } });
    fireEvent.change(screen.getByLabelText("X axis"), { target: { value: "region" } });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/Too many groups/));
    cleanup();
    render(<ChartView result={{ suppressed: true, note: "The filtered cohort has fewer than 10 rows; results are suppressed." }} chartType="bar" />);
    expect(screen.getByText(/results are suppressed/)).toBeTruthy();
    cleanup();
    render(<ChartView result={{ ...aggResult, rows: [] }} chartType="bar" />);
    expect(screen.getByText("No data for this selection.")).toBeTruthy();
  });
});

describe("KPI cards and exports", () => {
  it("computes a KPI through the engine and adds it", async () => {
    mockFetch(() => ({ result: { kind: "aggregate", measure_labels: ["count(rows)"], rows: [], totals: { n: 300, values: [300] } } }));
    const onAdd = vi.fn().mockResolvedValue(undefined);
    wrap(<KpiBuilder catalogue={cat} dashboards={[{ id: "dash1", name: "Sales" }]} onAdd={onAdd} />);
    await waitFor(() => expect(screen.getByTestId("kpi-value").textContent).toBe("300"));
    fireEvent.change(screen.getByLabelText("KPI aggregation"), { target: { value: "mean" } });
    expect(screen.getByText(/needs a numerical column/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("KPI column"), { target: { value: "amount" } });
    fireEvent.change(screen.getByLabelText("KPI dashboard"), { target: { value: "dash1" } });
    fireEvent.click(screen.getByText("Add KPI"));
    await waitFor(() => expect(onAdd).toHaveBeenCalledWith("dash1", expect.objectContaining({ kind: "kpi", spec: { query: { kind: "aggregate", measures: [{ agg: "mean", column: "amount" }] }, display: {} } })));
  });

  it("exports aggregated CSV through the API and switches versions only with confirmation", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("a,b\n", { headers: { "content-disposition": 'attachment; filename="x.csv"' } }));
    Object.assign(URL, { createObjectURL: vi.fn(() => "blob:x"), revokeObjectURL: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    await exportQueryCsv("v1", { kind: "aggregate", measures: [{ agg: "count" }] });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/versions\/v1\/analytics\/export$/);
    expect(init.method).toBe("POST");
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ id: "d" })));
    await analyticsApi.switchVersion("d", "v2");
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body))).toEqual({ version_id: "v2", confirm: true });
  });
});

describe("Copilot-generated charts", () => {
  it("shows the interpreted query and adds the chart to a dashboard", async () => {
    const fetchMock = mockFetch((url) => (url.endsWith("/dashboards") ? { items: [{ id: "dash1", name: "Sales", version_id: "v1" }] } : { id: "w1" }));
    wrap(
      <AnalysisResult
        datasetId="d1"
        versionId="v1"
        onAsk={vi.fn()}
        analysis={{ intent: "top_group", interpretation: "mean of amount per region", spec: { kind: "aggregate", group_by: [{ column: "region" }], measures: [{ agg: "mean", column: "amount" }] }, result: aggResult, chart: { type: "bar" } }}
      />,
    );
    expect(screen.getByTestId("interpreted-spec").textContent).toContain('"agg": "mean"');
    await waitFor(() => expect(screen.getByLabelText("Copilot dashboard").innerHTML).toContain("Sales"));
    fireEvent.change(screen.getByLabelText("Copilot dashboard"), { target: { value: "dash1" } });
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(/Added/));
    const post = fetchMock.mock.calls.find((c) => String(c[0]).endsWith("/dashboards/dash1/widgets"));
    expect(JSON.parse(String((post![1] as RequestInit).body)).spec.display.chart_type).toBe("bar");
  });

  it("offers clarification options for ambiguous questions", () => {
    mockFetch(() => ({ items: [] }));
    const onAsk = vi.fn();
    wrap(<AnalysisResult datasetId="d1" versionId="v1" onAsk={onAsk} analysis={{ status: "ambiguous", options: ["amount_net", "amount_gross"] }} />);
    fireEvent.click(screen.getByText("amount_net"));
    expect(onAsk).toHaveBeenCalledWith("Use the column amount_net");
  });
});
