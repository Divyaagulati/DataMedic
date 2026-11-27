"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { ChartBuilder, FilterEditor, stateFromSpec } from "@/components/analytics/builder";
import { ChartView } from "@/components/analytics/chart-view";
import { PivotTable } from "@/components/analytics/panels";
import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type ChartType, analyticsApi, type Dict, type Filter, useAnalyticsQuery, useCatalogue, withFilters } from "@/lib/analytics";
import { problemMessage } from "@/lib/api/problem";
import { downloadFile } from "@/lib/mvp";
import { useDataset, useVersionList } from "@/lib/queries";

const SPAN: Record<string, string> = { small: "lg:col-span-1", medium: "lg:col-span-2", large: "lg:col-span-3" };

function Widget({ w, versionId, filters, onEdit, onAction, first, last }: {
  w: Dict; versionId: string; filters: Filter[]; onEdit: () => void; onAction: (a: string) => void; first: boolean; last: boolean;
}) {
  const spec = w.spec as Dict;
  const query = withFilters(spec.query as Dict, filters);
  const res = useAnalyticsQuery(versionId, query);
  const display = (spec.display ?? {}) as Dict;
  return (
    <Card className={`${SPAN[String(w.size)] ?? SPAN.medium} break-inside-avoid`} data-testid="widget">
      <CardHeader className="flex flex-row items-start justify-between gap-2 pb-1">
        <CardTitle className="text-sm">{String(w.title)}</CardTitle>
        <div className="no-print flex flex-wrap gap-0.5 text-xs">
          <Button size="sm" variant="ghost" className="h-6 px-1.5" aria-label="Move up" disabled={first} onClick={() => onAction("up")}>↑</Button>
          <Button size="sm" variant="ghost" className="h-6 px-1.5" aria-label="Move down" disabled={last} onClick={() => onAction("down")}>↓</Button>
          <select aria-label="Widget size" className="h-6 rounded border bg-background text-xs" value={String(w.size)} onChange={(e) => onAction(`size:${e.target.value}`)}>
            <option value="small">S</option><option value="medium">M</option><option value="large">L</option>
          </select>
          {w.kind === "chart" ? <Button size="sm" variant="ghost" className="h-6 px-1.5" onClick={onEdit}>Edit</Button> : null}
          <Button size="sm" variant="ghost" className="h-6 px-1.5" onClick={() => onAction("duplicate")}>Duplicate</Button>
          <Button size="sm" variant="ghost" className="h-6 px-1.5" onClick={() => onAction("remove")}>Remove</Button>
        </div>
      </CardHeader>
      <CardContent>
        {res.error ? <p role="alert" className="text-sm text-destructive">{(res.error as Error).message}</p> : !res.data ? <p className="text-sm text-muted-foreground">Calculating…</p> :
          w.kind === "kpi" ? <ChartView result={res.data.result} chartType="kpi" /> :
          w.kind === "pivot" ? <PivotTable result={res.data.result} /> :
          <ChartView result={res.data.result} chartType={display.chart_type as ChartType} xLabel={display.x_label as string} yLabel={display.y_label as string} />}
      </CardContent>
    </Card>
  );
}

export function DashboardView() {
  const { datasetId, dashboardId } = useParams<{ datasetId: string; dashboardId: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const dataset = useDataset(datasetId);
  const versions = useVersionList(datasetId);
  const q = useQuery({ queryKey: ["analytics", "dashboard", dashboardId], queryFn: () => analyticsApi.dashboard(dashboardId) });
  const d = q.data as Dict | undefined;
  const versionId = d?.version_id as string | undefined;
  const catalogue = useCatalogue(versionId);
  const [editing, setEditing] = useState<Dict | null>(null);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState("");
  const [compat, setCompat] = useState<Dict | null>(null);
  const [filtersDraft, setFiltersDraft] = useState<Filter[] | null>(null);
  const reload = () => qc.invalidateQueries({ queryKey: ["analytics", "dashboard", dashboardId] });

  if (dataset.error || q.error) return <FormError message={dataset.error ? problemMessage(dataset.error) : (q.error as Error).message} />;
  if (!dataset.data || !d) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const widgets = (d.widgets as Dict[]) ?? [];
  const filters = filtersDraft ?? ((d.filters as Filter[]) ?? []);
  const vnum = versions.data?.find((v) => v.id === versionId)?.version_number;
  const head = dataset.data.head_version?.id;

  async function act(w: Dict, action: string) {
    setError(null);
    try {
      const i = widgets.indexOf(w);
      if (action === "remove") await analyticsApi.deleteWidget(dashboardId, String(w.id));
      else if (action === "duplicate") await analyticsApi.duplicateWidget(dashboardId, String(w.id));
      else if (action.startsWith("size:")) await analyticsApi.patchWidget(dashboardId, String(w.id), { size: action.slice(5) });
      else {
        const ids = widgets.map((x) => String(x.id));
        const j = action === "up" ? i - 1 : i + 1;
        [ids[i], ids[j]] = [ids[j], ids[i]];
        await analyticsApi.reorder(dashboardId, ids);
      }
      await reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="space-y-6">
      <div className="no-print"><DatasetNav datasetId={datasetId} name={dataset.data.name} status={dataset.data.status} active="analytics" /></div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Link href={`/datasets/${datasetId}/analytics`} className="no-print text-sm text-muted-foreground hover:text-foreground">← Analytics Studio</Link>
          <h2 className="text-2xl font-semibold">{String(d.name)}</h2>
          {d.description ? <p className="text-sm text-muted-foreground">{String(d.description)}</p> : null}
          <p className="text-xs text-muted-foreground">Bound to version v{String(vnum ?? "?")}{versionId !== head ? " · a newer version exists" : ""}</p>
        </div>
        <div className="no-print flex flex-wrap gap-2">
          <Button size="sm" onClick={() => setAdding((a) => !a)}>{adding ? "Close builder" : "Add chart"}</Button>
          <Button size="sm" variant="outline" onClick={() => window.print()}>Print / PDF</Button>
          <Button size="sm" variant="outline" onClick={() => downloadFile(`/api/v1/dashboards/${dashboardId}/export`, "dashboard.json").catch((e) => setError(e.message))}>Export JSON</Button>
          <Button size="sm" variant="outline" onClick={async () => { const r = await analyticsApi.duplicateDashboard(dashboardId, {}); router.push(`/datasets/${datasetId}/analytics/dashboards/${String((r.dashboard as Dict).id)}`); }}>Duplicate</Button>
          <Button size="sm" variant="outline" onClick={async () => { if (window.confirm("Delete this dashboard?")) { await analyticsApi.deleteDashboard(dashboardId); router.push(`/datasets/${datasetId}/analytics`); } }}>Delete</Button>
        </div>
      </div>
      <FormError message={error} />

      <Card className="no-print">
        <CardHeader className="pb-1"><CardTitle className="text-sm">Dashboard filters (applied to every widget)</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {catalogue.data ? <FilterEditor filters={filters} onChange={setFiltersDraft} columns={catalogue.data.columns} /> : null}
          {filtersDraft ? (
            <Button size="sm" onClick={async () => {
              try { await analyticsApi.patchDashboard(dashboardId, { filters: filtersDraft }); setFiltersDraft(null); await reload(); } catch (e) { setError((e as Error).message); }
            }}>Save filters</Button>
          ) : null}
        </CardContent>
      </Card>

      <Card className="no-print">
        <CardHeader className="pb-1"><CardTitle className="text-sm">Version</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <select aria-label="Target version" className="h-8 rounded-md border bg-background px-2 text-sm" value={target} onChange={(e) => { setTarget(e.target.value); setCompat(null); }}>
              <option value="">Choose another version…</option>
              {(versions.data ?? []).filter((v) => v.id !== versionId).map((v) => <option key={v.id} value={v.id}>v{v.version_number}</option>)}
            </select>
            <Button size="sm" variant="outline" disabled={!target} onClick={async () => setCompat(await analyticsApi.compatibility(dashboardId, target))}>Check compatibility</Button>
          </div>
          {compat ? (
            <div className="space-y-1">
              {((compat.widgets as Dict[]) ?? []).map((w) => (
                <p key={String(w.widget_id)} className="text-xs">
                  <Badge variant={w.compatible ? "secondary" : "destructive"} className="mr-1">{w.compatible ? "compatible" : "incompatible"}</Badge>
                  {String(w.title)}{(w.problems as string[]).length ? `: ${(w.problems as string[]).join(" ")}` : ""}
                </p>
              ))}
              {((compat.filter_problems as string[]) ?? []).map((p) => <p key={p} className="text-xs text-destructive">Filter: {p}</p>)}
              <div className="flex gap-2">
                <Button size="sm" disabled={!compat.compatible} onClick={async () => {
                  if (!window.confirm(`Switch this dashboard to v${String(compat.version_number)}? Its charts will be recomputed on that version.`)) return;
                  try { await analyticsApi.switchVersion(dashboardId, target); setCompat(null); await reload(); } catch (e) { setError((e as Error).message); }
                }}>Switch to this version</Button>
                <Button size="sm" variant="outline" onClick={async () => {
                  const r = await analyticsApi.duplicateDashboard(dashboardId, { version_id: target });
                  const skipped = r.skipped_widgets as string[];
                  if (skipped.length) window.alert(`Skipped incompatible widgets: ${skipped.join(", ")}`);
                  router.push(`/datasets/${datasetId}/analytics/dashboards/${String((r.dashboard as Dict).id)}`);
                }}>Duplicate onto it</Button>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {adding && catalogue.data ? (
        <ChartBuilder catalogue={catalogue.data} extraFilters={filters} onSave={async (widget) => { await analyticsApi.addWidget(dashboardId, widget); setAdding(false); await reload(); }} />
      ) : null}
      {editing && catalogue.data ? (
        <Card>
          <CardHeader className="pb-1"><CardTitle className="text-sm">Edit “{String(editing.title)}”</CardTitle></CardHeader>
          <CardContent>
            <ChartBuilder catalogue={catalogue.data} initial={stateFromSpec(editing.spec as Dict)} extraFilters={filters}
              onSave={async (widget) => { await analyticsApi.patchWidget(dashboardId, String(editing.id), { title: widget.title, spec: widget.spec }); setEditing(null); await reload(); }} />
            <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
          </CardContent>
        </Card>
      ) : null}

      {!widgets.length ? (
        <p className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">This dashboard is empty. Add a chart, or add KPI cards and pivot tables from the Analytics Studio.</p>
      ) : versionId ? (
        <div className="grid gap-4 lg:grid-cols-3">
          {widgets.map((w, i) => (
            <Widget key={String(w.id)} w={w} versionId={versionId} filters={(d.filters as Filter[]) ?? []} first={i === 0} last={i === widgets.length - 1}
              onEdit={() => setEditing(w)} onAction={(a) => act(w, a)} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
