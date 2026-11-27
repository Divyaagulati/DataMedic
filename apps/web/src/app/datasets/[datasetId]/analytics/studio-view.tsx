"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";

import { ChartBuilder } from "@/components/analytics/builder";
import { ComparePanel, DashboardsPanel, InsightsPanel, KpiBuilder, PivotBuilder } from "@/components/analytics/panels";
import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { VersionPicker } from "@/components/mvp/version-picker";
import { Badge } from "@/components/ui/badge";
import { problemMessage } from "@/lib/api/problem";
import { analyticsApi, type Dict, useCatalogue } from "@/lib/analytics";
import { useDataset, useVersionList } from "@/lib/queries";

const TABS = [
  { key: "build", label: "Chart builder" },
  { key: "kpi", label: "KPI & pivot" },
  { key: "insights", label: "Insights" },
  { key: "dashboards", label: "Dashboards" },
  { key: "compare", label: "Compare versions" },
] as const;

const TYPE_ICON: Record<string, string> = { numeric: "#", categorical: "Aa", boolean: "✓", datetime: "📅", text: "T", identifier: "id" };

export function StudioView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const dataset = useDataset(datasetId);
  const versions = useVersionList(datasetId);
  const [picked, setPicked] = useState<string>();
  const versionId = picked ?? dataset.data?.head_version?.id;
  const catalogue = useCatalogue(versionId);
  const [tab, setTab] = useState<(typeof TABS)[number]["key"]>("build");
  const [search, setSearch] = useState("");
  const qc = useQueryClient();
  const dashboards = useQuery({ queryKey: ["analytics", datasetId, "dashboards"], queryFn: () => analyticsApi.dashboards(datasetId) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["analytics", datasetId, "dashboards"] });
  const forVersion = ((dashboards.data?.items ?? []) as Dict[]).filter((d) => d.version_id === versionId);
  async function addWidget(id: string, widget: Dict) {
    await analyticsApi.addWidget(id, widget);
    await refresh();
  }

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const cat = catalogue.data;
  const cols = (cat?.columns ?? []).filter((c) => c.name.toLowerCase().includes(search.toLowerCase()));
  const counts = (cat?.columns ?? []).reduce<Record<string, number>>((a, c) => ({ ...a, [c.inferred_type]: (a[c.inferred_type] ?? 0) + 1 }), {});

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="analytics" />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <VersionPicker datasetId={datasetId} value={versionId} onChange={setPicked} />
        {cat?.privacy_mode ? (
          <Badge variant="outline" title={`Sensitive columns can only be counted; groups under ${cat.min_cell} rows are suppressed; scatter plots are binned.`}>
            Privacy mode
          </Badge>
        ) : null}
      </div>
      {ds.status !== "ready" ? <p className="text-sm text-muted-foreground">The studio opens once the dataset is ready.</p> : null}
      {catalogue.error ? <FormError message={(catalogue.error as Error).message} /> : null}
      {cat ? (
        <div className="grid gap-6 lg:grid-cols-[15rem_1fr]">
          <aside className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border bg-card p-3"><p className="text-xs text-muted-foreground">Rows</p><p className="text-lg font-semibold tabular-nums">{cat.rows.toLocaleString()}</p></div>
              <div className="rounded-lg border bg-card p-3"><p className="text-xs text-muted-foreground">Columns</p><p className="text-lg font-semibold tabular-nums">{cat.columns.length}</p></div>
            </div>
            <p className="text-xs text-muted-foreground">{Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ")}</p>
            <div className="rounded-lg border bg-card">
              <input aria-label="Search columns" className="w-full rounded-t-lg border-b bg-transparent px-3 py-2 text-sm" placeholder="Search columns…" value={search} onChange={(e) => setSearch(e.target.value)} />
              <ul className="max-h-[28rem] overflow-y-auto text-sm" aria-label="Columns">
                {cols.map((c) => (
                  <li key={c.name} className="flex items-center gap-2 border-b px-3 py-1.5 last:border-0" title={`${c.inferred_type}${c.roles.length ? ` · roles: ${c.roles.join(", ")}` : ""}${c.protected ? " · sensitive" : ""}`}>
                    <span className="w-6 text-center text-xs text-muted-foreground">{TYPE_ICON[c.inferred_type] ?? "?"}</span>
                    <span className="flex-1 truncate">{c.name}</span>
                    {c.protected ? <Badge variant="outline" className="px-1 text-[10px]">sensitive</Badge> : null}
                    {c.missing_pct ? <span className="text-[10px] text-muted-foreground">{c.missing_pct.toFixed(0)}%∅</span> : null}
                  </li>
                ))}
              </ul>
            </div>
          </aside>
          <section className="space-y-4">
            <nav aria-label="Studio sections" className="no-print flex gap-1 overflow-x-auto rounded-lg border bg-card p-1">
              {TABS.map((t) => (
                <button key={t.key} onClick={() => setTab(t.key)} aria-current={tab === t.key ? "page" : undefined}
                  className={"whitespace-nowrap rounded-md px-3 py-1.5 text-sm " + (tab === t.key ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted")}>
                  {t.label}
                </button>
              ))}
            </nav>
            {tab === "build" ? <ChartBuilder key={cat.version_id} catalogue={cat} dashboards={forVersion} onAdd={addWidget} /> : null}
            {tab === "kpi" ? (
              <div className="space-y-4">
                <KpiBuilder catalogue={cat} dashboards={forVersion} onAdd={addWidget} />
                <PivotBuilder catalogue={cat} dashboards={forVersion} onAdd={addWidget} />
              </div>
            ) : null}
            {tab === "insights" ? <InsightsPanel versionId={cat.version_id} catalogue={cat} /> : null}
            {tab === "dashboards" ? <DashboardsPanel datasetId={datasetId} versionId={cat.version_id} dashboards={(dashboards.data?.items ?? []) as Dict[]} onChange={refresh} /> : null}
            {tab === "compare" ? <ComparePanel versions={versions.data ?? []} defaultB={cat.version_id} /> : null}
          </section>
        </div>
      ) : catalogue.isPending && versionId ? <p className="text-sm text-muted-foreground">Loading columns…</p> : null}
    </div>
  );
}
