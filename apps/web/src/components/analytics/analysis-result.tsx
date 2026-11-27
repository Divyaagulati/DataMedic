"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { ChartView } from "@/components/analytics/chart-view";
import { Button } from "@/components/ui/button";
import { analyticsApi, type ChartType, type Dict } from "@/lib/analytics";

/** Chart type for a Copilot analysis (deterministic mapping from the interpreted intent). */
export function chartFor(analysis: Dict): ChartType | "kpi" | null {
  const t = (analysis.chart as Dict | undefined)?.type;
  switch (analysis.intent) {
    case "most_missing":
      return "missingness_heatmap";
    case "strongest_correlations":
      return "correlation_heatmap";
    case "relationship":
      return "scatter";
    case "histogram":
      return "histogram";
    case "kpi":
      return "kpi";
    case "trend":
      return "line";
    default:
      return t === "line" ? "line" : t ? "bar" : null;
  }
}

export function AnalysisResult({ analysis, datasetId, versionId, onAsk }: { analysis: Dict; datasetId: string; versionId: string; onAsk: (q: string) => void }) {
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const dashboards = useQuery({ queryKey: ["analytics", datasetId, "dashboards"], queryFn: () => analyticsApi.dashboards(datasetId) });
  const options = (analysis.options as string[] | undefined) ?? [];
  if (analysis.status) {
    return options.length ? (
      <div className="flex flex-wrap gap-2 pt-2" data-testid="clarify">
        {options.map((o) => (
          <Button key={o} size="sm" variant="outline" onClick={() => onAsk(`Use the column ${o}`)}>{o}</Button>
        ))}
      </div>
    ) : null;
  }
  if (analysis.intent === "compare_versions") {
    return <Link className="text-xs text-primary hover:underline" href={`/datasets/${datasetId}/analytics`}>Open the version comparison in Analytics Studio ↗</Link>;
  }
  const type = chartFor(analysis);
  const result = analysis.result as Dict | undefined;
  const spec = analysis.spec as Dict | undefined;
  const mine = ((dashboards.data?.items ?? []) as Dict[]).filter((d) => d.version_id === versionId);
  const kind = type === "kpi" ? "kpi" : "chart";
  return (
    <div className="space-y-2 pt-2">
      {result && type ? (
        <div className="rounded-lg border bg-background p-3">
          <ChartView result={result} chartType={type} title={String(analysis.interpretation ?? "")} />
        </div>
      ) : null}
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">Interpreted query</summary>
        <pre className="mt-1 overflow-x-auto rounded bg-muted p-2" data-testid="interpreted-spec">{JSON.stringify(spec, null, 2)}</pre>
      </details>
      {spec && type ? (
        <div className="flex flex-wrap items-center gap-2">
          <select aria-label="Copilot dashboard" className="h-8 rounded-md border bg-background px-2 text-xs" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">Add to dashboard…</option>
            {mine.map((d) => <option key={String(d.id)} value={String(d.id)}>{String(d.name)}</option>)}
          </select>
          <Button size="sm" variant="outline" disabled={!target} onClick={async () => {
            try {
              await analyticsApi.addWidget(target, {
                kind,
                title: String(analysis.interpretation ?? "Copilot chart").slice(0, 200),
                size: kind === "kpi" ? "small" : "medium",
                spec: { query: spec, display: kind === "kpi" ? {} : { chart_type: type } },
              });
              setMsg("Added to the dashboard.");
            } catch (e) {
              setMsg((e as Error).message);
            }
          }}>Add</Button>
          {msg ? <span className="text-xs text-muted-foreground" role="status">{msg}</span> : null}
        </div>
      ) : null}
    </div>
  );
}
