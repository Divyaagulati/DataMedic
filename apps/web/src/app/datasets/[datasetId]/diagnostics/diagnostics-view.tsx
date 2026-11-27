"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FindingCard, SeverityBadge } from "@/components/diagnostics/finding-card";
import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type Schemas } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import {
  countBy,
  filterFindings,
  SEVERITIES,
  SEVERITY_LABELS,
  STATUS_LABELS,
  STATUS_ORDER,
  type Filters,
  type FindingStatus,
  type FindingView,
} from "@/lib/diagnostics";
import {
  isActiveJob,
  useAnalysis,
  useDataset,
  useDomains,
  useRoles,
  useVersion,
} from "@/lib/queries";

import { SemanticsPanel } from "./semantics-panel";

const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";

export function DiagnosticsView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const queryClient = useQueryClient();
  const dataset = useDataset(datasetId);
  const versionId = dataset.data?.head_version?.id;
  const version = useVersion(versionId);
  const roles = useRoles(versionId);
  const domains = useDomains();
  const analysis = useAnalysis(versionId);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function run() {
    if (!versionId) return;
    setError(null);
    const { error: problem, response } = await api.POST("/api/v1/versions/{version_id}/analyses", {
      params: { path: { version_id: versionId } },
    });
    if (response.status !== 202) setError(problemMessage(problem));
    else setNotice(null);
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "analysis"] });
  }

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const state = analysis.data;
  const job = state?.job;
  const running = isActiveJob(job?.status);
  const latest = state?.run ?? null;

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="diagnostics" />
      <FormError message={error ?? (analysis.error ? problemMessage(analysis.error) : null)} />
      {!versionId ? (
        <FormNotice message="The dataset has no parsed version yet, so there is nothing to diagnose." />
      ) : null}
      <p className="text-sm text-muted-foreground">
        Diagnostics describe possible data-quality and modelling risks. A finding is a reason to
        look closer, not proof of an error: outliers can be valid observations, and associations
        are not causes. Nothing here changes your data.
      </p>

      <FormNotice message={notice} />
      {versionId && version.data && roles.data && domains.data ? (
        <SemanticsPanel
          key={`${roles.data.semantics_hash}-${roles.data.domain}`}
          datasetId={datasetId}
          versionId={versionId}
          domain={ds.domain}
          columns={version.data.columns}
          domains={domains.data}
          roles={roles.data}
          onNotice={setNotice}
        />
      ) : null}

      {versionId ? (
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
          <span>
            {latest
              ? `Last run ${new Date(latest.created_at).toLocaleString()} · domain ${latest.domain} · ${latest.summary.total} results`
              : running
                ? "Diagnostics are running in the background. Results appear here automatically."
                : job?.status === "failed"
                  ? `The last run failed: ${job.error?.code} ${job.error?.message ?? ""}`
                  : "No diagnostics yet."}
          </span>
          <Button size="sm" onClick={() => void run()} disabled={running}>
            {running ? "Running…" : "Run diagnostics"}
          </Button>
        </div>
      ) : null}

      {latest?.stale ? (
        <FormNotice message="The domain or roles changed after this run. These results may be out of date: run diagnostics again." />
      ) : null}
      {latest ? <Results run={latest} /> : null}
    </div>
  );
}

function Results({ run }: { run: Schemas["AnalysisRunOut"] }) {
  const findings = run.findings;
  const [filters, setFilters] = useState<Filters>({
    status: "detected",
    severity: "all",
    category: "all",
    domain: "all",
  });
  const detected = findings.filter((f) => f.status === "detected");
  const bySeverity = countBy(detected, (f) => f.severity);
  const byStatus = countBy(findings, (f) => f.status);
  const categories = [...new Set(findings.map((f) => f.category))].sort();
  const domains = [...new Set(findings.map((f) => f.domain))].sort();
  const visible = filterFindings(findings, filters);

  return (
    <>
      <section aria-label="Findings by severity" className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {SEVERITIES.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setFilters({ ...filters, status: "detected", severity: filters.severity === s ? "all" : s })}
            className={
              "rounded-lg border bg-card px-4 py-3 text-left hover:bg-muted/50 " +
              (filters.status === "detected" && filters.severity === s ? "ring-2 ring-ring" : "")
            }
            aria-pressed={filters.status === "detected" && filters.severity === s}
            aria-label={`${SEVERITY_LABELS[s]}: ${bySeverity.get(s) ?? 0} detected. Show only these.`}
          >
            <span className="block"><SeverityBadge severity={s} /></span>
            <span className="mt-1 block text-2xl font-semibold">{bySeverity.get(s) ?? 0}</span>
            <span className="block text-xs text-muted-foreground">detected</span>
          </button>
        ))}
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Findings</CardTitle>
          <CardDescription>
            Detected: an issue was found. Not detected: the check ran and found nothing.
            Not assessed: the check could not run (the reason is shown), so nothing is claimed.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div role="group" aria-label="Result status" className="flex flex-wrap gap-1">
            {STATUS_ORDER.map((status) => (
              <Button
                key={status}
                aria-pressed={filters.status === status}
                size="sm"
                variant={filters.status === status ? "default" : "outline"}
                onClick={() => setFilters({ ...filters, status: status as FindingStatus })}
              >
                {STATUS_LABELS[status]} ({byStatus.get(status) ?? 0})
              </Button>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 text-sm">
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Severity</span>
              <select className={SELECT} value={filters.severity}
                onChange={(e) => setFilters({ ...filters, severity: e.target.value as Filters["severity"] })}>
                <option value="all">All</option>
                {SEVERITIES.map((s) => <option key={s} value={s}>{SEVERITY_LABELS[s]}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Category</span>
              <select className={SELECT} value={filters.category}
                onChange={(e) => setFilters({ ...filters, category: e.target.value })}>
                <option value="all">All</option>
                {categories.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Module</span>
              <select className={SELECT} value={filters.domain}
                onChange={(e) => setFilters({ ...filters, domain: e.target.value })}>
                <option value="all">All</option>
                {domains.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
          </div>
          {visible.length === 0 ? (
            <p className="text-sm text-muted-foreground">No results match these filters.</p>
          ) : (
            <ul className="space-y-2" aria-label="Diagnostic results">
              {visible.map((f: FindingView) => (
                <FindingCard key={f.id} finding={f} />
              ))}
            </ul>
          )}
          {run.summary.check_errors ? (
            <p className="text-xs text-muted-foreground">
              {run.summary.check_errors} checks failed unexpectedly; the others completed. See
              “Check error”.
            </p>
          ) : null}
          <p className="text-xs text-muted-foreground">
            Engine {run.engine_version} · checks run on all rows unless a result says it used a
            sample. <SeverityBadge severity="info" /> results are context, not problems.
          </p>
        </CardContent>
      </Card>
    </>
  );
}
