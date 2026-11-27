"use client";

import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError, FormNotice } from "@/components/form-error";
import { ConfirmDialog } from "@/components/pipelines/approval-dialog";
import { type ComparisonResult, VersionComparison } from "@/components/pipelines/version-comparison";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, API_BASE_URL } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { STAGE_CHOICE, VERSION_KIND, type VersionSummary } from "@/lib/pipelines";
import {
  useAnalysis,
  useComparison,
  useDataset,
  useJob,
  usePipelines,
  useProfileState,
  useTransformationRun,
  useVersionList,
} from "@/lib/queries";

export function VersionBadges({ v, isHead, versions }: { v: VersionSummary; isHead: boolean; versions: VersionSummary[] }) {
  const restored = versions.find((x) => x.id === v.restored_from_version_id);
  return (
    <span className="flex flex-wrap gap-1">
      <Badge variant="outline">{VERSION_KIND[v.kind] ?? v.kind}</Badge>
      {isHead ? <Badge>Current</Badge> : null}
      {v.fitted_on_full_data ? (
        <Badge variant="destructive" title="Steps learned from every row: not for leakage-controlled model evaluation as stored">
          Fitted on all rows
        </Badge>
      ) : null}
      {v.has_model_spec ? <Badge variant="secondary">Modelling specification</Badge> : null}
      {restored ? <Badge variant="secondary">Restores v{restored.version_number}</Badge> : null}
    </span>
  );
}

function RunDetail({ runId }: { runId: string }) {
  const run = useTransformationRun(runId);
  if (!run.data) return <p className="text-xs text-muted-foreground">Loading the transformation…</p>;
  const r = run.data;
  return (
    <div className="space-y-2 text-sm">
      <h4 className="font-medium">Committed transformation</h4>
      <ol className="list-decimal pl-5">
        {r.steps.map((s, i) => (
          <li key={i}>
            {String(s.op)} {(s.columns as string[]).length ? `(${(s.columns as string[]).join(", ")})` : ""} —{" "}
            {STAGE_CHOICE[String(s.stage)]?.label ?? String(s.stage)}
          </li>
        ))}
      </ol>
      <h4 className="font-medium">Approvals</h4>
      <ul className="list-disc pl-5 text-xs">
        {r.approvals.map((a) => (
          <li key={a.id}>
            {new Date(a.created_at).toLocaleString()} — {a.statement}
          </li>
        ))}
        {r.approvals.length === 0 ? <li>None were required (low-risk steps only).</li> : null}
      </ul>
      <h4 className="font-medium">Step reports</h4>
      <ul className="list-disc pl-5 text-xs">
        {r.step_reports.map((s, i) => (
          <li key={i}>
            {String(s.op)}: {String(s.rows_in)} rows, {(s.columns_added as string[]).length} columns added,{" "}
            {(s.columns_removed as string[]).length} removed, {(s.warnings as unknown[]).length} warnings
          </li>
        ))}
      </ul>
      {(r.roles_report.lost as unknown[] | undefined)?.length ? (
        <p className="text-xs text-destructive">
          Roles not carried over:{" "}
          {(r.roles_report.lost as { column: string; role: string; reason: string }[])
            .map((l) => `${l.role} on ${l.column} (${l.reason})`)
            .join("; ")}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">Every column role was carried over.</p>
      )}
    </div>
  );
}

function AnalysisStatus({ versionId }: { versionId: string }) {
  const profile = useProfileState(versionId);
  const analysis = useAnalysis(versionId);
  const p = profile.data?.job?.status ?? (profile.data?.profile ? "succeeded" : "none");
  const a = analysis.data?.job?.status ?? (analysis.data?.run ? "succeeded" : "none");
  return (
    <p className="text-xs" role="status">
      Profiling: {p}
      {profile.data?.job?.error ? ` (${profile.data.job.error.code})` : ""} · Diagnostics: {a}
      {analysis.data?.job?.error ? ` (${analysis.data.job.error.code})` : ""}
    </p>
  );
}

export async function downloadCode(versionId: string, versionNumber: number): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/api/v1/versions/${versionId}/code`, { credentials: "include" });
  if (!response.ok) return false;
  const url = URL.createObjectURL(new Blob([await response.text()], { type: "text/x-python" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `pipeline_v${versionNumber}.py`;
  a.click();
  URL.revokeObjectURL(url);
  return true;
}

export function HistoryView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const dataset = useDataset(datasetId);
  const versions = useVersionList(datasetId);
  const pipelines = usePipelines(datasetId);
  const [selected, setSelected] = useState<string | null>(null);
  const [compareJob, setCompareJob] = useState<string | null>(null);
  const comparison = useComparison(compareJob);
  const [reverting, setReverting] = useState<VersionSummary | null>(null);
  const [revertJob, setRevertJob] = useState<string | null>(null);
  const revert = useJob(revertJob);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const list = versions.data ?? [];
  const head = ds.head_version?.id;
  const current = list.find((v) => v.id === (selected ?? head)) ?? null;

  async function newPipeline() {
    const { data, error: problem, response } = await api.POST("/api/v1/datasets/{dataset_id}/pipelines", {
      params: { path: { dataset_id: datasetId } },
      body: { name: "New pipeline", steps: [] },
    });
    if (!response.ok || !data) return setError(problemMessage(problem));
    router.push(`/datasets/${datasetId}/pipelines/${data.id}`);
  }

  async function compare(v: VersionSummary) {
    if (!v.parent_version_id) return;
    const { data, error: problem, response } = await api.POST("/api/v1/versions/{version_id}/compare/{other_version_id}", {
      params: { path: { version_id: v.parent_version_id, other_version_id: v.id } },
    });
    if (!response.ok || !data) return setError(problemMessage(problem));
    setCompareJob(data.job.id);
  }

  async function doRevert(v: VersionSummary) {
    const { data, error: problem, response } = await api.POST("/api/v1/datasets/{dataset_id}/revert", {
      params: { path: { dataset_id: datasetId }, header: { "Idempotency-Key": crypto.randomUUID() } },
      body: { version_id: v.id },
    });
    setReverting(null);
    if (!response.ok || !data) return setError(problemMessage(problem));
    setRevertJob(data.job.id);
    setNotice(`Reverting to version ${v.version_number}: a new version is being created.`);
    await queryClient.invalidateQueries({ queryKey: ["datasets", datasetId] });
  }

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="history" />
      <FormError message={error} />
      <FormNotice message={notice} />
      {revert.data && revert.data.status !== "queued" && revert.data.status !== "running" ? (
        <FormNotice message={`Revert ${revert.data.status}${revert.data.error ? `: ${revert.data.error.message}` : ""}.`} />
      ) : null}
      <p className="text-sm text-muted-foreground">
        Every version is immutable. Commits and reverts always create a new version; nothing is deleted or overwritten,
        and the original upload is never changed.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Versions</CardTitle>
          <CardDescription>Select a version to see how it was made.</CardDescription>
        </CardHeader>
        <CardContent>
          {list.length === 0 ? (
            <p className="text-sm text-muted-foreground">No version yet.</p>
          ) : (
            <ol className="space-y-1">
              {[...list].reverse().map((v) => (
                <li key={v.id} className="flex flex-wrap items-center gap-2">
                  <Button size="sm" variant={current?.id === v.id ? "default" : "ghost"} onClick={() => setSelected(v.id)}>
                    v{v.version_number}
                  </Button>
                  <VersionBadges v={v} isHead={v.id === head} versions={list} />
                  <span className="text-xs text-muted-foreground">
                    {v.n_rows} rows × {v.n_cols} columns · {new Date(v.created_at).toLocaleString()}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </CardContent>
      </Card>

      {current ? (
        <Card>
          <CardHeader>
            <CardTitle>Version {current.version_number}</CardTitle>
            <CardDescription>{VERSION_KIND[current.kind] ?? current.kind}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <AnalysisStatus versionId={current.id} />
            {current.fitted_on_full_data ? (
              <p className="text-xs">
                Steps of this version learned from all of its rows. Use the Experiment Lab or the Models workspace (in-fold
                fitting) for leakage-controlled model evaluation.
              </p>
            ) : null}
            <div className="flex flex-wrap gap-2">
              {current.parent_version_id ? (
                <Button size="sm" variant="outline" onClick={() => compare(current)}>
                  Compare with previous version
                </Button>
              ) : null}
              <Button
                size="sm"
                variant="outline"
                onClick={async () => {
                  if (!(await downloadCode(current.id, current.version_number)))
                    setError("The code for this version could not be generated.");
                }}
              >
                Download pipeline.py
              </Button>
              {current.id !== head ? (
                <Button size="sm" variant="destructive" onClick={() => setReverting(current)}>
                  Revert to this version…
                </Button>
              ) : null}
            </div>
            {current.transformation_run_id ? <RunDetail runId={current.transformation_run_id} /> : null}
            {compareJob ? (
              comparison.data?.result ? (
                <VersionComparison result={comparison.data.result as unknown as ComparisonResult} />
              ) : (
                <p className="text-xs text-muted-foreground" role="status">
                  Comparison: {comparison.data?.job.status ?? "starting"}
                  {comparison.data?.job.error ? ` — ${comparison.data.job.error.message}` : ""}
                </p>
              )
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Pipelines</CardTitle>
          <CardDescription>Drafts and committed pipelines. A pipeline always starts from the current version.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button size="sm" onClick={newPipeline} disabled={!head}>
            New pipeline
          </Button>
          <ul className="space-y-1 text-sm">
            {(pipelines.data ?? []).map((p) => (
              <li key={p.id}>
                <Link className="underline" href={`/datasets/${datasetId}/pipelines/${p.id}`}>
                  {p.name}
                </Link>{" "}
                <Badge variant="outline">{p.state}</Badge>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {reverting ? (
        <ConfirmDialog
          title={`Revert to version ${reverting.version_number}`}
          confirmLabel="Create revert version"
          onClose={() => setReverting(null)}
          onConfirm={() => doRevert(reverting)}
        >
          <p>
            A new version will be created with exactly the content of version {reverting.version_number}. No version is
            deleted or changed; you can revert again later.
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
