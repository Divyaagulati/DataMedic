"use client";

import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { StatusBadge } from "@/components/status-badge";
import { ModeBadge } from "@/components/training/run-config";
import { Comparison, Explainability, Performance, Reliability, Reproducibility } from "@/components/training/run-results";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { useDataset, useTrainingAudit, useTrainingRun } from "@/lib/queries";
import { isActive, MODE_INFO } from "@/lib/training";

export function RunView() {
  const { datasetId, runId } = useParams<{ datasetId: string; runId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const dataset = useDataset(datasetId);
  const run = useTrainingRun(runId);
  const done = Boolean(run.data?.completed_at);
  const audit = useTrainingAudit(runId, done);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (dataset.error || run.error) return <FormError message={problemMessage(dataset.error ?? run.error)} />;
  if (!dataset.data || !run.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const r = run.data;
  const job = r.job;
  const progress = (job?.progress ?? null) as { pct?: number; done?: number; total?: number } | null;

  async function cancel() {
    if (!job) return;
    setBusy(true);
    const { error: problem, response } = await api.POST("/api/v1/jobs/{job_id}/cancel", {
      params: { path: { job_id: job.id } },
    });
    setBusy(false);
    if (!response.ok) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["training-runs", runId] });
  }

  async function rerun() {
    setBusy(true);
    const { data, error: problem } = await api.POST("/api/v1/training-runs/{training_run_id}/rerun", {
      params: { path: { training_run_id: runId }, header: { "Idempotency-Key": crypto.randomUUID() } },
    });
    setBusy(false);
    if (data) router.push(`/datasets/${datasetId}/models/${data.training_run.id}`);
    else setError(problemMessage(problem));
  }

  async function reaudit() {
    setBusy(true);
    const { error: problem, response } = await api.POST("/api/v1/training-runs/{training_run_id}/audit", {
      params: { path: { training_run_id: runId } },
    });
    setBusy(false);
    if (!response.ok) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["training-runs", runId, "audit"] });
  }

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={dataset.data.name} status={dataset.data.status} active="models" />
      <Link href={`/datasets/${datasetId}/models`} className="text-sm underline">
        ← All training runs
      </Link>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-xl font-semibold">{r.name}</h2>
        <ModeBadge mode={r.evaluation_mode} />
        {job ? <StatusBadge status={job.status} /> : null}
        <span className="text-sm text-muted-foreground">
          {r.task_type} · target {r.target_column}
        </span>
        <span className="ml-auto flex gap-2">
          {job && isActive(job.status) ? (
            <Button size="sm" variant="outline" onClick={cancel} disabled={busy || job.status === "cancel_requested"}>
              {job.status === "cancel_requested" ? "Cancelling…" : "Cancel"}
            </Button>
          ) : null}
          {done ? (
            <Button size="sm" variant="outline" onClick={rerun} disabled={busy}>
              Re-run same configuration
            </Button>
          ) : null}
        </span>
      </div>
      {r.evaluation_mode === "exploratory" ? (
        <div role="alert" className="rounded-md border border-destructive p-3 text-sm text-destructive">
          <strong>Exploratory run.</strong> {MODE_INFO.exploratory.help}
        </div>
      ) : null}
      <FormError message={error} />
      {job && isActive(job.status) ? (
        <Card>
          <CardContent className="space-y-2 pt-6 text-sm">
            <p>
              Training… {progress?.pct ?? 0}% ({progress?.done ?? 0} of {progress?.total ?? "?"} model-fold steps).
              Cancellation takes effect between model fits.
            </p>
            <div className="h-2 rounded bg-muted">
              <div className="h-2 rounded" style={{ width: `${progress?.pct ?? 0}%`, background: "var(--viz-seq)" }} />
            </div>
          </CardContent>
        </Card>
      ) : null}
      {job && !isActive(job.status) && !done ? (
        <Card>
          <CardContent className="pt-6 text-sm">
            {job.status === "cancelled"
              ? "This run was cancelled; no results were stored."
              : `This run did not complete: ${job.error?.message ?? job.status}. No results were stored.`}
          </CardContent>
        </Card>
      ) : null}
      {done ? (
        <>
          <Comparison run={r} />
          <Performance run={r} />
          <Reliability audit={audit.data} onRerun={reaudit} busy={busy} />
          <Explainability run={r} />
          <Reproducibility run={r} />
        </>
      ) : null}
    </div>
  );
}
