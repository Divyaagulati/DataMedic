"use client";

import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError, FormNotice } from "@/components/form-error";
import { AddStepForm, ParamField } from "@/components/lab/pipeline-builder";
import { RiskBadge } from "@/components/lab/recommendation-card";
import { ConfirmDialog, StepApprovalDialog } from "@/components/pipelines/approval-dialog";
import { ImpactPreviewPanel } from "@/components/pipelines/impact-preview";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import {
  approvalState,
  canCommit,
  moveStep,
  nextActions,
  type Pipeline,
  type PipelineStep,
  type PreviewSummary,
  splitByStage,
  STAGE_CHOICE,
} from "@/lib/pipelines";
import type { Op } from "@/lib/preprocess";
import { isActiveJob, useDataset, useImpactPreview, useJob, useOps, usePipeline, useVersion } from "@/lib/queries";

const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";

export function StepEditor({
  pipeline,
  step,
  index,
  op,
  editable,
  onPatch,
  onRemove,
  onApprove,
}: {
  pipeline: Pipeline;
  step: PipelineStep;
  index: number;
  op: Op | undefined;
  editable: boolean;
  onPatch: (body: Record<string, unknown>) => void;
  onRemove: () => void;
  onApprove: () => void;
}) {
  const state = approvalState(pipeline, step);
  return (
    <li className="space-y-1 rounded-md border p-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium">
          {index + 1}. {step.op_title}
        </span>
        <RiskBadge risk={step.risk as "low" | "medium" | "high"} destructive={step.destructive} />
        {step.catalog_stage === "either" && editable ? (
          <select
            aria-label={`Stage of ${step.op}`}
            className={SELECT}
            value={step.stage}
            onChange={(e) => onPatch({ stage: e.target.value })}
          >
            <option value="data">{STAGE_CHOICE.data.label}</option>
            <option value="model">{STAGE_CHOICE.model.label}</option>
          </select>
        ) : (
          <Badge variant="secondary" title={STAGE_CHOICE[step.stage].help}>
            {STAGE_CHOICE[step.stage].label}
          </Badge>
        )}
        {state === "approved" ? <Badge variant="outline">Approved</Badge> : null}
        {state === "required" ? (
          <Button size="sm" onClick={onApprove} disabled={!pipeline.readiness.preview_current}>
            Approve…
          </Button>
        ) : null}
        {editable ? (
          <span className="ml-auto flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Move ${step.op} up`}
              disabled={moveStep(pipeline.steps, index, -1) === null}
              onClick={() => onPatch({ position: index - 1 })}
            >
              ↑
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Move ${step.op} down`}
              disabled={moveStep(pipeline.steps, index, 1) === null}
              onClick={() => onPatch({ position: index + 1 })}
            >
              ↓
            </Button>
            <Button size="sm" variant="ghost" aria-label={`Remove ${step.op}`} onClick={onRemove}>
              Remove
            </Button>
          </span>
        ) : null}
      </div>
      <p className="text-xs text-muted-foreground">{STAGE_CHOICE[step.stage].help}</p>
      {step.columns.length ? <p className="text-xs">Columns: {step.columns.join(", ")}</p> : null}
      {op && op.params.length && editable ? (
        <div className="flex flex-wrap gap-3">
          {op.params.map((p) => (
            <ParamField
              key={p.name}
              param={p}
              value={step.params[p.name] ?? p.default}
              onChange={(v) => {
                const params = { ...step.params };
                if (v === undefined) delete params[p.name];
                else params[p.name] = v;
                onPatch({ params });
              }}
            />
          ))}
        </div>
      ) : null}
    </li>
  );
}

export function PipelineView() {
  const { datasetId, pipelineId } = useParams<{ datasetId: string; pipelineId: string }>();
  const queryClient = useQueryClient();
  const dataset = useDataset(datasetId);
  const pipeline = usePipeline(pipelineId);
  const version = useVersion(pipeline.data?.source_version_id);
  const ops = useOps();
  const preview = useImpactPreview(pipelineId, pipeline.data?.preview_job?.status);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [approving, setApproving] = useState<PipelineStep | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [commitJob, setCommitJob] = useState<string | null>(null);
  const job = useJob(commitJob);
  const [busy, setBusy] = useState(false);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (pipeline.error) return <FormError message={problemMessage(pipeline.error)} />;
  if (!dataset.data || !pipeline.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const p = pipeline.data;
  const editable = p.state !== "committed" && p.state !== "archived";
  const byOp = new Map((ops.data ?? []).map((o) => [o.op, o]));
  const { data: dataSteps, model: modelSteps } = splitByStage(p.steps);
  const summary = (preview.data?.summary ?? null) as PreviewSummary | null;
  const previewRunning = isActiveJob(p.preview_job?.status);
  const columns = (version.data?.columns ?? []).map((c) => c.name);

  async function refresh() {
    await queryClient.invalidateQueries({ queryKey: ["pipelines", pipelineId] });
  }

  async function act(fn: () => Promise<{ error?: unknown; response: Response }>, done?: string) {
    setBusy(true);
    setError(null);
    const { error: problem, response } = await fn();
    setBusy(false);
    if (!response.ok) setError(problemMessage(problem));
    else if (done) setNotice(done);
    await refresh();
    return response.ok;
  }

  const patch = (step: PipelineStep, body: Record<string, unknown>) =>
    act(() =>
      api.PATCH("/api/v1/pipelines/{pipeline_id}/steps/{step_id}", {
        params: { path: { pipeline_id: pipelineId, step_id: step.id } },
        body,
      }),
    );

  async function commit() {
    setBusy(true);
    const { data, error: problem, response } = await api.POST("/api/v1/pipelines/{pipeline_id}/commit", {
      params: { path: { pipeline_id: pipelineId }, header: { "Idempotency-Key": crypto.randomUUID() } },
      body: { spec_hash: p.spec_hash },
    });
    setBusy(false);
    setCommitting(false);
    if (!response.ok || !data) return setError(problemMessage(problem));
    setCommitJob(data.job.id);
    await refresh();
  }

  const renderSteps = (steps: PipelineStep[]) => (
    <ol className="space-y-2">
      {steps.map((s) => (
        <StepEditor
          key={s.id}
          pipeline={p}
          step={s}
          index={p.steps.indexOf(s)}
          op={byOp.get(s.op)}
          editable={editable}
          onPatch={(body) => patch(s, body)}
          onRemove={() =>
            act(() =>
              api.DELETE("/api/v1/pipelines/{pipeline_id}/steps/{step_id}", {
                params: { path: { pipeline_id: pipelineId, step_id: s.id } },
              }),
            )
          }
          onApprove={() => setApproving(s)}
        />
      ))}
    </ol>
  );

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={dataset.data.name} status={dataset.data.status} active="history" />
      <FormError message={error} />
      <FormNotice message={notice} />
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-xl font-semibold">{p.name}</h2>
        <Badge variant="outline">{p.state}</Badge>
        {!p.source_is_head && editable ? <Badge variant="destructive">Source is no longer the current version</Badge> : null}
        <Link href={`/datasets/${datasetId}/history`} className="text-sm underline">
          Version history
        </Link>
      </div>
      <p className="text-sm text-muted-foreground">
        Nothing changes your dataset until you commit. A commit creates a new version; the current version and the
        original file stay unchanged, and you can revert later.
      </p>
      {p.state === "committed" && p.committed_version_id ? (
        <FormNotice message="This pipeline was committed into a new version. It can no longer change." />
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Steps</CardTitle>
          <CardDescription>Data-stage steps run first and change the saved data; modelling-only steps follow.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <section className="space-y-2">
            <h3 className="text-sm font-medium">Changes the saved data ({dataSteps.length})</h3>
            {dataSteps.length ? renderSteps(dataSteps) : <p className="text-xs text-muted-foreground">None.</p>}
          </section>
          <section className="space-y-2">
            <h3 className="text-sm font-medium">Modelling only — stored with the version, never applied to its data ({modelSteps.length})</h3>
            {modelSteps.length ? renderSteps(modelSteps) : <p className="text-xs text-muted-foreground">None.</p>}
          </section>
          {editable ? (
            <AddStepForm
              ops={ops.data ?? []}
              columns={columns}
              onAdd={(step) =>
                act(() =>
                  api.POST("/api/v1/pipelines/{pipeline_id}/steps", {
                    params: { path: { pipeline_id: pipelineId } },
                    body: step,
                  }),
                )
              }
            />
          ) : null}
          {p.validation && !p.validation.valid ? (
            <ul className="list-disc pl-5 text-sm text-destructive" role="alert">
              {(p.validation.errors as { code: string; message: string; step: number | null }[]).map((e, i) => (
                <li key={i}>
                  {e.step !== null ? `Step ${e.step + 1}: ` : ""}
                  {e.message}
                </li>
              ))}
            </ul>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Impact preview</CardTitle>
          <CardDescription>What a commit would change, computed on the current version. Nothing is saved.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {editable ? (
            <Button
              disabled={busy || previewRunning || !p.validation?.valid}
              onClick={() =>
                act(() =>
                  api.POST("/api/v1/pipelines/{pipeline_id}/impact-preview", {
                    params: { path: { pipeline_id: pipelineId } },
                  }),
                )
              }
            >
              {previewRunning ? "Computing…" : summary ? "Recompute preview" : "Generate preview"}
            </Button>
          ) : null}
          {p.preview_job?.status === "failed" ? (
            <p className="text-sm text-destructive" role="alert">
              {p.preview_job.error?.code}: {p.preview_job.error?.message}
            </p>
          ) : null}
          {summary ? <ImpactPreviewPanel pipelineId={pipelineId} summary={summary} current={Boolean(preview.data?.current)} /> : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Approve and commit</CardTitle>
          <CardDescription>High-risk steps need their own approval; medium-risk steps need one pipeline confirmation.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {nextActions(p).length ? (
            <ul className="list-disc pl-5 text-sm">
              {nextActions(p).map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          ) : null}
          {p.readiness.stale_approvals > 0 ? (
            <p className="text-xs text-muted-foreground">
              {p.readiness.stale_approvals} earlier approval(s) no longer apply because the pipeline or the dataset changed.
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            {p.readiness.confirmation_required && editable ? (
              <Button
                variant="outline"
                disabled={!p.readiness.preview_current || p.readiness.confirmation_valid}
                onClick={() => setConfirming(true)}
              >
                {p.readiness.confirmation_valid ? "Pipeline confirmed" : "Confirm pipeline…"}
              </Button>
            ) : null}
            {editable ? (
              <Button disabled={!canCommit(p) || busy} onClick={() => setCommitting(true)}>
                Commit as new version…
              </Button>
            ) : null}
          </div>
          {job.data ? (
            <p className="text-sm" role="status">
              Commit job: {job.data.status}
              {job.data.status === "failed" ? ` — ${job.data.error?.code}: ${job.data.error?.message}` : ""}
              {job.data.status === "succeeded" ? (
                <>
                  {" "}
                  — <Link className="underline" href={`/datasets/${datasetId}/history`}>see the new version</Link>
                </>
              ) : null}
            </p>
          ) : null}
          {p.approvals.length ? (
            <section className="space-y-1">
              <h4 className="text-sm font-medium">Approval history</h4>
              <ul className="space-y-1 text-xs">
                {p.approvals.map((a) => (
                  <li key={a.id}>
                    <Badge variant={a.valid ? "outline" : "secondary"}>{a.valid ? "valid" : "stale"}</Badge>{" "}
                    {new Date(a.created_at).toLocaleString()} — {a.statement}
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </CardContent>
      </Card>

      {approving ? (
        <StepApprovalDialog
          step={approving}
          impact={summary?.steps.find((s) => s.index === approving.position)}
          busy={busy}
          onClose={() => setApproving(null)}
          onApprove={async (ack) => {
            const ok = await act(
              () =>
                api.POST("/api/v1/pipelines/{pipeline_id}/steps/{step_id}/approve", {
                  params: { path: { pipeline_id: pipelineId, step_id: approving.id } },
                  body: { step_hash: approving.step_hash ?? "", spec_hash: p.spec_hash, acknowledge_destructive: ack },
                }),
              "Step approved.",
            );
            if (ok) setApproving(null);
          }}
        />
      ) : null}
      {confirming && summary ? (
        <ConfirmDialog
          title="Confirm the pipeline"
          confirmLabel="Confirm pipeline"
          busy={busy}
          onClose={() => setConfirming(false)}
          onConfirm={async () => {
            const ok = await act(
              () =>
                api.POST("/api/v1/pipelines/{pipeline_id}/confirm", {
                  params: { path: { pipeline_id: pipelineId } },
                  body: { spec_hash: p.spec_hash },
                }),
              "Pipeline confirmed.",
            );
            if (ok) setConfirming(false);
          }}
        >
          <p>
            {summary.totals.estimates_filled} missing values will be replaced by estimates, {summary.totals.observed_overwritten}{" "}
            observed values overwritten and {summary.totals.observed_re_expressed} re-expressed, in a new version.
          </p>
        </ConfirmDialog>
      ) : null}
      {committing ? (
        <ConfirmDialog
          title="Create a new version"
          confirmLabel="Commit"
          busy={busy}
          onClose={() => setCommitting(false)}
          onConfirm={commit}
        >
          <p>
            The {dataSteps.length} data-stage step(s) will be applied to every row and saved as a new version. The{" "}
            {modelSteps.length} modelling-only step(s) will be stored with it without changing its data. The current
            version and the original file stay unchanged.
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
