"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError, FormNotice } from "@/components/form-error";
import { ExperimentResults } from "@/components/lab/experiment-results";
import { PipelineBuilder } from "@/components/lab/pipeline-builder";
import { RecommendationCard } from "@/components/lab/recommendation-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import {
  addStep,
  type CandidateDraft,
  draftsFromSuggestions,
  experimentPayload,
  type LabSettings,
  newCandidateId,
  type RecommendationItem,
} from "@/lib/experiments";
import {
  isActiveJob,
  useDataset,
  useExperiment,
  useExperiments,
  useOps,
  useRecommendations,
  useRoles,
  useVersion,
} from "@/lib/queries";

type Dict = Record<string, unknown>;

const DEFAULT_SETTINGS: LabSettings = {
  name: "Experiment",
  models: ["dummy", "linear", "hgb"],
  method: null,
  nSplits: 5,
  seed: 0,
  clustering: false,
  nClusters: 3,
};

function List({ title, items, render }: { title: string; items: Dict[]; render: (d: Dict) => string }) {
  if (items.length === 0) return null;
  return (
    <section className="space-y-1">
      <h3 className="text-sm font-medium">{title}</h3>
      <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
        {items.map((d, i) => (
          <li key={i}>{render(d)}</li>
        ))}
      </ul>
    </section>
  );
}

export function LabView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const queryClient = useQueryClient();
  const router = useRouter();
  const dataset = useDataset(datasetId);
  const versionId = dataset.data?.head_version?.id;
  const version = useVersion(versionId);
  const roles = useRoles(versionId);
  const ops = useOps();
  const recs = useRecommendations(versionId);
  const experiments = useExperiments(versionId);
  const [selected, setSelected] = useState<string | null>(null);
  const experiment = useExperiment(selected);
  const [drafts, setDrafts] = useState<CandidateDraft[]>([]);
  const [settings, setSettings] = useState<LabSettings>(DEFAULT_SETTINGS);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const set = recs.data ?? null;
  const columns = (version.data?.columns ?? []).map((c) => c.name);
  const hasTarget = (roles.data?.roles ?? []).some((r) => r.role === "target");

  async function generate() {
    if (!versionId) return;
    setError(null);
    const { error: problem, response } = await api.POST("/api/v1/versions/{version_id}/recommendations", {
      params: { path: { version_id: versionId } },
    });
    if (!response.ok) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "recommendations"] });
  }

  async function setStatus(item: RecommendationItem, status: "open" | "dismissed") {
    const { error: problem, response } = await api.PATCH("/api/v1/recommendations/{recommendation_id}", {
      params: { path: { recommendation_id: item.id } },
      body: { status },
    });
    if (!response.ok) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "recommendations"] });
  }

  function addRecommendation(candidateId: string, item: RecommendationItem) {
    setDrafts((current) =>
      current.map((c) =>
        c.id === candidateId
          ? addStep(c, { op: item.op, columns: [...item.columns], params: { ...item.params } }, item.id)
          : c,
      ),
    );
    setNotice("Step added. Review its parameters in the pipeline builder below.");
  }

  async function run() {
    if (!versionId) return;
    setBusy(true);
    setError(null);
    const { data, error: problem, response } = await api.POST("/api/v1/versions/{version_id}/experiments", {
      params: { path: { version_id: versionId }, header: { "Idempotency-Key": crypto.randomUUID() } },
      body: experimentPayload(drafts, settings, set?.id ?? null),
    });
    setBusy(false);
    if (!response.ok || !data) {
      setError(problemMessage(problem));
      return;
    }
    setSelected(data.experiment.id);
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "experiments"] });
  }

  async function cancel() {
    const jobId = experiment.data?.job?.id;
    if (!jobId) return;
    await api.POST("/api/v1/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } });
    await queryClient.invalidateQueries({ queryKey: ["experiments", selected] });
  }

  async function toPipeline(candidateId: string) {
    const { data, error: problem, response } = await api.POST(
      "/api/v1/experiment-candidates/{candidate_id}/to-pipeline",
      { params: { path: { candidate_id: candidateId } } },
    );
    if (!response.ok || !data) return setError(problemMessage(problem));
    router.push(`/datasets/${datasetId}/pipelines/${data.id}`);
  }

  async function rerun() {
    if (!selected) return;
    const { data, error: problem, response } = await api.POST("/api/v1/experiments/{experiment_id}/rerun", {
      params: { path: { experiment_id: selected } },
    });
    if (!response.ok || !data) return setError(problemMessage(problem));
    setSelected(data.experiment.id);
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "experiments"] });
  }

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="lab" />
      {version.data?.fitted_on_full_data ? (
        <div role="alert" className="rounded-md border border-amber-500/60 p-3 text-sm" data-testid="fitted-warning">
          <strong>This version was fitted on all of its rows.</strong> Committed steps learned from every row,
          including rows the Lab uses as test rows, so scores here can be optimistic. For a leakage-controlled evaluation, use the{" "}
          <a className="underline" href={`/datasets/${datasetId}/models`}>
            Models
          </a>{" "}
          workspace, which replays those steps inside each training fold. Leakage control refers to the implemented
          split and preprocessing safeguards; it does not guarantee external validity, absence of bias or clinical
          reliability.
        </div>
      ) : null}
      <FormError message={error} />
      <FormNotice message={notice} />
      <p className="text-sm text-muted-foreground">
        The Experiment Lab proposes preprocessing from your diagnostics, profile and roles, and compares
        candidate pipelines on identical frozen folds. Recommendations are hypotheses to test, not
        guaranteed improvements. Nothing here changes your dataset or creates a new version.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Recommendations</CardTitle>
          <CardDescription>
            Ranked by transparent factors (evidence, task relevance, domain fit, risk, information loss,
            sample size, cost, assumptions). Open a recommendation to see each factor.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={generate} disabled={!versionId}>
              {set ? "Refresh recommendations" : "Generate recommendations"}
            </Button>
            {set?.stale ? (
              <Badge variant="destructive">Out of date: roles, domain or diagnostics changed</Badge>
            ) : null}
            {set ? (
              <span className="text-xs text-muted-foreground">
                Rules {set.rules_version} · {set.recommendations.length} recommendations · split plan:{" "}
                {String(set.split_plan.method)} {set.split_plan.n_splits ? `(${String(set.split_plan.n_splits)} folds)` : ""}
              </span>
            ) : null}
          </div>
          {set ? (
            <>
              <List
                title="Advisories (no operation can fix these)"
                items={set.advisories}
                render={(a) => `${String(a.title)}: ${String(a.detail)} ${(a.columns as string[]).join(", ")}`}
              />
              <ul className="space-y-3">
                {set.recommendations.map((item) => (
                  <RecommendationCard
                    key={item.id}
                    item={item}
                    candidates={drafts.map((d) => ({ id: d.id, label: d.label }))}
                    onAdd={addRecommendation}
                    onStatus={setStatus}
                  />
                ))}
              </ul>
              {set.conflicts.length > 0 ? (
                <section className="space-y-2">
                  <h3 className="text-sm font-medium">Trade-offs: options that compete for the same decision</h3>
                  {set.conflicts.map((c) => (
                    <div key={String(c.id)} className="rounded-md border p-3 text-sm">
                      <p className="font-medium">{String(c.topic)}</p>
                      <p>{String(c.trade_off)}</p>
                      <ul className="list-disc pl-5 text-muted-foreground">
                        {(c.options as Dict[]).map((o, i) => (
                          <li key={i}>
                            {String(o.label)}
                            {o.consequence ? ` — ${String(o.consequence)}` : ""}
                          </li>
                        ))}
                      </ul>
                      <p className="text-xs text-muted-foreground">{String(c.how_to_decide)}</p>
                    </div>
                  ))}
                </section>
              ) : null}
              <List
                title="Considered and not recommended"
                items={set.not_recommended}
                render={(n) => `${String(n.op)}${(n.columns as string[]).length ? ` (${(n.columns as string[]).join(", ")})` : ""}: ${String(n.reason)}`}
              />
              <List
                title="Not assessed (evidence or metadata missing)"
                items={set.not_assessed}
                render={(n) => `${String(n.topic)}: ${String(n.reason)}`}
              />
              <List title="Limitations" items={set.limitations.map((l) => ({ l }))} render={(d) => String(d.l)} />
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              Generate recommendations after diagnostics have run with your current roles.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Candidate pipelines</CardTitle>
          <CardDescription>Select operations, configure parameters, and compare several pipelines.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" disabled={!set} onClick={() => setDrafts(set ? draftsFromSuggestions(set) : [])}>
              Use suggested candidates
            </Button>
            <Button
              variant="ghost"
              onClick={() => setDrafts([{ id: newCandidateId(), label: "Candidate 1", steps: [], recommendationIds: [] }])}
            >
              Start from empty
            </Button>
          </div>
          <PipelineBuilder
            drafts={drafts}
            onDrafts={setDrafts}
            ops={ops.data ?? []}
            columns={columns}
            settings={settings}
            onSettings={setSettings}
            hasTarget={hasTarget}
            onRun={run}
            running={busy}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experiments</CardTitle>
          <CardDescription>Saved configurations and results for this version.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {(experiments.data ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">No experiments yet.</p>
          ) : (
            <ul className="space-y-1 text-sm">
              {(experiments.data ?? []).map((e) => (
                <li key={e.id} className="flex items-center gap-2">
                  <Button size="sm" variant={selected === e.id ? "default" : "ghost"} onClick={() => setSelected(e.id)}>
                    {e.name}
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    {new Date(e.created_at).toLocaleString()} · {e.job?.status ?? "—"}
                    {isActiveJob(e.job?.status) ? " (running)" : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {experiment.data ? <ExperimentResults experiment={experiment.data} onCancel={cancel} onRerun={rerun} onToPipeline={toPipeline} /> : null}
        </CardContent>
      </Card>
    </div>
  );
}
