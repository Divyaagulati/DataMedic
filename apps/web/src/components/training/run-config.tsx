"use client";

import { useEffect, useState } from "react";

import { FormError } from "@/components/form-error";
import { AddStepForm, StepRow } from "@/components/lab/pipeline-builder";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import type { Op } from "@/lib/preprocess";
import {
  DEFAULT_TRAINING,
  MODE_INFO,
  MODEL_KEYS,
  MODEL_LABELS,
  type ModelKey,
  type StepDraft,
  type TrainingPlan,
  type TrainingSettings,
  trainingPayload,
} from "@/lib/training";

const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";
const METHODS = [
  { value: "", label: "Recommended from roles" },
  { value: "stratified", label: "Stratified K-fold" },
  { value: "kfold", label: "K-fold" },
  { value: "group", label: "Group K-fold (entity role)" },
  { value: "stratified_group", label: "Stratified group K-fold (entity role)" },
  { value: "time", label: "Time-ordered (time role)" },
];

type VersionOption = { id: string; version_number: number; kind: string; fitted_on_full_data?: boolean };

export function ModeBadge({ mode }: { mode: string }) {
  const info = MODE_INFO[mode] ?? { label: mode, tone: "warn", help: "" };
  return (
    <Badge variant={info.tone === "bad" ? "destructive" : info.tone === "ok" ? "secondary" : "outline"} title={info.help}>
      {info.label}
    </Badge>
  );
}

export function NewTrainingRun({
  versions,
  headId,
  ops,
  onStarted,
}: {
  versions: VersionOption[];
  headId: string | undefined;
  ops: Op[];
  onStarted: (runId: string) => void;
}) {
  const [chosenVersion, setVersionId] = useState<string | undefined>(undefined);
  const versionId = chosenVersion ?? headId;
  const [settings, setSettings] = useState<TrainingSettings>(DEFAULT_TRAINING);
  const [steps, setSteps] = useState<StepDraft[] | null>(null);
  const [plan, setPlan] = useState<TrainingPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);


  useEffect(() => {
    if (!versionId) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      const { data, error: problem } = await api.POST("/api/v1/versions/{version_id}/training-plan", {
        params: { path: { version_id: versionId } },
        body: trainingPayload(settings, steps),
      });
      if (cancelled) return;
      if (data) {
        setPlan(data);
        setError(null);
      } else setError(problemMessage(problem));
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [versionId, settings, steps]);

  const modelSteps: StepDraft[] =
    steps ?? (plan?.preprocess_steps ?? []).map((s) => ({
      op: String(s.op),
      columns: (s.columns as string[]) ?? [],
      params: (s.params as Record<string, unknown>) ?? {},
    }));
  const blocked = plan?.evaluation_mode === "blocked" || plan?.eligibility.mode === "blocked";
  const replay = plan?.eligibility.replay as { steps?: { op: string; columns: string[] }[] } | null | undefined;
  const canStart = Boolean(plan && !plan.errors.length && (!blocked || settings.allowExploratory));

  function set<K extends keyof TrainingSettings>(key: K, value: TrainingSettings[K]) {
    setSettings((s) => ({ ...s, [key]: value }));
  }

  async function start() {
    if (!versionId) return;
    setBusy(true);
    setError(null);
    const { data, error: problem } = await api.POST("/api/v1/versions/{version_id}/training-runs", {
      params: { path: { version_id: versionId }, header: { "Idempotency-Key": crypto.randomUUID() } },
      body: trainingPayload(settings, steps),
    });
    setBusy(false);
    if (data) onStarted(data.training_run.id);
    else setError(problemMessage(problem));
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>New training run</CardTitle>
        <CardDescription>
          Evaluates a modelling procedure on frozen folds. Training never changes the dataset.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <section className="space-y-2">
          <h3 className="text-sm font-medium">1. Version, task and target</h3>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <label htmlFor="tr-version">Version</label>
            <select
              id="tr-version"
              className={SELECT}
              value={versionId ?? ""}
              onChange={(e) => {
                setVersionId(e.target.value);
                setSteps(null);
              }}
            >
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  v{v.version_number} ({v.kind}
                  {v.fitted_on_full_data ? ", fitted on all rows" : ""})
                </option>
              ))}
            </select>
            <Input
              aria-label="Run name"
              placeholder="Run name (optional)"
              className="h-8 w-56"
              value={settings.name}
              onChange={(e) => set("name", e.target.value)}
            />
          </div>
          {plan ? (
            <p className="text-sm">
              Task: <strong>{plan.task ?? "—"}</strong> · Target: <strong>{plan.target ?? "not declared"}</strong>{" "}
              · {plan.features.length} feature column(s)
              {plan.sensitive_columns.length ? ` (sensitive: ${plan.sensitive_columns.join(", ")})` : ""}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">Checking the configuration…</p>
          )}
          {plan?.features.length ? (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">Eligible predictors and excluded columns</summary>
              <p className="mt-1">Predictors: {plan.features.join(", ")}</p>
              {plan.excluded_columns.length ? (
                <p>
                  Never model inputs (roles):{" "}
                  {plan.excluded_columns.map((c) => `${c.column} (${(c.roles as string[]).join(", ")})`).join(", ")}
                </p>
              ) : null}
              {plan.datetime_columns.length ? <p>Ignored date/time columns: {plan.datetime_columns.join(", ")}</p> : null}
              <p>The &apos;weight&apos; role is not used for training or metrics.</p>
            </details>
          ) : null}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-medium">2. Evaluation strategy</h3>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <select
              aria-label="Split method"
              className={SELECT}
              value={settings.method ?? ""}
              onChange={(e) => set("method", (e.target.value || null) as TrainingSettings["method"])}
            >
              {METHODS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
            <label htmlFor="tr-folds">Folds</label>
            <Input
              id="tr-folds"
              type="number"
              min={2}
              max={10}
              className="h-8 w-16"
              value={settings.nSplits}
              onChange={(e) => set("nSplits", Math.min(10, Math.max(2, Number(e.target.value) || 5)))}
            />
            <label htmlFor="tr-seed">Seed</label>
            <Input
              id="tr-seed"
              type="number"
              min={0}
              className="h-8 w-20"
              value={settings.seed}
              onChange={(e) => set("seed", Math.max(0, Number(e.target.value) || 0))}
            />
          </div>
          {plan?.split ? (
            <p className="text-xs text-muted-foreground">
              Folds: {String(plan.split.method)}
              {plan.split.group ? `, grouped by ${String(plan.split.group)}` : ""}
              {plan.split.time ? `, ordered by ${String(plan.split.time)}` : ""}; duplicate rows are always kept on one
              side.
            </p>
          ) : null}
          {plan ? (
            <div className="rounded-md border p-3 text-sm" data-testid="eligibility">
              <div className="flex items-center gap-2">
                <span className="font-medium">Evaluation:</span>
                <ModeBadge mode={settings.allowExploratory && blocked ? "exploratory" : plan.eligibility.mode as string} />
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {MODE_INFO[plan.eligibility.mode as string]?.help}
              </p>
              {replay?.steps?.length ? (
                <p className="mt-1 text-xs">
                  Evaluated on version v{plan.evaluation_version_number}; replayed in each training fold:{" "}
                  {replay.steps.map((s) => `${s.op} (${s.columns.join(", ")})`).join(" → ")}
                </p>
              ) : null}
              {blocked ? (
                <div className="mt-2 space-y-1">
                  <ul className="list-disc pl-5 text-xs text-destructive">
                    {(plan.eligibility.reasons as { code: string; message: string }[]).map((r) => (
                      <li key={r.code}>{r.message}</li>
                    ))}
                  </ul>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={settings.allowExploratory}
                      onChange={(e) => set("allowExploratory", e.target.checked)}
                    />
                    Run as exploratory anyway: results will be permanently labelled as likely optimistic and never
                    reported as leakage-controlled.
                  </label>
                </div>
              ) : null}
            </div>
          ) : null}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-medium">3. Preprocessing inside each training fold</h3>
          <p className="text-xs text-muted-foreground">
            {steps === null
              ? "Using the version's stored modelling specification."
              : "Custom steps for this run."}{" "}
            Every learned step is fitted on training rows only; resampling changes training rows only.
          </p>
          <ol className="space-y-2">
            {modelSteps.map((s, i) => (
              <StepRow
                key={`${s.op}-${i}`}
                step={s}
                op={ops.find((o) => o.op === s.op)}
                onChange={(next) => setSteps(modelSteps.map((x, j) => (j === i ? next : x)))}
                onRemove={() => setSteps(modelSteps.filter((_, j) => j !== i))}
              />
            ))}
          </ol>
          {ops.length ? (
            <AddStepForm ops={ops} columns={plan?.features ?? []} onAdd={(s) => setSteps([...modelSteps, s])} />
          ) : null}
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-medium">4. Models and tuning</h3>
          <div className="flex flex-wrap gap-3 text-sm">
            {MODEL_KEYS.map((m) => (
              <label key={m} className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={settings.models.includes(m)}
                  onChange={(e) =>
                    set(
                      "models",
                      e.target.checked
                        ? MODEL_KEYS.filter((k) => k === m || settings.models.includes(k))
                        : settings.models.filter((k) => k !== m),
                    )
                  }
                />
                {MODEL_LABELS[m]}
              </label>
            ))}
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="tr-tuning">Settings</label>
            <select
              id="tr-tuning"
              className={SELECT}
              value={settings.tuning}
              onChange={(e) => set("tuning", e.target.value as TrainingSettings["tuning"])}
            >
              <option value="nested">Select from small fixed grids (nested cross-validation)</option>
              <option value="defaults">Default settings only</option>
            </select>
          </div>
          {plan ? (
            <p className="text-xs text-muted-foreground">
              About {plan.estimated_fits} model fits (limit {plan.fit_limit}); settings used:{" "}
              {String(plan.tuning.planned)}.
            </p>
          ) : null}
        </section>

        {plan?.warnings.length ? (
          <ul className="list-disc space-y-1 pl-5 text-xs text-muted-foreground">
            {plan.warnings.map((w, i) => (
              <li key={i}>{String(w.message)}</li>
            ))}
          </ul>
        ) : null}
        {plan?.errors.length ? (
          <ul className="list-disc space-y-1 pl-5 text-xs text-destructive" data-testid="plan-errors">
            {plan.errors.map((e, i) => (
              <li key={i}>{String(e.message)}</li>
            ))}
          </ul>
        ) : null}
        <FormError message={error} />
        <Button disabled={!canStart || busy || settings.models.length === 0} onClick={start}>
          {busy ? "Starting…" : "Start training"}
        </Button>
      </CardContent>
    </Card>
  );
}

export type { ModelKey };
