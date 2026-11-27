"use client";

import { useId, useState } from "react";

import { RiskBadge, StageBadge } from "@/components/lab/recommendation-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  addStep,
  type CandidateDraft,
  type LabSettings,
  MAX_CANDIDATES,
  newCandidateId,
  removeStep,
  type StepDraft,
} from "@/lib/experiments";
import { defaultParams, KIND_INFO, type Op, type OpParam } from "@/lib/preprocess";

const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";
const SPLIT_METHODS: { value: string; label: string }[] = [
  { value: "", label: "Recommended from roles" },
  { value: "stratified", label: "Stratified K-fold" },
  { value: "kfold", label: "K-fold" },
  { value: "group", label: "Group K-fold (entity role)" },
  { value: "stratified_group", label: "Stratified group K-fold (entity role)" },
  { value: "time", label: "Time-ordered (time role)" },
];

function parseParam(param: OpParam, raw: string | boolean): unknown {
  if (param.type === "bool") return Boolean(raw);
  const text = String(raw);
  if (param.type === "int") return text === "" ? undefined : Number.parseInt(text, 10);
  if (param.type === "float") return text === "" ? undefined : Number.parseFloat(text);
  if (param.type === "columns" || param.type === "text_list")
    return text.split(",").map((s) => s.trim()).filter(Boolean);
  if (param.type === "scalar" && text !== "" && !Number.isNaN(Number(text))) return Number(text);
  return text;
}

export function ParamField({
  param,
  value,
  onChange,
}: {
  param: OpParam;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const id = useId();
  const label = (
    <label htmlFor={id} className="text-xs" title={param.description}>
      {param.name}
      {param.required ? " *" : ""}
    </label>
  );
  if (param.type === "bool")
    return (
      <span className="flex items-center gap-1">
        <input id={id} type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {label}
      </span>
    );
  if (param.type === "choice")
    return (
      <span className="flex items-center gap-1">
        {label}
        <select id={id} className={SELECT} value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
          {(param.choices ?? []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </span>
    );
  const shown = Array.isArray(value) ? value.join(", ") : value === undefined || value === null ? "" : String(value);
  return (
    <span className="flex items-center gap-1">
      {label}
      <Input
        id={id}
        className="h-8 w-32"
        value={shown}
        type={param.type === "int" || param.type === "float" ? "number" : "text"}
        onChange={(e) => onChange(parseParam(param, e.target.value))}
      />
    </span>
  );
}

export function StepRow({
  step,
  op,
  onChange,
  onRemove,
}: {
  step: StepDraft;
  op: Op | undefined;
  onChange: (step: StepDraft) => void;
  onRemove: () => void;
}) {
  return (
    <li className="space-y-1 rounded-md border p-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-sm">{op?.title ?? step.op}</span>
        {op ? <StageBadge stage={op.stage} /> : null}
        {op ? <RiskBadge risk={op.risk} destructive={op.destructive} /> : null}
        <Button size="sm" variant="ghost" className="ml-auto" onClick={onRemove} aria-label={`Remove ${step.op}`}>
          Remove
        </Button>
      </div>
      {op ? <p className="text-xs text-muted-foreground">{KIND_INFO[op.kind]}</p> : null}
      {op?.uses_columns !== false ? (
        <p className="text-xs">Columns: {step.columns.join(", ") || "—"}</p>
      ) : null}
      {op && op.params.length > 0 ? (
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
                onChange({ ...step, params });
              }}
            />
          ))}
        </div>
      ) : null}
    </li>
  );
}

export function AddStepForm({
  ops,
  columns,
  onAdd,
}: {
  ops: Op[];
  columns: string[];
  onAdd: (step: StepDraft) => void;
}) {
  const [opName, setOpName] = useState(ops[0]?.op ?? "");
  const [cols, setCols] = useState<string[]>([]);
  const op = ops.find((o) => o.op === opName);
  return (
    <div className="flex flex-wrap items-end gap-2 rounded-md bg-muted/40 p-2">
      <span className="flex flex-col gap-1">
        <label className="text-xs" htmlFor="add-op">
          Operation
        </label>
        <select id="add-op" className={SELECT} value={opName} onChange={(e) => setOpName(e.target.value)}>
          {ops.map((o) => (
            <option key={o.op} value={o.op}>
              {o.title} ({o.op})
            </option>
          ))}
        </select>
      </span>
      {op?.uses_columns !== false ? (
        <span className="flex flex-col gap-1">
          <label className="text-xs" htmlFor="add-cols">
            Columns
          </label>
          <select
            id="add-cols"
            multiple
            className="h-20 rounded-md border bg-background px-2 text-sm"
            value={cols}
            onChange={(e) => setCols(Array.from(e.target.selectedOptions, (o) => o.value))}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </span>
      ) : null}
      <Button
        size="sm"
        variant="outline"
        disabled={!op}
        onClick={() => op && onAdd({ op: op.op, columns: op.uses_columns ? cols : [], params: defaultParams(op) })}
      >
        Add step
      </Button>
      {op ? <p className="w-full text-xs text-muted-foreground">{op.description}</p> : null}
    </div>
  );
}

export function PipelineBuilder({
  drafts,
  onDrafts,
  ops,
  columns,
  settings,
  onSettings,
  hasTarget,
  onRun,
  running,
}: {
  drafts: CandidateDraft[];
  onDrafts: (drafts: CandidateDraft[]) => void;
  ops: Op[];
  columns: string[];
  settings: LabSettings;
  onSettings: (settings: LabSettings) => void;
  hasTarget: boolean;
  onRun: () => void;
  running: boolean;
}) {
  const byOp = new Map(ops.map((o) => [o.op, o]));
  const update = (id: string, change: (c: CandidateDraft) => CandidateDraft) =>
    onDrafts(drafts.map((c) => (c.id === id ? change(c) : c)));
  const models = settings.clustering ? ["kmeans"] : ["dummy", "linear", "hgb"];
  const toggleModel = (m: string) =>
    onSettings({
      ...settings,
      models: settings.models.includes(m) ? settings.models.filter((x) => x !== m) : [...settings.models, m],
    });
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        The baseline (no preprocessing beyond each model&apos;s documented minimal handling) is always
        included. Every candidate is evaluated on the same frozen folds; every learned step is fitted
        on the training rows of each fold only. Nothing here changes your dataset.
      </p>
      <ol className="space-y-3">
        {drafts.map((c, i) => (
          <li key={c.id} className="space-y-2 rounded-lg border p-3">
            <div className="flex items-center gap-2">
              <label className="sr-only" htmlFor={`label-${c.id}`}>
                Candidate name
              </label>
              <Input
                id={`label-${c.id}`}
                className="h-8 max-w-sm"
                value={c.label}
                onChange={(e) => update(c.id, (x) => ({ ...x, label: e.target.value }))}
              />
              <span className="text-xs text-muted-foreground">Candidate {i + 1}</span>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto"
                onClick={() => onDrafts(drafts.filter((x) => x.id !== c.id))}
              >
                Delete candidate
              </Button>
            </div>
            {c.steps.length === 0 ? (
              <p className="text-xs text-muted-foreground">No steps yet: add recommended steps or choose an operation.</p>
            ) : null}
            <ol className="space-y-2">
              {c.steps.map((s, j) => (
                <StepRow
                  key={`${s.op}-${j}`}
                  step={s}
                  op={byOp.get(s.op)}
                  onRemove={() => update(c.id, (x) => removeStep(x, j))}
                  onChange={(step) =>
                    update(c.id, (x) => ({ ...x, steps: x.steps.map((y, k) => (k === j ? step : y)) }))
                  }
                />
              ))}
            </ol>
            <AddStepForm ops={ops} columns={columns} onAdd={(step) => update(c.id, (x) => addStep(x, step))} />
          </li>
        ))}
      </ol>
      <Button
        variant="outline"
        disabled={drafts.length >= MAX_CANDIDATES}
        onClick={() =>
          onDrafts([...drafts, { id: newCandidateId(), label: `Candidate ${drafts.length + 1}`, steps: [], recommendationIds: [] }])
        }
      >
        New candidate pipeline
      </Button>

      <fieldset className="space-y-2 rounded-lg border p-3">
        <legend className="px-1 text-sm font-medium">Experiment settings</legend>
        <div className="flex flex-wrap items-end gap-3">
          <span className="flex flex-col gap-1">
            <label className="text-xs" htmlFor="exp-name">
              Name
            </label>
            <Input id="exp-name" className="h-8 w-56" value={settings.name} onChange={(e) => onSettings({ ...settings, name: e.target.value })} />
          </span>
          <span className="flex flex-col gap-1">
            <label className="text-xs" htmlFor="exp-method">
              Folds
            </label>
            <select id="exp-method" className={SELECT} value={settings.method ?? ""} onChange={(e) => onSettings({ ...settings, method: e.target.value || null })}>
              {SPLIT_METHODS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </span>
          <span className="flex flex-col gap-1">
            <label className="text-xs" htmlFor="exp-splits">
              Number of folds
            </label>
            <Input id="exp-splits" type="number" min={2} max={10} className="h-8 w-20" value={settings.nSplits} onChange={(e) => onSettings({ ...settings, nSplits: Number(e.target.value) })} />
          </span>
          <span className="flex flex-col gap-1">
            <label className="text-xs" htmlFor="exp-seed">
              Seed
            </label>
            <Input id="exp-seed" type="number" min={0} className="h-8 w-24" value={settings.seed} onChange={(e) => onSettings({ ...settings, seed: Number(e.target.value) })} />
          </span>
          <span className="flex items-center gap-1">
            <input
              id="exp-clustering"
              type="checkbox"
              checked={settings.clustering}
              onChange={(e) =>
                onSettings({ ...settings, clustering: e.target.checked, models: e.target.checked ? ["kmeans"] : ["dummy", "linear", "hgb"] })
              }
            />
            <label className="text-xs" htmlFor="exp-clustering">
              Clustering (no target)
            </label>
          </span>
          {settings.clustering ? (
            <span className="flex flex-col gap-1">
              <label className="text-xs" htmlFor="exp-k">
                Clusters
              </label>
              <Input id="exp-k" type="number" min={2} max={20} className="h-8 w-20" value={settings.nClusters} onChange={(e) => onSettings({ ...settings, nClusters: Number(e.target.value) })} />
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-3 text-sm">
          {models.map((m) => (
            <span key={m} className="flex items-center gap-1">
              <input id={`model-${m}`} type="checkbox" checked={settings.models.includes(m)} onChange={() => toggleModel(m)} />
              <label htmlFor={`model-${m}`}>{m}</label>
            </span>
          ))}
        </div>
        {!hasTarget && !settings.clustering ? (
          <p className="text-xs text-destructive">
            No target is declared. Declare a target with a task type on the Diagnostics page, or run a
            clustering experiment.
          </p>
        ) : null}
        <Button
          onClick={onRun}
          disabled={running || drafts.length === 0 || settings.models.length === 0 || (!hasTarget && !settings.clustering)}
        >
          {running ? "Starting…" : `Run experiment (${drafts.length + 1} pipelines incl. baseline)`}
        </Button>
      </fieldset>
    </div>
  );
}
