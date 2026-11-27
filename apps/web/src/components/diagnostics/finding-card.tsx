"use client";

import { CircleCheck, CircleQuestionMark, CircleX, Info, OctagonAlert, TriangleAlert } from "lucide-react";
import { useId, useState } from "react";

import {
  evidenceLabel,
  evidenceScalar,
  SEVERITY_COLORS,
  SEVERITY_LABELS,
  type FindingView,
  type Severity,
} from "@/lib/diagnostics";
import { formatPct } from "@/lib/profile";

const SEVERITY_ICONS: Record<Severity, typeof Info> = {
  critical: OctagonAlert,
  high: TriangleAlert,
  medium: TriangleAlert,
  low: Info,
  info: Info,
};

/** Severity as icon + label (colour never carries meaning alone). */
export function SeverityBadge({ severity }: { severity: Severity }) {
  const Icon = SEVERITY_ICONS[severity];
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium">
      <Icon className="size-3.5" style={{ color: SEVERITY_COLORS[severity] }} aria-hidden />
      {SEVERITY_LABELS[severity]}
    </span>
  );
}

export function StatusIcon({ status }: { status: FindingView["status"] }) {
  if (status === "not_detected") return <CircleCheck className="size-4 text-muted-foreground" aria-hidden />;
  if (status === "not_assessed") return <CircleQuestionMark className="size-4 text-muted-foreground" aria-hidden />;
  if (status === "error") return <CircleX className="size-4 text-muted-foreground" aria-hidden />;
  return null;
}

function EvidenceValue({ value, depth }: { value: unknown; depth: number }) {
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted-foreground">none</span>;
    if (value.every((v) => v === null || typeof v !== "object")) {
      return <span>{value.map(evidenceScalar).join(", ")}</span>;
    }
    if (depth >= 2) return <span className="text-muted-foreground">{value.length} items</span>;
    return (
      <ul className="space-y-1">
        {value.slice(0, 20).map((item, i) => (
          <li key={i} className="rounded border px-2 py-1">
            <EvidenceValue value={item} depth={depth + 1} />
          </li>
        ))}
      </ul>
    );
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (depth >= 3) return <span className="text-muted-foreground">{entries.length} fields</span>;
    return (
      <dl className="grid grid-cols-[minmax(0,12rem)_1fr] gap-x-3 gap-y-0.5">
        {entries.map(([k, v]) => (
          // Only top-level keys are defined by the check and safe to prettify. Nested keys are
          // often data (class, segment or column names) and are shown verbatim.
          <EvidenceRow key={k} label={depth === 0 ? evidenceLabel(k) : k} value={v} depth={depth + 1} />
        ))}
      </dl>
    );
  }
  return <span className="tabular-nums">{evidenceScalar(value)}</span>;
}

function EvidenceRow({ label, value, depth }: { label: string; value: unknown; depth: number }) {
  return (
    <>
      <dt className="truncate text-muted-foreground" title={label}>
        {label}
      </dt>
      <dd className="min-w-0 break-words">
        <EvidenceValue value={value} depth={depth} />
      </dd>
    </>
  );
}

export function FindingCard({ finding }: { finding: FindingView }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const affected =
    finding.affected_rows !== null
      ? `${finding.affected_rows.toLocaleString("en")} rows${
          finding.affected_pct !== null ? ` (${formatPct(finding.affected_pct)})` : ""
        }`
      : null;
  return (
    <li className="rounded-lg border bg-card">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3 text-left hover:bg-muted/50"
      >
        {finding.status === "detected" ? (
          <SeverityBadge severity={finding.severity} />
        ) : (
          <StatusIcon status={finding.status} />
        )}
        <span className="min-w-0 flex-1 font-medium">{finding.title}</span>
        <span className="text-xs text-muted-foreground">
          {finding.category}
          {affected ? ` · ${affected}` : ""}
        </span>
        <span aria-hidden className="text-xs text-muted-foreground">
          {open ? "▲" : "▼"}
        </span>
      </button>
      {open ? (
        <div id={panelId} className="space-y-4 border-t px-4 py-3 text-sm">
          <p>{finding.explanation}</p>
          {finding.recommendation ? (
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Suggested action
              </h4>
              <p>{finding.recommendation}</p>
            </div>
          ) : null}
          {finding.columns.length ? (
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Columns
              </h4>
              <p className="break-words">{finding.columns.join(", ")}</p>
            </div>
          ) : null}
          {Object.keys(finding.evidence).length ? (
            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Evidence
              </h4>
              <div className="text-xs">
                <EvidenceValue value={finding.evidence} depth={0} />
              </div>
            </div>
          ) : null}
          {finding.sample_info ? (
            <p className="text-xs text-muted-foreground">
              Computed on a sample: {evidenceScalar(finding.sample_info["n"])} of{" "}
              {evidenceScalar(finding.sample_info["of"])} rows ({String(finding.sample_info["strategy"])}, seed{" "}
              {String(finding.sample_info["seed"])}).
            </p>
          ) : null}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Limitations and assumptions
            </h4>
            <ul className="list-disc pl-5 text-muted-foreground">
              {finding.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </div>
          <p className="text-xs text-muted-foreground">
            Check {finding.check_id} v{finding.check_version} · {finding.domain}
            {finding.error_code ? ` · ${finding.error_code}` : ""}
          </p>
        </div>
      ) : null}
    </li>
  );
}
