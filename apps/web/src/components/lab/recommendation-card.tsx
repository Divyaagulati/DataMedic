"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  EVIDENCE_LABELS,
  LOSS_LABELS,
  type RecommendationDetail,
  type RecommendationItem,
} from "@/lib/experiments";
import { RISK_INFO, STAGE_INFO } from "@/lib/preprocess";

function evidenceValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (value === null || value === undefined) return "—";
  return String(value);
}

export function RiskBadge({ risk, destructive }: { risk: "low" | "medium" | "high"; destructive?: boolean }) {
  return (
    <>
      <Badge variant={risk === "high" ? "destructive" : "outline"} title={RISK_INFO[risk].help}>
        {RISK_INFO[risk].label}
      </Badge>
      {destructive ? <Badge variant="destructive">Destructive</Badge> : null}
    </>
  );
}

export function StageBadge({ stage }: { stage: "data" | "model" | "either" }) {
  return (
    <Badge variant="secondary" title={STAGE_INFO[stage].help}>
      {STAGE_INFO[stage].label}
    </Badge>
  );
}

export function RecommendationCard({
  item,
  candidates,
  onAdd,
  onStatus,
}: {
  item: RecommendationItem;
  candidates: { id: string; label: string }[];
  onAdd: (candidateId: string, item: RecommendationItem) => void;
  onStatus: (item: RecommendationItem, status: "open" | "dismissed") => void;
}) {
  const d = item.detail as unknown as RecommendationDetail;
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState<string>(candidates[0]?.id ?? "");
  const dismissed = item.status === "dismissed";
  return (
    <li className={"rounded-lg border p-4 " + (dismissed ? "opacity-60" : "")}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">
            #{item.rank} · {d.op_title} · score {d.ranking.score}
          </p>
          <h3 className="font-medium">{d.title}</h3>
          <div className="flex flex-wrap gap-1">
            <StageBadge stage={item.stage as "data" | "model" | "either"} />
            <RiskBadge risk={item.risk as "low" | "medium" | "high"} destructive={item.destructive} />
            <Badge variant="outline">{EVIDENCE_LABELS[d.evidence_strength.level]}</Badge>
            <Badge variant="outline">{LOSS_LABELS[d.information_loss.level]}</Badge>
            {d.domain_compatibility.status === "caution" ? (
              <Badge variant="destructive">Domain caution</Badge>
            ) : null}
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide details" : "Why and what"}
        </Button>
      </div>
      <p className="mt-2 text-sm">{d.reason}</p>
      <p className="mt-1 text-xs text-muted-foreground">{d.applies_to}</p>
      {open ? (
        <div className="mt-3 space-y-3 text-sm">
          <section>
            <h4 className="font-medium">What it does</h4>
            <p>{d.what_it_does}</p>
          </section>
          <section>
            <h4 className="font-medium">Evidence</h4>
            <p className="text-muted-foreground">{d.evidence_strength.explanation}</p>
            <ul className="list-disc pl-5">
              {d.evidence.map((e, i) => (
                <li key={i}>
                  {e.column ? `${e.column}: ` : ""}
                  {e.metric.replaceAll("_", " ")} = {evidenceValue(e.value)}{" "}
                  <span className="text-muted-foreground">({e.source})</span>
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h4 className="font-medium">Expected benefit</h4>
            <p>{d.expected_benefit}</p>
            <p className="text-muted-foreground">{d.not_a_guarantee}</p>
          </section>
          <section>
            <h4 className="font-medium">Risks and information loss</h4>
            <p>{d.information_loss.detail}</p>
            <ul className="list-disc pl-5">
              {d.risks.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </section>
          <section>
            <h4 className="font-medium">Fit for this dataset</h4>
            <p>Domain: {d.domain_compatibility.detail}</p>
            <p>Task: {d.task_compatibility.detail}</p>
          </section>
          <section>
            <h4 className="font-medium">Decisions you need to make</h4>
            <ul className="list-disc pl-5">
              {d.required_decisions.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </section>
          <section>
            <h4 className="font-medium">Why it is ranked here</h4>
            <table className="text-xs">
              <tbody>
                {d.ranking.factors.map((f) => (
                  <tr key={f.factor}>
                    <td className="pr-2 font-mono">{f.points > 0 ? `+${f.points}` : f.points}</td>
                    <td className="pr-2">{f.factor.replace("_", " ")}</td>
                    <td className="text-muted-foreground">{f.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section>
            <h4 className="font-medium">Limitations</h4>
            <ul className="list-disc pl-5 text-muted-foreground">
              {d.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </section>
        </div>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {candidates.length > 0 ? (
          <>
            <label className="text-xs" htmlFor={`target-${item.id}`}>
              Add to
            </label>
            <select
              id={`target-${item.id}`}
              className="h-8 rounded-md border bg-background px-2 text-sm"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            >
              {candidates.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <Button size="sm" disabled={dismissed || !target} onClick={() => onAdd(target, item)}>
              Add step
            </Button>
          </>
        ) : (
          <span className="text-xs text-muted-foreground">Create a candidate pipeline to add steps.</span>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onStatus(item, dismissed ? "open" : "dismissed")}
        >
          {dismissed ? "Restore" : "Dismiss"}
        </Button>
      </div>
    </li>
  );
}
