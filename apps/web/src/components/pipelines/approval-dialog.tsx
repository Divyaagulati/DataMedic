"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { PipelineStep, PreviewStep } from "@/lib/pipelines";

/** A modal confirmation. Opening it is never an approval: only the confirm button is. */
export function ConfirmDialog({
  title,
  children,
  confirmLabel,
  requireAcknowledgement,
  acknowledgement,
  onConfirm,
  onClose,
  busy,
}: {
  title: string;
  children: React.ReactNode;
  confirmLabel: string;
  requireAcknowledgement?: boolean;
  acknowledgement?: string;
  onConfirm: (acknowledged: boolean) => void;
  onClose: () => void;
  busy?: boolean;
}) {
  const [ack, setAck] = useState(false);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="presentation">
      <div role="dialog" aria-modal="true" aria-label={title} className="w-full max-w-lg space-y-3 rounded-lg bg-background p-4 shadow-lg">
        <h3 className="text-lg font-semibold">{title}</h3>
        <div className="space-y-2 text-sm">{children}</div>
        {requireAcknowledgement ? (
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} />
            <span>{acknowledgement}</span>
          </label>
        ) : null}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant={requireAcknowledgement ? "destructive" : "default"}
            disabled={busy || (requireAcknowledgement && !ack)}
            onClick={() => onConfirm(ack)}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}

export function StepApprovalDialog({
  step,
  impact,
  onApprove,
  onClose,
  busy,
}: {
  step: PipelineStep;
  impact: PreviewStep | undefined;
  onApprove: (acknowledged: boolean) => void;
  onClose: () => void;
  busy?: boolean;
}) {
  return (
    <ConfirmDialog
      title={`Approve high-risk step: ${step.op_title}`}
      confirmLabel={step.destructive ? "Approve destructive step" : "Approve step"}
      requireAcknowledgement={step.destructive}
      acknowledgement="I understand this step is destructive: it overwrites or removes observed values in the new version, and they cannot be recovered from that version (earlier versions stay unchanged)."
      onConfirm={onApprove}
      onClose={onClose}
      busy={busy}
    >
      {step.stage === "model" ? (
        <p>This is a modelling-only step: it is stored with the new version and never changes the saved data.</p>
      ) : impact ? (
        <>
          <p>
            Affects <strong>{impact.rows_affected}</strong> rows.{" "}
            {impact.columns.map((c) => `${c.column}: ${c.observed_overwritten} overwritten, ${c.estimates_filled} estimated`).join("; ")}
          </p>
          <ul className="list-disc pl-5">
            {impact.information_loss.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </>
      ) : (
        <p>No preview is available for this step.</p>
      )}
      <p className="text-muted-foreground">
        The approval is bound to this exact step and pipeline. Any later edit, reordering or removal makes it stale.
      </p>
    </ConfirmDialog>
  );
}
