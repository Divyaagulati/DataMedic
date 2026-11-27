"use client";

import { useVersionList } from "@/lib/queries";

/** Version selector (defaults to the current head). */
export function VersionPicker({
  datasetId,
  value,
  onChange,
}: {
  datasetId: string;
  value: string | undefined;
  onChange: (id: string) => void;
}) {
  const versions = useVersionList(datasetId);
  const items = [...(versions.data ?? [])].reverse();
  if (!items.length) return null;
  return (
    <label className="no-print flex items-center gap-2 text-sm">
      <span className="text-muted-foreground">Version</span>
      <select
        aria-label="Version"
        className="h-8 rounded-md border bg-background px-2 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {items.map((v) => (
          <option key={v.id} value={v.id}>
            v{v.version_number} · {v.kind}
            {v.fitted_on_full_data ? " · fitted on all rows" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}
