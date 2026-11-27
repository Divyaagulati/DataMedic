import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";

export function DatasetNav({
  datasetId,
  name,
  status,
  active,
}: {
  datasetId: string;
  name: string;
  status: string;
  active:
    | "overview"
    | "profile"
    | "diagnostics"
    | "biomedical"
    | "analytics"
    | "lab"
    | "models"
    | "history"
    | "copilot"
    | "passport";
}) {
  const tabs = [
    { key: "overview", label: "Overview", href: `/datasets/${datasetId}` },
    { key: "profile", label: "Profile", href: `/datasets/${datasetId}/profile` },
    { key: "analytics", label: "Analytics Studio", href: `/datasets/${datasetId}/analytics` },
    { key: "diagnostics", label: "Diagnostics", href: `/datasets/${datasetId}/diagnostics` },
    { key: "biomedical", label: "Biomedical", href: `/datasets/${datasetId}/biomedical` },
    { key: "lab", label: "Experiment Lab", href: `/datasets/${datasetId}/lab` },
    { key: "models", label: "Models", href: `/datasets/${datasetId}/models` },
    { key: "history", label: "History", href: `/datasets/${datasetId}/history` },
    { key: "copilot", label: "Copilot", href: `/datasets/${datasetId}/copilot` },
    { key: "passport", label: "Passport & Export", href: `/datasets/${datasetId}/passport` },
  ] as const;
  return (
    <div className="space-y-4">
      <Link href="/datasets" className="text-sm text-muted-foreground hover:text-foreground">
        ← All datasets
      </Link>
      <div className="flex items-center justify-between gap-4">
        <h1 className="truncate text-2xl font-semibold tracking-tight">{name}</h1>
        <StatusBadge status={status} />
      </div>
      <nav aria-label="Dataset sections" className="no-print flex gap-1 overflow-x-auto border-b">
        {tabs.map((tab) => (
          <Link
            key={tab.key}
            href={tab.href}
            aria-current={active === tab.key ? "page" : undefined}
            className={
              "-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm transition-colors " +
              (active === tab.key
                ? "border-primary font-medium text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground")
            }
          >
            {tab.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
