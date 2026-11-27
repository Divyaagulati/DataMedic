"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { StatTile } from "@/components/charts/charts";
import { DatasetNav } from "@/components/dataset-nav";
import { FormError, FormNotice } from "@/components/form-error";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, unwrap, type Schemas } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { formatBytes } from "@/lib/format";
import { formatNumber, formatPct, TYPE_LABELS } from "@/lib/profile";
import { isActiveJob, useDataset, usePreview, useProfileState, useVersion } from "@/lib/queries";

export function DatasetDetail() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const dataset = useDataset(datasetId);
  const headId = dataset.data?.head_version?.id;
  const version = useVersion(headId);
  const preview = usePreview(headId);
  const profile = useProfileState(headId);
  const jobs = useQuery({
    queryKey: ["jobs", { datasetId }, dataset.data?.status, profile.data?.job?.status],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/jobs", { params: { query: { dataset_id: datasetId } } }))
        .items,
    enabled: dataset.isSuccess,
  });

  async function onParse() {
    setBusy(true);
    setActionError(null);
    const { error: problem, response } = await api.POST("/api/v1/datasets/{dataset_id}/parse", {
      params: { path: { dataset_id: datasetId } },
    });
    if (response.status !== 202) setActionError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["datasets"] });
    setBusy(false);
  }

  async function onDelete() {
    if (!window.confirm("Delete this dataset and all its stored files? This cannot be undone.")) {
      return;
    }
    setBusy(true);
    const { error: problem, response } = await api.DELETE("/api/v1/datasets/{dataset_id}", {
      params: { path: { dataset_id: datasetId } },
    });
    if (response.status === 202) {
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      router.push("/datasets");
    } else {
      setActionError(problemMessage(problem));
      setBusy(false);
    }
  }

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;

  const ds = dataset.data;
  const jobList = jobs.data ?? [];
  const lastIngest = jobList.find((j) => j.type === "ingest");
  const summary = profile.data?.profile?.summary;
  const profileJob = profile.data?.job;
  // "…" only while a profile is actually being computed; otherwise an honest "—".
  const pending = isActiveJob(profileJob?.status) ? "…" : "—";

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="overview" />
      <FormError message={actionError} />

      {ds.status === "validating" ? (
        <FormNotice message="Checking and parsing the file… this page updates automatically." />
      ) : null}
      {(ds.status === "rejected" || ds.status === "parse_failed") && lastIngest?.error ? (
        <FormError message={`${lastIngest.error.code}: ${lastIngest.error.message}`} />
      ) : null}
      {(ds.status === "validated" || ds.status === "parse_failed") && !ds.head_version ? (
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {ds.status === "validated"
                ? "This file was uploaded before parsing was available. Parse it now to create version 0."
                : "Parsing failed. If the problem was temporary you can try again; otherwise fix the file and upload it as a new dataset."}
            </p>
            <Button onClick={onParse} disabled={busy}>
              {ds.status === "validated" ? "Parse now" : "Retry parsing"}
            </Button>
          </CardContent>
        </Card>
      ) : null}

      {ds.head_version ? (
        <section aria-label="Summary" className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <StatTile label="Rows" value={formatNumber(ds.head_version.n_rows)} />
          <StatTile label="Columns" value={formatNumber(ds.head_version.n_cols)} />
          <StatTile
            label="Missing cells"
            value={summary ? formatPct(summary.missing_cells_pct) : pending}
            detail={summary ? `${formatNumber(summary.missing_cells)} of ${formatNumber(summary.n_cells)}` : undefined}
          />
          <StatTile
            label="Duplicate rows"
            value={summary ? formatNumber(summary.duplicate_rows) : pending}
            detail={summary ? formatPct(summary.duplicate_rows_pct) : undefined}
          />
          <StatTile
            label="Constant columns"
            value={summary ? formatNumber(summary.constant_columns.length) : pending}
          />
        </section>
      ) : null}

      {ds.head_version ? (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-4">
            <CardTitle>Profile</CardTitle>
            <Link href={`/datasets/${datasetId}/profile`} className="text-sm underline">
              Open profiling dashboard →
            </Link>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {profile.data?.profile
              ? `Profiled ${new Date(profile.data.profile.created_at).toLocaleString()}: types, missing values, distributions, correlations and class balance.`
              : isActiveJob(profileJob?.status)
                ? "Profiling is running in the background…"
                : profileJob?.status === "failed"
                  ? `Profiling failed (${profileJob.error?.code}). Open the dashboard to run it again.`
                  : "No profile yet."}
          </CardContent>
        </Card>
      ) : null}

      {preview.data ? <PreviewCard preview={preview.data} /> : null}
      {preview.error ? <FormError message={problemMessage(preview.error)} /> : null}

      <Card>
        <CardHeader>
          <CardTitle>File and version</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-[11rem_1fr] gap-y-2 text-sm">
            <dt className="text-muted-foreground">Original file</dt>
            <dd className="break-all">{ds.filename}</dd>
            <dt className="text-muted-foreground">Size</dt>
            <dd>{formatBytes(ds.size_bytes)}</dd>
            <dt className="text-muted-foreground">Detected format</dt>
            <dd>{ds.detected_format ?? "—"}</dd>
            <dt className="text-muted-foreground">Uploaded</dt>
            <dd>{new Date(ds.created_at).toLocaleString()}</dd>
            {version.data ? <VersionFacts version={version.data} /> : null}
          </dl>
          <p className="mt-4 text-xs text-muted-foreground">
            The original upload is stored unchanged. Version 0 is a parsed copy; later versions
            will never overwrite it.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Jobs</CardTitle>
        </CardHeader>
        <CardContent>
          {jobList.length === 0 ? (
            <p className="text-sm text-muted-foreground">No jobs yet.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {jobList.map((j) => (
                <li key={j.id} className="flex items-center justify-between gap-4">
                  <span>
                    {j.type} · {new Date(j.queued_at).toLocaleString()}
                    {j.error ? <span className="text-muted-foreground"> · {j.error.code}</span> : null}
                  </span>
                  <StatusBadge status={j.status} />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
      <Button variant="destructive" onClick={onDelete} disabled={busy}>
        Delete dataset
      </Button>
    </div>
  );
}

function VersionFacts({ version }: { version: Schemas["VersionOut"] }) {
  const options = version.parse_options ?? {};
  const report = version.parse_report ?? {};
  const delimiter = options["delimiter"] === "\t" ? "tab" : (options["delimiter"] as string | undefined);
  const facts: [string, string][] = [
    ["Version", `v${version.version_number} (${version.kind})`],
    ["Shape", `${formatNumber(version.n_rows)} rows × ${formatNumber(version.n_cols)} columns`],
  ];
  if (options["encoding"]) facts.push(["Encoding", String(options["encoding"])]);
  if (delimiter) facts.push(["Delimiter", delimiter]);
  if (options["sheet"]) {
    const count = Number(report["sheet_count"] ?? 1);
    facts.push(["Sheet", `${options["sheet"]}${count > 1 ? ` (first of ${count}; other sheets were not imported)` : ""}`]);
  }
  if (options["json_orient"]) facts.push(["JSON layout", String(options["json_orient"])]);
  const dropped = Number(report["dropped_empty_rows"] ?? 0);
  if (dropped) facts.push(["Empty rows skipped", formatNumber(dropped)]);
  const renamed = Number(report["renamed_columns"] ?? 0);
  if (renamed) facts.push(["Columns renamed", `${renamed} (blank or duplicate headers)`]);
  if (report["header_looks_like_data"]) {
    facts.push([
      "Header check",
      "The first row is mostly numbers. If the file has no header row, its first data row was used as column names (choosing header options arrives with re-parse).",
    ]);
  }
  facts.push(["Content hash", version.content_hash.slice(0, 16) + "…"]);
  return (
    <>
      {facts.map(([k, v]) => (
        <FactRow key={k} term={k} value={v} />
      ))}
    </>
  );
}

function FactRow({ term, value }: { term: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="break-all">{value}</dd>
    </>
  );
}

function PreviewCard({ preview }: { preview: Schemas["PreviewOut"] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Preview</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="max-h-[32rem] overflow-auto rounded-md border">
          <table className="w-full text-xs">
            <caption className="sr-only">First rows of the dataset</caption>
            <thead className="sticky top-0 bg-muted">
              <tr>
                {preview.columns.map((c) => (
                  <th key={c.name} scope="col" className="px-3 py-2 text-left align-bottom font-medium">
                    <div className="max-w-[14rem] truncate" title={c.name}>
                      {c.name}
                    </div>
                    <div className="font-normal text-muted-foreground">{TYPE_LABELS[c.inferred_type]}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, i) => (
                <tr key={i} className="border-t">
                  {row.map((cell, j) => (
                    <td key={j} className="max-w-[16rem] truncate px-3 py-1.5 tabular-nums" title={cell === null ? "missing" : String(cell)}>
                      {cell === null ? <span className="text-muted-foreground">—</span> : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted-foreground">
          First {preview.rows.length} of {formatNumber(preview.n_rows_total)} rows. “—” marks a
          missing value. Visible to you only.
        </p>
      </CardContent>
    </Card>
  );
}
