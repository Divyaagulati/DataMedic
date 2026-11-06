"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { FormError } from "@/components/form-error";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, unwrap } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { ACTIVE_DATASET_STATUSES, formatBytes } from "@/lib/format";

const POLL_MS = 2000;

export function DatasetDetail() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const dataset = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/datasets/{dataset_id}", {
          params: { path: { dataset_id: datasetId } },
        }),
      ),
    refetchInterval: (query) =>
      query.state.data && ACTIVE_DATASET_STATUSES.has(query.state.data.status) ? POLL_MS : false,
  });
  const jobs = useQuery({
    queryKey: ["jobs", { datasetId }, dataset.data?.status],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/jobs", { params: { query: { dataset_id: datasetId } } }))
        .items,
    enabled: dataset.isSuccess,
  });

  async function onDelete() {
    if (!window.confirm("Delete this dataset and all its stored files? This cannot be undone.")) {
      return;
    }
    setDeleting(true);
    const { error: problem, response } = await api.DELETE("/api/v1/datasets/{dataset_id}", {
      params: { path: { dataset_id: datasetId } },
    });
    if (response.status === 202) {
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      router.push("/datasets");
    } else {
      setDeleteError(problemMessage(problem));
      setDeleting(false);
    }
  }

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;

  const ds = dataset.data;
  const jobList = jobs.data ?? [];
  const lastIngest = jobList.find((j) => j.type === "ingest");

  return (
    <div className="space-y-6">
      <Link href="/datasets" className="text-sm underline">
        ← All datasets
      </Link>
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">{ds.name}</h1>
        <StatusBadge status={ds.status} />
      </div>
      <FormError message={deleteError} />
      <Card>
        <CardHeader>
          <CardTitle>File</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-[10rem_1fr] gap-y-2 text-sm">
            <dt className="text-muted-foreground">Original file</dt>
            <dd>{ds.filename}</dd>
            <dt className="text-muted-foreground">Size</dt>
            <dd>{formatBytes(ds.size_bytes)}</dd>
            <dt className="text-muted-foreground">Detected format</dt>
            <dd>{ds.detected_format ?? "—"}</dd>
            <dt className="text-muted-foreground">Uploaded</dt>
            <dd>{new Date(ds.created_at).toLocaleString()}</dd>
          </dl>
          {lastIngest?.error ? (
            <div className="mt-4">
              <FormError message={`${lastIngest.error.code}: ${lastIngest.error.message}`} />
            </div>
          ) : null}
          {ds.status === "validated" ? (
            <p className="mt-4 text-sm text-muted-foreground">
              The file passed safety and format validation. Parsing, diagnostics and
              recommendations arrive in later phases.
            </p>
          ) : null}
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
                <li key={j.id} className="flex items-center justify-between">
                  <span>
                    {j.type} · {new Date(j.queued_at).toLocaleString()}
                  </span>
                  <StatusBadge status={j.status} />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
      <Button variant="destructive" onClick={onDelete} disabled={deleting}>
        {deleting ? "Deleting…" : "Delete dataset"}
      </Button>
    </div>
  );
}
