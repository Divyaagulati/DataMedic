"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";

import { useAuth } from "@/components/auth-provider";
import { FormError, FormNotice } from "@/components/form-error";
import { StatusBadge } from "@/components/status-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api, unwrap } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { ACTIVE_DATASET_STATUSES, formatBytes } from "@/lib/format";

import { UploadCard } from "./upload-card";

const POLL_MS = 2000;

export function DatasetsView() {
  const { user, loading } = useAuth();
  const queryClient = useQueryClient();
  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => unwrap(await api.GET("/api/v1/datasets")).items,
    enabled: Boolean(user),
    // Poll only while some dataset is still being processed.
    refetchInterval: (query) =>
      query.state.data?.some((d) => ACTIVE_DATASET_STATUSES.has(d.status)) ? POLL_MS : false,
  });

  if (loading) return null;
  if (!user) return <FormNotice message="Please sign in to see your datasets." />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
          <p className="text-sm text-muted-foreground">
            Upload → Understand → Diagnose → Experiment → Transform → Train → Audit → Export
          </p>
        </div>
        {datasets.data?.length ? (
          <div className="flex gap-3 text-sm">
            <div className="rounded-lg border bg-card px-4 py-2">
              <p className="text-xs text-muted-foreground">Datasets</p>
              <p className="text-lg font-semibold tabular-nums">{datasets.data.length}</p>
            </div>
            <div className="rounded-lg border bg-card px-4 py-2">
              <p className="text-xs text-muted-foreground">Ready</p>
              <p className="text-lg font-semibold tabular-nums">
                {datasets.data.filter((d) => d.status === "ready").length}
              </p>
            </div>
            <div className="rounded-lg border bg-card px-4 py-2">
              <p className="text-xs text-muted-foreground">Rows</p>
              <p className="text-lg font-semibold tabular-nums">
                {datasets.data.reduce((n, d) => n + (d.head_version?.n_rows ?? 0), 0).toLocaleString()}
              </p>
            </div>
          </div>
        ) : null}
      </div>
      {!user.email_verified ? (
        <FormNotice message="Verify your email address (check your inbox) before uploading data." />
      ) : (
        <UploadCard
          onUploaded={() => void queryClient.invalidateQueries({ queryKey: ["datasets"] })}
        />
      )}
      <FormError message={datasets.error ? problemMessage(datasets.error) : null} />
      <Card>
        <CardHeader>
          <CardTitle>Your datasets</CardTitle>
        </CardHeader>
        <CardContent>
          {datasets.isPending ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : !datasets.data || datasets.data.length === 0 ? (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              <p className="font-medium text-foreground">No datasets yet</p>
              <p>Upload a CSV, Excel, Parquet or JSON file above. Use synthetic or non-sensitive data in this demo.</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>File</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Rows × columns</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.data.map((d) => (
                  <TableRow key={d.id}>
                    <TableCell>
                      <Link href={`/datasets/${d.id}`} className="font-medium text-primary hover:underline">
                        {d.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{d.filename}</TableCell>
                    <TableCell>{formatBytes(d.size_bytes)}</TableCell>
                    <TableCell className="tabular-nums">
                      {d.head_version
                        ? `${d.head_version.n_rows.toLocaleString()} × ${d.head_version.n_cols}`
                        : "—"}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={d.status} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
