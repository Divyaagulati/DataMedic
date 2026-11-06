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
      <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
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
            <p className="text-sm text-muted-foreground">No datasets yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>File</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.data.map((d) => (
                  <TableRow key={d.id}>
                    <TableCell>
                      <Link href={`/datasets/${d.id}`} className="font-medium underline">
                        {d.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{d.filename}</TableCell>
                    <TableCell>{formatBytes(d.size_bytes)}</TableCell>
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
