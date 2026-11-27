"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { StatusBadge } from "@/components/status-badge";
import { ModeBadge, NewTrainingRun } from "@/components/training/run-config";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { problemMessage } from "@/lib/api/problem";
import { useDataset, useOps, useTrainingRuns, useVersionList } from "@/lib/queries";
import { headlineView, STATUS_INFO } from "@/lib/training";

export function ModelsView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const router = useRouter();
  const dataset = useDataset(datasetId);
  const versions = useVersionList(datasetId);
  const runs = useTrainingRuns(datasetId);
  const ops = useOps();

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const numbers = new Map((versions.data ?? []).map((v) => [v.id, v.version_number]));

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="models" />
      {ds.status !== "ready" ? (
        <p className="text-sm text-muted-foreground">Models can be trained once the dataset is ready.</p>
      ) : (
        <NewTrainingRun
          versions={[...(versions.data ?? [])].reverse()}
          headId={ds.head_version?.id}
          ops={ops.data ?? []}
          onStarted={(id) => router.push(`/datasets/${datasetId}/models/${id}`)}
        />
      )}
      <Card>
        <CardHeader>
          <CardTitle>Training runs</CardTitle>
          <CardDescription>
            The headline is the highest observed mean of the primary metric (mean ± standard deviation over the outer
            test folds) among the compared models, excluding the dummy baseline. It is optimistic because it was chosen on
            the same folds, and no model is declared the winner. Exploratory runs are not leakage-controlled.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {runs.error ? <FormError message={problemMessage(runs.error)} /> : null}
          {runs.data && runs.data.length === 0 ? (
            <p className="text-sm text-muted-foreground">No training runs yet.</p>
          ) : null}
          {runs.data?.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Version</TableHead>
                  <TableHead>Evaluation</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Headline</TableHead>
                  <TableHead>Audit</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.data.map((r) => {
                  const audit = r.audit ? STATUS_INFO[r.audit.status] : null;
                  const h = headlineView(r.headline);
                  return (
                    <TableRow key={r.id}>
                      <TableCell>
                        <Link className="underline" href={`/datasets/${datasetId}/models/${r.id}`}>
                          {r.name}
                        </Link>
                      </TableCell>
                      <TableCell>v{numbers.get(r.version_id) ?? "?"}</TableCell>
                      <TableCell>
                        <ModeBadge mode={r.evaluation_mode} />
                      </TableCell>
                      <TableCell>{r.job ? <StatusBadge status={r.job.status} /> : "—"}</TableCell>
                      <TableCell className="tabular-nums" title={h.title}>
                        {h.text}
                        {h.qualifier ? (
                          <div>
                            <Badge variant="destructive">{h.qualifier}</Badge>
                          </div>
                        ) : null}
                      </TableCell>
                      <TableCell>
                        {audit ? (
                          <Badge variant={audit.tone === "bad" ? "destructive" : audit.tone === "ok" ? "secondary" : "outline"}>
                            {audit.label}
                          </Badge>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
