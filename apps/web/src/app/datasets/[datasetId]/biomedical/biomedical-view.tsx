"use client";

import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FindingCard } from "@/components/diagnostics/finding-card";
import { FormError } from "@/components/form-error";
import { VersionPicker } from "@/components/mvp/version-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import type { FindingView } from "@/lib/diagnostics";
import { type Dict, fmtNum, SEVERITY_ORDER, useBiomedical } from "@/lib/mvp";
import { useDataset } from "@/lib/queries";

function NotAssessed({ section }: { section: Dict | undefined }) {
  return (
    <p className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
      <Badge variant="outline" className="mr-2">
        not assessed
      </Badge>
      {String(section?.reason ?? "Not available.")}
    </p>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

const sevVariant = (s: string) => (s === "critical" || s === "high" ? "destructive" : "outline");

export function BiomedicalView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const dataset = useDataset(datasetId);
  const [picked, setPicked] = useState<string>();
  const versionId = picked ?? dataset.data?.head_version?.id;
  const ws = useBiomedical(versionId);
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;

  async function run() {
    if (!versionId) return;
    setBusy(true);
    setError(null);
    const { error: problem, response } = await api.POST("/api/v1/versions/{version_id}/biomedical", {
      params: { path: { version_id: versionId } },
    });
    setBusy(false);
    if (!response.ok) setError(problemMessage(problem));
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId, "biomedical"] });
  }

  const w = ws.data;
  const s = (w?.summary ?? null) as Dict | null;
  const running = w?.job && ["queued", "running"].includes(w.job.status);
  const cohort = (s?.cohort ?? {}) as Dict;
  const patients = (cohort.patients ?? {}) as Dict;
  const target = (s?.target ?? {}) as Dict;
  const biomarkers = (s?.biomarkers ?? {}) as Dict;
  const comparisons = (s?.comparisons ?? {}) as Dict;
  const batches = (s?.batches ?? {}) as Dict;
  const confounding = (s?.confounding as Dict[]) ?? [];
  const small = (s?.small_cohort as Dict[]) ?? [];
  const entities = (s?.entities ?? {}) as Dict;
  const reliability = [...((w?.model_reliability as Dict[]) ?? [])].sort(
    (a, b) => (SEVERITY_ORDER[String(a.severity)] ?? 9) - (SEVERITY_ORDER[String(b.severity)] ?? 9),
  );

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="biomedical" />
      <div role="note" className="rounded-lg border border-amber-500/50 bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
        <strong>Research use only.</strong> Descriptive statistics of this dataset. Not clinical evidence; no causal or
        diagnostic claims. Groups with fewer than 10 units are suppressed.
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <VersionPicker datasetId={datasetId} value={versionId} onChange={setPicked} />
        <div className="flex items-center gap-2">
          {w?.stale ? <Badge variant="outline">roles changed since this summary</Badge> : null}
          {ds.domain !== "biomedical" ? (
            <Badge variant="outline" title="Set the domain to biomedical on the Diagnostics tab for biomedical checks and roles">
              domain: {ds.domain}
            </Badge>
          ) : null}
          <Button size="sm" onClick={run} disabled={busy || Boolean(running) || ds.status !== "ready"}>
            {running ? "Summarising…" : s ? "Refresh summary" : "Run biomedical summary"}
          </Button>
        </div>
      </div>
      <FormError message={error} />
      {ws.error ? <FormError message={problemMessage(ws.error)} /> : null}
      {!s && !running ? (
        <Card>
          <CardContent className="space-y-2 pt-6 text-sm">
            <p className="font-medium">No biomedical summary yet.</p>
            <p className="text-muted-foreground">
              Declare the <em>target</em>, the <em>patient identifier</em> (bio.patient_id or group) and any{" "}
              <em>batch or site</em> column on the{" "}
              <Link className="underline" href={`/datasets/${datasetId}/diagnostics`}>
                Diagnostics
              </Link>{" "}
              tab, then run the summary. Missing roles are reported as not assessed, never as passed.
            </p>
          </CardContent>
        </Card>
      ) : null}
      {s ? (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Rows" value={Number(cohort.rows ?? 0).toLocaleString()} />
            <Stat
              label="Unique patients"
              value={patients.status === "assessed" ? Number(patients.unique).toLocaleString() : "—"}
              hint={patients.status === "assessed" ? `column ${String(patients.column)}` : "no patient role"}
            />
            <Stat
              label="Repeated patients"
              value={patients.status === "assessed" ? Number(patients.repeated_patients).toLocaleString() : "—"}
              hint={
                patients.status === "assessed"
                  ? `median ${String((patients.observations_per_patient as Dict)?.median)} obs/patient`
                  : undefined
              }
            />
            <Stat
              label="Minority class share"
              value={target.imbalance ? `${(Number((target.imbalance as Dict).minority_share) * 100).toFixed(1)}%` : "—"}
              hint={target.status === "assessed" ? `target ${String(target.column)}` : "no target"}
            />
          </section>

          {confounding.length || small.length ? (
            <Card>
              <CardHeader>
                <CardTitle>Warnings</CardTitle>
                <CardDescription>Potential confounding, batch effects and small-cohort limitations.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {confounding.map((c, i) => (
                  <div key={i} className="flex gap-2 text-sm">
                    <Badge variant={sevVariant(String(c.severity))}>{String(c.severity)}</Badge>
                    <span>{String(c.message)}</span>
                  </div>
                ))}
                {small.map((c, i) => (
                  <div key={`s${i}`} className="flex gap-2 text-sm">
                    <Badge variant="outline">limitation</Badge>
                    <span>{String(c.message)}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          ) : null}

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Cohort</CardTitle>
                <CardDescription>Patients and repeated observations.</CardDescription>
              </CardHeader>
              <CardContent>
                {patients.status === "assessed" ? (
                  <dl className="grid grid-cols-2 gap-y-1 text-sm">
                    <dt className="text-muted-foreground">Single-observation patients</dt>
                    <dd className="tabular-nums">{String(patients.single_observation_patients)}</dd>
                    <dt className="text-muted-foreground">Observations per patient</dt>
                    <dd className="tabular-nums">
                      {String((patients.observations_per_patient as Dict).min)}–{String((patients.observations_per_patient as Dict).max)}
                    </dd>
                    <dt className="text-muted-foreground">Rows without patient id</dt>
                    <dd className="tabular-nums">{String(patients.rows_without_id)}</dd>
                    {patients.patients_with_changing_target !== undefined ? (
                      <>
                        <dt className="text-muted-foreground">Patients whose outcome changes</dt>
                        <dd className="tabular-nums">{String(patients.patients_with_changing_target)}</dd>
                      </>
                    ) : null}
                  </dl>
                ) : (
                  <NotAssessed section={patients} />
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Target distribution</CardTitle>
                <CardDescription>Class balance (counts under 10 suppressed).</CardDescription>
              </CardHeader>
              <CardContent>
                {target.status !== "assessed" ? (
                  <NotAssessed section={target} />
                ) : target.classes ? (
                  <ul className="space-y-2 text-sm">
                    {(target.classes as Dict[]).map((c) => {
                      const total = (target.classes as Dict[]).reduce((a, x) => a + Number(x.n ?? 0), 0);
                      const pct = c.n ? (100 * Number(c.n)) / total : 0;
                      return (
                        <li key={String(c.label)}>
                          <div className="flex justify-between">
                            <span>{String(c.label)}</span>
                            <span className="tabular-nums">{c.suppressed ? "<10" : String(c.n)}</span>
                          </div>
                          <div className="h-2 rounded bg-muted">
                            <div className="h-2 rounded bg-primary" style={{ width: `${pct}%` }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <pre className="text-xs">{JSON.stringify(target.distribution, null, 1)}</pre>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Biomarker comparisons</CardTitle>
              <CardDescription>
                Descriptive effect sizes across outcomes ({String(comparisons.note ?? "")}); q-values adjust for multiple
                comparisons (Benjamini–Hochberg). Associations are not causal.
              </CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              {comparisons.status !== "assessed" ? (
                <NotAssessed section={comparisons} />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Biomarker</TableHead>
                      <TableHead>Effect</TableHead>
                      <TableHead>Value</TableHead>
                      <TableHead>Groups (n, mean)</TableHead>
                      <TableHead>p</TableHead>
                      <TableHead>q</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {((comparisons.items as Dict[]) ?? []).map((c) => (
                      <TableRow key={String(c.column)}>
                        <TableCell className="font-medium">{String(c.column)}</TableCell>
                        {c.status === "assessed" ? (
                          <>
                            <TableCell>{String((c.effect as Dict).name).replaceAll("_", " ")}</TableCell>
                            <TableCell className="tabular-nums">{fmtNum((c.effect as Dict).value)}</TableCell>
                            <TableCell className="text-xs">
                              {((c.groups as Dict[]) ?? []).map((g) => `${String(g.label)}: ${String(g.n)}, ${fmtNum(g.mean)}`).join(" · ") ||
                                (c.n ? `n=${String(c.n)}` : "")}
                            </TableCell>
                            <TableCell className="tabular-nums">{fmtNum(c.p_value, 4)}</TableCell>
                            <TableCell className="tabular-nums">{fmtNum(c.q_value, 4)}</TableCell>
                          </>
                        ) : (
                          <TableCell colSpan={5} className="text-xs text-muted-foreground">
                            not assessed: {String(c.reason)}
                          </TableCell>
                        )}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Biomarker distributions, missingness and outliers</CardTitle>
              <CardDescription>Outliers by the 1.5 × IQR rule (3 × IQR in brackets); flagged, never removed.</CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              {biomarkers.status !== "assessed" ? (
                <NotAssessed section={biomarkers} />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Column</TableHead>
                      <TableHead>Missing</TableHead>
                      <TableHead>Median [IQR]</TableHead>
                      <TableHead>Mean ± SD</TableHead>
                      <TableHead>Outliers</TableHead>
                      <TableHead title="Mean within-patient SD / overall SD">Within-patient SD ratio</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {((biomarkers.items as Dict[]) ?? []).map((b) => (
                      <TableRow key={String(b.column)}>
                        <TableCell className="font-medium">{String(b.column)}</TableCell>
                        <TableCell className="tabular-nums">{fmtNum(b.missing_pct, 1)}%</TableCell>
                        <TableCell className="tabular-nums">
                          {fmtNum(b.median)} [{fmtNum(b.q1)}, {fmtNum(b.q3)}]
                        </TableCell>
                        <TableCell className="tabular-nums">
                          {fmtNum(b.mean)} ± {fmtNum(b.sd)}
                        </TableCell>
                        <TableCell className="tabular-nums">
                          {String(b.outliers_iqr_1_5 ?? "—")} ({String(b.outliers_iqr_3 ?? "—")})
                        </TableCell>
                        <TableCell className="tabular-nums">{fmtNum(b.within_patient_sd_ratio, 2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Site and batch effects</CardTitle>
              <CardDescription>Association of each batch/site with the outcome and with each biomarker.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {batches.status !== "assessed" ? (
                <NotAssessed section={batches} />
              ) : (
                ((batches.items as Dict[]) ?? []).map((b) => {
                  const assoc = b.outcome_association as Dict | null;
                  return (
                    <div key={String(b.column)} className="space-y-2">
                      <p className="text-sm font-medium">
                        {String(b.column)} · {String(b.n_groups)} levels
                        {assoc ? ` · outcome ${String(assoc.name).replaceAll("_", " ")} ${fmtNum(assoc.value)} (p ${fmtNum(assoc.p_value, 4)})` : ""}
                      </p>
                      <div className="flex flex-wrap gap-2 text-xs">
                        {((b.groups as Dict[]) ?? []).map((g) => (
                          <Badge key={String(g.label)} variant="outline">
                            {String(g.label)}: {g.suppressed ? "<10" : String(g.n)}
                          </Badge>
                        ))}
                      </div>
                      <ul className="text-xs text-muted-foreground">
                        {((b.biomarker_shifts as Dict[]) ?? []).slice(0, 6).map((x) => (
                          <li key={String(x.column)}>
                            {String(x.column)}: epsilon² {fmtNum(x.value)} (p {fmtNum(x.p_value, 4)})
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })
              )}
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Patient-level leakage and identifiers</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {(entities.patient_leakage as Dict)?.status === "assessed" ? (
                  <p>{String((entities.patient_leakage as Dict).message)}</p>
                ) : (
                  <NotAssessed section={entities.patient_leakage as Dict} />
                )}
                {((entities.possible_identifiers as Dict[]) ?? []).length ? (
                  <p>
                    <Badge variant="outline" className="mr-2">
                      possible
                    </Badge>
                    Undeclared columns that look like entity keys (evidence, not confirmed):{" "}
                    {((entities.possible_identifiers as Dict[]) ?? []).map((k) => String(k.column)).join(", ")}
                  </p>
                ) : null}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Model reliability (biomedical)</CardTitle>
                <CardDescription>From stored training runs and their audits.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {reliability.map((r, i) => (
                  <div key={i} className="flex gap-2">
                    <Badge variant={r.status === "detected" ? sevVariant(String(r.severity)) : "outline"}>
                      {r.status === "not_assessed" ? "not assessed" : String(r.severity)}
                    </Badge>
                    <span>
                      <strong>{String(r.title)}</strong>
                      {r.run_name ? ` (${String(r.run_name)})` : ""}: {String(r.detail)}
                    </span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Biomedical diagnostics findings</CardTitle>
          <CardDescription>From the latest diagnostics run of this version.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {!w?.diagnostics_available ? (
            <p className="text-sm text-muted-foreground">Diagnostics have not been run on this version.</p>
          ) : (w?.biomedical_findings ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No biomedical checks ran (set the dataset domain to biomedical on the Diagnostics tab).
            </p>
          ) : (
            (w?.biomedical_findings ?? []).map((f) => <FindingCard key={f.id} finding={f as FindingView} />)
          )}
        </CardContent>
      </Card>
      {s ? (
        <ul className="list-disc space-y-1 pl-5 text-xs text-muted-foreground">
          {((s.limitations as string[]) ?? []).map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
