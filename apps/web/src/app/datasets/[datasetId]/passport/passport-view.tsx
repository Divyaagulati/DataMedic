"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { VersionPicker } from "@/components/mvp/version-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { type Dict, downloadFile, fmtNum, usePassport } from "@/lib/mvp";
import { useDataset } from "@/lib/queries";

function Section({ title, data, children }: { title: string; data: Dict | undefined; children: React.ReactNode }) {
  return (
    <section className="passport-section space-y-2 border-t pt-4">
      <h2 className="text-lg font-semibold">{title}</h2>
      {data && data.status === "not_available" ? (
        <p className="text-sm text-muted-foreground">
          <Badge variant="outline" className="mr-2">
            not available
          </Badge>
          {String(data.reason)}
        </p>
      ) : (
        children
      )}
    </section>
  );
}

function Findings({ items, empty }: { items: Dict[]; empty: string }) {
  if (!items.length) return <p className="text-sm text-muted-foreground">{empty}</p>;
  return (
    <ul className="space-y-1 text-sm">
      {items.map((f, i) => (
        <li key={i}>
          <Badge variant={f.severity === "critical" || f.severity === "high" ? "destructive" : "outline"} className="mr-2">
            {String(f.severity)}
          </Badge>
          {f.basis === "statistical" ? <Badge variant="outline" className="mr-2">suspicion</Badge> : null}
          {String(f.title)}
          {(f.columns as string[])?.length ? <span className="text-muted-foreground"> — {(f.columns as string[]).slice(0, 4).join(", ")}</span> : null}
        </li>
      ))}
    </ul>
  );
}

function ExportPanel({ versionId }: { versionId: string }) {
  const [ack, setAck] = useState(false);
  const [includeSensitive, setIncludeSensitive] = useState(false);
  const [ackSensitive, setAckSensitive] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function csv() {
    setError(null);
    const { data, error: problem } = await api.POST("/api/v1/versions/{version_id}/exports", {
      params: { path: { version_id: versionId } },
      body: { format: "csv", acknowledged: ack, include_sensitive: includeSensitive, acknowledge_sensitive: ackSensitive },
    });
    if (!data) return setError(problemMessage(problem));
    setExcluded(data.excluded_sensitive_columns);
    setPreparing(true);
    let job = data.job;
    for (let i = 0; i < 300 && ["queued", "running"].includes(job.status); i++) {
      await new Promise((r) => setTimeout(r, 1500));
      const next = await api.GET("/api/v1/jobs/{job_id}", { params: { path: { job_id: job.id } } });
      if (!next.data) break;
      job = next.data;
    }
    setPreparing(false);
    if (job.status !== "succeeded") return setError(job.error?.message ?? "The export did not finish.");
    try {
      await downloadFile(`/api/v1/exports/${job.id}/download`, "export.csv");
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  async function get(path: string, name: string) {
    setError(null);
    try {
      await downloadFile(path, name);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  return (
    <Card className="no-print">
      <CardHeader>
        <CardTitle>Exports</CardTitle>
        <CardDescription>
          Files are generated for your account only and downloaded directly; no shareable links are created. The source
          version is never modified.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2 rounded-lg border p-4">
          <p className="font-medium">Cleaned dataset (CSV)</p>
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} className="mt-1" />
            I understand this file contains row-level data and I am responsible for storing it securely.
          </label>
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" checked={includeSensitive} onChange={(e) => setIncludeSensitive(e.target.checked)} className="mt-1" />
            Include sensitive columns (excluded by default)
          </label>
          {includeSensitive ? (
            <label className="flex items-start gap-2 text-sm text-destructive">
              <input type="checkbox" checked={ackSensitive} onChange={(e) => setAckSensitive(e.target.checked)} className="mt-1" />
              I confirm I am authorised to export sensitive values.
            </label>
          ) : null}
          <Button size="sm" onClick={csv} disabled={!ack || preparing || (includeSensitive && !ackSensitive)}>
            {preparing ? "Preparing…" : "Download CSV"}
          </Button>
          {excluded.length ? (
            <p className="text-xs text-muted-foreground">Excluded sensitive columns: {excluded.join(", ")}</p>
          ) : null}
          <p className="text-xs text-muted-foreground">
            Text cells that start with =, +, -, @ are prefixed with an apostrophe to prevent spreadsheet formula injection.
          </p>
        </div>
        <div className="space-y-2 rounded-lg border p-4">
          <p className="font-medium">Reports and code</p>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => window.print()}>
              Print / Save as PDF
            </Button>
            <Button size="sm" variant="outline" onClick={() => get(`/api/v1/versions/${versionId}/passport?download=true`, "passport.json")}>
              JSON report
            </Button>
            <Button size="sm" variant="outline" onClick={() => get(`/api/v1/versions/${versionId}/code`, "pipeline.py")}>
              Python pipeline (pipeline.py)
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            pipeline.py reproduces this version from its original file with the committed steps. It contains no data
            values, credentials or storage paths.
          </p>
        </div>
        <div className="md:col-span-2">
          <FormError message={error} />
        </div>
      </CardContent>
    </Card>
  );
}

export function PassportView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const dataset = useDataset(datasetId);
  const [picked, setPicked] = useState<string>();
  const versionId = picked ?? dataset.data?.head_version?.id;
  const passport = usePassport(versionId);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;
  const p = passport.data;
  const v = (p?.version ?? {}) as Dict;
  const quality = (p?.quality ?? {}) as Dict;
  const qs = (quality.summary ?? {}) as Dict;
  const diag = (p?.diagnostics ?? {}) as Dict;
  const domain = (p?.domain_findings ?? {}) as Dict;
  const recs = (p?.recommendations ?? {}) as Dict;
  const history = (p?.history ?? {}) as Dict;
  const exps = (p?.experiments ?? {}) as Dict;
  const models = (p?.models ?? {}) as Dict;
  const repro = (p?.reproducibility ?? {}) as Dict;
  const roles = ((p?.roles as Dict)?.roles ?? {}) as Record<string, string[]>;
  const bio = domain.biomedical_summary as Dict | undefined;

  return (
    <div className="space-y-6">
      <div className="no-print space-y-6">
        <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="passport" />
        <VersionPicker datasetId={datasetId} value={versionId} onChange={setPicked} />
        {versionId ? <ExportPanel versionId={versionId} /> : null}
      </div>
      {passport.error ? <FormError message={String((passport.error as Error).message)} /> : null}
      {!p ? (
        <p className="text-sm text-muted-foreground">Loading the passport…</p>
      ) : (
        <article className="passport space-y-4 rounded-xl border bg-card p-6 shadow-sm">
          <header className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-primary">DataMedic · Data Passport</p>
              <h1 className="text-2xl font-semibold">{String((p.dataset as Dict).name)}</h1>
              <p className="text-sm text-muted-foreground">
                Version {String(v.number)} ({String(v.kind)}) · domain {String((p.dataset as Dict).domain)} · generated{" "}
                {new Date(String(p.generated_at)).toLocaleString()}
              </p>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <p>Passport {String(p.passport_version)} · engine {String(p.engine_version)}</p>
              <p className="font-mono">content {String(v.content_hash).slice(0, 16)}…</p>
            </div>
          </header>
          <p className="rounded-md bg-muted/60 p-3 text-xs text-muted-foreground">
            Built from stored analyses only. Not a certification: a check that was not assessed is not a pass, statistical
            suspicions are not confirmed problems, and exploratory evaluations are likely optimistic. There is no overall
            reliability score.
          </p>

          <Section title="1. Dataset overview" data={undefined}>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm md:grid-cols-4">
              <dt className="text-muted-foreground">Rows</dt>
              <dd className="tabular-nums">{Number(v.rows).toLocaleString()}</dd>
              <dt className="text-muted-foreground">Columns</dt>
              <dd className="tabular-nums">{String(v.columns)}</dd>
              <dt className="text-muted-foreground">Fitted on all rows</dt>
              <dd>{v.fitted_on_full_data ? "yes" : "no"}</dd>
              <dt className="text-muted-foreground">Protected columns</dt>
              <dd>{((p.protected_columns as string[]) ?? []).join(", ") || "none"}</dd>
            </dl>
            <p className="text-sm">
              <span className="text-muted-foreground">Roles: </span>
              {Object.entries(roles)
                .map(([r, c]) => `${r}: ${c.join(", ")}`)
                .join(" · ") || "none declared"}
            </p>
          </Section>

          <Section title="2. Quality, missingness and outliers" data={quality}>
            <p className="text-sm">
              {String(qs.missing_cells_pct ?? "—")}% missing cells · {String(qs.duplicate_rows ?? "—")} duplicate rows ·{" "}
              {((qs.constant_columns as string[]) ?? []).length} constant column(s)
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Missing</TableHead>
                  <TableHead>Unique</TableHead>
                  <TableHead>Median (IQR)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {((quality.columns as Dict[]) ?? []).slice(0, 40).map((c) => (
                  <TableRow key={String(c.name)}>
                    <TableCell>{String(c.name)}</TableCell>
                    <TableCell>{String(c.inferred_type)}</TableCell>
                    <TableCell className="tabular-nums">{fmtNum(c.missing_pct, 1)}%</TableCell>
                    <TableCell className="tabular-nums">{String(c.n_unique ?? "—")}</TableCell>
                    <TableCell className="tabular-nums">
                      {fmtNum(c.median)} ({fmtNum(c.iqr)})
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <p className="text-xs text-muted-foreground">Outlier findings appear under diagnostics (detected issues).</p>
          </Section>

          <Section title="3. Diagnostics" data={diag}>
            {diag.stale ? <Badge variant="outline">roles changed since this run</Badge> : null}
            <p className="text-sm text-muted-foreground">
              Detected {((diag.detected as Dict[]) ?? []).length} · not detected {String(diag.not_detected_count ?? 0)} · not
              assessed {((diag.not_assessed as Dict[]) ?? []).length} · check errors {((diag.check_errors as Dict[]) ?? []).length}
            </p>
            <Findings items={(diag.detected as Dict[]) ?? []} empty="No issue detected among the assessed checks." />
            {((diag.not_assessed as Dict[]) ?? []).length ? (
              <details className="text-sm">
                <summary className="cursor-pointer">Not assessed ({((diag.not_assessed as Dict[]) ?? []).length})</summary>
                <ul className="list-disc pl-5 text-muted-foreground">
                  {((diag.not_assessed as Dict[]) ?? []).map((f, i) => (
                    <li key={i}>
                      {String(f.title)}: {String(f.not_assessed_reason ?? "")}
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </Section>

          <Section title="4. Domain-specific findings" data={domain}>
            <Findings items={((domain.findings as Dict[]) ?? []).filter((f) => f.status === "detected")} empty="No domain-specific issue detected." />
            {bio ? (
              <div className="space-y-1 text-sm">
                <p className="font-medium">Biomedical summary</p>
                <p>
                  {(bio.cohort as Dict)?.patients && ((bio.cohort as Dict).patients as Dict).status === "assessed"
                    ? `${String(((bio.cohort as Dict).patients as Dict).unique)} unique patients, ${String(((bio.cohort as Dict).patients as Dict).repeated_patients)} with repeated observations.`
                    : "Patients not assessed (no patient role)."}
                </p>
                <ul className="list-disc pl-5">
                  {((bio.confounding as Dict[]) ?? []).map((c, i) => (
                    <li key={i}>{String(c.message)}</li>
                  ))}
                </ul>
                <p className="text-xs text-muted-foreground">{String(bio.disclaimer)}</p>
              </div>
            ) : null}
          </Section>

          <Section title="5. Preprocessing recommendations" data={recs}>
            <ul className="space-y-1 text-sm">
              {((recs.items as Dict[]) ?? []).map((r, i) => (
                <li key={i}>
                  <Badge variant="outline" className="mr-2">
                    {String(r.status)}
                  </Badge>
                  {String(r.title ?? r.op)} — {((r.columns as string[]) ?? []).join(", ") || "all rows"} ({String(r.stage)}, {String(r.risk)} risk)
                </li>
              ))}
            </ul>
          </Section>

          <Section title="6. Transformation history and provenance" data={history}>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Kind</TableHead>
                  <TableHead>Rows × cols</TableHead>
                  <TableHead>Committed steps</TableHead>
                  <TableHead>Approvals</TableHead>
                  <TableHead>Content hash</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {((history.versions as Dict[]) ?? []).map((h) => {
                  const t = h.transformation as Dict | null;
                  return (
                    <TableRow key={String(h.version_number)}>
                      <TableCell>v{String(h.version_number)}</TableCell>
                      <TableCell>{String(h.kind)}</TableCell>
                      <TableCell className="tabular-nums">
                        {String(h.rows)} × {String(h.columns)}
                      </TableCell>
                      <TableCell className="text-xs">{t ? ((t.steps as Dict[]) ?? []).map((s) => String(s.op)).join(", ") : "—"}</TableCell>
                      <TableCell>{t ? String(t.approvals) : "—"}</TableCell>
                      <TableCell className="font-mono text-xs">{String(h.content_hash).slice(0, 12)}…</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Section>

          <Section title="7. Experiment comparisons" data={exps}>
            {((exps.items as Dict[]) ?? []).slice(0, 3).map((e, i) => (
              <div key={i} className="text-sm">
                <p className="font-medium">
                  {String(e.name)} · {String(e.task_type)} · {String(e.primary_metric ?? "")}
                </p>
                <ul className="list-disc pl-5 text-muted-foreground">
                  {((e.comparisons as Dict[]) ?? []).slice(0, 6).map((c, j) => (
                    <li key={j}>
                      {String(c.candidate)} vs {String(c.baseline)} ({String(c.model)}): {String(c.interpretation)}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">Comparisons describe the same folds; they are not significance tests.</p>
          </Section>

          <Section title="8. Model performance, reliability audit and explainability" data={models}>
            {((models.items as Dict[]) ?? []).slice(0, 5).map((m, i) => {
              const h = m.headline as Dict;
              const audit = m.audit as Dict | null;
              return (
                <div key={i} className="space-y-1 rounded-md border p-3 text-sm">
                  <p className="font-medium">
                    {String(m.name)}{" "}
                    <Badge variant={m.evaluation_mode === "exploratory" ? "destructive" : "secondary"}>{String(m.evaluation_label)}</Badge>
                  </p>
                  <p>
                    {h.state === "available"
                      ? `Highest observed ${String(h.label)} ${fmtNum(h.mean)} ± ${fmtNum(h.sd)} (${String(h.model)}; ${h.higher_is_better ? "higher" : "lower"} is better; dummy ${fmtNum(h.baseline_mean)}). Not a declared winner; optimistic.`
                      : `${String(h.state)}: ${String(h.reason ?? "")}`}
                  </p>
                  {audit ? (
                    <>
                      <p>
                        Audit: <strong>{String(audit.status)}</strong> — {String(audit.limitation ?? "")}
                      </p>
                      <Findings items={(audit.detected as Dict[]) ?? []} empty="No audit check detected an issue among those assessed." />
                    </>
                  ) : (
                    <p className="text-muted-foreground">Audit not available.</p>
                  )}
                  {((m.models as Dict[]) ?? [])
                    .filter((x) => x.model === h.model)
                    .map((x) => (
                      <p key={String(x.model)} className="text-xs text-muted-foreground">
                        Top permutation importance ({String(x.model)}, not causal):{" "}
                        {((x.top_importance as Dict[]) ?? []).map((t) => `${String(t.column)} ${fmtNum(t.mean)}`).join(", ") || "—"}
                      </p>
                    ))}
                </div>
              );
            })}
          </Section>

          <Section title="9. Reproducibility" data={undefined}>
            <dl className="grid gap-y-1 text-xs sm:grid-cols-[12rem_1fr]">
              <dt className="text-muted-foreground">Content hash</dt>
              <dd className="break-all font-mono">{String(v.content_hash)}</dd>
              <dt className="text-muted-foreground">File SHA-256</dt>
              <dd className="break-all font-mono">{String(v.file_sha256)}</dd>
              <dt className="text-muted-foreground">Code</dt>
              <dd>pipeline.py (Exports) — {String(repro.note)}</dd>
            </dl>
          </Section>

          <Section title="10. Scientific limitations" data={undefined}>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {((p.limitations as string[]) ?? []).map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </Section>
        </article>
      )}
    </div>
  );
}
