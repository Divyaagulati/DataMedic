import Link from "next/link";

import { Button } from "@/components/ui/button";

const WORKFLOW = [
  { step: "Upload", text: "CSV, Excel, Parquet or JSON, parsed in a sandbox into an immutable version 0." },
  { step: "Explore", text: "Analytics Studio: no-code charts, KPI cards, pivot tables, dashboards and questions in plain English." },
  { step: "Understand", text: "Profiling and automated, evidence-based insights with methods and limitations." },
  { step: "Diagnose", text: "Universal and domain checks with evidence, severity and honest not-assessed states." },
  { step: "Experiment", text: "Compare preprocessing candidates on frozen, leakage-safe folds." },
  { step: "Transform", text: "Approved, hash-bound pipelines create new immutable versions." },
  { step: "Train", text: "Baseline AutoML with nested selection and in-fold preprocessing." },
  { step: "Audit", text: "The ML Reliability Auditor: leakage, overfitting, calibration, dummy baselines." },
  { step: "Export", text: "Data Passport, JSON report, cleaned CSV and reproducible pipeline.py." },
];

const FEATURES = [
  {
    title: "Privacy by design",
    text: "Row-level security per user, private storage, sensitive-column protection across versions and results, and no dataset content sent to external AI.",
  },
  {
    title: "Domain intelligence",
    text: "Biomedical, financial, business and industrial modules. Biomedical Intelligence Lite covers cohorts, biomarkers, batch effects and patient-level leakage.",
  },
  {
    title: "Leakage-controlled ML",
    text: "Grouped and time-aware folds, in-fold fitting, replay of committed steps and explicit exploratory labels — no overconfident claims.",
  },
  {
    title: "Data Passport",
    text: "One consolidated, printable report of quality, findings, provenance, experiments, models, audits and limitations — without an arbitrary score.",
  },
  {
    title: "Analytics Studio",
    text: "Interactive charts, filters, KPI cards, pivot tables and saved dashboards on immutable versions — with privacy-safe aggregation and honest sampling notes.",
  },
  {
    title: "Conversational analytics",
    text: "Ask “average sales by region” and get a calculated answer, the interpreted query and a chart. Numbers come from the query engine, never from guesswork.",
  },
  {
    title: "Reproducible by default",
    text: "Content hashes, immutable versions, approvals, recorded seeds and library versions, and generated Python code for every version.",
  },
];

export default function Home() {
  return (
    <div className="space-y-16 pb-8">
      <section className="relative -mx-4 -mt-8 overflow-hidden bg-[var(--brand-navy)] px-4 py-20 text-white sm:rounded-b-3xl">
        <div
          aria-hidden
          className="pointer-events-none absolute -right-24 -top-24 h-96 w-96 rounded-full bg-[var(--brand-teal)] opacity-20 blur-3xl"
        />
        <div className="relative mx-auto max-w-4xl space-y-6 text-center">
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[var(--brand-teal)]">
            Data intelligence · ML reliability
          </p>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Trustworthy data and models, from upload to audit.
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-white/75">
            DataMedic is a privacy-first, domain-aware data intelligence and ML reliability platform. It diagnoses your
            data, tests fixes without leakage, trains baselines honestly and tells you what you can — and cannot — trust.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Button asChild size="lg" className="bg-[var(--brand-teal)] text-[var(--brand-navy)] hover:bg-[var(--brand-teal)]/90">
              <Link href="/signup">Create a free account</Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-white/30 bg-transparent text-white hover:bg-white/10 hover:text-white">
              <Link href="/login">Sign in</Link>
            </Button>
          </div>
          <p className="text-xs text-white/50">
            Demonstration MVP: use synthetic or non-sensitive data. Research use only; not a medical device.
          </p>
        </div>
      </section>

      <section className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-semibold tracking-tight">One workflow, end to end</h2>
          <p className="text-muted-foreground">Every step is versioned, explained and reproducible.</p>
        </div>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {WORKFLOW.map((w, i) => (
            <li key={w.step} className="relative rounded-xl border bg-card p-4 shadow-sm">
              <div className="mb-2 flex items-center gap-2">
                <span className="grid h-7 w-7 place-items-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
                  {i + 1}
                </span>
                <span className="font-semibold">{w.step}</span>
                {i < WORKFLOW.length - 1 ? (
                  <span aria-hidden className="ml-auto text-[var(--brand-teal)]">
                    →
                  </span>
                ) : null}
              </div>
              <p className="text-sm text-muted-foreground">{w.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {FEATURES.map((f) => (
          <div key={f.title} className="rounded-xl border bg-card p-5 shadow-sm">
            <div className="mb-3 h-1 w-10 rounded bg-[var(--brand-teal)]" />
            <h3 className="font-semibold">{f.title}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{f.text}</p>
          </div>
        ))}
      </section>

      <section className="rounded-2xl border bg-secondary/60 p-8 text-center">
        <h2 className="text-xl font-semibold">Honest by design</h2>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-muted-foreground">
          A check that could not run is reported as not assessed, never as passed. Statistical suspicions are labelled as
          such. Exploratory evaluations are flagged as optimistic. There is no single magic reliability score.
        </p>
        <Button asChild className="mt-4">
          <Link href="/signup">Start with a synthetic dataset</Link>
        </Button>
      </section>
    </div>
  );
}
