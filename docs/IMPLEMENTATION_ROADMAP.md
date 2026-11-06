# DataMedic — Implementation Roadmap

| | |
|---|---|
| Status | Revision 2. **Phase 1 starts only after explicit approval.** |
| Assumption | One developer, ~15–20 h/week. Estimates are rough, about ±30%. |
| Principle | Every phase ends with a demoable vertical slice, green tests and updated docs. Golden datasets grow alongside checks. |

## 1. Release tiers

### MVP (Phases 1–10, ~24 weeks)
Everything in [PRD.md §10 "MVP"](PRD.md): accounts and isolation, quotas, streaming upload and hardened ingest, roles, universal and biomedical clinical diagnostics, recommendations, experiments, impact preview, approvals, commit, linear versions and revert, leakage audit, Data Health Passport, ML-readiness and validated export.

### Version 2 (after the MVP works)
YData Profiling (fg-data-profiling), PyOD and Evidently adapters; forks and version graph; version deletion; locked holdout, bootstrap CIs, calibration, random forest; opt-in LLM gateway; TOTP MFA; PDF passport; SSE; presigned large uploads; sandbox container; biomedical omics module (Scanpy, ComBat) with `worker-bio`; aggregation ops.

### Future Enhancements
Teams, sharing and SSO; public API; financial, business and industrial modules; connectors; on-prem distribution; entry-point plugin discovery; hyper-parameter tuning.

## 2. MVP phases

| # | Phase | ~Weeks | Demo at the end |
|---|---|---|---|
| 1 | Foundations & compatibility spike | 1.5 | `docker compose up` runs every service; CI green |
| 2 | Authentication, isolation, quotas | 2 | Two users cannot see each other's data |
| 3 | Upload, ingest, versions v0, jobs | 3 | Upload any of the 5 formats → preview |
| 4 | Roles + check framework + core universal checks | 3 | First diagnosis |
| 5 | Remaining universal checks + biomedical clinical module | 2.5 | Biomedical diagnosis |
| 6 | Recommendations, op registry, impact preview, approvals, commit, revert | 3.5 | Clean → v1 → revert |
| 7 | Splits + experiments | 3 | "Did cleaning help?" |
| 8 | Leakage audit + label noise | 1.5 | Leak caught |
| 9 | Passport, readiness, codegen, export validation | 2.5 | Full journey with passport |
| 10 | Hardening & launch | 1.5 | Portfolio-ready deployment |

### Phase 1 status (2026-11-06) — implemented, as scoped by the owner
Phase 1 was widened by the owner to include parts of Phases 2–3. Delivered:
- Repo, uv-locked backend (Python 3.12), Next.js 16 frontend, docker-compose, CI workflow.
- Auth (signup, email verification, login/logout, server-side sessions, CSRF, progressive login delay, password reset, session list/revoke).
- `owner_id` tenancy with PostgreSQL RLS, owner-consistency and immutability triggers, `claim_job()`.
- Private storage interface with a development filesystem adapter (the S3 adapter is pending).
- Streamed upload (200 MB), magic-byte and XLSX-container validation in an `ingest` job (dataset → `validated`). **Parsing to version 0 is not done** and moves to Phase 3.
- Jobs table, Celery publisher (job-ID-only messages), worker runner, requeue and reaper, polling API, soft delete + purge.
- Health endpoints and the route-generated user-isolation test matrix.

Not done in Phase 1, and moved to later phases: the dependency compatibility spike for ML libraries (C1–C5, C8, C10; first needed in Phase 4), `plan_limits` / `user_usage` tables (limits are settings for now), audit events, the per-IP rate limit, deletion receipts and tombstones, and the S3 storage adapter.

### Phase 1 — Foundations & compatibility spike (original plan)
- Initialise a **dedicated git repository in this directory**, because the current enclosing repository is the user's home directory.
- Monorepo layout ([ARCHITECTURE.md §4](ARCHITECTURE.md)); `uv` workspace; Python 3.12; Next.js + TypeScript + Tailwind + shadcn/ui per the official shadcn/ui installation docs.
- docker-compose: postgres, valkey, storage (dev backend), mail catcher, api, worker, beat, web.
- **Compatibility spike** (blocking): smoke scripts using APIs copied from the official docs for C1–C9 (§4).
- **Storage decision**: dev backend must support multipart upload, `copy_object`, prefix listing and deletion, and ideally prefix-scoped credentials.
- CI: ruff, mypy (strict for engine and contracts), ESLint, tsc, pytest, Vitest, SBOM, license gate, pip-audit, npm audit, OpenAPI drift.
- Problem-details framework; log scrubber; `request_id` middleware.
- **Exit**: `docs/COMPATIBILITY_REPORT.md` + committed lockfiles; compose healthy; CI green.

### Phase 2 — Authentication, isolation, quotas
- Users, sessions, CSRF, signup/verify/login/logout/reset, progressive delays, rate limits (Valkey).
- `owner_id` + RLS scaffolding, the `app.user_id` session hook, `authorize()` function, DB roles.
- **AuthZ matrix harness** generated from the route table.
- `plan_limits`, `user_usage`, admission-control helpers, `/account/usage`.
- Frontend: auth pages, app shell, settings (account, security, usage).
- **Exit**: auth e2e green; RLS fail-closed test; AuthZ matrix running in CI.

### Phase 3 — Upload, ingest, versions v0, jobs
- `POST /datasets` + streaming `PUT /content` (multipart to storage, SHA-256 while streaming, 200 MB enforcement).
- Jobs table, `claim_job`, JSON-only Celery, requeue and reaper, polling endpoint, cancellation, admission limits.
- Ingest worker: magic bytes, ZIP inspection, defusedxml, xlrd, subprocess parsing with rlimits, cell and column limits, `_dm_row_id`, canonical Parquet, `file_sha256` + `content_hash`, preview artifact.
- Reparse (new version, `REPARSE_NOT_ALLOWED` rule), sheets, parse options UI.
- Deletion: soft delete, purge, receipts, prefix verification.
- `fixtures/malicious`; broker-message privacy test.
- **Exit**: upload wizard works for all formats; malicious fixtures rejected; deletion verified.

### Phase 4 — Roles + check framework + core universal checks
- Roles API and UI (with `task_type` on the target), role suggestions, semantics hash and staleness.
- Check framework: requirements, `not_run`, sampling, error isolation, cache, persistence.
- Ports with MVP adapters (`BuiltinProfiler`, `SklearnIsolationForest`, `ScipyDistributionTests`).
- About 10 universal checks (schema, missingness, duplicates, variance, cardinality, target).
- Findings UI (health summary, tabs, evidence, limitations, stale banner).
- Golden universal datasets + clean control; registry-wide `not_run` honesty test.
- `DomainModule` interface + in-code registry + **stub `fin` module test**.
- **Exit**: first diagnosis demo; golden tests green.

### Phase 5 — Remaining universal checks + biomedical clinical module
- Remaining universal checks (categories, dates, outliers, distribution, correlation and VIF, encoding, units, PII scan).
- `datamedic_biomedical`: roles, suggesters, the clinical checks in [DOMAIN_MODULES.md §4.3](DOMAIN_MODULES.md), reference-range API and UI, PHI extension.
- Biomedical golden datasets; domain contract test kit.
- **Exit**: biomedical journey 1 (PRD §8) demo.

### Phase 6 — Recommendations, op registry, impact preview, approvals, commit, revert
- Rule engine with universal and biomedical rules; recommendations UI.
- Op registry (MVP catalogue, including risk and destructive flags); `_dm_row_id` preservation property tests.
- Pipeline builder; `spec_hash` maintenance; validation; **impact preview job + UI**; **Destructive** badges; high-risk approval dialog; pipeline confirmation.
- Commit worker with re-verification, advisory lock, step reports, fitted-state HMAC, version creation, auto re-analysis.
- Revert (server-side copy), version timeline, detail, SciPy comparison.
- **Exit**: bypass tests (unapproved, stale, edited, concurrent, non-head source) all rejected; revert demo.

### Phase 7 — Splits + experiments
- Split recommendation; frozen `_dm_row_id` fold assignment (grouped and time-aware).
- Cohort + duplicate-aware grouping before splitting; experiment runner (candidates × models × folds); in-fold fitting; row removal on training rows only; intact test folds; minimal model preprocessing; metrics; training-rows-removed reporting; comparability check (`NOT_DIRECTLY_COMPARABLE`); paired per-fold differences between comparable candidates; warnings; risk labelling.
- **Spy-transformer leakage tests**; experiment-isolation test (no version writes).
- Experiment UI; "candidate → pipeline" handoff.
- **Exit**: on `bio_clinical_patient_leak`, random CV is inflated and group CV is not, and the UI shows both.

### Phase 8 — Leakage audit + label noise
- Leakage checks ([ML_ENGINE_DESIGN.md §6](ML_ENGINE_DESIGN.md)), including biomedical patient overlap; remediation rules.
- Cleanlab label-noise check on OOF probabilities. If the schedule slips, this moves to V2 and stays unregistered.
- **Exit**: seeded leaks detected; unassessable checks report `not_run`.

### Phase 9 — Passport, readiness, codegen, export
- ML-readiness checklist (RD-01…RD-11) and `/readiness`.
- Passport assembly (JSON), Jinja2 HTML, passport page + history + downloads.
- Codegen from `replay_chain`, per-op templates, requirements pinning, manifest.
- Export bundle; subprocess replay validation with tolerance; CSV formula-injection warning; streamed downloads; expiry.
- Codegen-equivalence tests per op and per golden lineage (including reparse and revert); hostile-column-name tests.
- **Exit**: full PRD §8 journeys 1–5 pass end to end in Playwright.

### Phase 10 — Hardening & launch
- Performance benchmarks at reference sizes; tune sampling budgets and worker limits.
- Accessibility (WCAG 2.2 AA), CSP and headers, threat-model test review.
- Deployment: managed Postgres and S3-compatible storage, TLS, reverse proxy with body limit, SMTP provider, backups + restore test.
- README, architecture diagram, demo video, license inventory.
- **Exit**: public demo deployment with synthetic data only.

## 3. Version 2 order (indicative)
1. Ports → YData Profiling (fg-data-profiling) / PyOD / Evidently adapters (one compatibility spike).
2. Holdout, bootstrap CIs, calibration, random forest.
3. TOTP MFA; SSE; PDF passports.
4. Forks, version graph and version deletion.
5. Presigned large-file uploads; sandbox container.
6. Opt-in LLM gateway with the canary CI test.
7. Biomedical omics module + `worker-bio` + ComBat safeguards; aggregation ops.

## 4. Dependencies needing compatibility checks

| # | Item | Tier | Why |
|---|---|---|---|
| C1 | pandas 2.3.x vs 3.x decision | MVP | pandas 3.x is current; Feature-engine's support for it is unverified; the V2 profiler requires `<3.0` |
| C2 | NumPy 2.x across scikit-learn, XGBoost, Feature-engine, Cleanlab, statsmodels | MVP | |
| C3 | scikit-learn 1.9.x vs Feature-engine 1.9.x, Cleanlab 2.9.x, XGBoost sklearn API | MVP | sklearn API evolution (`set_output`, tags) |
| C4 | Python 3.12 wheels for every MVP dependency | MVP | Floor set by XGBoost |
| C5 | Cleanlab license in the pinned release (Apache-2.0) | MVP | Earlier AGPL releases |
| C6 | Celery/kombu with Valkey: ack-late, visibility timeout, JSON serializer | MVP | Native `valkey://` scheme unmerged at research time |
| C7 | Dev S3 backend: multipart, `copy_object`, prefix ops, scoped credentials | MVP | MinIO community archived |
| C8 | XGBoost Linux wheel with `nvidia-nccl-cu13` (image size) | MVP | |
| C9 | Next.js / React / Tailwind majors vs the shadcn/ui CLI | MVP | |
| C10 | SciPy `false_discovery_control` availability in the pinned version | MVP | Used for FDR (with an in-house fallback) |
| C11 | YData Profiling (fg-data-profiling): rename, `pandas<3.0`, `numpy<2.6`, `scipy<1.19`, config keys for disabling samples | V2 | |
| C12 | PyOD + numba on Python 3.12 / NumPy 2.x | V2 | |
| C13 | Evidently 0.7.x API | V2 | Frequent API changes |
| C14 | Scanpy 1.12.x + anndata (`pp.combat` with covariates) | V2 | |

## 5. Risks

| Risk | Mitigation |
|---|---|
| False positives erode trust | Golden suites with clean controls; PRD §9 targets; visible limitations |
| Codegen drift from platform results | Equivalence tests per op and lineage; tolerance-based validation |
| Scope creep | Tiers are fixed in the PRD; V2 items are not started before MVP Phase 10 |
| Library churn | Ports and adapters; pinned lockfile; compatibility report |
| Schedule overrun | Pre-agreed cut order: (1) Cleanlab label noise → V2, (2) `universal.scale.units` and `bio.cohort.subgroup_balance` → V2, (3) XGBoost baseline → V2 (HGB remains) |
