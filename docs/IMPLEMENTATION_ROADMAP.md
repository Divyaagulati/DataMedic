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

### Phase 2 status (2026-11-09) — "Dataset parsing and universal profiling", as scoped by the owner
The owner re-scoped the second delivery. Authentication and isolation (this table's Phase 2) were already delivered in Phase 1. Phase 2 therefore took the parsing half of Phase 3, plus a descriptive profile behind the `Profiler` port from Phase 4. It does not include the check/findings framework.

Delivered:
- **Parsing to version 0** for CSV/TSV/TXT, XLSX, XLS, Parquet and JSON (array of records, JSON Lines, column-oriented, `{"key": [records]}` wrappers; nested objects flattened as `a.b`). The parse runs in a **sandbox subprocess** (`python -I`, minimal environment, single-threaded native libraries). On Linux it runs under RLIMIT_AS, CPU, FSIZE and CORE limits, with a wall-clock timeout everywhere. Cell (20M), column (2,000) and JSON-depth (10) limits apply. Child output is never logged.
- Canonicalisation: unique non-empty column names (originals kept), a documented missing-token list, conservative type inference (numeric, boolean, ISO-8601 datetime, categorical, text, identifier, empty), `_dm_row_id`, canonical Parquet, `file_sha256` + `content_hash` (§9.3 of ARCHITECTURE).
- `dataset_versions` (v0, kind `ingest`), `datasets.head_version_id`, a preview artifact (**first 20 rows**, owner decision; was 100). All are insert-only, RLS-protected and checked by triggers.
- **Profile job** (`profile`, queue `analysis`) chained after ingest and re-runnable: `BuiltinProfiler` behind the `Profiler` port. It stores row/column counts, missing counts and percentages, duplicate rows, unique counts, constant/empty columns, numeric statistics (mean, median, std, min, max, quartiles, IQR, skewness, kurtosis, histogram), frequency tables (class distributions), datetime ranges, text lengths, and Pearson/Spearman correlations. Results go in `version_profiles` (PostgreSQL), with limitations and sampling disclosed.
- API: version list/detail, preview, profile read/re-run, and re-queue parsing for uploads without a version.
- Web: dataset overview (summary tiles, preview, parse facts, jobs) and the profiling dashboard (type mix, missing values, column explorer with histograms and frequencies, target-column class distribution, correlation heatmap, limitations).

Phase 2 review (2026-11-09): profile statistics verified against hand-calculated values. Fixed: Celery replaced the worker's log scrubber and formatter; Parquet timestamps in non-nanosecond units gave 1970 histogram edges; identifier-like integers received means and histograms; all-distinct measurements were labelled identifiers; single letters such as a lone `F` were read as boolean; colliding JSON keys silently dropped a value; and UI percentages could round 99.6 % up to 100 %.

Known limitations and technical debt (accepted for now):
- Worker memory: a maximum-size file needs about 1 GB to profile and up to the 4 GB sandbox limit to parse. With two concurrent worker processes on an 8 GB host, two maximum-size uploads at once can exhaust memory. Mitigate by lowering `parse_memory_limit_mb` or running ingest with concurrency 1 before deployment.
- The profile job loads the whole version into memory; no streaming or chunked profiling.
- Status never shows `parsing`: ingest is a single transaction.
- A failed upload's partial objects are removed only by the purge, not at failure time.
- The stale-job reaper (`JOB_STALE`) does not reset a dataset left in `validating`.
- Identifier and boolean detection are heuristics; per-column overrides arrive with roles and re-parse.
- Frequency tables can still contain identifying values for small datasets (fewer than 20 rows, where identifier detection does not apply). They are capped and owner-only.

Not done (still with their original phases): re-parse with user options and the parse-options/sheet UI (sheet = first sheet for now), the privacy/PII scan at ingest, roles and a persisted target role (the target picker is a view-only selection), the check/findings framework and severities, Cramér's V for categorical associations, `fixtures/malicious` as a corpus (hostile cases are in the unit tests), quotas tables, and the S3 adapter.

### Phase 2 hardening (2026-11-09) — done
- **Workers split:** `worker-ingest` (concurrency 1, parse sandbox capped at 2,500 MB) and `worker` (analysis + maintenance, concurrency 1, per-process `RLIMIT_DATA` 2,000 MB). There are container caps of 3 GB and 2.5 GB, and a cap hit fails the job with `LIMIT_EXCEEDED`. Verified in the Linux image: the cap raises `MemoryError`, and diagnostics on 19M cells peak at about 1.15 GB. See README "Resource limits".
- **Stale-job recovery:** after reaping, datasets stuck in `validating` with no active ingest job move to `parse_failed` (retryable), `ready` (if version 0 exists), or `upload_pending` (no original stored, after 1 h). Ingest re-checks under a row lock before creating version 0, so no retry or late duplicate can create a second version.
- These resolve the two Phase 2 review items, the worker-memory risk and the stale-reaper gap.

### Phase 3 status (2026-11-09) — "Intelligent diagnostics", as scoped by the owner
Owner decisions:
- ECOD and COPOD are implemented in-house behind the `OutlierDetector` port. PyOD stays out of the MVP, as DEVELOPMENT_GUIDE.md requires.
- The financial, business and industrial modules move into the MVP (PRD FR-4.4a).

This delivers most of the original Phase 4 and part of Phase 5:
- **Check framework:** specs, honest `not_assessed`, error isolation, finding validation, sampling budgets (expensive checks ≤ 20,000 rows, correlations ≤ 200,000) disclosed on findings, and a registry with a composition root.
- **Checks:** 17 universal checks and 4 domain modules with 19 checks (ML_ENGINE_DESIGN §2.0, DOMAIN_MODULES §4.8).
- **Roles and domains:** column roles with task type and validation, domain selection, semantics hash and stale runs.
- **Runs:** analysis job chained after ingest and re-runnable; runs and findings in PostgreSQL (migration 0004, RLS, insert-only, honesty CHECKs).
- **Diagnostics UI:** domain and roles editor, severity summary, status, category, severity and module filters, and expandable findings with evidence, suggested action and limitations.
- **Tests:** synthetic golden datasets with planted issues and clean controls for every check, registry-wide `not_assessed` honesty, stub-module extensibility, detector properties, and API/RLS/isolation tests.

Not done:
- Role suggestions.
- The paginated findings and affected-rows endpoints (row IDs appear only as capped evidence examples).
- The universal checks listed as missing in ML_ENGINE_DESIGN §2.0.
- The biomedical checks listed as missing in DOMAIN_MODULES §4.8, including the reference-range table.
- Recommendations, the PII/PHI scan, and copying roles to new versions (only version 0 exists).

### Phase 3 manual-verification review (2026-11-09)
Fixed:
- **Industrial sensor checks.** `ind.sensors.drift` crashed (`IndexError`) and `ind.sensors.shift` silently computed on a 2-D slice when one column carried two sensor roles. Found by fuzzing every check with edge-case data and roles: 0 errors in 1,881 other findings.
- **Roles hidden after a domain change.** Roles from a previous domain were invisible in the role editor (the dropdown showed "Feature (no role)") and made saving fail. They are now shown and flagged.
- **Vanishing confirmations.** "Roles saved" and "Domain changed" disappeared at once, because the panel remounts after a save.
- **Role errors without a column.** Role validation errors did not name the column.
- **Data keys rewritten.** Nested evidence keys that are data (class, segment and column names) were prettified, e.g. `no_show` was shown as "No show".
- **Small shares shown as 0.0%.** Affected-row shares under 0.05% displayed as "0.0%".
- **Binary columns called skewed.** 0/1 indicator columns were reported as "strongly asymmetric".

Still incomplete or known limits (Phase 3 UI and diagnostics):
- **Role editor.** It assigns one role per column; several roles per column are possible only through the API, and such columns are read-only in the UI.
- **Other-domain roles.** These are kept after a domain change: they are flagged, not removed, and ignored by the new domain's checks.
- **Target column.** The profile page's target picker is separate from the declared `target` role.
- **Overview page.** It does not summarise diagnostics; they appear only on the Diagnostics tab.
- **Missing values on wide tables.** `missingness.column_rate` reports one finding per affected column, with no grouping.
- **Maintenance delays.** With one analysis worker, maintenance tasks (requeue, reaper) wait while a long analysis runs.
- **Browser tests.** No automated browser end-to-end tests (Playwright) yet: the frontend is covered by component tests, and the manual checklist (Phase 3 report) covers the browser.

### Phase 4 status (2026-11-17) — "Intelligent preprocessing engine", as scoped by the owner
Scope: roadmap Phases 6 and 7 plus the codegen part of Phase 9, split into Parts A–E ([PHASE4_PLAN.md](PHASE4_PLAN.md)). Owner decisions D1–D6 are recorded there. Each part needs approval before the next starts.

**Phase 4 complete (2026-11-25), closure approved by the owner.** All five parts are delivered and verified; Part E closed after the owner's manual browser verification and the closure checks below.

### Phase numbering
Owner milestones and the original roadmap phases below differ. Owner Phase 4 = roadmap Phases 6 and 7 plus the codegen part of 9. **Owner Phase 5 = roadmap Phase 8 (leakage audit, generalised into the Reliability Auditor) plus PRD §6.8 model evaluation and explainability**; label noise (Cleanlab) is not part of it. The roadmap's own "Phase 5" (remaining universal checks and the biomedical clinical module) and Phase 9 (passport, readiness, export) are still open and will be scheduled as owner milestones.

### Phase 8 — Analytics Studio and Advanced Data Intelligence (2026-11-27)
**Approved as one milestone on 2026-11-27; implemented; owner browser verification completed; closed 2026-11-27.** Closure fixes: the Copilot routed "most important insights about my dataset" to feature importance (now a stored-insights topic; feature importance only for explicit model/feature questions); editing a saved widget did not restore its sorting, limit or quantile. Closure verification: backend targeted unit 44 passed and integration (analytics, MVP, isolation matrix) 13 passed; frontend 105 passed; ruff, format, strict mypy, tsc, ESLint and Docker builds pass; live smoke 15/15; migration 0009 current; pre-existing versions unchanged. The earlier Phase 8 regression (1,447 passed) stands: the closure changes touch only the Copilot router topic list and a frontend helper. Design: ARCHITECTURE.md §18; API §18; migration 0009.
- Query engine (pandas, decision A1), Analytics Studio UI with Recharts 3.11.1, KPI cards, pivot tables, dashboards, automated insights (worker), NL analytics Copilot, version comparison, exports, passport integration. Hugging Face integration deferred by decision.
- Verification (2026-11-27): backend unit suite plus analytics, MVP, isolation-matrix (all new routes), database-security, dataset-job, diagnostics, training-privacy and version-security integration suites 1,447 passed, 4 skipped (pre-existing), 0 failed; frontend 104 passed (18 files); ruff, format, strict mypy, tsc, ESLint and the Docker production builds pass. Live dev stack: Phase 8 smoke 14/14 (all 17 steps; the Copilot's answer equals an independent pandas computation) and the MVP smoke 22/22; pre-existing versions, hashes and training configurations unchanged; dev database migrated to 0009. Production rehearsal (`docker-compose.prod.yml`, HTTPS via Caddy, rehearsal database upgraded 0008 → 0009): 22/22 and 14/14; idle memory api 146 MiB, worker 292 MiB, web 87 MiB, db 50 MiB.

### MVP completion — owner Phase 6/7 + deployment (2026-11-26)
**Approved by the owner as one milestone on 2026-11-26; implemented the same day; awaiting the owner's acceptance.** Phase 5 approved 2026-11-26.
- Biomedical Intelligence Lite (`engine/bio/summary.py`, `biomedical/`, Biomedical tab); Data Passport and Copilot (`reports/`); exports (`exports/`, `reports/router.py`); migration 0008; professional UI (landing page, navy/teal design system, dashboard, print styles); production configuration (`docker-compose.prod.yml`, `deploy/`, `docs/DEPLOYMENT.md`); authenticated TLS SMTP.
- Fixed on the way: a biomedical p-value of exactly 0 treated as "not significant" (falsy `or`), and the worker's handler loader skipping modules when one handler module had been imported first.
- Deferred: optional external-LLM Copilot, server-side PDF rendering (print-to-PDF is provided), advanced omics, arm64 image verification.
- Verification (2026-11-27): backend unit suite plus training, privacy, Lab, pipeline and MVP integration suites 1,400 passed, 4 skipped (pre-existing), 0 failed; isolation matrix (new routes included), database security and dataset-job suites 32 passed; frontend 96 passed (17 files); ruff, format, strict mypy, tsc, ESLint, Docker production builds and compose config pass. End-to-end smoke on a synthetic biomedical cohort: 22/22 on the dev stack and 22/22 on a local production rehearsal of `docker-compose.prod.yml` (clean database, HTTPS through Caddy, `__Host-` secure cookies, only ports 80/443 published, security headers verified). Existing development datasets, versions, hashes, runs and evaluations unchanged; dev database migrated to 0008.
- Public deployment: configuration ready; not yet deployed (needs the owner's server, DNS name and SMTP account; docs/DEPLOYMENT.md §3).

### Phase 5 status (2026-11-25) — "Baseline AutoML and ML Reliability Auditor", as scoped by the owner
**Implemented 2026-11-25. Owner browser verification completed 2026-11-26 (conditional acceptance); closure corrections implemented 2026-11-26 ([PHASE5_PLAN.md](PHASE5_PLAN.md) §19); awaiting the owner's final approval.**
- Engine `engine/ml/`: model catalogue and fixed grids; eligibility (direct, replay from the eligible ancestor, blocked); in-fold replay with a fidelity proof; nested selection on frozen folds; metrics, curves, calibration, binned residuals, group-aware bootstrap; permutation, coefficient and impurity importance; error slices; 20 audit checks and the evaluation-status rule.
- Migration 0007: `training_runs`, `model_evaluations`, audit runs in `analysis_runs`, job types `train` and `reliability_audit`, artifact kind `oof_predictions`; RLS, triggers, grants.
- API: model catalogue, training plan, create / list / detail / rerun, audit run and read.
- Frontend: Models tab (configuration with eligibility, preprocessing, models and tuning; run page with progress, cancellation, comparison, performance charts, reliability, explainability, reproducibility, rerun); Lab warning for full-data-fitted versions.
- No new dependencies (no XGBoost, SHAP or Cleanlab).
- Verification (2026-11-25):
  - backend 1,506 passed, 4 skipped (pre-existing `test_check_framework.py:91`), 0 failed, including 177 database integration tests; 58 new Phase 5 tests;
  - frontend 84 passed (16 files); ruff, format, strict mypy, tsc, ESLint, the production build and `docker compose config` pass;
  - live stack: migration 0007 applied to the existing database without a reset (users, datasets and versions preserved); smoke test 24/24 (binary nested training of 4 models on 1,500 rows in 77 s, multiclass, regression, audit, explainability, frozen folds and identical rerun, cancellation, exact replay proof of a fitted version, blocked and exploratory evaluation, cross-tenant 404s, 69 earlier stored objects unchanged, no column names in logs);
  - manual browser verification by the owner: completed 2026-11-26, all nine categories.
- Closure corrections (2026-11-26, PHASE5_PLAN §19): list headline fixed (direction-aware, explicit states, exploratory qualified); sensitive-column protection re-applied when serving historical results (training-time, current and derived sensitivity); Lab HGB internal early stopping disabled; key-like undeclared-entity audit signal; "leakage-controlled" terminology; uncertainty and reproducibility wording.
- Closure verification (2026-11-26): full backend regression 1,523 passed, 1 failed, 4 skipped (pre-existing `test_check_framework.py:91`); the failure was a pre-existing flaky Phase 4 assertion (`"4242" not in` a response that contained a random UUID `…-4242-…`), fixed together with the same pattern in a pipeline test by using non-hex markers, and both re-run green. Targeted: 7 new privacy/headline integration tests and 63 training, Lab, isolation-matrix and database-security integration tests passed; 11 new unit tests and 73 ML/audit/Lab unit tests passed. Frontend 91 passed (16 files); ruff, format, strict mypy, tsc, ESLint, the production build and `docker compose config` pass. Live stack (images rebuilt, containers recreated, no volume touched): 20/20 focused checks; datasets, versions, content hashes, runs, configurations, evaluations, roles and stored objects unchanged (one new audit run from the verification).
- Not measured: performance at the 200,000-row / 400-fit limits.

**Part A — engine architecture and missing-value handling: approved 2026-11-18** (after an independent review that fixed nullable-dtype handling, KNN cost caps, and warning/exception texts reaching logs).

**Part B — outliers, encoding and scaling: approved 2026-11-18** (after an audit; fixes listed below).

**Part E — pipelines, approvals, commit, revert, comparison and generated code: complete 2026-11-25** (owner decisions E1–E5 approved 2026-11-25; owner manual browser verification passed 2026-11-25).
- Migration 0006:
  - new tables `pipelines`, `pipeline_steps`, `approvals`, `transformation_runs`, `version_events`;
  - `dataset_versions` gains `model_pipeline_spec` and the deferred FK to runs;
  - new job types, artifact kinds, and the one-mutation-per-dataset index;
  - RLS and triggers;
  - maintenance read grants for orphan reconciliation.
- Engine: `stages.py` (explicit data/model stage per step, hashing), `commit.py` (preview and materialise through `commit_transform`), `compare.py`, `pipeline_code.py` (exact-hash `pipeline.py`, rendered without data libraries).
- Backend: `pipelines/` (API service with `commit_readiness`, worker jobs `impact_preview`, `commit`, `revert` and `compare_versions`, lineage and roles); hourly `reconcile_orphans`; version cap 20 (E2); impact preview limited to 1,000 rows per step with redaction (E1); model-stage steps stored as `model_pipeline_spec` (E3); automatic profiling and diagnostics after commit and revert (E4).
- Frontend: History tab (versions, lineage, runs, approvals, comparison, revert, `pipeline.py`); pipeline page (stage-grouped step editor, preview, approval dialogs, commit); "To pipeline" in the Experiment Lab.
- Verification:
  - backend 1,446 passed, 4 skipped (pre-existing), 0 failed, including 156 database integration tests;
  - frontend 74 passed;
  - ruff, strict mypy, tsc, ESLint, the production build and `docker compose config` pass;
  - the live-stack smoke test passed all 28 checks: migration 0006 applied to the existing database without a reset, the full commit/revert workflow, unchanged original and earlier version files, generated-code equivalence and cross-tenant 404s.
- Generated code reproduces the exact content hash on the committing platform; across operating systems, floats agree within 1 ulp (checked with `allclose`).
- Owner manual browser verification (2026-11-25), all passed:
  - version history and revert lineage; transformation details and approval history; version comparison;
  - Experiment Lab candidate to editable pipeline; data-stage and model-stage separation;
  - impact preview and affected-row inspection; sensitive-value redaction;
  - medium-risk pipeline confirmation; stale approvals after a parameter edit;
  - a commit creating a new version (3,000 rows, 9 columns), then a revert creating another (3,000 rows, 5 columns) with every earlier version kept;
  - `pipeline.py` download.
- Closure checks (2026-11-25):
  - **Stage assignment.** Lab candidate steps carry no stage (the experiment `StepIn` has only `op`, `columns` and `params`), so conversion always assigns stages automatically: the catalogue stage, and `data` for `either` ops except after the first model-stage step. A stage the owner chooses on a pipeline step (create or edit) is stored as chosen and never reassigned; an invalid order is reported as `STAGE_ORDER`, not moved.
  - **Version quota.** The cap is checked under the dataset row lock in the API (commit via `commit_readiness`, and revert) and again in the worker under the lock just before the version is inserted. The one-mutation-per-dataset index means only one commit or revert per dataset can be queued or running at a time.
  - Two targeted integration tests were added (an API commit refused at the cap; the worker refusing a commit and a revert when the cap is reached after queueing), and the Lab conversion test now also checks that explicit stages are kept. `tests/integration/test_pipelines_api.py`: 17 passed. No defect was found and no application code changed, so the full suite was not re-run.
- Known non-blocking limitations:
  - After switching the selected versions, the previous comparison result stays visible until a new comparison runs.
  - Repeated experiment re-runs accumulate "Re-run of" prefixes in names.
  - Performance at the maximum supported dataset size has not been measured (tested up to 3,000 rows on the live stack).
  - High-risk destructive approvals and byte-level immutability are covered by automated tests and the API smoke test, not checked independently in the browser.

**Part D — recommendation engine and Experiment Lab: approved 2026-11-25** (live-stack smoke test passed: migration 0005 applied, 2-candidate 3-fold experiment, re-run, cancellation, no version changed).
- Deterministic recommendation engine (`engine/preprocess/recommend/`, standard library): evidence-based rules over findings, profile and roles (never column names), domain policies, validation of every proposed step, `not_assessed` / `not_recommended` / advisories, transparent factor ranking, conflict groups with trade-offs and "do nothing", suggested candidates and a role-based split plan (ML_ENGINE_DESIGN §3.5).
- Experiment Lab engine (`engine/preprocess/experiment/`): candidates on one stored frozen split, in-fold fitting through `fit_resample`/`transform`, baseline models (dummy, linear, HGB; k-means for clustering), classification/regression/clustering metrics with fold statistics, calibration and per-class results, descriptive comparisons with the baseline (no winner), data-quality changes, selection stability with noise probes, failure isolation, cancellation (ML_ENGINE_DESIGN §5.0).
- Persistence: migration 0005 (recommendation sets, recommendations, recommendation findings, split definitions with `fold_assignment` artifacts, experiments, candidates, results), RLS, owner/parent/version triggers, insert-only grants, complete-once trigger. Job type `experiment` on queue `evaluate`.
- API: recommendations (generate, latest, status), experiments (create with `Idempotency-Key`, list, detail, identical re-run). Isolation matrix and DB security tests extended.
- Frontend: Experiment Lab tab (recommendations with evidence, risk, stage and ranking factors; trade-offs; pipeline builder with parameters; experiment settings; status, cancellation, side-by-side metrics, fold scores, comparisons with the baseline, data quality, selection stability, warnings and limitations).
- Commit guard: `PreprocessExecutor.commit_transform` refuses model-stage steps (closes the Part C enforcement gap; the pinned test now expects the refusal).
- Bug found while testing: the jobs list `type` filter rejected `experiment`.

**Part C — distribution transforms, feature selection and class imbalance: approved 2026-11-24** (after an independent audit; fixes listed below).
- Ops (ML_ENGINE_DESIGN §3.4):
  - `transform.log`, `transform.log1p`, `transform.yeo_johnson`, `transform.quantile` (stage `either`);
  - `select.variance`, `select.correlation` (proposal only), `select.mutual_information`, `select.univariate`, `select.drop_columns` (model stage; report by default; domain-role columns never removed automatically);
  - `imbalance.class_weight`, `imbalance.random_oversample`, `imbalance.random_undersample`, `imbalance.smote` (model stage, new kind `training_only`: only `fit_resample` builds a resampled training table; transform and commit paths never resample).
- Frozen fold splitter `engine/preprocess/split.py` (stratified, group, stratified-group, time; duplicate-aware). Feature lineage on every transform result. Executor post-conditions: protected columns never change; a step removes only columns it was given.
- Dependency: imbalanced-learn 0.14.2 (+ sklearn-compat 0.1.6), contract-tested. Catalogue API gains `uses_columns`, `needs_target` and kind `training_only`; typed client regenerated; no migrations, persistence or UI.
- Bugs found while testing and fixed with regression tests:
  - Yeo-Johnson with λ = 1 was not an exact identity (rounding);
  - quantile failed when the seeded subsample was smaller than `n_quantiles`;
  - synthetic rows used NaN instead of None for missing text, unlike canonical tables.
- Independent audit (2026-11-18), fixed with regression tests (ML_ENGINE_DESIGN §3.4, "Part C audit"):
  - quantile ties at the minimum/maximum forced to the output ends;
  - correlation, mutual information and ANOVA wrong or "not assessed" at magnitudes near 1e±300;
  - SMOTE over its memory budget at its own caps (new cell caps for resampling and correlation);
  - false `NO_FEATURES` refusal after a drop-mode selector;
  - unsanitised or misleading splitter errors;
  - missing fold class-count reporting;
  - `_dm_sample_weight` missing from lineage;
  - duplicate column names accepted;
  - per-cell Python loop in step reports.
- Audit decisions:
  - SMOTE on data with patient, group, identifier or time roles requires `acknowledge_dependent_rows`;
  - selection instability between folds is now stated for every selector.
- Part E gap (`fit_transform` accepting model-stage steps): closed in Part D by `commit_transform`.
- Ops: `outliers.flag`, `outliers.cap`, `outliers.signed_log`, `outliers.isolation_forest_flag`, `encode.onehot`, `encode.ordinal`, `encode.frequency`, `scale.standard`, `scale.minmax`, `scale.robust`, `scale.none` (ML_ENGINE_DESIGN §3.3). Encoding and scaling are model-stage only; no op removes rows.
- Validation refactored: op-specific rules and schema effects in `rules.py`, shared types in `state.py` (Part A behaviour unchanged, verified by its tests). Step reports gained `effects` (counts only).
- No migrations, persistence or UI; the typed API client was regenerated.
- Audit (2026-11-18), fixed:
  - silent NaN, inf or zeros at extreme magnitudes in scaling, imputation and outlier bounds (now `NUMERIC_OVERFLOW`);
  - Isolation Forest's `"auto"` threshold flagging 13–60% of clean rows (replaced by an explicit share, default 1%);
  - identical or constant columns flagging every row;
  - unbounded scoring memory inside `fit`;
  - one-hot `error` policy ignored when an `__other__` column existed;
  - one-hot names copying control characters, overlong text or `__other__` from data.

  Signed log is documented as invertible up to floating-point rounding.
- `engine/preprocess/`: standard-library specs, catalogue and validation; one runtime implementation shared with generated code; a scikit-learn `Pipeline` executor with a fit-on-rows / transform-any API; step reports with sensitive-column redaction; code generation.
- Ops: `impute.mean`, `impute.median`, `impute.mode`, `impute.constant`, `impute.knn`, `impute.iterative`, `missing.indicator` (ML_ENGINE_DESIGN §3.2).
- API: `GET /api/v1/ops`. Web: typed client, `useOps`, `lib/preprocess.ts` (labels and tooltip text).
- Tests:
  - catalogue, validation, and per-op correctness against hand-calculated values;
  - edge cases: empty and tiny tables, columns empty in the fitting rows, infinities, ties;
  - Hypothesis property tests;
  - leakage tests with a negative control;
  - codegen equivalence in an isolated subprocess, including hostile column names;
  - scikit-learn contract tests and API integration.
- Results: backend 541 passed (4 pre-existing parametrised skips); frontend 48 passed.
- Not in Part A: recommendations, experiments, splits, persistence, approvals, versions and UI pages (Parts D–E); other op families (Parts B–C).

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
