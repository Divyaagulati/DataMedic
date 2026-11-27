# DataMedic — Phase 4 Plan: Intelligent Preprocessing Engine

| | |
|---|---|
| Status | Approved 2026-11-17 with decisions D1–D6 (§2.3). Part A approved 2026-11-18. Part B approved 2026-11-18. Part C approved 2026-11-24 (after an independent audit). Part D approved 2026-11-25 (live-stack smoke test passed; owner browser check). Part E plan and decisions E1–E5 approved 2026-11-25. Part E complete 2026-11-25 (owner manual browser verification passed). **Phase 4 complete.** Phase 5 not started. |
| Date | 2026-11-17 |
| Baseline | Phases 1–3 delivered. Backend: 280 passed, 105 skipped (integration tests skip without `TEST_ADMIN_DATABASE_URL`). Frontend: 9 files, 43 tests passed. |
| Scope mapping | Owner Phase 4 = roadmap Phase 6 (recommendations, op registry, impact preview, approvals, commit, revert) + Phase 7 (splits, experiments) + the codegen part of Phase 9. Passport, readiness and export bundles stay in later phases. |

Workflow: raw version → diagnostics (exists) → recommendations → pipeline configuration → experiments → before/after comparison → approval → new immutable version → reproducible Python code.

---

## 1. What exists and is reused

| Area | Existing component | Reuse in Phase 4 |
|---|---|---|
| Engine purity | `datamedic.engine` + `tests/unit/test_engine_boundary.py` | All preprocessing logic goes in `engine/preprocess/`; the boundary test covers it automatically |
| Row identity | `ROW_ID = "_dm_row_id"` (`engine/__init__.py`) | Every op preserves it; folds, reports and affected rows are keyed by it |
| Content hash | `engine/hashing.content_hash` | Reproducibility tests; commit writes it on new versions |
| Spec/impl split | `engine/checks/spec.py` (stdlib only) + `"module:function"` impls | Same pattern for ops: catalogue importable by the API without pandas |
| Domain registry | `engine/registry.py`, `domains/bootstrap.py` | Domain modules may later register ops and rules; core never imports them |
| Roles | `RoleMap`, `RoleSpec.is_feature`, `is_sensitive` | Validation: never transform target, `id`, `group`, `time`, `weight`, `exclude`; redact learned values for sensitive columns |
| Findings | `findings` rows with structured `evidence` JSON | Input to deterministic recommendation rules |
| Profiles | `version_profiles.columns` (missing %, skewness, min, quartiles, unique counts) | Input to rules (zero/negative checks for log, skew, cardinality) |
| Versions | `dataset_versions` (kinds `transform`, `revert` already allowed by CHECKs), `head_version_id`, ingest's row-lock pattern in `datasets/jobs.py` | Commit and revert follow the same lock-then-insert pattern |
| Jobs | `jobs` table, `claim_job`, `register_handler`, `register_failure_hook`, follow-up publishing | New handlers `experiment`, `impact_preview`, `commit`, `revert` |
| Storage | `ObjectStorage`, `version_object_key`, `artifact_object_key`, sha256-verified reads | New version Parquet files and artifacts (fold assignment, impact preview, experiment report) |
| Errors | `DomainError` factory in `errors.py`, problem+json | New codes `APPROVAL_REQUIRED`, `APPROVAL_STALE` (already in ARCHITECTURE §13) |
| Isolation | RLS + owner-consistency triggers (migration 0004 helpers), route-generated AuthZ matrix | Every new table and route |
| Frontend | `DatasetNav`, TanStack Query polling hooks (`lib/queries.ts`), generated client, shadcn/ui, `charts.tsx`, `FindingCard` | New Preprocessing and History tabs reuse all of these |

Design already settled in the docs (and followed here): the op registry with server-derived risk, destructive flag and kind (ML_ENGINE_DESIGN §3.1); hash-bound approvals verified in the API and the worker (§4, API §8); frozen `_dm_row_id` folds with duplicate-aware grouping (§5.2); row removal only on training rows; `NOT_DIRECTLY_COMPARABLE` (§5.3.1); template-only codegen (§9).

## 2. Architectural changes and owner decisions

### 2.1 New concept: op stage (`data` vs `model`)
The docs define risk, destructive flag and kind per op. Phase 4 adds a **stage**:

| Stage | Meaning | Examples | Can be written into a version? |
|---|---|---|---|
| `data` | Cleaning that changes what the dataset *says* | Trim/cast, missing indicators, row drops, outlier caps, imputation (when materialised) | Yes, after approval |
| `model` | Preparation that belongs inside a model's training pipeline | Scaling, encoding, feature selection, resampling, class weights | **No.** Stored with the version as a modeling-pipeline spec and emitted as a scikit-learn `Pipeline` that is fitted in-fold |
| `either` | Valid in both | Imputation, Yeo-Johnson, quantile transform | User chooses; `model` is recommended for ML use |

Reasons: SMOTE creates rows (violates invariant 6 and leaks if done before splitting); scaling, encoding and feature selection fitted on the full version leak into any later evaluation. **Decision D1** (below) confirms this.

### 2.2 Engine layout (new package `engine/preprocess/`)
```
engine/preprocess/
  spec.py        # stdlib only: OpSpec, ParamSpec, enums (Risk, Kind, Stage, Family), StepSpec,
                 #   canonical JSON, step_hash / spec_hash
  catalog.py     # stdlib only: every OpSpec; the API lists and validates from this
  validate.py    # stdlib only: steps vs version schema + roles -> errors / warnings
  frame.py       # pandas helpers: nullable dtypes <-> float arrays, ROW_ID passthrough
  base.py        # DMTransformer(BaseEstimator, TransformerMixin): fit/transform on DataFrames,
                 #   learned_params() -> JSON, never mutates input
  executor.py    # build sklearn Pipeline from steps; fit(train) / transform(any) -> frame + reports
  report.py      # StepReport: rows in/out, cells changed, missing before/after, dtype changes,
                 #   learned params, warnings, fit rows, duration
  ops/impute.py  # Part A
  ops/outliers.py, ops/encode.py, ops/scale.py                  # Part B
  ops/transform.py, ops/select.py, ops/imbalance.py, split.py  # Part C
  recommend/     # Part D: stdlib-only rules over findings + profile + roles
  experiment.py  # Part D: candidates x folds, data-quality metrics, optional baseline models
  compare.py     # Part E: before/after comparison
  codegen/       # per-op templates (fragments from Part A; full pipeline.py in Part E)
```
**Revised during Part A:** ops do *not* wrap a `ColumnTransformer` (it reorders columns, and `SimpleImputer` leaves `None` unfilled in object columns; see ML_ENGINE_DESIGN §3.2). Each op selects its own columns and delegates statistics to scikit-learn estimators. Originally planned: each op wraps its scikit-learn estimator in a `ColumnTransformer` over the selected columns (`remainder="passthrough"`, `verbose_feature_names_out=False`, `set_output(transform="pandas")`), and a candidate is a scikit-learn `Pipeline` of those steps. `_dm_row_id` passes through untouched and the original column order is restored.

The **leakage primitive** is the executor's split API: `fit(frame.loc[train_ids])` learns, and `transform(frame)` never learns. Experiments, commits and tests all go through it.

### 2.3 Owner decisions needed (recommendations first)

| # | Decision | Recommendation | Alternative |
|---|---|---|---|
| D1 | Fitted ops in committed versions | `data`/`either` ops may be materialised: fitted on the full source version, flagged `fitted_on_full_data` with a visible warning. `model` ops are never materialised; they are stored as the version's modeling-pipeline spec and emitted as in-fold code. | Materialise everything (simpler, but leaks into later evaluation), or materialise nothing fitted |
| D2 | Libraries | scikit-learn plus small in-house fitted transformers (IQR/MAD caps, frequency encoding, log). No Feature-engine. imbalanced-learn 0.14.x (MIT) added in Part C after a contract test, for SMOTE and random over/undersampling. | Feature-engine as ML_ENGINE_DESIGN §3.1 lists (adds statsmodels; codegen must then mirror its internals) |
| D3 | Models in the Experiment Lab | Data-quality metrics always, plus optional baseline cross-validation with `dummy`, `linear` (logistic/ridge) and `hgb`. XGBoost deferred (not installed; image size). No tuning. | Data-quality metrics only (cannot show model-performance change, which §7 of the brief asks to separate) |
| D4 | Fitted state | Learned parameters stored as JSON in step reports; no joblib/pickle anywhere in Phase 4. Code generation refits. | joblib + HMAC as ARCHITECTURE §9.5 allows |
| D5 | Integration tests | Owner provides `TEST_ADMIN_DATABASE_URL` (PostgreSQL superuser URL) so the 105 skipped integration tests and all new RLS/isolation tests run. | Run unit tests only (RLS, approvals and isolation stay untested) |
| D6 | Dev dependency | Add `hypothesis` (MPL-2.0, dev only; already listed in ARCHITECTURE §11.2) for property tests (row-ID preservation, no input mutation). | Hand-written parametrised cases |

**Owner answers (2026-11-17):** D1, D2, D3, D4 and D6 approved as recommended. D3 adds that comparisons use identical frozen splits and must separate data-quality from model-performance changes. D4 adds that sensitive-column learned values never appear in reports, logs or API responses. D5: the owner keeps `TEST_ADMIN_DATABASE_URL` local, pointed at a disposable instance with a test-only administrator (README "Disposable test database"); integration tests are never skipped to avoid this.

Doc conflicts this plan resolves once approved: Feature-engine in ML_ENGINE_DESIGN §3.1 (D2); joblib fitted state in ARCHITECTURE §9.5 (D4); XGBoost in the experiment model list (D3); the roadmap's phase numbering (scope mapping above).

## 3. Files to modify (whole phase)

| File | Change | Part |
|---|---|---|
| `backend/src/datamedic/engine/__init__.py` | `ENGINE_VERSION` bump | A |
| `backend/src/datamedic/engine/versions.py` | Track scikit-learn and SciPy (later imbalanced-learn) | A, C |
| `backend/src/datamedic/api/app.py` | Register `ops`, `recommendations`, `experiments`, `pipelines` routers | A, D, E |
| `backend/src/datamedic/config.py` | Preprocessing limits (KNN/iterative caps, max candidates, max versions per dataset) | A, D, E |
| `backend/src/datamedic/errors.py` | `APPROVAL_REQUIRED`, `APPROVAL_STALE`, `OP_INVALID`, `OP_LIMIT_EXCEEDED` | A, E |
| `backend/src/datamedic/db/models.py` | New tables; widen `JOB_TYPES`, `ARTIFACT_KINDS`, artifact `owner_type` | D, E |
| `backend/src/datamedic/jobs/registry.py` | Job types `experiment` (queue `evaluate`), `impact_preview`, `commit`, `revert` (queue `execute`) | D, E |
| `backend/src/datamedic/workers/tasks.py`, `workers/runner.py` | New Celery tasks; register new handler modules | D, E |
| `docker-compose.yml` | Worker consumes `analysis,evaluate,execute,maintenance` | D |
| `backend/src/datamedic/versions/{service,router,schemas}.py` | Lineage, transformation run detail, revert, version comparison | E |
| `backend/src/datamedic/analyses/service.py` | Chain profile + diagnostics after commit/revert; copy roles to new versions (tracking dropped/added columns) | E |
| `backend/pyproject.toml`, `uv.lock` | `hypothesis` (dev, A); `imbalanced-learn` (C) | A, C |
| `backend/tests/integration/test_isolation_matrix.py` | `KNOWN_PARAMS` += `recommendation_id`, `split_id`, `experiment_id`, `pipeline_id`, `step_id`, `transformation_run_id`; JSON bodies | D, E |
| `backend/tests/integration/test_db_security.py` | RLS and grants for the new tables | D, E |
| `apps/web/src/components/dataset-nav.tsx` | Tabs: Preprocessing, History | D, E |
| `apps/web/src/lib/queries.ts`, `lib/api/openapi.json`, `lib/api/schema.d.ts` | Hooks; regenerated client | A, D, E |
| `apps/web/src/app/datasets/[datasetId]/dataset-detail.tsx` | Head version, version count, link to History | E |
| `apps/web/src/components/diagnostics/finding-card.tsx` | "See recommendations" link | D |
| `docs/*.md`, `README.md`, `DEVELOPMENT_GUIDE.md` | Keep design and code in agreement at the end of each part | A–E |

## 4. New files

| Part | Backend | Frontend |
|---|---|---|
| A | `engine/preprocess/{__init__,spec,catalog,validate,frame,base,executor,report}.py`, `engine/preprocess/ops/{__init__,impute}.py`, `engine/preprocess/codegen/{__init__,templates}.py`, `ops/router.py` + `ops/schemas.py` (catalogue API); tests `unit/test_preprocess_spec.py`, `test_preprocess_validate.py`, `test_impute_ops.py`, `test_preprocess_leakage.py`, `test_preprocess_codegen.py`, `test_sklearn_contract.py` | `lib/preprocess.ts` (op labels, risk/stage helpers) + test |
| B | `ops/{outliers,encode,scale}.py`, tests per family | — |
| C | `ops/{transform,select,imbalance}.py`, `engine/preprocess/split.py`, `test_imblearn_contract.py`, tests per family, fold-leakage tests | — |
| D | `engine/preprocess/recommend/{rules,engine}.py`, `engine/preprocess/experiment.py`, `recommendations/` and `experiments/` modules (router/service/schemas/jobs), migration `0005_recommendations_experiments.py`, integration tests | `app/datasets/[datasetId]/preprocessing/` (page, recommendations, configurator, experiment lab, comparison), `components/preprocessing/*`, `lib/experiments.ts` + tests |
| E | `pipelines/` module (router/service/schemas/jobs), `engine/preprocess/compare.py`, `engine/preprocess/codegen/pipeline_py.py`, migration `0006_pipelines_commit.py`, integration tests (approval bypass, immutability, revert, concurrency) | `app/datasets/[datasetId]/history/`, `components/preprocessing/{impact-preview,approval-dialog,before-after}.tsx` + tests |

## 5. Database changes

No migration in Parts A–C. They are pure engine plus a read-only catalogue endpoint.

**Migration 0005 (Part D, implemented; see DATABASE_SCHEMA "Phase 4 Part D implementation notes"):** follows DATABASE_SCHEMA §3.4 and §3.6 with the deviations recorded there (recommendation sets; splits created by the experiment job; results per candidate).
- `recommendations`, `recommendation_findings`. Generated synchronously in the API from findings, profile and roles; persisted for traceability; insert-only except `status`.
- `split_definitions` (fold map stored as a `fold_assignment` artifact), `experiments`, `experiment_candidates`, `experiment_results` (DATABASE_SCHEMA calls it `model_results`; it also holds data-quality metrics).
- `jobs.type` CHECK += `experiment`; `artifacts.kind` += `fold_assignment`, `experiment_report`; `artifacts.owner_type` += `split_definition`, `experiment`.
- RLS, owner-consistency and version-in-dataset triggers, grants (`SELECT, INSERT` plus `UPDATE(status)` on recommendations only).

**Migration 0006 (Part E):** follows DATABASE_SCHEMA §3.5.
- `pipelines`, `pipeline_steps`, `approvals` (insert-only), `transformation_runs`, `version_events` (insert-only).
- `dataset_versions`: FK `transformation_run_id → transformation_runs`; new nullable `model_pipeline_spec jsonb` (D1).
- `jobs.type` += `impact_preview`, `commit`, `revert`; partial unique index: one `commit`/`revert` job per dataset while queued or running.
- `artifacts.kind` += `impact_preview`; `owner_type` += `pipeline`, `transformation_run`.
- Approval rows store `approved_hash`, `impact_shown`, the displayed `statement`, `approved_by` and timestamp. Transformation runs store `spec_snapshot`, `spec_hash`, `approval_ids`, step reports with learned parameters, seeds, library versions, `executed_by`, timestamps and parent/result version IDs.

## 6. API changes

| Part | Endpoints |
|---|---|
| A | `GET /api/v1/ops?domain=`: catalogue `{op, version, family, stage, risk, destructive, kind, title, description, when_to_use, downsides, params_schema, column_types}` |
| D | `GET /analyses/{rid}/recommendations`, `PATCH /recommendations/{id}` (dismiss), `GET /versions/{vid}/splits/recommended`, `POST /versions/{vid}/splits`, `GET /splits/{sid}`, `POST /splits/{sid}/experiments`, `GET /versions/{vid}/experiments`, `GET /experiments/{eid}` |
| E | Pipelines exactly as API_SPECIFICATION §8 (create, steps CRUD, validate, impact preview, step approve, confirm, commit with `Idempotency-Key`), `POST /experiment-candidates/{cid}/to-pipeline`, `GET /transformation-runs/{tid}`, `POST /datasets/{id}/revert`, `GET /versions/{vid}/code` (generated `pipeline.py`, owner only, `no-store`) |

All return 404 for another user's resources. Risk, destructive flag, kind and stage always come from the catalogue, never from the request.

## 7. Security and privacy

- **No external services.** No LLM, no telemetry; rules are deterministic templates. All statistics are computed by the engine.
- **Learned parameters are data.** A median, mode or fill value is derived from cell values. They are stored only in owner-scoped, RLS-protected rows, never logged, and redacted in UI and reports for `sensitive`-role columns.
- **Broker messages** stay `{"job_id"}`. Pipeline specs live in Postgres and are loaded under RLS by the worker.
- **Never trust the client.** Owner comes from the session; hashes are recomputed server-side; approvals are verified in the API **and** again in the worker before reading data; source must equal head under the per-dataset row lock.
- **Immutability.** Nothing writes under `raw/` or overwrites a version object; the commit worker writes a new key only. Tests assert the original's bytes and every prior version's `file_sha256` are unchanged.
- **Code generation.** Templates only; parameters as `repr` of JSON primitives; column names as string literals; tested with hostile column names (quotes, newlines, `__import__`).
- **Resource limits.** KNN and iterative imputation are O(n²)/iterative: capped by config (initially ≤ 50,000 fit rows, ≤ 100 predictor columns) with `OP_LIMIT_EXCEEDED` and a reason, inside the existing worker memory caps.
- **Isolation.** New tables get RLS, owner triggers and AuthZ-matrix coverage. New route params extend `KNOWN_PARAMS`.

## 8. Testing strategy

| Layer | Tests |
|---|---|
| Op contract (every op) | Input frame never mutated; `_dm_row_id` preserved and unique; row count rules respected; column order stable; output dtypes as documented; deterministic for a fixed seed (identical `content_hash`) |
| Correctness | Learned statistics equal hand-calculated values on small fixtures (median, mean, mode ties, constant, KNN neighbours, iterative convergence) |
| **Leakage** | (1) Spy transformer: `_dm_row_id`s seen in `fit` ⊆ training rows. (2) Perturbation: changing test-row values changes neither learned parameters nor transformed training rows. (3) Test rows are filled with the training statistic, not their own. (4) From Part C, feature selection and resampling run only inside training folds; test-fold row IDs and counts are unchanged. (5) The target is never a predictor for imputation. |
| Validation | Unknown op/param, wrong column type, target/`id`/`_dm_row_id` columns rejected, constant of the wrong type, empty dataset, 1–3 rows, column all-missing in the fit rows, KNN `k` > fit rows (reduced with a warning), limits exceeded |
| Codegen equivalence | Each op's template run in a `python -I` subprocess gives the engine's result (`allclose` for floats, exact otherwise) |
| Library contract | Pinned scikit-learn (and later imbalanced-learn) signatures and behaviours the ops rely on |
| Integration (Part D/E) | RLS, AuthZ matrix, job handlers, job failure (crash → failed job, no partial version), approval bypass (unapproved, stale, edited, concurrent, non-head), version creation, revert, original immutability, experiment writes no versions |
| Frontend | Vitest component tests for recommendations, configurator, lab, comparison, approval dialog. No Playwright yet (Phase 3 gap). |
| Regression | Full backend and frontend suites after every part; ruff, mypy (strict on engine), eslint, tsc |

## 9. Part A — engine architecture and missing-value handling

**Goal:** the preprocessing engine foundation and every missing-value op, with leakage-proof fit/transform separation, validation, step reports and codegen fragments. No database or UI change beyond the read-only catalogue.

### 9.1 Ops

| Op | Implementation | Column types | Stage | Kind | Risk |
|---|---|---|---|---|---|
| `impute.mean` | `SimpleImputer(strategy="mean")` | numeric | either | fitted | medium |
| `impute.median` | `SimpleImputer(strategy="median")` | numeric | either | fitted | medium |
| `impute.mode` | `SimpleImputer(strategy="most_frequent")`; ties → smallest value (scikit-learn rule), reported | numeric, categorical, boolean | either | fitted | medium |
| `impute.constant` | `SimpleImputer(strategy="constant", fill_value=…)`; value type-checked against the column | numeric, categorical, boolean | either | stateless_column | medium |
| `impute.knn` | `KNNImputer(n_neighbors=k, weights)` over numeric feature predictors. Predictors standardised with statistics from the fit rows before distances, then inverse-transformed (disclosed) | numeric | either | fitted | medium |
| `impute.iterative` | `IterativeImputer(max_iter, random_state=seed, sample_posterior=False)` (scikit-learn marks it experimental; disclosed). Convergence warnings captured into the step report | numeric | either | fitted | medium |
| `missing.indicator` | Adds `<column>__missing` (boolean) for an explicit column list. Stateless, so train and test always get the same columns | any feature | data | stateless_column | low |

Rules:
- The target, `id`, `group`, `time`, `weight` and `exclude` columns and `_dm_row_id` are never imputed and never used as KNN/iterative predictors.
- Datetime, text and identifier columns are rejected with a reason.
- Integer columns imputed with mean/median/KNN/iterative become float, reported as a dtype change. There is no silent rounding.
- A column with no observed value in the fit rows fails the step with `FIT_COLUMN_EMPTY`; it is never silently skipped.

### 9.2 Step report (per step)
Rows in/out; columns added; per column: missing before/after, cells imputed, learned value(s) (redacted for sensitive columns), dtype change; fit-row count; warnings (ties, convergence, reduced `k`, dtype change); duration.

### 9.3 Explanations for the UI
Each catalogue entry carries deterministic `description`, `when_to_use` and `downsides` text. For example, mean imputation is "sensitive to skew and outliers; shrinks variance"; KNN is "slow on large data; needs comparable scales"; iterative is "can overfit relationships; experimental in scikit-learn". Part D's rules use these together with diagnostics evidence (missing %, skewness, sample size, task type, domain) so that mean imputation is never the default for skewed columns.

### 9.4 API
`GET /api/v1/ops` (authenticated; catalogue only; no data access), with its isolation-matrix entry and regenerated OpenAPI client.

### 9.5 Done when
All Part A tests in §8 pass, including leakage tests (1)–(3) and (5); there are codegen-equivalence tests for all seven ops; regression suites are green; ruff, mypy (strict) and tsc are clean; ML_ENGINE_DESIGN, ARCHITECTURE, API_SPECIFICATION and IMPLEMENTATION_ROADMAP are updated; and remaining limitations are listed.

### 9.6 Not in Part A
Recommendations, experiments, splits, persistence, approvals, versions and UI pages (Parts D–E); other op families (Parts B–C).

## 10. Later parts (summary)

- **B:** `outliers.flag` (IQR / z / MAD, fitted thresholds), `outliers.cap_iqr`, `outliers.winsorize` (percentile caps, destructive, high risk), `outliers.robust_transform` (rank/quantile), `outliers.isolation_forest_flag` (flag only), `rows.drop_by_ids` (destructive, explicit approval). Every outlier op reports affected rows and an information-loss estimate; none deletes rows automatically. Encoding: one-hot (`handle_unknown="ignore"`, capped levels), ordinal (only with an explicit user order), frequency (fitted on training rows). Scaling: standard, min-max, robust, none, with model-sensitivity notes.
- **C:** log/log1p (validated against zero and negative values in the fit rows and at transform), Yeo-Johnson, quantile; feature selection (variance, correlation, mutual information, univariate tests) as `model` stage, in-fold, never removing the target, with warnings for biomarker and domain-role columns; imbalance (class weights, random over/undersampling, SMOTE via imbalanced-learn) as `model` stage only, applied inside training folds; frozen fold splitter (stratified / group / time, duplicate-aware), since in-fold tests need it.
- **D:** recommendation rules (problem, columns, transformation, reason, benefit, downside, approval needed, evidence, alternatives including "do nothing"); Experiment Lab (≤ 3 candidates + "no preprocessing", data-quality metrics per candidate, optional baseline models per D3, comparability rules, no "best pipeline" claim from scores alone); Preprocessing workspace UI.
- **E:** pipelines, impact preview, per-step and pipeline approvals, commit worker, new versions with lineage, revert by server-side copy, roles copied forward, automatic re-profiling and diagnostics, before/after comparison (missing %, outliers, distributions, dimensions, changed values, rows affected, information loss, history), generated `pipeline.py`.


## 11. Part E — implementation plan (approved 2026-11-25; complete 2026-11-25)

**Closure (2026-11-25):** the owner's manual browser verification passed, and the closure checks on stage assignment and the version quota found no defect. The verified items, closure checks and non-blocking limitations are recorded in IMPLEMENTATION_ROADMAP ("Part E").

**Owner approval (2026-11-25):** this plan and decisions E1–E5 as recommended (affected-row values for up to 1,000 rows per step, owner only and redacted; a cap of 20 versions per dataset; model-stage steps kept as `model_pipeline_spec`; automatic re-profiling and diagnostics; a single milestone).

**Implementation notes and deviations** (details in ML_ENGINE_DESIGN §4.0 and §9; API §5 and §8; DATABASE_SCHEMA §3.5):
- Each step stores an explicit chosen stage, included in its hash. Data steps must precede model steps. `either` ops default to data, except after a model step in a converted Lab candidate.
- High-risk model-stage steps also need an approval (conservative reading of invariant 7).
- Every commit requires a current impact preview, even with only low-risk steps; the preview records the expected content hash, and the commit must reproduce it.
- `pipeline.py` is rendered in the API (engine sources read as text); no new job type.
- Orphan objects are removed by a new hourly maintenance task (`reconcile_orphans`). Storage gained `delete` (never for `raw/`) and `modified_at`.
- Hashes are hex text, as elsewhere in the engine. Approvals keep the step id without a foreign key, so history outlives deleted steps.
- Not built: archiving pipelines, deleting pipelines, version deletion and forks (out of scope).
- Lab candidates carry no stage, so conversion always assigns stages automatically (as above). Stages chosen explicitly on a pipeline step are kept as chosen.
- Known non-blocking limitations: a previous comparison result stays visible after switching the selected versions; "Re-run of" prefixes accumulate; performance at the maximum dataset size is not measured; destructive high-risk approvals and byte-level immutability are verified by automated tests, not in the browser.

**Goal.** Turn an approved preprocessing pipeline into a new immutable dataset version, with impact preview, hash-bound approvals, lineage, revert, before/after comparison and a reproducible `pipeline.py`. Parts A–D are reused unchanged. Part E adds the write path only.

### 11.1 What is reused (no redesign)

| Need | Existing component |
|---|---|
| Executing steps | `PreprocessExecutor.commit_transform` (refuses model-stage steps; closes the Part C gap) |
| Validation, step and spec hashes, risk, destructive flag, stage | `validate_steps`, `spec.step_hash` / `spec_hash`, catalogue |
| Step reports, lineage, redaction | `StepReport`, `feature_lineage`, `redact` |
| Version write | Ingest's pattern in `datasets/jobs.py`: write the Parquet object, compute `file_sha256` and `content_hash`, insert the version under a dataset row lock, chain the profile and diagnostics jobs |
| Jobs, cancellation, failure hooks, idempotency | `jobs` table, `register_handler`, `register_failure_hook`, `Idempotency-Key` handling (as in experiments) |
| Pipelines from the lab | `experiment_candidates.steps` and `recommendations` |
| Code | `codegen.render_steps` plus a loader and saver header |
| UI | Lab components (step rows, risk and stage badges), `DatasetNav`, query hooks |

### 11.2 Scope

1. **Pipelines.** Create a pipeline from scratch, from an experiment candidate, or from recommendations, always against the current head. Steps can be added, edited, removed and reordered.
   - The service recomputes `spec_hash` on every change and resets the state to `draft` (DATABASE_SCHEMA §4 rule 5).
   - `POST /pipelines/{pid}/validate`.
   - **Model-stage steps (D1)** stay in the pipeline as the version's `model_pipeline_spec`: stored, shown and emitted in code, but never executed into the data. Only `data` and `either` steps are materialised.
2. **Impact preview** (job `impact_preview`, queue `execute`): a dry run of `commit_transform` on the head version. It writes only an `impact_preview` artifact bound to `spec_hash`, containing per step:
   - rows before and after;
   - columns added and removed;
   - values changed per column;
   - before/after summary statistics;
   - up to 1,000 affected row ids.
3. **Approvals (insert-only).**
   - Every **high-risk** step needs its own approval, bound to its `step_hash` and the previewed `spec_hash`. Destructive steps also need `acknowledge_destructive`.
   - A **pipeline confirmation** bound to `spec_hash` is needed if any medium-risk step exists.
   - Any edit invalidates both (`409 APPROVAL_STALE`).
   - Each approval stores the exact impact numbers and statement that were displayed.
4. **Commit** (job `commit`, queue `execute`, `Idempotency-Key`). The API verifies all hashes, approvals and that the source is still the head. The worker verifies everything again before reading data. Then it:
   - runs `commit_transform`;
   - writes the new Parquet object;
   - inserts a `transform` version (`fitted_on_full_data` when any fitted step ran) together with a `transformation_run` (spec snapshot, approval ids, step reports, seeds, library versions) in one transaction, under the dataset row lock;
   - moves the head, copies roles forward, and chains re-profiling and diagnostics.

   One mutation job per dataset is allowed at a time (partial unique index). A crash leaves no version (an object written without a version row is removed by reconciliation).
5. **Revert** (job `revert`): creates a new `revert` version whose content is a server-side copy of a chosen earlier version, with a `version_events` row. History stays linear and nothing is deleted.
6. **Before/after comparison** (`engine/preprocess/compare.py`, job `compare_versions`), counts and statistics only:
   - schema diff and row-count diff;
   - per-column missing share, outlier counts and distribution summary;
   - values changed and rows affected;
   - information-loss indicators;
   - KS test (numeric) and chi-square test (categorical) with effect sizes and limitations.
7. **History UI** (new History tab):
   - version list with lineage and badges for revert and fitted-on-full-data;
   - transformation run detail (spec, approvals, step reports);
   - revert action;
   - before/after comparison.

   A new Pipeline page (reusing the Lab step builder) shows the impact preview, the approval dialogs (with "Destructive" wording) and the commit action, plus a "To pipeline" button on lab candidates.
8. **Generated code:** `GET /versions/{vid}/code` (owner only, `no-store`) returns `pipeline.py`. It replays the version's chain: the original parse options, then every committed step, with refitting and no pickles, and includes the model-stage spec as a separate in-fold function. A test checks equivalence with the stored version (`content_hash`, or `allclose` for floats).

### 11.3 Out of scope (later phases)
- Export bundles and replay validation in a sandbox (Phase 9).
- Passport and readiness checklist.
- New row-removal ops (`rows.*`): none exist in the catalogue, and none are added.
- Version deletion, forks from non-head versions, holdout sets.
- AI Copilot.

### 11.4 Database (migration 0006; DATABASE_SCHEMA §3.5)
- **New tables:**
  - `pipelines`, `pipeline_steps` (unique `(pipeline_id, position)`, deferrable);
  - `approvals` (insert-only);
  - `transformation_runs` (insert-only; `result_version_id` unique);
  - `version_events` (insert-only).
- **`dataset_versions`:** FK `transformation_run_id → transformation_runs`; new nullable `model_pipeline_spec jsonb`. The immutability trigger is unchanged.
- **Enumerations:** `jobs.type` gains `impact_preview`, `commit`, `revert`, `compare_versions`, with a partial unique index of one queued or running `commit`/`revert` per dataset. `artifacts.kind` gains `impact_preview` and `version_comparison`; owner types `pipeline` and `transformation_run`.
- **Security:** RLS, owner, parent-consistency and version-in-dataset triggers on every table, reusing the 0005 functions. Grants: `pipelines` / `pipeline_steps` editable; everything else insert-only.
- **Limit:** a version cap per dataset (`QUOTA_EXCEEDED`); see decision E2.

### 11.5 API (API_SPECIFICATION §8 and §5, as designed)
- **Pipelines:** create; get; add, edit, delete and reorder steps; validate; impact preview (start and read, plus affected rows); approve a step; confirm the pipeline; commit; transformation runs.
- **Lab bridge:** `POST /experiment-candidates/{cid}/to-pipeline`.
- **Versions:** `POST /datasets/{id}/revert`; `POST /versions/{a}/compare/{b}` and its result; `GET /versions/{vid}/code`.
- **Isolation matrix:** new parameters `pipeline_id`, `step_id`, `transformation_run_id`, `candidate_id`.

### 11.6 Tests
- **Approval bypass:** commit refused when:
  - an approval is missing;
  - an approval is stale (step edited, reordered or deleted after approval);
  - the approved preview belongs to an older spec;
  - the source is no longer the head (a concurrent commit or revert);
  - another mutation is running;
  - hashes are forged in the request.

  The worker also refuses when approvals are deleted or invalidated between queueing and running.
- **Immutability:** the original file and every earlier version's `file_sha256` are byte-identical after commits and reverts; no object is overwritten; a revert copies content and never deletes.
- **Stage enforcement:** model-stage steps are never materialised (the version data equals the data steps alone; the model-stage spec is stored); resampled rows can never appear in a version.
- **Crash safety:** a worker crash at each stage leaves no partial version or run, and the head is unchanged.
- **Concurrency:** two commits on one head produce one success and one conflict.
- **Lineage and roles:** roles copied forward (dropped columns lose roles and are reported); re-profile and diagnostics chained.
- **Code:** `pipeline.py` reproduces the committed version in a `python -I` subprocess, including after a revert.
- **Isolation, plus UI:** RLS, grants, triggers and the isolation matrix. Vitest for the approval dialog, impact preview, history and revert.
- **Smoke:** the same live-stack smoke script, extended to cover pipeline, preview, approve, commit, compare and revert.

### 11.7 Owner decisions needed

| # | Decision | Recommendation | Alternative |
|---|---|---|---|
| E1 | Affected-row values in the impact preview | Show before/after values for up to 1,000 affected rows per step: owner only, `no-store`, sensitive columns redacted, values not logged. Approvals are hard to make meaningful without seeing what changes | Counts and statistics only |
| E2 | Versions per dataset | Cap of 20 (configurable), `QUOTA_EXCEEDED` beyond it | No cap until plans exist |
| E3 | Lab candidates with model-stage steps | "To pipeline" keeps them as the version's `model_pipeline_spec` (per D1) and shows that they will not change the data | Drop model-stage steps when converting |
| E4 | After commit or revert | Automatically re-profile and re-run diagnostics on the new version (2 jobs) | Leave it to the user |
| E5 | Delivery | One part with an internal order (DB/engine → API → worker → UI → docs) and one review at the end, as for Parts A–D | Split into E1 (pipelines, preview, approvals) and E2 (commit, revert, compare, code, UI), each reviewed |

### 11.8 Done when
- All tests in §11.6 pass, plus the full backend (including integration) and frontend suites, ruff, strict mypy, tsc, ESLint, the production build and Compose validation.
- The live-stack smoke test passes.
- The docs (ML_ENGINE_DESIGN §4 and §9, API §5 and §8, DATABASE §3.5, roadmap, this plan) are updated.
- A limitations report is delivered with an APPROVE / DO NOT APPROVE recommendation.
