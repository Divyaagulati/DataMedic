# DataMedic — API Specification (MVP)

| | |
|---|---|
| Status | Revision 2, MVP-scoped |
| Base path | `/api/v1` |
| Format | JSON. Errors use `application/problem+json` (RFC 9457). Uploads use `application/octet-stream`. |
| Source of truth | FastAPI-generated OpenAPI 3.1 → generated TypeScript client. This document is the design contract. |

## 1. Conventions

- **Auth**: session cookie `__Host-dm_session` (HttpOnly, Secure, SameSite=Lax, Path=/). Unsafe methods require `X-CSRF-Token` from `GET /auth/csrf`.
- **Ownership**: every resource belongs to the authenticated user. Another user's resource returns **404**.
- **IDs**: UUID strings. **Timestamps**: RFC 3339 UTC.
- **Pagination**: `?limit=50&cursor=…` → `{items, next_cursor}`.
- **Async**: `202 Accepted` → `{job: Job, …subject}`. Clients poll `GET /jobs/{id}`, and the response includes `poll_after_ms`.
- **Idempotency**: job-creating POSTs accept `Idempotency-Key` (UUID). Repeats return the original response for 24 h.
- **Limits**: `429 JOB_LIMIT_EXCEEDED` or `RATE_LIMITED` (with `Retry-After`); `409 QUOTA_EXCEEDED`; `413 UPLOAD_TOO_LARGE`; `422 LIMIT_EXCEEDED` (cells or columns, from ingest via the job error).
- **Data-bearing responses** (previews, finding rows, impact rows, downloads) send `Cache-Control: no-store`.

### 1.1 Shared schemas

```ts
type Severity = "info" | "low" | "medium" | "high" | "critical";
type FindingStatus = "passed" | "warning" | "failed" | "not_run" | "error";
type Risk = "low" | "medium" | "high";
type OpKind = "stateless_row" | "stateless_column" | "fitted" | "row_filter_rule" | "row_filter_fitted";
type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancel_requested" | "cancelled";
type TaskType = "binary" | "multiclass" | "regression";

interface Job {
  id: string; type: string; status: JobStatus;
  progress: { phase: string; pct: number } | null;
  error: { code: string; message: string; retryable: boolean } | null;
  subject: { type: string; id: string } | null;
  queued_at: string; started_at: string | null; finished_at: string | null;
  poll_after_ms: number;
}

interface Step { op: string; op_version?: string; params: Record<string, unknown>; columns?: string[]; }

interface StepView extends Step {
  id: string; position: number;
  risk: Risk; destructive: boolean; kind: OpKind;      // server-derived
  step_hash: string;
  approval: { id: string; valid: boolean; created_at: string } | null;
  from_recommendation_id: string | null;
}

interface Finding {
  id: string; check_id: string; check_version: string; domain: string; category: string;
  status: FindingStatus; severity: Severity; title: string;
  explanation: string;                                  // deterministic template text
  evidence: Record<string, unknown>; limitations: string[];
  not_run_reason: string | null; error_code: string | null;
  columns: string[]; has_row_refs: boolean;
  sampled: boolean; sample_info: { n: number; seed: number; strategy: string } | null;
}

interface Problem {
  type: string; title: string; status: number; detail?: string; instance?: string;
  code: string; request_id: string; errors?: { pointer: string; code: string; message?: string }[];
}
```

## 2. Auth and account

| Method & path | Request | Response / errors |
|---|---|---|
| `GET /auth/csrf` | — | `200 {csrf_token}` |
| `POST /auth/signup` | `{email, password, display_name?}` | `202` (identical whether or not the email exists); a verification email is sent |
| `POST /auth/verify-email` | `{token}` | `204` · `AUTH_INVALID` |
| `POST /auth/login` | `{email, password}` | `200 {user}` + Set-Cookie (session rotated) · `AUTH_INVALID` (generic) · `RATE_LIMITED` (progressive delay) |
| `POST /auth/logout` | — | `204` |
| `POST /auth/password-reset/request` | `{email}` | `202` always |
| `POST /auth/password-reset/confirm` | `{token, new_password}` | `204`, revokes all sessions |
| `GET /auth/me` | — | `200 {user: {id, email, display_name, email_verified, plan}}` |
| `GET /auth/sessions` · `DELETE /auth/sessions/{id}` | — | list · `204` |
| `GET /account/usage` | — | `200 {limits: {...plan_limits}, usage: {storage_bytes, dataset_count, running_jobs, queued_jobs}}` |
| `DELETE /account` | `{password, confirm: "DELETE"}` | `202 {deletion_request}` |
| `GET /deletion-requests/{id}` | — | Status + receipt |
| `GET /audit-events` | — | The user's own security events |

## 3. Datasets and upload

### 3.1 Create dataset (reserve upload)
`POST /datasets`
```json
{ "name": "ICU cohort 2024", "filename": "cohort.xlsx", "size_bytes": 18234112 }
```
`201 { "dataset": { "id": "…", "name": "ICU cohort 2024", "status": "upload_pending", "domain": "generic" }, "max_upload_bytes": 209715200 }`

Errors:
- `413 UPLOAD_TOO_LARGE` (declared size > 200 MB);
- `415 UNSUPPORTED_FORMAT` (extension not in csv/tsv/txt/xlsx/xls/parquet/json/jsonl; a UX pre-check only);
- `409 QUOTA_EXCEEDED` (dataset count or storage headroom);
- `403 FORBIDDEN` (email not verified).

### 3.2 Upload content (streamed)
`PUT /datasets/{id}/content`
- Headers: `Content-Type: application/octet-stream`, `Content-Length` (required), `X-CSRF-Token`.
- Body: the raw file bytes.
- `202 { "job": Job(type="ingest"), "dataset": {..., "status": "validating"} }`
- Errors:
  - `411 LENGTH_REQUIRED`;
  - `413 UPLOAD_TOO_LARGE` (header, or the byte count exceeding the declared length);
  - `409 CONFLICT` (content already uploaded);
  - `422 VALIDATION_FAILED` (`Content-Length` ≠ declared size).

### 3.3 Read and manage
| Method & path | Notes |
|---|---|
| `GET /datasets?status&limit&cursor` | `{id, name, domain, status, head_version: {id, version_number, n_rows, n_cols}, readiness_status?, created_at}` |
| `GET /datasets/{id}` | Detail + `health_summary` (latest non-stale analysis on head) + `domain_changed_at` |
| `PATCH /datasets/{id}` | `{name?, domain?}`. Changing `domain` sets `domain_changed_at`, and existing runs become stale by semantics hash. |
| `GET /datasets/{id}/sheets` | `{sheets: [{name, n_rows_estimate}]}` (XLSX/XLS) |
| `POST /datasets/{id}/reparse` | `{parse_options: {delimiter?, encoding?, sheet?, header_row?, json_orient?}}` → `202 {job}`. Creates a **new `reparse` version** (D3). `409 REPARSE_NOT_ALLOWED` once any transform or revert version exists. |
| `GET /versions/{vid}/preview` | `{version_id, columns: [{name, dtype, inferred_type}], rows, n_rows_total, truncated}` (≤ 20 rows, cells ≤ 200 chars, `_dm_row_id` hidden; read from the preview artifact, sha256-verified) |
| `POST /datasets/{id}/parse` | **Phase 2.** Queues parsing for a stored upload that has no version (status `validated`, e.g. uploaded in Phase 1, or `parse_failed`) → `202 {dataset, job}`. `409 CONFLICT` otherwise. Not a reparse: no options, no new version kind. |
| `GET /datasets/{id}/versions` | (also listed in §5) |
| `GET /versions/{vid}/profile` | **Phase 2.** `{profile: Profile \| null, job: Job \| null}`: the latest stored profile of the version and the latest `profile` job (poll it while active). `Profile` = `{id, version_id, job_id, created_at, profiler, profiler_version, summary, columns[], correlations, sampling, limitations, engine_version, library_versions}`; see DATABASE_SCHEMA `version_profiles`. |
| `POST /versions/{vid}/profile` | **Phase 2.** `202 {job}`: run profiling again (new row; old rows kept). `409 CONFLICT` while a profile job for the version is queued or running. |

`DatasetOut` (list and detail) carries `head_version: {id, version_number, kind, n_rows, n_cols, created_at} | null`.
| `DELETE /datasets/{id}` | `202 {deletion_request}` |

## 4. Schema, roles, domains

| Method & path | Request / response |
|---|---|
| `GET /versions/{vid}/schema` | `{columns: [{name, original_name, dtype, inferred_type, n_missing, n_unique}]}` |
| `GET /versions/{vid}/roles` | `{roles: [{column, role, task_type?, source}], suggestions: [{column, role, confidence, reason}], semantics_hash}` |
| `PUT /versions/{vid}/roles` | `{roles: [{column, role, task_type?}]}`. `task_type` is required for and only allowed on `target` (D7). At most one target. Returns `{semantics_hash, stale_runs: n}`. |
| `GET /domains` | `[{key, name, version, roles: [...], check_count, op_count}]` |
| `GET /datasets/{id}/reference-ranges` | Biomedical: `{items: [{column_name, unit, low, high, source}]}` |
| `PUT /datasets/{id}/reference-ranges` | `{items: [{column_name, unit?, low, high, source}]}` (≤ 500 rows, `source` required). Used by `bio.labs.plausibility`. Changing it changes the semantics hash. |

**Phase 3 implementation** (paths under `/api/v1`):

| Method & path | Notes |
|---|---|
| `GET /domains` | `{items: [{key, name, version, description, roles: RoleSpec[], check_count}]}`; `generic` = universal checks only |
| `GET /checks?domain=` | Catalogue `{items: [{id, version, domain, category, title, description, requirements: {roles, roles_any, task_types, min_rows}, limitations, expensive}]}` |
| `PATCH /datasets/{id}` | `{name?, domain?}` → `DatasetOut` (with `domain_changed_at`). Unknown domain → `422` (`UNKNOWN_DOMAIN`). |
| `GET /versions/{vid}/roles` | `{version_id, domain, roles: [{column, role, task_type}], available_roles: RoleSpec[], semantics_hash}` |
| `PUT /versions/{vid}/roles` | `{roles: [{column, role, task_type?}]}` replaces the set. `422 VALIDATION_FAILED` with per-item codes: `COLUMN_NOT_FOUND`, `ROLE_NOT_AVAILABLE` (another domain's role), `ROLE_TYPE_MISMATCH`, `DUPLICATE_ROLE`, `TOO_MANY`, `TASK_TYPE_REQUIRED`, `TASK_TYPE_MISMATCH` (binary needs 2 values, multiclass 3–50, regression numeric), `TASK_TYPE_NOT_ALLOWED`. Suggestions are not implemented. |

## 5. Versions

| Method & path | Notes |
|---|---|
| `GET /datasets/{id}/versions` | Linear list: `{id, version_number, kind, parent_version_id, restored_from_version_id, n_rows, n_cols, content_hash, is_head, fitted_on_full_data, created_at}` |
| `GET /versions/{vid}` | Plus lineage (replay chain summary), transformation run (spec, approvals, step reports), library versions |
| `POST /datasets/{id}/revert` | `{version_id, reason?}` → `202 {job}`. Creates a new `revert` version copying `version_id`'s content. `409 CONFLICT` if another mutation job is active. `409 QUOTA_EXCEEDED` on the version cap. |
| `POST /versions/{a}/compare/{b}` | `202 {job}` → result: schema diff, row-count diff, per-column SciPy tests (KS for numeric, chi-square for categorical) with p-values, effect sizes and limitations |
| `GET /version-comparisons/{job_id}` | Comparison result |

**Phase 4 Part E implementation** (2026-11-25):
- `GET /datasets/{id}/versions` items gain `parent_version_id`, `restored_from_version_id`, `transformation_run_id`, `fitted_on_full_data` and `has_model_spec`.
- `GET /versions/{vid}` adds `model_pipeline_spec`.

| Method & path | Notes |
|---|---|
| `POST /datasets/{id}/revert` | `{version_id}` + optional `Idempotency-Key` → `202 {job}` (`200` for a repeated key). Errors: `409 CONFLICT` (already current, or another commit/revert active), `409 QUOTA_EXCEEDED` |
| `GET /datasets/{id}/version-events` | `{items: [{id, event: created \| reverted, version_id, related_version_id, details: {roles, …}, created_at}]}` |
| `POST /versions/{vid}/compare/{other_version_id}` | `202 {job}`: counts and statistics only |
| `GET /version-comparisons/{job_id}` | `{job, result \| null}` (ML_ENGINE_DESIGN §4.0) |
| `GET /versions/{vid}/code` | `text/x-python` attachment `pipeline_v{n}.py` (owner only, `no-store`) |

Individual version deletion is a V2 feature. In the MVP, versions are removed with their dataset.

## 6. Analyses (diagnostics, leakage audit)

`POST /versions/{vid}/analyses`
```json
{ "kind": "diagnostics", "suites": ["universal", "biomedical"], "checks": null, "split_definition_id": null, "config": { "seed": 42 } }
```
`202 { "job": Job, "analysis_run": { "id": "…" } }`

- `kind`: `diagnostics` | `leakage_audit` | `privacy_scan`.
- For `leakage_audit`, `split_definition_id` is optional. Without it, split-dependent checks use the recommended split. If no split can be derived, they report `not_run`.
- `409 DATASET_NOT_READY`.

| Method & path | Notes |
|---|---|
| `GET /versions/{vid}/analyses` | `[{id, kind, suites, job: {status}, summary, stale: bool, created_at}]` |
| `GET /analyses/{rid}` | Inputs, summary (by status, severity, domain; `check_errors`), `stale`, engine and library versions |
| `GET /analyses/{rid}/findings?domain&severity&status&category&column&limit&cursor` | `Finding[]` |
| `GET /findings/{fid}` | Full finding |
| `GET /findings/{fid}/rows?limit&cursor` | `{rows: [{row_id, values: {...}}]}` (owner only, `no-store`) |
| `GET /checks?domain` | Catalogue: `{id, version, domain, category, description, requirements, general_limitations}` |

**Phase 3 implementation**: `POST /versions/{vid}/analyses` takes no body and runs the dataset domain's full check set → `202 {job}`, or `409 CONFLICT` while one is active. Ingest also chains one automatically. `GET /versions/{vid}/analyses/latest` → `{run: AnalysisRun | null, job: Job | null, current_semantics_hash}`. `GET /analyses/{rid}` → `AnalysisRun`: `{id, version_id, job_id, domain, created_at, stale, roles_snapshot, summary: {total, by_status, detected_by_severity, detected_by_domain, detected_by_category, check_errors}, config, engine_version, library_versions, findings: Finding[]}`. Findings are embedded, ordered by check; the paginated findings endpoint, `/findings/{fid}` and `/findings/{fid}/rows` are not built yet. Finding = `{id, check_id, check_version, domain, category, status, severity, title, explanation, evidence, columns, affected_rows, affected_pct, recommendation, limitations, not_assessed_reason, error_code, sampled, sample_info}`.

## 7. Recommendations (proposal only)

| Method & path | Notes |
|---|---|
| `GET /analyses/{rid}/recommendations?domain&risk&destructive&status` | List |
| `GET /recommendations/{id}` | `{id, rule_id, domain, title, rationale, proposed_steps: [{...Step, risk, destructive, kind}], risk, destructive, expected_effect, alternatives, finding_ids, status}` |
| `PATCH /recommendations/{id}` | `{status: "dismissed"}` |

No endpoint under `/recommendations` changes data.

**Phase 4 Part D implementation** (2026-11-24). Recommendations are proposals and never change data.

| Method & path | Notes |
|---|---|
| `POST /versions/{vid}/recommendations` | Generate from the current diagnostics, profile, roles and domain. `201` new set, `200` unchanged inputs (same set), `409 CONFLICT` without current diagnostics. Body: `RecommendationSet` |
| `GET /versions/{vid}/recommendations` | `{set: RecommendationSet \| null}` (latest; `stale` when its inputs changed) |
| `PATCH /recommendations/{rid}` | `{status: "open" \| "used" \| "dismissed"}` |

`RecommendationSet` = `{id, version_id, analysis_run_id, profile_id, rules_version, engine_version, domain, task_type, created_at, stale, recommendations, conflicts, candidates, split_plan, not_assessed, not_recommended, advisories, inputs, limitations}`:
- `recommendations` items are `{id, key, rank, status, op, stage, risk, destructive, columns, params, detail}`. `detail` is the full recommendation (ML_ENGINE_DESIGN §3.5): reason, evidence, expected benefit, risks, information loss, domain and task compatibility, required decisions, evidence strength, limitations, alternatives and ranking factors.
- `conflicts` items are `{id, group, topic, trade_off, options, allow_none, how_to_decide}`.
- `candidates` items are `{key, label, rationale, steps, recommendation_ids, skipped}`.

## 8. Pipelines: impact preview, approval, commit

**Phase 4 Part A implementation**: `GET /api/v1/ops?family=` (authenticated; no data access) → `{items: [{op, version, family, title, description, when_to_use, downsides[], limitations[], stage, kind, risk, destructive, column_types[], adds_columns, max_columns, uses_columns, needs_target, params: [{name, type, description, default, required, minimum, maximum, choices}]}]}`. `stage` is `data | model | either` (PHASE4_PLAN §2.1). `kind` adds `training_only` in Part C (resampling and class weights: only the training table changes). `uses_columns = false` means the op works on whole training rows and takes no columns; `needs_target` means it reads the target of the training rows. The target and task type are resolved from the declared roles, never accepted as request parameters. `type` is `int | float | bool | choice | scalar | columns | text | text_list` (the last two since Part B). Families: `missing`, `outliers`, `encoding`, `scaling`, `transform`, `selection`, `imbalance` (the last three since Part C). The other endpoints in this section arrive in Parts D–E.

| Method & path | Request | Response / errors |
|---|---|---|
| `GET /ops?domain` | — | `[{op, version, domain, risk, destructive, kind, description, params_schema, reversible_note}]` |
| `POST /datasets/{id}/pipelines` | `{name, from_recommendation_ids?, from_experiment_candidate_id?}` | `201 Pipeline`. `source_version_id` = current head. |
| `GET /pipelines/{pid}` | — | `{id, name, state, source_version_id, source_is_head, steps: StepView[], spec_hash, validation_report, impact_preview: {valid, job_id?}, approvals_needed: {steps: [step_id], pipeline: bool}}` |
| `POST /pipelines/{pid}/steps` | `Step & {position?}` | `201 StepView`. Recomputes `spec_hash` and resets `state` to `draft` (D9). |
| `PATCH /pipelines/{pid}/steps/{sid}` | `{params?, columns?, position?}` | `200`. New hash. Invalidates that step's approval and any pipeline confirmation. |
| `DELETE /pipelines/{pid}/steps/{sid}` | — | `204` (recomputes the hash) |
| `POST /pipelines/{pid}/validate` | — | `200 {valid, errors, warnings}` → `state=validated` |
| `POST /pipelines/{pid}/impact-preview` | — | `202 {job}` (dry run on the source version) |
| `GET /pipelines/{pid}/impact-preview` | — | `{spec_hash, per_step: [{step_id, op, risk, destructive, rows_before, rows_after, rows_removed, cols_added, cols_removed, values_changed: {col: n}, before_after_stats: {col: {before: {...}, after: {...}}}, sample_row_ids: [...]}]}` |
| `GET /pipelines/{pid}/impact-preview/rows?step_id&limit&cursor` | — | Affected rows' before/after values (owner only, `no-store`) |
| `POST /pipelines/{pid}/steps/{sid}/approve` | `{step_hash, spec_hash, acknowledge_destructive?: true}` | `201 Approval`. Only for `risk=high`. Requires a valid impact preview for `spec_hash`. `acknowledge_destructive` is required when the step is destructive. `409 APPROVAL_STALE` on a hash mismatch. |
| `POST /pipelines/{pid}/confirm` | `{spec_hash}` | `201 Approval(scope=pipeline)`. Required if any medium-risk step exists. |
| `POST /pipelines/{pid}/commit` | `{spec_hash}` + `Idempotency-Key` | `202 {job, transformation_run}`. Errors: `409 APPROVAL_REQUIRED`, `409 APPROVAL_STALE`, `409 CONFLICT` (source is no longer head, or another mutation is active), `422 VALIDATION_FAILED`, `409 QUOTA_EXCEEDED`. |
| `GET /transformation-runs/{tid}` | — | Run detail, step reports, `result_version_id` |

**Enforcement**: `commit` recomputes every step hash and the spec hash, then requires a valid step approval for every high-risk step, a valid confirmation if any medium-risk step exists, and source = head. The worker repeats all of these checks before reading data.

**Phase 4 Part E implementation** (2026-11-25). All responses are `no-store`; others' resources are 404. Hashes in requests are only compared with server-computed ones.

| Method & path | Request → response |
|---|---|
| `GET /datasets/{id}/pipelines` | `{items: PipelineSummary[]}` |
| `POST /datasets/{id}/pipelines` | `{name, steps: [{op, columns, params, stage?, recommendation_id?}]}` → `201 Pipeline` (source = head) |
| `POST /experiment-candidates/{cid}/to-pipeline` | `201 Pipeline` (candidate's steps, parameters and order; stages as in ML_ENGINE_DESIGN §4.0; the experiment is unchanged) |
| `GET /pipelines/{pid}` | `Pipeline` |
| `POST /pipelines/{pid}/steps` | `{op, columns, params, stage?, position?}` → `201 Pipeline` |
| `PATCH /pipelines/{pid}/steps/{sid}` | `{params?, columns?, stage?, position?}` → `Pipeline` |
| `DELETE /pipelines/{pid}/steps/{sid}` | → `Pipeline` (200, not 204, so the client gets the new hashes) |
| `POST /pipelines/{pid}/validate` | → `{valid, errors, warnings, data_steps, model_steps, spec_hash}` |
| `POST /pipelines/{pid}/impact-preview` | → `202 {job}` |
| `GET /pipelines/{pid}/impact-preview` | → `{current, job, summary}` |
| `GET /pipelines/{pid}/impact-preview/rows?step&offset&limit` | → `{step_index, columns, redacted_columns, rows: [{row_id, before, after}], rows_affected, rows_listed, rows_truncated, next_offset}` |
| `POST /pipelines/{pid}/steps/{sid}/approve` | `{step_hash, spec_hash, acknowledge_destructive}` → `201 Approval`. `409 APPROVAL_STALE` (hash mismatch or no current preview); `422 ACKNOWLEDGEMENT_REQUIRED`; `422 APPROVAL_NOT_NEEDED` (not high risk) |
| `POST /pipelines/{pid}/confirm` | `{spec_hash}` → `201 Approval(scope=pipeline)` |
| `POST /pipelines/{pid}/commit` | `{spec_hash}` + **required** `Idempotency-Key` → `202 {job, transformation_run_id}` (`200` for a repeated key). Errors: `409 APPROVAL_REQUIRED`, `409 APPROVAL_STALE`, `409 CONFLICT` (source not head, already committed, or another mutation active), `409 QUOTA_EXCEEDED`, `422 VALIDATION_FAILED`, `422 IDEMPOTENCY_KEY_REQUIRED` |
| `GET /transformation-runs/{tid}` | `{id, pipeline_id, source_version_id, result_version_id, spec_hash, steps, data_ops, model_pipeline_spec, step_reports, roles_report, approvals, seeds, engine_version, library_versions}` |

Response shapes:
- `Pipeline` = `{id, dataset_id, name, state, source_version_id, source_is_head, committed_version_id, spec_hash, validation, steps, approvals, readiness, preview_job, from_experiment_candidate_id}`.
- `steps` items: `{id, position, op, op_version, op_title, catalog_stage, stage, columns, params, risk, destructive, kind, step_hash, approved}`.
- `approvals` items: `{id, scope, approved_hash, spec_hash, impact_shown, statement, acknowledge_destructive, valid}`.
- `readiness` = `{ready, preview_current, high_risk_steps, confirmation_required, confirmation_valid, stale_approvals, problems}`.

Job types `impact_preview`, `commit` and `revert` run on queue `execute`; `compare_versions` runs on `analysis`.

## 9. Experiments (isolated; never write versions)

`GET /versions/{vid}/splits/recommended` → `{strategy, reason, params, warnings}`

`POST /versions/{vid}/splits`
```json
{ "strategy": "stratified_group_kfold", "n_splits": 5, "group_column": "patient_id", "stratify_column": "diagnosis", "seed": 42 }
```
`201 {split_definition, cohort, warnings}`. Folds are frozen by `_dm_row_id`. Whole groups (declared group role ∪ duplicate clusters) are assigned together. Warnings such as `GROUP_ROLE_IGNORED` and `TIME_ROLE_IGNORED` appear when an override contradicts the roles.

`POST /splits/{sid}/experiments`
```json
{
  "models": ["dummy", "linear", "hgb", "xgb"],
  "metrics": ["roc_auc", "average_precision", "balanced_accuracy", "brier"],
  "candidates": [
    { "label": "A: impute labs", "steps": [ { "op": "impute.median", "params": {}, "columns": ["creatinine"] } ] },
    { "label": "B: A + drop duplicates", "steps": [ { "op": "impute.median", "params": {}, "columns": ["creatinine"] }, { "op": "rows.drop_exact_duplicates", "params": {} } ] }
  ]
}
```
`202 {job, experiment}`

- "No preprocessing" is always included automatically.
- Up to 3 candidates and 4 models (`QUOTA_EXCEEDED` otherwise).
- Candidates may include **high-risk and destructive steps without approval**. The response and results label them.
- Target and task type come from the version's target role. `422 VALIDATION_FAILED` if no target is assigned.

| Method & path | Notes |
|---|---|
| `GET /experiments/{eid}` | `{job, cohort: {rows, excluded_missing_target, duplicate_groups}, summary: {leaderboard: [{candidate, model, metric: {mean, sd}}] (comparable candidates only), paired_differences: [{a, b, model, metric, per_fold: [..], mean_diff, n_folds, test_rows}] (comparable pairs only), not_comparable: [{candidate, reason, folds_affected, rows_unscored}], warnings}, candidates: [{id, label, max_risk, has_destructive, comparable, training_rows_removed: {per_fold: [..], per_step: {..}, total, mean}, effects}]}` |
| `GET /experiments/{eid}/results?candidate&model` | Per-fold metrics |
| `POST /experiment-candidates/{cid}/to-pipeline` | `201 Pipeline` built from the candidate's steps against the current head. The normal impact-preview and approval flow applies. |

Warning codes: `SUSPICIOUSLY_HIGH_SCORE`, `FEW_SAMPLES_PER_CLASS_PER_FOLD`, `DUMMY_COMPETITIVE`, `FITTED_ON_FULL_DATA_SOURCE`, `ROWSET_DIFFERS` (retained training rows differ between candidates), `NOT_DIRECTLY_COMPARABLE` (a candidate was not scored on the full shared test cohort), `DUPLICATE_GROUPS_APPLIED` (duplicates were grouped before splitting), `GROUP_ROLE_IGNORED`, `TIME_ROLE_IGNORED`.

**Phase 4 Part D implementation** (2026-11-24). The split is chosen per experiment, not by separate split endpoints:

| Method & path | Notes |
|---|---|
| `POST /versions/{vid}/experiments` | `ExperimentIn` + optional `Idempotency-Key` → `202 {experiment, job}` (`200` with the original for a repeated key) |
| `GET /versions/{vid}/experiments` | `{items: ExperimentSummary[]}` (latest 100) |
| `GET /experiments/{eid}` | `ExperimentSummary` plus `config`, `recommendation_set_id`, `split`, `summary`, `library_versions` and `candidates` |
| `POST /experiments/{eid}/rerun` | `202`: a new experiment with exactly the stored configuration (roles snapshot, split, seeds, candidates) |
| `POST /jobs/{jid}/cancel` | Cancels a queued experiment, or a running one before its next fold |

`ExperimentIn` = `{name, candidates, models?, task?, n_clusters?, split?, seed?, recommendation_set_id?}`:
- `candidates`: 1–6 items `{label, steps, recommendation_ids?}`; the baseline is added automatically.
- `models?`: `dummy`, `linear`, `hgb`; with `task: "clustering"`, `kmeans` only.
- `n_clusters?`: 2–20, required for clustering.
- `split?`: `{method?, n_splits (2–10), seed}`. With no `method`, it is chosen from roles. Group and time columns always come from roles.

Other response shapes:
- `ExperimentSummary` = `{id, version_id, name, task_type, target_column, models, seed, config_hash, rerun_of_id, created_at, completed_at, job}`.
- `split` = `{id, method, n_splits, seed, target_column, group_column, time_column, assignment_sha256, cohort, fold_stats, warnings}`.
- `summary` = `{task, primary_metric, metrics, metric_info, classes, positive_class, cohort, folds, comparisons, warnings, split_warnings, limitations, model_notes}`.
- `candidates` items are `{id, key, label, is_baseline, steps, spec_hash, max_risk, has_destructive, stages, recommendation_ids, result}`, where `result` = `{status, comparable, comparability_reason, metrics, folds, data_quality, selection_stability, failures, confusion_matrix, per_class, …}`.

Errors:
- `422 VALIDATION_FAILED`, with JSON-pointer errors per candidate step (validator codes), `TARGET_REQUIRED`, `MODEL_INVALID`, `SPLIT_ROLE_MISSING`, `DUPLICATE_LABEL` or `EMPTY_CANDIDATE`;
- `409 QUOTA_EXCEEDED`: more than 6 candidates, more than 200,000 rows or 5,000,000 cells, or more than 300 model fits;
- `409 DATASET_NOT_READY`.

Experiments never write versions, and no candidate step is ever committed. The comparison is descriptive: no winner is selected.

## 10. Data Health Passport

| Method & path | Notes |
|---|---|
| `POST /versions/{vid}/passports` | `202 {job, passport: {id}}` |
| `GET /versions/{vid}/passports` | History: `[{id, readiness_status, created_at, outdated: bool}]`. `outdated` is true when newer analyses or experiments exist, or the semantics hash changed. |
| `GET /passports/{pid}` | `{id, version, readiness_status, readiness: [{id, title, status, evidence}], content: {...}, source_analysis_run_ids, source_experiment_ids, created_at}` |
| `GET /passports/{pid}/download?format=html\|json` | Streamed file (`Content-Disposition: attachment`, `no-store`). Audited. |
| `GET /versions/{vid}/readiness` | Live checklist computed from metadata and the latest non-stale runs (not a stored passport) |

## 11. Exports

| Method & path | Notes |
|---|---|
| `POST /versions/{vid}/exports` | `{formats: ["parquet","csv"], csv_formula_escape: false, keep_row_id: false}` → `202 {job, export}`. Builds the passport, `pipeline.py`, `requirements.txt` and `manifest.json`, then validates by replay. |
| `GET /exports/{xid}` | `{job, validation_status, validation_report, manifest, passport_id, warnings, expires_at}` |
| `GET /exports/{xid}/download` | Streamed ZIP through the API (`attachment`, `no-store`). Audited. |
| `DELETE /exports/{xid}` | `204` |

## 12. Jobs

| Method & path | Notes |
|---|---|
| `GET /jobs?status&type&dataset_id&limit&cursor` | |
| `GET /jobs/{jid}` | `Job` (with `poll_after_ms`) |
| `POST /jobs/{jid}/cancel` | `202` |

## 13. Health

`GET /healthz` (liveness), `GET /readyz` (internal), `GET /meta/version` → `{api_version, engine_version}`.

## 16. Training and reliability audit (Phase 5, implemented 2026-11-25)

All owner-scoped (others' resources → 404); no row values or per-row predictions in any response.

| Method & path | Notes |
|---|---|
| `GET /ml/models` | Models, grids, defaults, metric definitions and conventions, the 20 audit checks, status rules |
| `POST /versions/{vid}/training-plan` | Body = `TrainingRunIn`. Advisory: `{valid, errors[], warnings[], task, target, features, excluded_columns, datetime_columns, sensitive_columns, models, tuning{requested, planned, fallback_reason, grids}, estimated_fits, fit_limit, split, eligibility{mode, evaluation_version_id, reasons[], replay}, evaluation_mode, evaluation_version_id, preprocess_steps, spec_hash}`. Nothing stored. |
| `POST /versions/{vid}/training-runs` | `Idempotency-Key`. Body `TrainingRunIn = {name?, models? ⊆ dummy,linear,random_forest,hgb, tuning{mode: nested\|defaults}, split{method?, n_splits 2–10, seed}, preprocess_steps? (null = the version's modelling spec), seed, allow_exploratory}` → `202 {training_run, job}` (200 on key replay). `422` (errors with pointers: `TARGET_REQUIRED`, `TASK_TARGET_MISMATCH`, `TOO_FEW_ROWS`, `NO_FEATURES`, step errors), `409 EVALUATION_BLOCKED`, `409 QUOTA_EXCEEDED` (fits, rows, cells, runs per dataset). |
| `GET /datasets/{id}/training-runs` | Summaries: mode, job, latest audit status and `headline = {state: available\|unavailable\|pending\|failed\|cancelled, metric, label, higher_is_better, model, mean, sd, n_folds, total_folds, baseline_mean (dummy), exploratory, note, exploratory_note, reason}`: the highest observed fold mean of the primary metric among compared non-dummy models, respecting the metric's direction; never a declared winner (closure 2026-11-26) |
| `GET /training-runs/{rid}` | Configuration, eligibility, replay proof, split, tuning, preprocessing, summary, library versions, OOF artifact SHA-256, `evaluations[]` (metrics: folds / reasons / pooled / bootstrap; details: curves, calibration, residuals, confusion, per-class, error slices, training scores; fold results; selection; importance; failures), and `privacy = {policy, protected_columns: [{column, reasons: sensitive_at_training\|sensitive_now\|derived_from_sensitive}], withheld_error_slices, target_values_withheld}`. Sensitive-column protection is re-applied on every read; stored rows are not modified |
| `POST /training-runs/{rid}/rerun` | Identical stored configuration (roles snapshot, split, seeds, models, tuning, steps); exploratory stays exploratory |
| `POST /training-runs/{rid}/audit` | Re-run the audit of a completed run → `202 {job}`; `409` while one is active |
| `GET /training-runs/{rid}/audit` | `{run: {summary.evaluation_status, config, …} \| null, findings: Finding[], stale, job}` |

Worker failures (job `error.code`): `CONFIG_CHANGED`, `INTEGRITY_FAILED`, `TASK_TARGET_MISMATCH`,
`CLASS_TOO_RARE`, `TOO_FEW_ROWS`, `LIMIT_EXCEEDED`, `REPLAY_NOT_ESTABLISHED`, `JOB_CANCELLED` (status
`cancelled`). Diagnostics endpoints (`/versions/{vid}/analyses/latest`, `/analyses/{id}`) never return audit runs.

## 14. Status codes

| Code | Use |
|---|---|
| 200/201/204 | Success |
| 202 | Job accepted |
| 400 | Malformed request |
| 401 | `AUTH_REQUIRED`, `AUTH_INVALID` |
| 403 | `CSRF_FAILED`, `FORBIDDEN` |
| 404 | Not found, or not yours |
| 409 | `CONFLICT`, `APPROVAL_REQUIRED`, `APPROVAL_STALE`, `DATASET_NOT_READY`, `REPARSE_NOT_ALLOWED`, `QUOTA_EXCEEDED` |
| 411 | `LENGTH_REQUIRED` |
| 413 | `UPLOAD_TOO_LARGE` |
| 415 | `UNSUPPORTED_FORMAT` |
| 422 | `VALIDATION_FAILED`, `LIMIT_EXCEEDED` |
| 429 | `RATE_LIMITED`, `JOB_LIMIT_EXCEEDED` |
| 500 | `INTERNAL` |
| 503 | Dependency unavailable (retryable) |

Ingest failures such as `FORMAT_MISMATCH`, `MALICIOUS_FILE`, `PARSE_FAILED` and `LIMIT_EXCEEDED` are reported in the ingest job's `error.code`, and the dataset `status` changes to `rejected` or `parse_failed`.

## 15. V2 additions (not in the MVP)

`/auth/mfa/*`, `/privacy/settings` + `/privacy/egress-log` + LLM explanation endpoint, `/jobs/{id}/events` (SSE), version forks (`POST /versions/{vid}/pipelines` from a non-head version), `DELETE /versions/{vid}`, holdout endpoints, presigned large-file upload, PDF passports.

## 17. MVP completion (Phase 6/7, implemented 2026-11-26)

All owner-scoped (another user's resource → 404), in the AuthZ matrix, responses `Cache-Control: no-store`.

| Endpoint | Behaviour |
|---|---|
| `POST /versions/{vid}/biomedical` | Queue a descriptive biomedical summary (job `biomedical_summary`) → `202 {job}`; `409` while one is active |
| `GET /versions/{vid}/biomedical` | `{summary (cohort, target, biomarkers, comparisons, batches, covariates, confounding, entities, small_cohort, disclaimer, limitations) \| null, summary_created_at, stale, job, diagnostics_available, biomedical_findings[], model_reliability[]}` |
| `GET /versions/{vid}/passport[?download=true]` | The Data Passport JSON (sections with `status: available \| not_available` and reasons; no score); `download=true` adds `Content-Disposition` with a sanitised filename |
| `POST /versions/{vid}/copilot` | `{question: 1–500 chars}` → `{assistant {name, kind: "built-in rule-based assistant", note}, topic, answer[], references[{section, label, href}], suggestions[]}`. Read-only |
| `POST /versions/{vid}/exports` | `{format: "csv", acknowledged: true, include_sensitive?, acknowledge_sensitive?}` → `202 {job, excluded_sensitive_columns}`; `422 ACK_REQUIRED`, `422 SENSITIVE_ACK_REQUIRED` |
| `GET /exports/{job_id}/download` | Streams the finished CSV (owner only; 404 otherwise). No storage URL is ever exposed |

## 18. Analytics Studio (Phase 8, implemented 2026-11-27)

Owner-scoped (another user's resource → 404), in the AuthZ matrix, `Cache-Control: no-store`.
Specification format and limits: ARCHITECTURE.md §18.

| Endpoint | Behaviour |
|---|---|
| `GET /versions/{vid}/analytics/columns` | `{rows, columns[{name, inferred_type, n_unique, missing_pct, roles, protected}], privacy_mode, min_cell, roles}` |
| `POST /versions/{vid}/analytics/query` | `{spec}` → `{result}`; `422` with pointers for invalid specs; `409 QUOTA_EXCEEDED` above the size limits; `429 ANALYTICS_BUSY` |
| `POST /versions/{vid}/analytics/export` | Same body → CSV of an aggregate, pivot, histogram, box, correlation or missingness result (formula-neutralised; scatter points refused) |
| `GET /versions/{vid}/compare/{other_version_id}` | Shared-bin comparison: dimensions, added/removed columns, row populations, per-column missingness, numeric stats and histograms, category counts, correlation changes, transformation effects |
| `POST /versions/{vid}/insights` · `GET /versions/{vid}/insights` | Queue (job `insights`) / read stored insights `{summary{insights[], counts, not_assessed, limitations, unit, sampled}, stale, job}` |
| `GET/POST /datasets/{id}/dashboards` | List / create `{version_id, name, description?, filters?}` |
| `GET/PATCH/DELETE /dashboards/{did}` | Read with widgets / rename, describe, set filters / delete |
| `POST /dashboards/{did}/duplicate` | `{version_id?, name?}` → `{dashboard, skipped_widgets}` |
| `GET /dashboards/{did}/compatibility?version_id=` | Per-widget compatibility with another version (defaults to its own) |
| `POST /dashboards/{did}/switch-version` | `{version_id, confirm: true}`; refused unless fully compatible |
| `GET /dashboards/{did}/export` | Dashboard specification JSON (`format: datamedic.dashboard/1`; no data) |
| `POST /dashboards/{did}/widgets` · `PATCH/DELETE …/widgets/{wid}` · `POST …/widgets/{wid}/duplicate` · `POST /dashboards/{did}/reorder` | Widget CRUD (`kind: chart|kpi|pivot`, `size: small|medium|large`, `spec {query, display}`) |

`POST /versions/{vid}/copilot` now also answers analytical questions with
`analysis = {intent, interpretation, spec, result, chart, extra}` or `{status: ambiguous|unknown_column|unsupported|invalid, options?}`.
