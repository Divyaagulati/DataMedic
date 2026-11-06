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
| `GET /versions/{vid}/preview` | `{columns, rows, truncated}` (≤ 100 rows, `_dm_row_id` hidden) |
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

## 5. Versions

| Method & path | Notes |
|---|---|
| `GET /datasets/{id}/versions` | Linear list: `{id, version_number, kind, parent_version_id, restored_from_version_id, n_rows, n_cols, content_hash, is_head, fitted_on_full_data, created_at}` |
| `GET /versions/{vid}` | Plus lineage (replay chain summary), transformation run (spec, approvals, step reports), library versions |
| `POST /datasets/{id}/revert` | `{version_id, reason?}` → `202 {job}`. Creates a new `revert` version copying `version_id`'s content. `409 CONFLICT` if another mutation job is active. `409 QUOTA_EXCEEDED` on the version cap. |
| `POST /versions/{a}/compare/{b}` | `202 {job}` → result: schema diff, row-count diff, per-column SciPy tests (KS for numeric, chi-square for categorical) with p-values, effect sizes and limitations |
| `GET /version-comparisons/{job_id}` | Comparison result |

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

## 7. Recommendations (proposal only)

| Method & path | Notes |
|---|---|
| `GET /analyses/{rid}/recommendations?domain&risk&destructive&status` | List |
| `GET /recommendations/{id}` | `{id, rule_id, domain, title, rationale, proposed_steps: [{...Step, risk, destructive, kind}], risk, destructive, expected_effect, alternatives, finding_ids, status}` |
| `PATCH /recommendations/{id}` | `{status: "dismissed"}` |

No endpoint under `/recommendations` changes data.

## 8. Pipelines: impact preview, approval, commit

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
