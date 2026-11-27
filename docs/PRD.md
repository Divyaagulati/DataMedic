# DataMedic — Product Requirements Document

| | |
|---|---|
| Status | Revision 2, approved direction. Awaiting approval to start Phase 1. |
| Date | 2026-11-06 |
| Release tiers | **MVP** → **Version 2 (V2)** → **Future Enhancements** (defined in §10) |
| Related | [ARCHITECTURE.md](ARCHITECTURE.md), [ML_ENGINE_DESIGN.md](ML_ENGINE_DESIGN.md), [DOMAIN_MODULES.md](DOMAIN_MODULES.md), [PRIVACY_AND_SECURITY.md](PRIVACY_AND_SECURITY.md), [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) |

---

## 1. Product summary

DataMedic is a privacy-first, domain-aware data intelligence and ML reliability platform. A user uploads a tabular dataset, declares its domain and the roles of key columns, and DataMedic:

1. **Diagnoses** data-quality problems, with evidence, severity and stated limitations for every finding.
2. **Recommends** preprocessing steps, with a rationale linked to the findings and a risk classification.
3. **Experiments**: evaluates candidate preprocessing pipelines in isolated, leakage-safe experiments that never change stored data.
4. **Applies** only the steps the user approved, producing a new immutable dataset version.
5. **Audits** for data leakage.
6. **Issues a Data Health Passport** and exports a validated, ML-ready dataset with reproducible Python code.

The flagship domain is **biomedical** (clinical tabular data in the MVP). The architecture supports future **financial**, **business** and **industrial** modules.

### 1.1 Core differentiators (must be preserved in every tier)
| Differentiator | MVP embodiment |
|---|---|
| Domain-aware diagnostics | Universal suite + biomedical clinical suite, separated in data and UI |
| Preprocessing experiments | Compare up to 3 candidate pipelines under identical leakage-safe folds |
| Leakage-safe evaluation | Fitted steps refit per training fold; group- and time-aware splits; leakage audit |
| Privacy | Owner-only access, no external LLM or third-party processing of dataset content |
| Reproducibility | Immutable versions, replayable lineage, generated `pipeline.py` validated by replay |
| **Data Health Passport** | Per-version, immutable report covering identity, lineage, diagnostics, transformations and approvals, experiments, leakage, ML-readiness and limitations |

## 2. Problem statement

Practitioners spend most of an ML project on data preparation, and most ML reliability failures start in the data: leakage, the same patient across splits, batch effects confounded with outcome, preprocessing fitted before splitting, and label noise. Profiling tools describe data but do not:

- tie findings to risk-classified remediation;
- prove a preprocessing choice helps, under leakage-safe evaluation;
- keep a reversible, auditable lineage;
- understand domain semantics;
- keep sensitive data inside the platform boundary.

## 3. Target users

| Persona | Need |
|---|---|
| **Biomedical data scientist** (primary) | Catch patient-level leakage, batch confounding and PHI. Produce defensible preprocessing for publications. |
| **ML engineer** | Reproducible pipelines and evidence that cleaning improves generalisation |
| **Analyst / domain scientist** | Plain-language findings, safe defaults and guard rails against destructive edits |

## 4. Goals and non-goals

### Goals
- **G1 Trustworthy diagnostics.** Evidence, severity and limitations on every finding. Unmet requirements produce `not_run`.
- **G2 Non-destructive workflow.** Immutable original and versions, with any earlier state restorable.
- **G3 Leakage-safe by construction.**
- **G4 Privacy by design.** No dataset content leaves the platform.
- **G5 Reproducibility.** Exported code replays the version's lineage from the original file. Its output must match the platform version's *content hash*, or match within a documented float tolerance, or the export is labelled `unvalidated`.
- **G6 Biomedical-grade first domain** (clinical tabular).

### Non-goals (all tiers unless moved by a later decision)
Open-ended AutoML (search over model families and pipelines, Bayesian optimisation, ensembling, stacking); model deployment, serving and trained-model download; streaming data; image, signal, free-text NLP and raw sequencing data; regulatory certification; automatic application of any transformation to stored data without review.

## 5. Core requirements and how they are satisfied

| # | Requirement | Satisfied by | Verified by |
|---|---|---|---|
| R1 | Each user accesses only their own datasets and results | `owner_id` on every record; owner-scoped queries in the service layer; Postgres Row-Level Security on `app.user_id`; 404 for other users' resources; storage is not reachable by browsers | Auto-generated AuthZ matrix over every endpoint; RLS tests |
| R2 | Raw datasets never sent to external LLM APIs | **MVP: no LLM integration exists at all.** V2: opt-in, aggregate-only gateway ([PRIVACY_AND_SECURITY.md §3](PRIVACY_AND_SECURITY.md)) | MVP: dependency and network review. V2: canary egress tests. |
| R3 | Original upload unchanged | Write-once raw object, SHA-256 recorded while streaming, DB trigger blocks updates, no code path writes to `raw/` | Unit tests; hash verified on every read |
| R4 | Every processed version traceable and reversible | Immutable, linear version history. Each version records its kind, parent, spec, approvals, library versions and hashes. Revert creates a new version restoring earlier content. | Lineage and replay tests |
| R5 | High-risk transformations require explicit approval | Risk and `destructive` flag defined per operation in the registry. **Approval is required to commit any high-risk step to a version.** Experiments may include high-risk steps without approval because they never write versions, and they are labelled. | Bypass tests |
| R6 | Evidence, severity, limitations on every diagnostic | `Finding` schema + DB constraints | Contract tests |
| R7 | Never claim a check ran without the data it needs | Declared requirements → `not_run` with reason; sampling disclosed | Registry-wide honesty test |
| R8 | Leakage-safe ML | In-fold fitting, row-ID-based frozen folds, group/time-aware splitters, leakage audit | Spy-transformer tests |
| R9 | CSV, XLSX, XLS, Parquet, JSON | Hardened ingest adapters | Format and malicious-fixture suites |
| R10 | Biomedical flagship | Biomedical clinical module in the MVP | Biomedical golden datasets |
| R11 | Extensible to financial, business, industrial | Domain module interface + in-code registry, plus a stub module test | CI test: the stub registers without core changes |

## 6. Functional requirements

Each requirement is tagged with its tier: **[MVP]**, **[V2]** or **[Future]**.

### 6.1 Accounts and access
- FR-1.1 [MVP] Sign up with email and password, verify email, log in and log out. Password reset by a single-use token.
- FR-1.2 [MVP] Every dataset and derived record is owned by exactly one user (`owner_id`). There is no sharing.
- FR-1.3 [MVP] Server-side sessions. The user can list and revoke active sessions.
- FR-1.4 [MVP] Account deletion purges all of the user's data (§6.11).
- FR-1.5 [V2] TOTP MFA. [Future] Teams, sharing and SSO.

### 6.2 Upload and ingest
- FR-2.1 [MVP] Upload CSV, TSV (CSV with a tab delimiter), XLSX, XLS, Parquet and JSON (records, lines or column-oriented). **Maximum 200 MB per file**, streamed through the API.
- FR-2.2 [MVP] Detect encoding, delimiter, header row and sheet. The user can override them and re-parse. Re-parsing creates a **new version**, and the original bytes are never touched.
- FR-2.3 [MVP] Multi-sheet workbooks: the user selects one sheet per dataset.
- FR-2.4 [MVP] Nested JSON is flattened with a declared column-naming scheme. Nesting beyond a limit is rejected.
- FR-2.5 [MVP] Content type is validated by magic bytes. Macro-enabled formats and files over the decompression, cell or column limits are rejected.
- FR-2.6 [MVP] The original is stored byte-for-byte. The parsed table becomes **version 0** (canonical Parquet) with a hidden stable row identifier `_dm_row_id` assigned in file order.
- FR-2.7 [MVP] A preview of the first 20 rows is shown to the owner only (reduced from 100 by owner decision in Phase 2).
- FR-2.8 [MVP] A PHI/PII scan runs at ingest, and its findings are shown before any other action.
- FR-2.9 [MVP] Uploads count against the user's storage quota (§6.13).

### 6.3 Domain and column semantics
- FR-3.1 [MVP] Domain: `generic` or `biomedical`. Changing the domain marks existing domain findings **stale**.
- FR-3.2 [MVP] Universal roles: `target`, `id`, `group`, `time`, `weight`, `exclude`, `sensitive`. Biomedical roles are listed in [DOMAIN_MODULES.md §4.2](DOMAIN_MODULES.md).
- FR-3.3 [MVP] Role suggestions are shown and never auto-applied.
- FR-3.4 [MVP] Task type (`binary`, `multiclass`, `regression`) is stored with the target role of each version and confirmed by the user. Changing roles or task type marks dependent findings stale.

### 6.4 Diagnostics
- FR-4.1 [MVP] Universal diagnostic suite ([ML_ENGINE_DESIGN.md §2.1](ML_ENGINE_DESIGN.md)).
- FR-4.2 [MVP] Biomedical clinical suite, listed and labelled separately.
- FR-4.3 [MVP] Every finding has `check_id`, `check_version`, `status`, `severity`, a title, an explanation, structured `evidence`, affected columns, the number and percentage of affected rows (when applicable), a suggested action, `limitations` and sampling metadata. Owner decision, Phase 3 (2026-11-09):
  - `status` is `detected | not_detected | not_assessed | error`, replacing `passed | warning | failed | not_run | error`.
  - `severity` is `critical | high | medium | low | info`, where `info` is shown as "Informational".
  - Row references are evidence examples for now; a row-reference artifact comes later.
- FR-4.4a [MVP, owner decision Phase 3] Domain modules for **financial**, **business** and **industrial** data are in the MVP alongside biomedical. They were moved from Future Enhancements by the owner on 2026-11-09. Domain checks run only for the dataset's selected domain, and report `not_assessed` when their required roles are not declared.
- FR-4.4 [MVP] `not_run` findings state the unmet requirement.
- FR-4.5 [MVP] The health summary shows counts by status, severity and domain. There is no unexplained single score.
- FR-4.6 [MVP] Analysis runs are tied to a version and a **semantics hash** (domain + roles + task type + domain reference data such as biomedical reference ranges). A run whose hash differs from the current one is shown as stale.
- FR-4.7 [V2] Full profiling report (YData Profiling / fg-data-profiling), multivariate outlier ensembles (PyOD) and drift reports (Evidently), each added through existing adapter interfaces.

### 6.5 Recommendations (proposal only)
- FR-5.1 [MVP] Deterministic rules turn findings into recommendations.
- FR-5.2 [MVP] Each recommendation includes:
  - proposed declarative steps;
  - the findings it is linked to;
  - a rationale;
  - the expected effect;
  - a risk class and the `destructive` flag;
  - alternatives, including the consequence of doing nothing.
- FR-5.3 [MVP] Recommendations never modify data.
- FR-5.4 [MVP] The user assembles recommended or manually chosen steps into a pipeline draft.

### 6.6 Preprocessing experiments (isolated, never write versions)
- FR-6.1 [MVP] The user evaluates up to **3 candidate pipelines** (plus "no preprocessing") against a source version. Candidates may contain high-risk and destructive steps **without approval**, because experiments never change stored data.
- FR-6.2 [MVP] Candidates containing destructive or high-risk steps are **clearly labelled** in results, with their per-fold effect (e.g. "removed 37 training rows per fold on average").
- FR-6.3 [MVP] Experiments are leakage-safe as described in [ML_ENGINE_DESIGN.md §5](ML_ENGINE_DESIGN.md). Training-fold preprocessing and row-removal operations are fitted and applied using training data only.
- FR-6.4 [MVP] Validation and test folds stay intact. No candidate removes test rows, so difficult examples are never silently dropped from evaluation.
- FR-6.5 [MVP] Rows that may belong to the same patient or entity (declared group role, exact and normalised duplicates) are grouped before splitting.
- FR-6.6 [MVP] Results report the number of training rows removed by each candidate, per fold and per step. `ROWSET_DIFFERS` flags candidates whose retained training rows differ.
- FR-6.7 [MVP] If two candidates cannot be evaluated on the same test cohort, the comparison is flagged `NOT_DIRECTLY_COMPARABLE`. No score improvement is presented, and the candidate is excluded from ranking and paired differences.

### 6.7 Approval and commit (making changes permanent)
- FR-7.1 [MVP] A pipeline is validated against parameter schemas, column existence, types and ordering rules.
- FR-7.2 [MVP] Before any approval, an **impact preview** shows each step's effect on the source version: rows removed, columns removed or added, values changed per column, and before/after summary statistics. The owner can inspect affected row IDs.
- FR-7.3 [MVP] Destructive operations (removing rows or columns, or overwriting observed values) carry a visible **Destructive** label everywhere they appear.
- FR-7.4 [MVP] **Every high-risk step requires its own explicit approval** to be committed, bound to the step's hash. Medium-risk steps require one pipeline-level confirmation, bound to the pipeline's hash. Editing a step invalidates its approval.
- FR-7.5 [MVP] Committing produces a new immutable version, a transformation record and a diff summary. Versions produced by fitted steps are flagged `fitted_on_full_data`.
- FR-7.6 [MVP] The server and the worker both verify approvals. No API path commits a high-risk step without a valid approval.

### 6.8 Model evaluation (separate from data cleaning)
- FR-8.1 [MVP] Baseline models: dummy, logistic or ridge, random forest and histogram gradient boosting (owner decision P1, Phase 5). XGBoost stays deferred (owner decision P2; not installed).
- FR-8.1a [MVP, owner decision P1] Bounded hyper-parameter selection: nested cross-validation over fixed grids of at most four settings per model, inner folds built from outer training rows only, with a recorded fallback to default settings for small cohorts or above the fit budget.
- FR-8.2 [MVP] Split strategy is chosen from roles (group, time or stratified). The user may override it, with warnings.
- FR-8.3 [MVP] Metrics are reported as mean ± sd across folds, with per-fold paired differences between candidates. Suspiciously high scores are flagged.
- FR-8.4 [MVP, owner decision P1] Calibration assessment (reliability curve, ECE, Brier score) and group-aware bootstrap intervals on pooled out-of-fold predictions (unavailable for time-ordered splits). [V2] Locked holdout with an access log.
- FR-8.6 [MVP, owner decision P3] A version fitted on all of its rows receives a leakage-controlled evaluation only by replaying its committed steps inside each training fold from its eligible ancestor, after proving the replay faithful; otherwise leakage-controlled evaluation is blocked, and an explicit, permanently labelled exploratory run is the only alternative.
- FR-8.7 [MVP] ML Reliability Auditor (20 checks; confirmed, configuration and statistical findings; `not_assessed` when metadata is missing; a rule-based evaluation status, never a numeric score) and explainability (held-out permutation importance, coefficients, tree importances, error slices).
- FR-8.5 [MVP] Evaluation never writes versions.

### 6.9 Leakage audit
- FR-9.1 [MVP] Detect single-feature target leakage, identifier leakage, duplicates across folds, groups across folds, undeclared repeated entities, time-ordering problems, and preprocessing fitted on full data.
- FR-9.2 [MVP] Biomedical checks: patient overlap across folds, batch confounded with target, post-outcome variables (when `bio.outcome_time` is declared).
- FR-9.3 [MVP] Each leakage finding links to a remediation recommendation.

### 6.10 Versioning
- FR-10.1 [MVP] **Linear history.** Each new version's parent is the current head. Version kinds: `ingest` (v0), `reparse`, `transform` and `revert`.
- FR-10.2 [MVP] **Revert** creates a new version whose content is a copy of an earlier version, so history stays linear and nothing is mutated.
- FR-10.3 [MVP] The version detail page shows kind, parent, spec, approvals, executor, library versions, row and column deltas, file hash and content hash.
- FR-10.4 [MVP] Compare two versions: schema diff, row-count diff and per-column distribution tests (SciPy). [V2] Evidently drift reports.
- FR-10.5 [V2] Forks and branches from non-head versions, a version-graph UI, and deletion of individual versions. The data model already supports a DAG.

### 6.11 Deletion and retention
- FR-11.1 [MVP] Deleting a dataset removes all its versions, analyses, experiments, passports, exports and artifacts.
- FR-11.2 [MVP] Deletion is soft at first (immediately inaccessible), then purged within 24 h. Storage objects and DB rows are removed, a receipt is issued and a non-identifying tombstone is kept.
- FR-11.3 [MVP] Backup retention (30 days) is disclosed.

### 6.12 Data Health Passport and export
- FR-12.1 [MVP] **Data Health Passport**: an immutable, per-version report generated on request and on export. It contains:
  - dataset identity (name, file hash, content hash, size, shape);
  - the full lineage chain, with each step's risk, destructive flag and approval;
  - semantics (domain, roles, task type);
  - the diagnostics summary and findings by domain, including `not_run` items;
  - leakage audit results;
  - experiment results, if any;
  - the **ML-readiness checklist** (§6.12.1);
  - reproducibility information (engine and library versions, seeds, export validation status);
  - consolidated limitations.
- FR-12.2 [MVP] The passport is viewable at `/datasets/[datasetId]/versions/[versionId]/passport` and downloadable as self-contained HTML and JSON. [V2] PDF export.
- FR-12.3 [MVP] A passport lists the analysis and evaluation runs it summarises. If newer runs exist, the page offers to regenerate it, and old passports remain.
- FR-12.4 [MVP] The export ZIP contains the data (Parquet and/or CSV), `pipeline.py`, `requirements.txt`, `manifest.json`, `passport.html` and `passport.json`.
- FR-12.5 [MVP] **Export validation**: the worker runs `pipeline.py` against the original in a resource-limited subprocess and compares the output's content hash (with float tolerance) to the version's. The result is `validated`, `validated_within_tolerance` or `unvalidated` (with a per-column diff).
- FR-12.6 [MVP] Generated code uses only public APIs of the pinned pandas, scikit-learn and Feature-engine versions.
- FR-12.7 [MVP] CSV formula-injection risk (cells starting with `=`, `+`, `-`, `@`, tab or carriage return) is detected and warned about. Escaping is opt-in, and data is never altered silently.
- FR-12.8 [MVP] Downloads are streamed through the API after an authorisation check. Exports expire after 30 days.

#### 6.12.1 ML-readiness checklist [MVP]
Evaluated per version, from version metadata and the latest *non-stale* analysis. Each item returns `pass | warn | fail | not_applicable` with evidence. The overall result is `ready`, `ready_with_warnings` or `not_ready`. Export is allowed in every state, and the passport states the result prominently.

| ID | Item | Fail / warn rule |
|---|---|---|
| RD-01 | Target assigned, task type confirmed, no missing target values | fail if absent or any missing target |
| RD-02 | No missing values in model features | warn if NaNs remain only in numeric features (usable by NaN-native models such as HGB and XGBoost); fail if NaNs remain in categorical, boolean or text features |
| RD-03 | No constant or all-missing features | fail |
| RD-04 | No `id`, `exclude` or `sensitive`-role columns, and no ID-like columns, among the features | fail |
| RD-05 | All features numeric, boolean or categorical (no raw free text or unparsed dates) | fail |
| RD-06 | No open high or critical leakage findings | fail (critical) / warn (high) |
| RD-07 | Diagnostics current: the latest analysis matches the current semantics hash | fail if stale or missing |
| RD-08 | Enough samples per class for the recommended CV (≥ n_splits per class; ≥ 10 recommended) | fail / warn |
| RD-09 | No conflicting-label duplicates | warn |
| RD-10 | Group role declared when repeated entities are detected | fail if `leakage.groups.undeclared` failed |
| RD-11 | Export reproducibility validated | warn if `validated_within_tolerance`; fail if `unvalidated` (export only) |

### 6.13 Quotas and job limits [MVP]
Configurable per plan. Only one plan (`free`) exists in the MVP.

| Limit | Default |
|---|---|
| Max upload size | 200 MB |
| Max total storage per user (originals + versions + artifacts + exports) | 5 GB |
| Max datasets per user | 25 |
| Max versions per dataset | 30 |
| Max cells per table | 20,000,000 |
| Max columns per table | 2,000 |
| Max concurrently running jobs per user | 2 |
| Max queued jobs per user | 10 |
| Max candidates per experiment | 3 (+ "no preprocessing") |
| Max models per experiment | 4 |
| Job time limits | ingest 10 min · analysis 20 min · execute 15 min · evaluate 30 min · export 15 min |

Exceeding a limit returns `QUOTA_EXCEEDED` (storage, datasets, versions), `LIMIT_EXCEEDED` (cells, columns, size) or `JOB_LIMIT_EXCEEDED`. The settings page shows usage against limits.

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Ingest + v0 for a 100 MB CSV in under 90 s (p95). Universal + biomedical diagnostics on 1M × 100 in under 10 min, using disclosed sampling for expensive checks. |
| Reliability | Jobs survive worker restarts (at-least-once delivery, idempotent tasks). |
| Security | OWASP ASVS L2 as a design target. TLS in production. Encryption at rest in production. |
| Privacy | Data minimisation, owner-only access, no third-party scripts on authenticated pages, no dataset content in logs or job messages. |
| Reproducibility | Every result records engine version, check versions, library versions and seeds. |
| Accessibility | WCAG 2.2 AA. |
| Browsers | Last two versions of Chrome, Firefox, Safari and Edge. |

## 8. Key user journeys (MVP)

1. **Diagnose**: sign up → upload `cohort.xlsx` → pick sheet → domain *biomedical* → confirm roles (`patient_id`, `diagnosis` as target with task type binary, `site` as batch) → diagnostics show, for example:
   - "Patients have multiple rows (avg 1.8) — use group-aware evaluation (high)"
   - "Site strongly associated with diagnosis (Cramér's V 0.62) (critical)"
2. **Experiment**: from recommendations, build candidate A (median-impute labs) and candidate B (A + drop exact duplicates [Destructive, high-risk]) → run the experiment with group 5-fold CV → see paired per-fold differences, with B labelled destructive. No approval is needed and no version is created.
3. **Commit**: open candidate A as a pipeline → impact preview → confirm the medium-risk step → execute → v1. If B were committed, its high-risk step would require explicit approval after an impact preview.
4. **Passport and export**: generate the passport for v1 → ML-readiness `ready_with_warnings` → export → validation `validated` → download the ZIP.
5. **Undo**: revert to v0 → v2 is created with v0's content, and history stays linear.

## 9. Success metrics

- Golden datasets: 100% of seeded issues detected at or above expected severity. Clean controls produce **no high or critical findings and at most one medium finding per check**.
- 100% of exports are labelled with a validation status, and 100% of passports include the readiness checklist.
- Zero cross-user access findings in security testing.
- Time from upload to first diagnosis under 5 minutes for typical datasets.

## 10. Release tiers

### MVP (must build first)
- Accounts, sessions, CSRF, owner isolation with RLS, quotas and job limits.
- Streaming upload (≤ 200 MB) and hardened ingest of all 5 formats, with v0, row IDs, preview, reparse and deletion.
- Domain selection and roles, and the universal suite of 24 checks using built-in, scikit-learn, SciPy and Cleanlab implementations (no YData Profiling, PyOD or Evidently).
- Biomedical clinical suite.
- Recommendations, op registry, pipeline builder, impact previews, destructive labelling, approvals and commit to version.
- Linear versioning with revert, and a version comparison using SciPy tests.
- Preprocessing experiments, leakage-safe CV, 4 baseline models and paired per-fold comparison.
- Leakage audit, plus the Cleanlab label-noise check (the first item cut if the schedule slips).
- Data Health Passport (HTML/JSON), ML-readiness checklist, export with replay validation.
- Polling for job status, one worker image, no external LLM.

### Version 2 (after MVP works)
- YData Profiling (fg-data-profiling) report, PyOD ensembles, Evidently drift, all through existing adapters.
- Forks and branches with a version graph, and deletion of individual versions.
- Locked holdout. (Bootstrap intervals, calibration assessment and random forest moved into the MVP by owner decision P1, Phase 5.)
- Opt-in aggregate-only LLM explanations through the egress gateway.
- TOTP MFA, PDF passports, SSE progress.
- Direct-to-storage uploads for files over 200 MB, a hardened export sandbox container, per-patient aggregation ops.
- **Biomedical omics module**: Scanpy QC, normalisation, highly variable genes, and ComBat with safeguards in a separate worker image.

### Future Enhancements
- Teams, sharing and SSO; a public API.
- ~~Financial, business and industrial domain modules.~~ Moved into the MVP (owner decision, 2026-11-09; FR-4.4a). Their deeper features (e.g. look-ahead bias detection, funnel consistency, maintenance-event leakage) remain future work.
- Data connectors, an on-prem distribution (requires a licence re-review), plugin discovery via entry points, open-ended hyper-parameter search (bounded selection is in the MVP, FR-8.1a), XGBoost, SHAP.

## 11. Open questions

| # | Question | Current assumption |
|---|---|---|
| Q1 | Production object storage | Managed S3-compatible service. Dev backend chosen in Phase 1 ([ARCHITECTURE.md §11.4](ARCHITECTURE.md)). |
| Q2 | Broker | Valkey, subject to the Phase 1 spike |
| Q3 | Hosting region | Single region |
| Q4 | Email delivery provider (production) | SMTP abstraction. Provider chosen before Phase 10 deployment. |
| Q5 | On-prem distribution | Not planned. Affects the GPL/AGPL analysis. |
| Q6 | pandas 2.3.x vs 3.x for the MVP | 2.3.x (see the roadmap's compatibility checks) |
