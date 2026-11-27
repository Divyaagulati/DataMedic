# DataMedic — Phase 5 Plan: Baseline AutoML and ML Reliability Auditor

| | |
|---|---|
| Status | Approved 2026-11-25 with decisions P1–P4. Implemented 2026-11-25. Owner browser verification completed 2026-11-26 (conditional acceptance); **closure corrections implemented 2026-11-26 (§19), awaiting the owner's final approval.** Phase 6 not started. |
| Date | 2026-11-25 |
| Baseline | Phases 1–4 complete; the owner approved the Phase 4 closure on 2026-11-25. Last full backend run: 1,446 passed, 4 skipped (pre-existing). Two closure integration tests were added afterwards (the 17 pipeline tests pass). Frontend 74 passed. Migrations 0001–0006. |
| Delivery | One milestone, implemented autonomously in the dependency-aware internal order of §15. It stops only for a genuine blocker. |
| Scope mapping | Owner Phase 5 = PRD §6.8 model evaluation, plus PRD §6.9 / roadmap Phase 8 leakage audit, generalised into the Reliability Auditor, plus explainability. Roadmap "Phase 5" (remaining universal checks and the biomedical clinical module) is **not** this phase (§2.3). |

Workflow: dataset version → task and target → evaluation strategy and eligibility → preprocessing specification → training with bounded model selection → model comparison → reliability audit → explainability → results dashboard and reproducibility record.

The aim is not the highest score. It is an honest, reproducible estimate of how a modelling procedure generalises, with the evidence for or against trusting that estimate.

---

## 1. Current architecture and reusable components

The repository is the source of truth; the documents were checked against it.

| Need | Existing component (verified in code) | Phase 5 use |
|---|---|---|
| Frozen folds | `engine/preprocess/split.py` `make_splits` (kfold, stratified, group, stratified_group, time; duplicate-aware blocks via union-find), `experiment/folds.py` encode/decode, `split_definitions` table with a SHA-256-verified `fold_assignment` artifact, reused by config hash | Outer evaluation folds, reused unchanged. Inner tuning folds are built by the same splitter on outer-training rows. |
| In-fold preprocessing | `PreprocessExecutor.fit_resample(train)` / `transform(test)`: fitted ops learn on training rows only; `training_only` ops (resampling, class weights) change only the training table; test rows are never removed | The preprocessing specification for every fold and every tuning fit. |
| Models | `experiment/models.py`: `feature_kinds`, `model_matrix`, `build_model` (dummy, linear, hgb, kmeans) with documented minimal in-fold preprocessing | Extended with a `params` argument and `random_forest`. The Lab's allowed model list is unchanged. |
| Metrics | `experiment/metrics.py`: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier, ECE (binary), confusion and per-class; MAE, RMSE, R², median AE; `None` + reason when undefined | Reused as-is; adds MSE, curves, binned residuals and bootstrap intervals. |
| Runner patterns | `experiment/runner.py`: cohort, label encoding to integer codes, per-fold and per-model failure isolation, `NOT_DIRECTLY_COMPARABLE` check of scored test rows, cancellation before each fold, `threadpool_limits(1)`, results only in the final transaction | Same contract for the training runner. |
| Lineage | `pipelines/versioning.replay_chain` (ingest base + transformation runs; reverts follow the restored version), `transformation_runs.spec_snapshot` with stages and data-step hashes, `dataset_versions.fitted_on_full_data`, `model_pipeline_spec` | Evaluation eligibility and in-fold replay of full-data-fitted steps (§5.2). |
| Content hashing | `engine/hashing.content_hash`, `commit.finalise` (Parquet round-trip) | Replay-equivalence proof (§5.2). |
| Findings | `findings` table and its constraints (detected ⇒ evidence + action; limitations required; `not_assessed` ⇒ reason), `analysis_runs`, `engine/checks/framework.py` error isolation, `FindingCard` UI | Reliability-audit findings use the same rows, constraints and card. |
| Existing leakage diagnostics | `universal.leakage.target_proxy`, `universal.leakage.identifier`, `universal.target.imbalance`, `universal.duplicates.exact_rows` | Inputs to the audit (latest non-stale run), never duplicated or altered. |
| Jobs | `jobs` (status, progress, heartbeats, cancel), `register_handler`, follow-up jobs, idempotency keys, admission limits | New job types `train` and `reliability_audit`. |
| Isolation | RLS + owner/same-dataset triggers (0004–0006 helpers), auto-generated AuthZ matrix, `test_db_security.py` | Every new table and route. |
| Orphan cleanup | `reconcile_orphans` (hourly, 24 h grace) | Out-of-fold artifacts written before a failed final transaction. |
| Frontend | `DatasetNav`, TanStack polling hooks, generated client, shadcn/ui, dependency-free `charts.tsx` (BarList, Histogram, CorrelationHeatmap, StatTile, tooltips), Lab `ParamField`/`AddStepForm`, `FindingCard`, `StatusBadge` | The ML workspace. |

### 1.1 Gaps found in the existing code that matter for Phase 5
1. **The Experiment Lab does not warn when its source version is `fitted_on_full_data`.** ML_ENGINE_DESIGN §5.5 specifies `FITTED_ON_FULL_DATA_SOURCE`, but no code emits it. Phase 5 adds this warning: a few lines in `experiments/service.py` and the result view, with a test. It is the same safeguard Phase 5 depends on. No other Lab behaviour changes.
2. `HistGradientBoosting*` uses `early_stopping="auto"`, which for more than 10,000 rows holds out a *random* (not group-aware) 10% of the training rows. This never touches test rows, so it is not test leakage. It does make the internal stopping decision ignore groups. Phase 5 sets `early_stopping=False` explicitly. The Lab keeps its documented defaults; the limitation is recorded in ML_ENGINE_DESIGN §5.0.
3. Step reports record `n_fit_rows`, not fit row ids. Fit boundaries are guaranteed structurally by the executor API; Phase 5 adds runtime assertions (§6, A10) and spy-transformer tests.

---

## 2. Final recommended scope

### 2.1 In scope
- **Tasks:** binary classification, multiclass classification (3–50 classes) and regression. The target and task type come from the version's confirmed target role; the data must agree with them (§5.1).
- **Models** (scikit-learn only):
  - `dummy` (prior / median);
  - `linear` (logistic regression / ridge);
  - `random_forest` (tree-based);
  - `hgb` (histogram gradient boosting).
- **Bounded hyper-parameter selection** by **nested cross-validation** over small fixed grids (§5.4). It is resource-bounded, deterministic and off for very small cohorts.
- **Evaluation:** frozen outer folds (stratified, k-fold, group, stratified-group, time), identical test rows for every model, in-fold preprocessing, inner folds inside outer training rows only.
- **Full-data-fitted versions:** eligibility analysis, in-fold replay from an eligible ancestor with a proof of replay equivalence, and otherwise either a block or an explicitly labelled exploratory run (§5.2).
- **Metrics and visualisations (§5.5):**
  - fold mean ± sd, min and max;
  - group-aware bootstrap intervals on pooled out-of-fold predictions;
  - ROC and PR curves, confusion matrix, per-class results;
  - calibration reliability curve with Brier score and ECE;
  - binned actual-versus-predicted and residual views.
- **Reliability Auditor (§6):** 20 checks, each with a basis (`confirmed`, `configuration` or `statistical`), severity, evidence, explanation, limitations and corrective action, plus honest `not_assessed` results. There is no numeric reliability score; a rule-based evaluation status is shown with its method.
- **Explainability (§7):**
  - permutation importance on held-out folds, computed on original columns;
  - linear coefficients and random-forest impurity importance mapped back to original columns;
  - feature-level error slices.
- **Reproducibility:** stored configuration and `config_hash`, split hash, seeds, library versions, replay record and out-of-fold prediction artifact. A rerun reproduces identical metrics.
- **ML workspace UI:** a Models tab with a guided configuration flow, run progress and cancellation, and a results dashboard (§11).

### 2.2 Explicit exclusions (with reasons)

| Excluded | Reason |
|---|---|
| Model persistence, download, prediction API, deployment | Pickle/joblib is forbidden; scikit-learn has no safe general serialiser in the pinned stack; serving is a PRD non-goal. Fold models are discarded after scoring. Learned linear coefficients are stored as JSON for display only. |
| A "final model" refit on all rows | Its score could not be estimated without bias, and it could not be persisted. The reported performance is that of the **procedure** (nested CV), as §5.4 states. |
| XGBoost | Not installed (D3). Its Linux wheel pulls a large NCCL dependency, and HGB already provides gradient boosting. Kept behind the model catalogue for V2 (§16, decision P2). |
| SHAP, LIME, partial dependence, interaction effects | New heavy dependencies (SHAP pulls numba) and extra compute; permutation importance plus model-specific importances meet the brief. Not causal in any case. |
| Locked holdout with access log | V2 (FR-8.4). Nested CV separates selection from evaluation without sacrificing scarce rows. |
| Open-ended AutoML (search over model families and pipelines, Bayesian optimisation, ensembling, stacking) | Scientifically fragile on small data, expensive, and contrary to the "procedure, not leaderboard" goal. |
| Clustering in the training workflow | It stays in the Experiment Lab; supervised reliability checks do not apply. |
| Cleanlab label noise | Not installed and not in the brief; it stays a deferred roadmap item. |
| Biomedical-specific leakage (batch confounding, post-outcome variables) | Biomedical specialist modules are excluded. `bio.patient_id` already acts as the group role. Post-outcome checks report `not_assessed`. |
| Fairness or subgroup metrics on sensitive attributes | Not requested; needs separate design and consent. Sensitive-role columns are excluded from error slices. |
| Using the `weight` role in training or metrics | Not designed for Phase 5; disclosed as a limitation. |
| Passport, readiness checklist, export bundle, training-code generation | Roadmap Phase 9 (later owner phase). Reproduction is by rerun from the stored configuration. |
| External LLM / Copilot, Phase 6–7 features, production deployment | Out of scope by instruction. |
| Phase 4 cosmetic issues ("Re-run of" names; stale comparison after switching versions) | Deferred; they do not interfere with Phase 5. |

### 2.3 Conflicts with existing documents (raised, not silently resolved)

| # | Conflict | Recommended resolution |
|---|---|---|
| C1 | PRD §4 non-goals list "hyper-parameter tuning or AutoML"; PRD §10 and the roadmap list tuning as a Future enhancement. The brief asks for baseline AutoML and hyper-parameter selection. | Approve **bounded nested-CV selection over fixed small grids** (§5.4). Record it in the PRD as an owner decision; open-ended AutoML stays a non-goal. (P1) |
| C2 | PRD FR-8.4 and §10 put calibration plots, bootstrap CIs and random forest in V2. | Pull in calibration assessment, group-aware bootstrap intervals and random forest. Keep the locked holdout in V2. (P1) |
| C3 | PRD FR-8.1 lists XGBoost as MVP and CLAUDE.md lists it in the pinned baseline, but D3 deferred it and it is not installed. CLAUDE.md also lists Cleanlab and statsmodels, which are not installed. | Keep XGBoost deferred (P2). Correct the CLAUDE.md "pinned baseline" line to the installed libraries. |
| C4 | Roadmap "Phase 5" = remaining universal checks + biomedical clinical module. | Add a scope-mapping note to the roadmap. Those items remain open roadmap work outside this milestone. |
| C5 | ML_ENGINE_DESIGN §5.5 documents `FITTED_ON_FULL_DATA_SOURCE`, which is not implemented. | Implement it in the Lab (§1.1). This is the only Phase 4 code touched. |
| C6 | DATABASE_SCHEMA §3.6 designed `model_results.oof_artifact_id`; Part D stores no out-of-fold predictions. | Training runs store out-of-fold predictions as an owner-only artifact (§9). The Lab is unchanged. |
| C7 | API §6 plans `kind: leakage_audit` on version analyses. | Leakage checks need a model, folds and lineage, so they run as a **reliability audit of a training run** (`analysis_runs.kind = 'reliability_audit'`). The diagnostics leakage checks stay where they are. |
| C8 | ARCHITECTURE §9.5 allows joblib + HMAC fitted state. | Already superseded by D4. No model persistence (above). |

---

## 3. Explicit exclusions summary
§2.2 and §2.3 are authoritative. In one line: no persistence or serving, no XGBoost, no SHAP, no holdout, no open-ended AutoML, no clustering training, no label noise, no biomedical specialist checks, no fairness metrics, no export/codegen of training, no LLM, nothing from Phases 6–7.

---

## 4. End-to-end workflow

1. **Dataset version.** The user opens *Models* on a dataset. The head version is preselected; any version can be chosen.
2. **Task and target.** Read from the version's target role (`task_type` confirmed). If missing, the page links to the roles editor; nothing is guessed. Server-side validation of the data against the task (§5.1).
3. **Evaluation strategy.** `POST /versions/{vid}/training-plan` (synchronous, no job, standard library only) returns:
   - the default split from roles (§5.3), with any user override and its warnings;
   - **eligibility**: `unbiased_direct`, `unbiased_replay` (with the ancestor and steps to replay), or `blocked` with reasons and the `exploratory` option;
   - estimated fits and time class; blocking errors (rare classes, too few rows, limits).
4. **Preprocessing specification.** It defaults to the replayed data steps (if any, shown read-only as "replayed in each fold") followed by the version's `model_pipeline_spec`. The user can edit the model-stage part with the existing step editor. Validation uses the existing validator; stage order is enforced.
5. **Models and tuning.** Choose models (default: all four), the tuning mode (`nested` default where feasible, otherwise `defaults` with the reason), outer folds (default 5) and the seed (default 0).
6. **Train.** `POST /versions/{vid}/training-runs` (Idempotency-Key) → job `train` on queue `evaluate`. Progress shows fold, model and phase; cancellation uses the existing jobs endpoint.
7. **Comparison.** A table of every model, with metric mean ± sd, interval, per-fold values and paired differences against the dummy. **No winner is declared.** The note states that choosing the best of several models on the same folds is optimistic.
8. **Reliability audit.** Job `reliability_audit` follows automatically (and can be re-run). Findings are grouped as confirmed violations, statistical suspicions and not assessed, with the evaluation status (§6.3).
9. **Explainability.** Importances per model on held-out folds, coefficients, and error slices, each with stated limitations.
10. **Reproducibility.** Configuration, hashes, seeds, library versions, the replay record, and *Re-run* (identical results expected).

Training never writes a dataset version (invariant 9). Tests count versions and list `versions/` before and after.

---

## 5. ML evaluation methodology

### 5.1 Task and target validation (API plan step, re-checked in the worker)

| Rule | Failure |
|---|---|
| Exactly one target role with a confirmed `task_type` in `binary` / `multiclass` / `regression` | `TARGET_REQUIRED` |
| Cohort = rows with a non-missing target. Excluded rows are counted and reported. | — |
| Binary: exactly 2 classes in the cohort | `TASK_TARGET_MISMATCH` (with the observed class count) |
| Multiclass: 3–50 classes | `TASK_TARGET_MISMATCH` / `LIMIT_EXCEEDED` |
| Regression: numeric target with ≥ 10 distinct values and a non-constant cohort | `TASK_TARGET_MISMATCH` |
| Every class has ≥ `n_splits` rows (stratified/group methods) | `CLASS_TOO_RARE`, naming the class count only (never labels of a sensitive target); suggests fewer folds or merging classes in a pipeline |
| Cohort ≥ 50 rows; ≥ 1 usable feature after excluding roles (`id`, `group`, `time`, `weight`, `exclude`, `sensitive`, target) | `TOO_FEW_ROWS` / `NO_FEATURES` |
| Limits (§13) | `QUOTA_EXCEEDED` before queueing |

Datetime features are not model inputs (as in the Lab) and are reported. The `weight` role is not applied and is disclosed.

### 5.2 Full-data-fitted versions: eligibility and in-fold replay

A version with `fitted_on_full_data = true` contains values learned from every row, including future test rows. Imputation medians and Yeo-Johnson λ are examples. Training directly on it overstates performance (`leakage.pipeline.fit_on_full_data`). Phase 5 never does this silently.

**Eligibility** (`engine/ml/eligibility.py`, standard library; inputs are lineage facts):
1. `fitted_on_full_data = false` → **`unbiased_direct`**: evaluate on the version itself.
2. Otherwise, take `replay_chain(version)` = (ingest base, ordered transformation runs; reverts follow the restored version). Find the **eligible ancestor**: the latest version in the chain whose own `fitted_on_full_data` is false. It always exists, because v0 is never fitted. The **replay prefix** is the concatenation of the data-stage steps of every run after the ancestor, in lineage order, from their `spec_snapshot`s.
3. The prefix is re-validated against the ancestor's schema and roles with the existing validator. Validation tracks created columns, so later steps may use columns made by earlier ones. Every step's recomputed hash must equal its recorded hash. Row-removal ops cannot occur: none were committable in Phase 4. A row-count change would fail eligibility.
4. **Replay-equivalence proof** (worker, before any training): fit the prefix on **all** ancestor rows with `commit_transform`, finalise through Parquet, and compare the content hash with the evaluated version's. Equal → the specification provably describes the version, and the evaluation is **`unbiased_replay`**. The ancestor's rows are evaluated with the prefix refitted inside each training fold. The model-stage specification follows the prefix.
   - Floats can differ in the last bit across platforms. The worker image is the platform that committed, so exact equality is expected. A numeric `allclose(rtol=1e-9, atol=1e-12)` match with exact non-float columns is accepted and recorded as `equivalent_within_tolerance`.
5. If any of 2–4 fails (missing run, hash mismatch, validation failure, equivalence failure), the evaluation is **`blocked`**, with the reason as a confirmed finding. The user may explicitly request **`exploratory`** (`allow_exploratory: true`). That trains on the fitted version directly. Every result, chart and export of the run carries an "Exploratory — performance likely optimistic" banner. The audit reports `leakage.pipeline.fit_on_full_data` as a **confirmed critical violation**, and the evaluation status (§6.3) can never be better than `exploratory`.

Stateless committed steps (casts, missing indicators) do not set `fitted_on_full_data`, so versions built only from them are `unbiased_direct`.

The split is defined on the **evaluation source** (the version itself, or the ancestor). Rows and `_dm_row_id`s are identical along the chain (no row removal), so the fold map is the same cohort. Duplicate grouping is computed on the ancestor's values, before any learned transformation.

### 5.3 Outer split (frozen)
- **Default from roles** (ML_ENGINE_DESIGN §5.1):
  - group (or `bio.patient_id`) → `stratified_group` (classification) / `group`;
  - time → `time` (with group exclusivity if both);
  - otherwise `stratified` / `kfold`.
- Duplicate clusters are always grouped (§5.2 of the ML design).
- Overrides are allowed with the existing warnings (`GROUP_ROLE_IGNORED`, `TIME_ROLE_IGNORED`). The audit turns an ignored role into a **confirmed configuration violation**.
- 2–10 folds, default 5. Time splits: expanding window; rows before the first test window are never tested (reported; out-of-fold coverage < 100%).
- Stored and reused through `split_definitions` by config hash (shared with the Lab when configurations match). Every model and every tuning configuration uses the identical outer folds; the runner verifies that every test row of every fold is scored (otherwise `NOT_DIRECTLY_COMPARABLE`, excluded from comparison).

### 5.4 Model selection without test-set influence (nested CV)
For each outer fold *k*:
1. **Outer training rows** = cohort minus fold *k*. **Outer test rows** are untouched until step 4.
2. **Inner folds:** `make_splits` on the outer-training rows with the same method family, 3 folds, seed `seed + 1000 + k`. Groups and duplicate blocks are respected; time order is respected (inner test windows lie after inner training windows, all inside the outer training window).
3. For each grid configuration: for each inner fold, fit the preprocessing specification and the model on inner-training rows and score inner-test rows on the **primary metric** (balanced accuracy, or ROC-AUC for binary when defined; RMSE for regression). Select the configuration with the best inner mean. Ties go to the earlier (simpler) grid entry.
4. Refit the preprocessing and the selected configuration on all outer-training rows; predict outer-test rows; compute metrics.
5. Record the selected configuration per fold. The summary reports selection frequency and states that **the reported score estimates the selection procedure, not one fixed configuration**.

**Grids** (fixed, documented, ≤ 4 configurations per model; all other parameters fixed):

| Model | Classification | Regression |
|---|---|---|
| `dummy` | `prior` (no grid) | `median` (no grid) |
| `linear` | `LogisticRegression(C ∈ {0.01, 0.1, 1, 10}, max_iter=2000)` | `Ridge(alpha ∈ {0.1, 1, 10, 100})` |
| `random_forest` | `RandomForestClassifier(n_estimators=100, max_depth ∈ {6, 12}, min_samples_leaf ∈ {2, 10}, n_jobs=1)` | `RandomForestRegressor` (same) |
| `hgb` | `HistGradientBoostingClassifier(learning_rate ∈ {0.05, 0.1}, max_leaf_nodes ∈ {15, 31}, early_stopping=False)` | `HistGradientBoostingRegressor` (same) |

Every estimator gets `random_state = seed`. Parameter names are confirmed against scikit-learn 1.9.1 in a contract test before use (CLAUDE.md library rule).

**Tuning modes:**
- `nested` (default when feasible);
- `defaults` (one configuration per model, Lab-equivalent). It is used automatically, with the reason, when every class does not have ≥ 3 rows in every outer-training inner fold, or when the cohort has < 200 rows, or when the fit budget would be exceeded.

The user may also choose `defaults`.

Fits = outer folds × Σ_models (configurations × inner folds + 1). Defaults with 5 folds and all four models: 5 × (1 + 3·(4·3+1)) = 200 fits. The estimate is shown before queueing.

### 5.5 Metrics and visualisations
All metrics are computed on **outer test rows only**, per fold, then mean, sd, min, max and per-fold values (reusing `metrics.py`). Undefined values are `None` with a reason, never 0. For example, ROC-AUC is undefined when a fold lacks a class, and R² when the test target is constant.

- **Binary:**
  - accuracy, balanced accuracy, precision, recall, F1 (positive class = second label in sorted order, disclosed; class names coded for a sensitive target);
  - ROC-AUC, PR-AUC, log loss, Brier score, ECE (10 bins);
  - confusion matrix (summed over folds);
  - ROC and PR curves from pooled out-of-fold scores, with per-fold AUC shown, at ≤ 200 points each;
  - calibration reliability curve (10 quantile bins, observed rate, mean predicted, count per bin).
- **Multiclass:**
  - macro and weighted precision, recall and F1; balanced accuracy; accuracy;
  - one-vs-rest macro ROC-AUC and PR-AUC when every class appears in each test fold;
  - per-class table; confusion matrix;
  - per-class one-vs-rest curves for ≤ 10 classes;
  - calibration: top-label ECE with a reliability curve.
- **Regression:**
  - MAE, MSE, RMSE, R², median AE;
  - actual versus predicted as a **binned 2-D count grid** (≤ 40 × 40 cells) with a y = x reference;
  - residuals versus predicted (binned) and a residual histogram;
  - residual quantiles.

Charts are aggregates only: no per-row values reach the browser.

**Uncertainty:**
- fold mean ± sd, with min and max;
- for the primary and headline metrics, a **group-aware percentile bootstrap** (1,000 resamples of split groups, seeded) over pooled out-of-fold predictions.

It is labelled as reflecting test-sample variability only. It ignores training variability and the fact that predictions come from k different fold models, so it is not a guarantee. No p-values, because fold scores are not independent. Paired per-fold differences against the dummy and between models get the existing descriptive verdicts (`negligible`, `consistently_better`, `within_fold_variation`, …). A multiple-comparison caution is shown with more than two non-dummy models.

---

## 6. Reliability Auditor

### 6.1 Design
- **Package:** `engine/ml/audit/`. Specs are standard library (id, version, basis, requirements, method, limitations), like `engine/checks/spec.py`; implementations are worker-only.
- **Context:** `AuditContext` = the training run's configuration and summary, out-of-fold predictions, the frozen split, the cohort frame of the evaluation source, roles, lineage/eligibility facts, and the latest non-stale diagnostics findings.
- **Output:** findings with the existing schema and DB constraints. The status is `detected | not_detected | not_assessed | error`, and severity, evidence, explanation, columns, affected rows, recommendation, limitations and sampling are all required by constraint.
  - `evidence.basis` is one of:
    - `confirmed`: a fact computed from data or lineage, e.g. a group present in train and test;
    - `configuration`: a fact about the chosen setup, e.g. a time role ignored;
    - `statistical`: a suspicion from a threshold on an estimate.
  - Titles say "detected" only for `confirmed` and `configuration`, and "possible" or "suspected" for `statistical`.
- **Honesty:** unmet requirements give `not_assessed` with the reason, and are never `not_detected`. A registry-wide test asserts this, like the existing diagnostics honesty test.
- **Isolation:** each check runs isolated; an exception becomes `error` with only the exception type. Sampling is disclosed.

### 6.2 Checks

| ID | Basis | Method | Requirements → otherwise `not_assessed` | Severity |
|---|---|---|---|---|
| A1 `leakage.target.single_feature` | statistical | Per original feature (≤ 200, top-variance beyond, disclosed): out-of-fold depth-3 decision tree on the frozen outer folds, refit per fold. Flag binary ROC-AUC ≥ 0.98, multiclass balanced accuracy ≥ 0.98, or regression R² ≥ 0.95, and any feature whose single-feature score is ≥ 95% of the best model's. | target, ≥ 1 feature | critical if ≥ 0.995, else high |
| A2 `leakage.target.derived` | statistical | Near-deterministic relation: categorical purity ≥ 99% with categories ≪ rows; numeric \|Spearman\| ≥ 0.999 with the target, or within-class constant values. | target | critical |
| A3 `leakage.identifier` | statistical | ID-like model inputs (unique ratio ≥ 0.95, integer sequences or code strings) that carry target signal (A1 score or order association); reuses the diagnostics heuristic. | target | high |
| A4 `leakage.duplicates.cross_fold` | confirmed | Exact (feature + target hash) and normalised near-duplicate rows on both sides of any outer fold. By construction 0 for DataMedic splits; counts are reported either way. Feature-identical rows with different labels across folds are listed as info. | split | high if > 0 |
| A5 `leakage.groups.cross_fold` | confirmed | Groups (group role / `bio.patient_id`) present in both training and test of any fold; per-fold counts. | group role | critical if > 0 |
| A6 `leakage.groups.undeclared` | statistical | Repeated entities without a group role: ID-like columns with repeats (mean rows per value ≥ 1.2) or repeated feature signatures. | — | high |
| A7 `leakage.temporal.ordering` | configuration / confirmed | Time role with a non-time split → configuration violation. With a time split: verify max(train time) < min(test time) per fold. | time role (otherwise `not_assessed`; an info note if datetime columns exist without a time role) | critical (violation) |
| A8 `leakage.temporal.post_outcome` | — | Needs outcome-time and feature-time roles, which only domain modules declare. | outcome/feature time roles → `not_assessed` in Phase 5 | — |
| A9 `leakage.pipeline.fit_on_full_data` | confirmed | Lineage: an `exploratory` run on a fitted version → violation. `unbiased_replay` → `not_detected`, with the ancestor, replayed step hashes and equivalence result as evidence. | lineage | critical |
| A10 `leakage.pipeline.boundaries` | confirmed | Runtime assertions recorded per fold: every learned step's `n_fit_rows` = training rows; `training_only` steps ran only in `fit_resample`; test row ids unchanged and complete; no synthetic ids among test rows; sample weights never used in scoring. | — | critical if violated |
| A11 `leakage.tuning.isolation` | confirmed | Every inner fold ⊆ outer-training rows; no outer test row was used in selection; the selected configuration was chosen before outer scoring. `defaults` mode → `not_detected` with "no selection performed". | — | critical if violated |
| A12 `leakage.pipeline.target_encoding` | configuration | Specification inspection for target-derived encodings. None exist in the catalogue → `not_detected`, "no target-using encoders in this pipeline". | spec | critical |
| A13 `reliability.overfitting` | statistical | Training-fold score versus outer test score per model. The gap thresholds are model-aware: a random forest's training score is near 1 by design, so gaps are judged against the dummy and linear models. | supervised | medium |
| A14 `reliability.fold_variability` | statistical | Primary-metric sd > 0.05 (bounded) or coefficient of variation > 0.2, or a fold range > 0.15. | ≥ 3 folds | medium |
| A15 `reliability.class_imbalance` | confirmed (data property) | Minority share < 10% (medium) / < 2% (high); accuracy versus majority-rate baseline; recommends balanced accuracy and PR-AUC. | classification | medium/high |
| A16 `reliability.sample_size` | statistical | Test rows per fold < 30; minority events per feature < 10 (binary); rows per feature < 10 (regression); any class < 5 per test fold. | — | high/medium |
| A17 `reliability.suspicious_performance` | statistical | ROC-AUC / balanced accuracy / R² ≥ 0.99, or a large margin over the dummy together with any A1–A6 detection; links to those findings. | — | high |
| A18 `reliability.dummy_competitive` | statistical | The best non-dummy model is within fold variation of the dummy on the primary metric. | — | medium |
| A19 `reliability.calibration` | statistical | Binary ECE > 0.1 (medium) / > 0.2 (high); Brier versus prevalence baseline. Notes that resampling and class weights distort probabilities when the specification contains them. Multiclass: top-label ECE. | probabilities, ≥ 100 test rows | medium/high |
| A20 `reliability.metadata` | configuration | One finding listing missing metadata that blocks assessment: no group role while A6 suspects repeats; datetime columns without a time role; roles changed since the diagnostics run (semantics hash); diagnostics missing or stale; target role without confirmation. | — | info/medium |

The latest non-stale diagnostics findings `universal.leakage.target_proxy` and `universal.leakage.identifier` are shown alongside, as cross-references, not re-derived.

Each detected finding names a corrective action, for example:
- declare a group role;
- use a time split;
- exclude a column (via roles);
- evaluate from the eligible ancestor;
- reduce folds;
- prefer balanced metrics;
- recalibrate or interpret probabilities cautiously.

### 6.3 Summary indicator (not a score)
**Evaluation status**, by deterministic rules shown in the UI:
- `blocked`: any confirmed or configuration violation of critical severity (A4, A5, A7, A9 in exploratory, A10, A11, A12);
- `exploratory`: the run was explicitly exploratory;
- `concerns`: any high or critical statistical suspicion;
- `no_issues_detected_among_assessed`: otherwise. It always shows the count and list of `not_assessed` checks.

Limitations text: "This status summarises the checks that ran. It is not a certification of model validity; absence of a detection is not proof of absence."

No numeric score is produced, because no validated weighting of these heterogeneous risks exists.

---

## 7. Explainability methodology
1. **Permutation importance** (primary; model-agnostic).
   - For each outer fold and model, permute one **original input column** in the outer test rows, run the fold's fitted preprocessing and model, and record the drop in the primary metric. Use 5 repeats with a seeded permutation, and mean ± sd across folds and repeats.
   - Permuting original columns before the fold pipeline maps importance to original columns by construction. Created features (missing indicators, one-hot columns) are attributed to their source column.
   - Budgets: ≤ 5,000 test rows per fold (seeded sample, disclosed) and ≤ 100 columns (beyond that, the top 100 by model-specific importance or variance, disclosed).
   - Implemented in-house, because the fold pipeline is not a single scikit-learn estimator. A contract test checks it against `sklearn.inspection.permutation_importance` on a plain model.
2. **Model-specific:**
   - linear coefficients on standardised inputs, with one-hot coefficients aggregated per original column (L2 norm; multiclass per class);
   - random-forest impurity importance, labelled as biased towards high-cardinality and continuous features. HGB has no built-in importance in scikit-learn, so it shows permutation importance only.

   Mapping uses `ColumnTransformer` output groups and the executor's feature lineage (Part C).
3. **Error slices.**
   - For the 5 most important original features, show the primary error metric per bin or category: quantile bins for numeric features, the top 10 categories for categorical ones.
   - Each slice needs ≥ 30 test rows; smaller ones are merged into "other".
   - Sensitive-role columns are never sliced, and slice labels are bin edges or category names already visible to the owner.
   - Classification adds the per-class error profile from the confusion matrix.
4. **Stated limitations on every view:**
   - importance is predictive association under this model and data, not causation;
   - correlated features share or mask importance;
   - permutation can create unrealistic rows;
   - importance can differ between folds and models (shown).

SHAP is not justified for this milestone (§2.2).

---

## 8. Backend architecture

```
engine/ml/                       # pure engine (no FastAPI/SQLAlchemy/Celery/boto3)
  spec.py        # stdlib: model catalogue (keys, tasks, grids, notes), metric specs, tuning modes
  plan.py        # stdlib: config validation, split default, fit estimate, limits -> errors/warnings
  eligibility.py # stdlib: evaluation mode from lineage facts (direct / replay / blocked)
  replay.py      # pandas: build the replay prefix, the equivalence proof
  models.py      # build_estimator(key, task, kinds, params, seed) on experiment/models.py helpers
  tuning.py      # inner splits, grid evaluation, selection records
  runner.py      # outer loop, OOF predictions, per-model results, progress, cancellation
  metrics.py     # reuses experiment/metrics.py; adds MSE, curves, binned residuals, bootstrap
  explain.py     # permutation importance, coefficients, impurity, mapping, error slices
  audit/spec.py, audit/checks/*.py, audit/run.py
datamedic/training/              # service layer
  router.py, schemas.py, service.py (API: plan, create, list, detail, rerun, audit),
  jobs.py (worker handlers `train`, `reliability_audit`)
```

- **Jobs:**
  - `train`: queue `evaluate`, 3,600 s;
  - `reliability_audit`: queue `evaluate`, 1,200 s, chained after a successful `train` and re-runnable.

  The broker carries only `job_id`; parameters live in `jobs.params`. Status lives only in `jobs`.
- **Worker flow (`train`):**
  1. load the run's stored configuration;
  2. verify `config_hash` and the source content hash;
  3. re-run eligibility (lineage may not have changed, as versions are immutable, but roles snapshot from the config is used);
  4. replay-equivalence proof if needed;
  5. load or create the frozen split;
  6. run outer folds with cancellation checks before each fold × model, recording progress (`{phase, fold, model, done, total}`) through the separate session, as in the Lab;
  7. write the out-of-fold artifact (Parquet: `_dm_row_id`, fold, model, prediction, class probabilities);
  8. in the final transaction: model evaluations, summary, `completed_at`, artifact row, follow-up audit job.

  A crash, failure or cancellation stores no partial results. The orphaned object is reconciled after 24 h. A retried completed run returns `skipped`, and a complete-once trigger forbids changes.
- **Determinism:** fixed seeds everywhere, `n_jobs=1`, `threadpool_limits(1)`, sorted row order, inner-split seeds derived from the run seed. A rerun must reproduce identical metrics on the same platform (test).
- **Failures:** a model failure in a fold isolates that model and fold (`partial`, excluded from comparison). Preprocessing failure fails that fold for all models. `MemoryError` fails the job as `LIMIT_EXCEEDED`. Error messages carry engine codes or exception types only.
- **Errors:** new `DomainError` codes `TASK_TARGET_MISMATCH`, `CLASS_TOO_RARE`, `TOO_FEW_ROWS`, `EVALUATION_BLOCKED`, `REPLAY_NOT_ESTABLISHED`; existing `QUOTA_EXCEEDED`, `NO_FEATURES`, `CONFIG_CHANGED`.

---

## 9. Database schema and migration plan (migration 0007, expand-only)

**`training_runs`** (configuration written by the API; results written once by the worker):
- `id`, `owner_id`, `dataset_id`, `dataset_version_id` (selected), `evaluation_version_id` (direct or ancestor), `job_id` UNIQUE, `name`;
- `task_type` (`binary | multiclass | regression`), `target_column`, `evaluation_mode` (`unbiased_direct | unbiased_replay | exploratory`), `replay` jsonb (ancestor, run ids, step hashes; equivalence result once proven);
- `split_config` jsonb, `split_definition_id` (nullable until done), `models` jsonb, `tuning` jsonb (mode, grids, inner folds), `preprocess_steps` jsonb (with stages), `spec_hash`;
- `config` jsonb (complete reproducible configuration including the roles snapshot), `config_hash`, `source_content_hash`, `seed`, `rerun_of_id`;
- `summary` jsonb, `oof_artifact_id`, `engine_version`, `library_versions`, `created_by`, `created_at`, `completed_at`.
- CHECKs: enumerations; `(completed_at IS NULL) = (summary IS NULL)`; `evaluation_mode = 'unbiased_replay'` ⇔ `evaluation_version_id <> dataset_version_id`.
- Trigger: complete-once (no change after completion).

**`model_evaluations`** (insert-only, one per model per run): `training_run_id`, `model_key`, `status` (`completed | partial | failed`), `comparable` (+ reason CHECK), `metrics` (per-metric fold statistics and bootstrap interval), `fold_results`, `selection` (configuration per fold, frequency), `curves` (bounded), `calibration`, `residuals` (binned), `confusion`, `per_class`, `importance`, `error_slices`, `failures`, `timings`. UNIQUE (`training_run_id`, `model_key`).

**Existing tables widened:**
- `analysis_runs`: `kind` CHECK += `reliability_audit`; new nullable `training_run_id` FK. CHECK: `kind = 'reliability_audit'` ⇔ `training_run_id IS NOT NULL`. Findings rows are reused unchanged.
- `jobs.type` += `train`, `reliability_audit`.
- `artifacts.kind` += `oof_predictions`; `artifacts.owner_type` += `training_run`.

**Security and integrity:**
- RLS (`FORCE`) and owner triggers on the new tables;
- `enforce_version_in_dataset` on both version columns;
- `enforce_same_dataset_parent` for split, artifact, rerun, job and evaluation-to-run references;
- grants: `SELECT, INSERT, DELETE` for `dm_app` (DELETE only for dataset purge, which deletes Phase 5 rows first), plus a column-limited `UPDATE` of the worker-written result columns, guarded by the complete-once trigger;
- maintenance read grant on `artifacts(object_key)` already exists (orphans).

No table references a dataset version as an output.

**Values stored:**
- out-of-fold predictions (owner-only artifact, never served raw);
- class names as in the Lab (coded for a sensitive target);
- bin edges and category names in slices (sensitive columns excluded);
- linear coefficients as JSON.

No pickles.

---

## 10. API specification (all owner-scoped; another user's resource → 404)

| Method & path | Purpose |
|---|---|
| `GET /ml/models` | Catalogue: keys, supported tasks, grids, notes, limitations (from `engine/ml/spec.py`) |
| `POST /versions/{vid}/training-plan` | Body = draft configuration. Returns the validated configuration, split default and warnings, eligibility (mode, ancestor, replay steps, blocking reasons), fit estimate, errors. No job, no writes. |
| `POST /versions/{vid}/training-runs` | `Idempotency-Key` required. Body: `{name, models?, tuning: {mode}, split: {method?, n_splits, seed}, preprocess_steps?, seed, allow_exploratory}` → `202 {job, training_run}`. `409 EVALUATION_BLOCKED` unless `allow_exploratory`; `409 QUOTA_EXCEEDED`; `422` validation errors with pointers. |
| `GET /datasets/{id}/training-runs` | List: id, name, version, mode, status (from jobs), headline metric, evaluation status, created_at |
| `GET /training-runs/{rid}` | Configuration, summary, model evaluations, warnings, limitations, reproducibility record, job |
| `POST /training-runs/{rid}/rerun` | `Idempotency-Key`; identical configuration (`rerun_of_id`) |
| `POST /training-runs/{rid}/audit` | Re-run the reliability audit (e.g. after role changes) → `202 {job}` |
| `GET /training-runs/{rid}/audit` | Latest audit run with findings, evaluation status, staleness against current semantics |
| `POST /jobs/{id}/cancel` | Existing |

- The new path parameter `rid` (training run) is added to the isolation matrix `KNOWN_PARAMS`.
- Responses never contain row values or per-row predictions.
- Admission limits apply (running/queued jobs per user).

---

## 11. Frontend design (Next.js 16 conventions; read `node_modules/next/dist/docs/` first)

- **Navigation:** a new **Models** tab in `DatasetNav`.
- **`/datasets/[datasetId]/models`:** a run list (status, mode badge, headline metric ± sd, evaluation status) and **New training run**.
- **New run flow** (one page, sections, live plan validation through `training-plan`):
  1. *Task and target*: read-only summary from roles, with a link to the roles editor if missing.
  2. *Evaluation*:
     - split method (default from roles, override with warnings), folds, seed;
     - an **eligibility panel**: direct, "replayed in each fold from version N" (lists the steps), or a blocked explanation with an explicit *Run as exploratory* checkbox and its consequences.
  3. *Preprocessing*: read-only replayed steps plus an editable model-stage list (reuses `AddStepForm`, `ParamField`, stage and risk badges).
  4. *Models and tuning*: model checkboxes with notes, tuning mode, the fit estimate.
  5. *Review and start*.
- **`/datasets/[datasetId]/models/[runId]`:**
  - Header: status, progress (phase, fold, model) and **Cancel**; an evaluation-mode banner (prominent amber for exploratory); evaluation status with its method.
  - **Comparison:** table per model (mean ± sd, bootstrap interval, min–max), per-fold dot strip, paired differences against the dummy with verdicts. No "best" badge; optimism note.
  - **Performance** (model selector):
    - metric table;
    - ROC and PR curves; confusion matrix heat grid; per-class table;
    - calibration curve with ECE and Brier;
    - regression: binned actual-versus-predicted, residual views.
  - **Reliability:** findings grouped by *Confirmed violations*, *Statistical suspicions*, *Not assessed* (reusing `FindingCard`, with a basis badge); diagnostics cross-references; re-run audit.
  - **Explainability:** permutation importance bars with sd whiskers, per model; coefficients; error slices; limitations box.
  - **Reproducibility:** configuration and hashes, split definition, seeds, library versions, replay proof, selected configurations per fold, **Re-run**.
- **Charts:** new dependency-free primitives in `components/charts/` (`LineChart` for ROC/PR/calibration, `HeatGrid` for the confusion matrix and binned scatter, `DotStrip`, `IntervalBar`). They follow the existing chart rules: a tooltip on every mark, a table equivalent, `--viz-*` tokens. The dataviz guidance is applied at implementation.
- **Data and UI conventions:**
  - `lib/training.ts` (formatting, verdict and basis labels, status rules) with tests;
  - `lib/queries.ts` hooks (polling while active);
  - generated client only; no `dangerouslySetInnerHTML`.
  - Undefined metrics render as "—", with the reason in a tooltip, never 0.

---

## 12. Security and privacy safeguards
- **Isolation:** `owner_id` + RLS on every new row; service-layer owner scoping; 404 across tenants; AuthZ matrix covers every new route; DB security tests for RLS, grants, triggers and constraints.
- **Data leaving the platform:** no dataset content leaves it; there is no LLM. Logs carry ids, codes and counts only, never values or column names. Library warnings are reduced to their category (as in the Lab).
- **Broker:** carries only `job_id`, JSON serializer.
- **Immutability:** training and audits never write versions or `raw/`; tests assert version counts and the `versions/` listing.
- **Persistence:** no pickle/joblib anywhere; no model artifacts; out-of-fold predictions are Parquet written by the platform, read only by the platform, and SHA-256-verified on read.
- **Sensitive values:** sensitive-target class names coded; sensitive columns excluded from slices; no per-row values in API responses; charts are aggregates.
- **Client trust:** risk, stage, kind and model grids come from server catalogues; the client cannot submit estimator parameters outside the grids.
- **Resource safety:** limits enforced before queueing and memory pre-checks in the worker (§13).

---

## 13. Resource limits and dependency decisions

| Setting (new, configurable) | Default |
|---|---|
| `training_max_rows` | 200,000 |
| `training_max_cells` | 5,000,000 |
| `training_max_fits` | 400 |
| `training_max_features` | 500 (after excluding roles) |
| `permutation_max_rows_per_fold` / `permutation_max_columns` / repeats | 5,000 / 100 / 5 |
| `audit_single_feature_max_columns` / `audit_max_rows` | 200 / 50,000 (seeded sample, disclosed) |
| `bootstrap_resamples` | 1,000 |
| Max runs per dataset | 50 (`QUOTA_EXCEEDED`; nothing deleted) |
| Job time limits | `train` 3,600 s, `reliability_audit` 1,200 s |

- **Random forest memory:** depth ≤ 12, 100 trees, `min_samples_leaf ≥ 2` bound the node count. A worker memory pre-check (rows × features × a model factor) refuses with `LIMIT_EXCEEDED` before fitting. Only one fold model is alive at a time.
- **Dependencies:** **none new.** scikit-learn 1.9.1, NumPy, pandas, SciPy and threadpoolctl are already locked. XGBoost, SHAP, Optuna and Cleanlab are not added. The license gate is unchanged.

---

## 14. Testing and acceptance criteria

All fixtures are synthetic, generated with known properties.

**Engine unit tests:**
- **Classification and regression:** end-to-end on separable, noisy and null-signal data. Null-signal data must score near chance (ROC-AUC within 0.5 ± 0.1); the dummy must be competitive on it and A18 must detect that.
- **Determinism:** same configuration → identical metrics, selections and out-of-fold predictions; different seed → different folds.
- **Split integrity:** every cohort row tested exactly once (non-time); identical outer test ids for every model and configuration; scored-row check.
- **Group and temporal isolation:** no group on both sides in outer or inner folds; time order holds in outer and inner folds.
- **Train-only preprocessing:** spy transformers record fit row ids ⊆ training rows for every fitted op and every inner and outer fit; test rows never passed to `fit`.
- **Resampling boundaries:** synthetic rows only in training tables; test count unchanged; weights never used in scoring.
- **Tuning isolation:** a spy records every row used in selection; no outer test id appears; a planted "test-peeking" variant fails the A11 assertion.
- **Leakage detection** (each seeded and verified):
  - target copy → A1/A2 critical;
  - target-derived proxy → A2;
  - ID sorted by outcome → A3;
  - duplicated rows with a user-overridden random split → A4;
  - patients across folds with the group role ignored → A5 and A7-style configuration violation;
  - undeclared repeats → A6;
  - random split on time data → A7;
  - exploratory fitted version → A9.

  Clean controls produce no high or critical findings.
- **Full-data-fitted restrictions:** a v1 with median imputation → `unbiased_replay` with equivalence proven, and the evaluation equals a manual in-fold evaluation from v0. A hash-mismatch lineage → `blocked`; with `allow_exploratory`, results are labelled and A9 is critical. Revert chains follow the restored version.
- **Metrics:** undefined metrics → `None` + reason (single-class fold, constant target), never 0; bootstrap is deterministic and group-aware.
- **Unsupported tasks:** `TASK_TARGET_MISMATCH`, `CLASS_TOO_RARE`, `TOO_FEW_ROWS`, `NO_FEATURES`; text and datetime ignored and reported.
- **Failures and cancellation:** an injected model failure → `partial`, excluded from comparison; cancellation before a fold → cancelled, no rows written; `MemoryError` → `LIMIT_EXCEEDED`.
- **Explainability:** a planted informative feature ranks first and a pure-noise feature is near 0; original-column mapping is correct for one-hot and missing-indicator features; the contract test against `sklearn.inspection.permutation_importance`.
- **Audit honesty:** every check spec has limitations; `not_assessed` when requirements are unmet (registry-wide); no "no issue found" for a check that did not run.
- **Library contract tests:** every estimator parameter used, on the pinned scikit-learn.

**Integration (real test database):**
- API → worker → results;
- Idempotency-Key replay; rerun reproduces metrics;
- quota and limits before queueing;
- audit chained and re-runnable;
- DB constraints (complete-once, mode/version CHECK, findings constraints for audit rows);
- RLS and grants on new tables (extending `test_db_security.py`);
- isolation matrix with `rid` (404 across tenants);
- dataset purge removes Phase 5 rows and artifacts;
- no version writes; broker message contains only the job id;
- orphan reconciliation of an out-of-fold object after a simulated crash.

**Frontend:** component tests for the configuration flow (eligibility states, exploratory confirmation), comparison table (no winner, undefined metrics), charts (tooltips and table equivalents), findings grouping by basis, cancellation button, and the `lib/training.ts` rules.

**Live-stack smoke test** (scripted against the running stack, new synthetic accounts; existing data untouched):
1. migration 0007 applied without reset; services healthy;
2. binary run (group split, nested);
3. regression run;
4. multiclass run;
5. a fitted v1 evaluated by replay;
6. a blocked lineage, then an exploratory run;
7. cancellation;
8. a seeded-leak dataset detected;
9. rerun equality;
10. cross-tenant 404;
11. no new versions and earlier objects byte-identical;
12. no values in logs.

A manual browser checklist is delivered for the owner.

**Regression:** full backend suite (with `TEST_ADMIN_DATABASE_URL`), ruff, format, strict mypy, frontend tests, typecheck, lint, production build, `docker compose config`.

**Acceptance:** every item above passes; docs updated (§15 step 12); limitations reported; a final report with an APPROVE / DO NOT APPROVE recommendation.

---

## 15. Internal implementation sequence (no owner checkpoints)

1. **Contracts:** scikit-learn contract tests for every estimator and parameter in the grids; `engine/ml/spec.py`, `plan.py`, `eligibility.py` with unit tests.
2. **Engine core:** `models.py` (random forest, params), `tuning.py`, `runner.py`, `metrics.py` extensions; leakage-boundary and determinism tests.
3. **Replay:** `replay.py` with the equivalence proof and the v0-versus-v1 equality tests.
4. **Explainability:** `explain.py` and its tests.
5. **Auditor:** specs, checks and the context; seeded-leak golden tests, clean controls, honesty test.
6. **Migration 0007**, models and DB security tests.
7. **Service layer:** `training/` (plan, create, rerun, audit, jobs), registry, config limits, errors, purge; integration and isolation tests.
8. **Lab gap fix:** the `FITTED_ON_FULL_DATA_SOURCE` warning with a test.
9. **Frontend:** OpenAPI export and client generation; lib and hooks; chart primitives; pages and components; tests.
10. **Full regression and fixes.**
11. **Live-stack rebuild and the smoke test**; benchmarks at representative sizes (5k, 20k, 100k rows) recorded honestly.
12. **Docs:** ML_ENGINE_DESIGN (§5 training, §6 auditor), API_SPECIFICATION, DATABASE_SCHEMA, ARCHITECTURE (modules, jobs), PRD (decisions C1–C3), IMPLEMENTATION_ROADMAP, README, CLAUDE.md (Phase 5 invariants, corrected pinned baseline); final report.

---

## 16. Owner decisions (one consolidated approval; defaults recommended)

| # | Decision | Recommended default |
|---|---|---|
| P1 | Scope changes against the PRD: bounded nested-CV hyper-parameter selection (C1); calibration assessment, group-aware bootstrap intervals and random forest moved from V2 (C2); locked holdout stays V2. | Approve; the PRD is updated to record it. |
| P2 | XGBoost | Stay deferred; HGB provides boosting. Revisit with a compatibility spike (CPU-only wheel, image size) later. |
| P3 | Full-data-fitted versions | Leakage-controlled evaluation (formerly worded "unbiased", §19) only by proven in-fold replay from the eligible ancestor; otherwise blocked, with exploratory runs only on explicit opt-in and permanently labelled. |
| P4 | Touching Phase 4 code | Only the `FITTED_ON_FULL_DATA_SOURCE` Lab warning (C5). |

Approving this plan approves these defaults. All other technical choices above are defaults within the approved scope.

---

## 17. Risks, limitations and mitigations

| Risk / limitation | Mitigation |
|---|---|
| Compute and memory on the reference machine (about 1.3 GB free) | Fit budget and estimate before queueing; one thread; random-forest depth bounds; memory pre-check; one fold model alive; benchmarks recorded; limits configurable |
| Nested CV cost | ≤ 4 configurations, 3 inner folds, automatic `defaults` for small cohorts or over budget |
| Optimism from choosing among models on the same folds | Stated in the UI; no winner; paired differences descriptive only |
| Bootstrap intervals understate uncertainty | Labelled as test-sample variability only; fold dispersion shown beside them |
| Statistical audit checks give false positives or negatives | `basis` distinguishes suspicion from fact; thresholds documented; clean controls in tests; limitations on every finding |
| Replay equivalence may fail across platforms | Proof runs in the worker image (committing platform); tolerance path recorded; otherwise blocked, never silently accepted |
| Time-split coverage < 100% out-of-fold | Coverage reported; curves and bootstrap computed on tested rows only |
| Importance misread as causation | Limitations on every view; no causal language |
| Very wide or high-cardinality data | Feature and category caps (one-hot top 20), sampling disclosed |
| Determinism differences across library versions | Library versions recorded; rerun equality asserted on the same platform only |
| No model persistence (users may expect a downloadable model) | Explained in the UI and docs; reproduction by rerun; persistence would need a safe format decision later |
| Schedule | Estimated complexity is **large**: comparable to Phase 4 Parts D and E combined. About 20–25 new backend modules, 1 migration, 2 pages and about 10 components, and roughly 150–200 new tests. Delivered as one autonomous milestone. |


---

## 18. Implementation notes and deviations (2026-11-25)

Implemented as planned, with these refinements (details in ML_ENGINE_DESIGN §5.6 and §6.0, API §16,
DATABASE_SCHEMA "Phase 5 implementation notes"):

- **Engine layout.** `engine/ml/` holds `spec.py`, `eligibility.py` and `audit_spec.py` (standard library,
  imported by the API) and `models.py`, `replay.py`, `runner.py`, `metrics.py`, `explain.py`, `audit.py`
  (worker). The auditor is two modules rather than a package.
- **Fidelity and safety are separate.** Eligibility re-validates every committed step to its recorded hash
  and refuses row-changing steps, steps that transformed the current target, and group/time roles on
  columns created by committed steps. The worker then proves fidelity by refitting the replayed commits on
  all ancestor rows (exact content hash, or the documented `allclose` tolerance). Replay applies each commit
  in turn, fitted on the fold's training rows, with the same Parquet round trip as the platform between
  commits.
- **Permutation importance** permutes, with one shared row permutation, every model input derived from an
  original column (its lineage), in the fold's fitted model-input table. This equals permuting the original
  column for single-column operations; for multi-column operations it is attributed to the written column
  (disclosed). Inputs whose lineage does not resolve to one column are reported as `unresolved:<name>`.
- **Bootstrap intervals** are unavailable for time-ordered splits (temporal dependence; no block bootstrap),
  and when there are fewer than 10 independent groups.
- **Evaluation status.** For an exploratory run, the acknowledged full-data fit gives `exploratory`; any other
  critical confirmed or configuration violation still gives `blocked`.
- **Limits** (settings): 200,000 rows, 5,000,000 cells, 400 fits, 500 features, 50 runs per dataset, a
  900 MB worker memory estimate (rows × features × 8 bytes × 12), audit samples of 50,000 rows and 200
  single-feature columns.
- **Cancellation** is checked before every outer fold, every model and after the last fit; a fit already in
  progress is not interrupted (documented in the UI). A cancelled run stores no results.
- **Task mismatch.** The roles API already refuses a task type the data contradicts; training re-checks the
  rule in the plan and in the worker (class counts, rare classes).
- **Unchanged Phase 4 code** except the Lab's `FITTED_ON_FULL_DATA_SOURCE` warning (P4), a `kind` filter on
  the diagnostics and recommendations queries so audits never appear as diagnostics, and the exported
  `StepRow` UI component.
- Training-code generation, model persistence and the locked holdout remain excluded (§2.2).

## 19. Final closure: browser verification and corrections (2026-11-26)

The owner completed the manual browser verification of all nine categories (version eligibility and
fold-wise replay; configuration and limits; cancellation with no results; classification comparison and
charts; Reliability Auditor; explainability and reproducibility; exploratory warnings and the critical
leakage finding; regression results; the Lab's full-data warning) and conditionally accepted Phase 5,
subject to the investigations below. Each was investigated against the code and the live database.

| # | Concern | Finding | Resolution |
|---|---|---|---|
| 1 | Headline column showed "—" for every run | **Defect.** `training.router.list_training_runs` built `TrainingRunSummaryOut` without the model evaluations, so `headline` was always `null` in the list (the detail page passed them). The old builder also always took the largest mean, ignoring direction, and gave no reason when no headline existed. | `training.schemas.headline_of`: the highest observed fold mean of the primary metric among compared models (dummy excluded), respecting `higher_is_better`, ties to the earlier model; explicit states `available`, `unavailable` (with reason), `pending`, `failed`, `cancelled`; the dummy's mean as a reference; exploratory runs carry an `exploratory_note` and a red "Exploratory: likely optimistic" badge in the list. `service.list_for_dataset` loads the evaluations of the listed runs in one owner-scoped query. The optimism note is kept; no winner is declared. |
| 2 | Exploratory regression run showed error slices for the sensitive column `code` and "roles changed since this audit" | **Explained by intentional test configuration, plus a real gap.** The smoke test (step 12) replaced version 1's roles with only `b: target (regression)` to exercise unsafe evaluation, ran the exploratory run, then restored the roles (stored run `config.roles = {"target": ["b"]}`, `sensitive_columns = []`; current roles rewritten 6 s after the run was created). Training honoured the roles it had. The gap: historical results were served with training-time sensitivity only, so a column sensitive *now* (or on another version, or derived from a sensitive column by committed steps) could still be shown. | `training.privacy` (new): a column is protected if it was sensitive at training time, is sensitive now on any version of the dataset, or is derived from such a column through committed lineage (`transformation_runs.step_reports.columns_added`). `GET /training-runs/{rid}` filters a copy of the results: error slices of protected columns are withheld; for a protected target, class names become "class 1, 2, …" and residual axis values are withheld. The response's `privacy` object discloses what was withheld and why. Stored rows, configurations and metrics are never rewritten. New runs (and re-runs) apply the same union at creation, so the engine never computes those slices. The badge now reads "roles changed since training" and explains that the run, its audit and audit re-runs keep the training-time roles. |
| 3 | Exploratory run used `patient_id` as a predictor, ordinary K-fold | **Explained by intentional test configuration.** Same roles change: `patient_id` had no `group` role in that run's snapshot, so it was a feature and the split resolved to `kfold`. The binary nested run (roles `group: patient_id`) used `stratified_group`. Verified that declared non-feature roles never become predictors: the API (`service.features_of`), the worker (`PreprocessExecutor.feature_columns`, from the stored roles snapshot), replay (same executor) and preprocessing (`validate.py` refuses steps on protected columns, `COLUMN_PROTECTED`); inner folds use the same group/time-aware splitter on outer training rows only (`test_group_and_time_constraints_hold_in_inner_and_outer_folds`). Historical runs keep their snapshot; re-runs reuse it. | Two evidence-based refinements: `leakage.groups.undeclared` now also flags a *key-like feature* (integer-valued or text, at least 20 distinct values and 5% of rows, at least 1.2 rows per value) without an entity role, as a **medium "possible"** finding stating it is not a confirmed identifier (audit version 1.1.0; a medium suspicion does not change the evaluation status). Its pass message is scoped to the inspected signals. The training plan warns `DERIVED_FROM_PROTECTED` when a feature was derived by committed steps from a column that now has a non-feature role. |
| 4 | Gradient-boosting early stopping and group/time boundaries | **Phase 5 training: no defect** (`engine/ml/models.build_estimator` sets `early_stopping=False`, 100 iterations). **Experiment Lab: confirmed defect** (`engine/preprocess/experiment/models.build_model`, `hgb`): scikit-learn 1.9.1's default `early_stopping="auto"` switches on above 10,000 training rows and draws a random (stratified) 10% validation split from the training rows, ignoring declared groups, time order and duplicate clusters. Outer test rows were never involved. | Lab `hgb` now sets `early_stopping=False` (the library's fixed `max_iter=100`), the smallest correction. Lab results on 10,000 training rows or fewer are unchanged (`'auto'` was already off there). Tests patch scikit-learn's internal `train_test_split` to fail and prove that neither builder draws a split on more than 10,000 grouped or time-ordered training rows; a control proves the guard is effective. |
| 5 | "Unbiased" terminology | Overstated. | Labels are now "Leakage-controlled evaluation", "Leakage-controlled evaluation (replayed)", "Exploratory only" and "Blocked", with the scope: the implemented split and preprocessing safeguards, not a guarantee of external validity, absence of bias or clinical reliability. Updated in the UI, tooltips, the Lab warning, API messages (`EVALUATION_BLOCKED` title, eligibility reasons, audit texts) and the docs. Internal identifiers `unbiased_direct` / `unbiased_replay` are unchanged (no migration). Texts already stored in historical runs are not rewritten. |
| 6 | Dummy ROC-AUC 0.500 ± 0.000 versus a checklist expecting "unavailable" | **The checklist was wrong; the metric is correct.** `DummyClassifier(strategy="prior")` gives every row the same probability; all scores tie, so ROC-AUC is exactly 0.5 whenever both classes are among the test rows (the repository's smoke test already asserted 0.5). ROC-AUC is unavailable with a reason only when a fold's test rows hold one class. | A metric convention line states this; tests cover both cases. No metric was changed. |
| 7 | Uncertainty and reproducibility semantics | Bootstrap intervals resample whole split groups and duplicate clusters and are unavailable for time-ordered folds (unchanged). | The comparison now states that "±" is the fold standard deviation and the bracketed range a separate 95% bootstrap interval of pooled out-of-fold predictions, and discloses dependence (grouped resampling; no group declared means rows are treated as independent and intervals may be too narrow; time-ordered means no interval). The reproducibility statement is scoped to the same data version, training-time roles, folds, seeds, settings, library and engine versions; the training-time roles are shown. Exploratory runs keep their warning on every view and the critical `leakage.pipeline.fit_on_full_data` finding on every audit re-run (tested). |

**Remaining limitations.** Sensitivity is matched by column name across versions (a renamed copy is not
followed unless committed lineage records it). Derived-column lineage covers DataMedic-committed steps only;
processing done before upload is unknown. Columns derived from a now-protected column are warned about, not
removed automatically. The key-like entity signal is a heuristic and can match measurements with many
repeated integer values (hence medium severity and explicit wording). Stored message texts of historical
runs (eligibility reasons, audit explanations) keep the wording of their time.

**Final test evidence (2026-11-26).** Full backend regression: 1,523 passed, 1 failed, 4 skipped
(pre-existing). The failure was a pre-existing flaky Phase 4 test: it asserted that the marker "4242" was
absent from a response that happened to contain a random UUID with "4242"; the marker (and the same
pattern in a pipeline test) is now non-hex, and both tests pass. Targeted integration (training,
privacy, Lab, isolation matrix, database security): 70 passed. Frontend: 91 passed. ruff, format, strict
mypy, tsc, ESLint, production build and compose config pass. Live stack after rebuilding the images and
recreating the containers (volumes untouched): 20/20 focused checks on the existing smoke-test runs;
existing datasets, versions, hashes, runs, evaluations, roles and stored objects unchanged.
