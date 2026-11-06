# DataMedic — ML Engine Design

| | |
|---|---|
| Status | Revision 2, MVP-scoped |
| Package | `datamedic_engine` (pure Python: no DB, HTTP, Celery or boto3 imports) |

```
version ─► 1 DIAGNOSE ─findings─► 2 RECOMMEND ─steps─► candidate pipelines
                                                         │                 │
                                     (no approval needed)▼                 ▼ (approval required for high risk)
                                              3 EXPERIMENT          4 COMMIT ─► new version
                                              metrics only           (only writer of versions)
                                                         │
                         5 LEAKAGE AUDIT (diagnostic family; inspects data, splits, specs)
                         6 READINESS + PASSPORT (summarises everything for a version)
```

| Subsystem | Reads | Writes | May change stored data? |
|---|---|---|---|
| Diagnose | Version | Findings | No |
| Recommend | Findings | Proposals | No |
| Experiment | Version, specs, split | Metrics, OOF artifacts | **No**, even with high-risk steps |
| Commit | Head version, approved spec | New version | Yes, only with valid approvals |
| Leakage audit | Version, split, lineage | Findings | No |
| Readiness / Passport | Version metadata, runs | Passport | No |

---

## 1. Foundations

### 1.1 Row identity (D1)
- Ingest and reparse add `_dm_row_id` (int64, 0…n-1 in source order).
- **Every op must preserve `_dm_row_id`** on surviving rows. Ops never create rows in the MVP; row-creating aggregation ops are V2 and will define new-ID derivation.
- Row references in findings, impact previews, fold assignments and OOF predictions use `_dm_row_id`.
- `_dm_row_id` is excluded from features, statistics and checks. It is dropped at export unless `keep_row_id=true`.
- Property test: for every op, `out._dm_row_id ⊆ in._dm_row_id` and the IDs are unique.

### 1.2 Check context
```python
@dataclass(frozen=True)
class CheckContext:
    frame: pd.DataFrame          # read-only; tests assert no mutation
    schema: VersionSchema
    roles: RoleMap               # includes target task_type
    domain: str
    split: FoldAssignment | None # for split-dependent leakage checks
    lineage: LineageInfo         # kinds, fitted_on_full_data, specs (for spec-inspection checks)
    budget: Budget
    rng_seed: int
    cache: SharedComputations    # encoded matrix, OOF probabilities, etc.
```

### 1.3 Requirements and honest `not_run`
```python
@dataclass(frozen=True)
class Requirements:
    roles: frozenset[str] = frozenset()        # e.g. {"target"}, {"bio.batch", "target"}
    task_types: frozenset[TaskType] | None = None
    min_rows: int = 1
    min_rows_per_class: int | None = None
    column_types: frozenset[str] | None = None
    needs_split: bool = False
    optional_packages: frozenset[str] = frozenset()
```
Requirements are evaluated before `run()`. The first unmet one yields `not_run` with a specific reason. Data-dependent preconditions (e.g. all values missing) may also return `not_run` from inside `run()`.

### 1.4 Finding rules
- Severity comes from documented, configurable thresholds per check.
- `limitations` list sampling, heuristics, assumptions and blind spots.
- Evidence holds statistics and row IDs, not row dumps. Example values are capped and omitted for `sensitive` columns.
- `check_version` is bumped whenever the logic **or the adapter implementation** changes (e.g. the V2 PyOD swap).

### 1.5 Sampling
Each check declares a cost class, which the budget converts to a max sample size. Samples are seeded, stratified by target and group-preserving, and `{n, seed, strategy}` is recorded.

### 1.6 Ports (extensibility for deferred libraries)
| Port | MVP adapter | V2 adapter | Used by |
|---|---|---|---|
| `Profiler` | `BuiltinProfiler`: per-column stats with pandas/NumPy | YData Profiling (fg-data-profiling) for an optional full HTML report, samples disabled | Schema view, distribution checks |
| `OutlierDetector` | `SklearnIsolationForest` (`sklearn.ensemble.IsolationForest`) | PyOD ensemble (e.g. `IForest`, `ECOD`) | `universal.outliers.multivariate` |
| `DriftComparator` | `ScipyDistributionTests`: `scipy.stats.ks_2samp`, `scipy.stats.chi2_contingency` | Evidently `Report([DataDriftPreset()])` | Version comparison |
| `LabelNoiseDetector` | `CleanlabDetector` (`cleanlab.filter.find_label_issues`) | — | `universal.label_noise` |

A port returns DataMedic's own result types, so swapping adapters changes no callers. The check ID stays the same and `check_version` changes.

---

## 2. Diagnostics

### 2.1 Universal checks (MVP)

| Check ID | Category | Method | Requirements | Key limitations |
|---|---|---|---|---|
| `universal.schema.type_inference` | schema | Semantic types; numeric-as-text | — | Heuristic; locale formats |
| `universal.schema.mixed_types` | consistency | Values parsing into several types | — | |
| `universal.missingness.column_rate` | missingness | Per-column missing rate incl. disguised tokens; suspected sentinels (e.g. -999) flagged separately | — | Sentinels are suggestions |
| `universal.missingness.pattern` | missingness | Co-missingness; missing-indicator association with observed columns | ≥ 1 column with missing values | Cannot prove MAR/MNAR |
| `universal.missingness.target_association` | missingness | Missing indicator vs target | target | Association only |
| `universal.duplicates.exact_rows` | duplicates | Row hashing (excluding `_dm_row_id` and `id`-role columns) | — | Float equality |
| `universal.duplicates.near_rows` | duplicates | Hashing after normalisation (trim, case, rounding) | — | No fuzzy matching |
| `universal.duplicates.columns` | duplicates | Identical or 1:1-mapped columns | — | |
| `universal.duplicates.conflicting_labels` | label_quality | Identical features, different targets | target | Exact match only |
| `universal.variance.constant` | variance | Constant and quasi-constant | — | Threshold |
| `universal.cardinality.id_like` | cardinality | Near-unique columns | — | Heuristic |
| `universal.categories.inconsistent` | consistency | Case and whitespace variants | categorical column | No semantic matching |
| `universal.categories.rare` | cardinality | Rare levels | categorical column | |
| `universal.datetime.parse_issues` | consistency | Unparseable, mixed, impossible and future dates | date-like column | DD/MM ambiguity reported, not resolved |
| `universal.outliers.univariate` | outliers | Robust z (median/MAD), IQR fences | numeric column | Outliers ≠ errors |
| `universal.outliers.multivariate` | outliers | `OutlierDetector` port (MVP: IsolationForest) | ≥ 2 numeric columns | Unsupervised; contamination assumed; sampled |
| `universal.distribution.shape` | distribution | Skew, zero-inflation, heavy tails | numeric column | Descriptive |
| `universal.correlation.redundancy` | redundancy | Pearson/Spearman, Cramér's V, VIF (statsmodels) | ≥ 2 columns | Pairwise; VIF unstable under perfect collinearity |
| `universal.target.imbalance` | target | Class counts; minority per planned fold | target, classification | |
| `universal.target.quality` | target | Missing targets; single class; target outliers | target | |
| `universal.label_noise` | label_quality | Cleanlab on **out-of-fold** probabilities (`cross_val_predict` with the recommended split) | target, classification, ≥ 10 rows per class | Candidates only; model-dependent; never auto-drops. **The first MVP item cut if the schedule slips** (it would then be unregistered, not falsely reported). |
| `universal.privacy.pii_scan` | privacy | PRIVACY_AND_SECURITY §2.1 | — | Heuristic |
| `universal.encoding.text_issues` | consistency | Mojibake, control characters, whitespace | text column | |
| `universal.scale.units` | consistency | Scale jumps suggesting mixed units | numeric column | Suggestive |

### 2.2 Domain checks
See [DOMAIN_MODULES.md](DOMAIN_MODULES.md). Findings carry `domain=<key>` and appear under a separate tab. A domain check may *add* findings that reference universal ones, but never alters or suppresses them.

### 2.3 Staleness
Each run stores the `semantics_hash` (domain + roles + task type + domain reference data, e.g. biomedical reference ranges). When it differs from the current hash, the run is stale: it is displayed with a banner, excluded from readiness, and the passport says so.

---

## 3. Recommendations (proposal only)

```python
class RecommendationRule(Protocol):
    id: str; version: str; domain: str
    triggers: set[str]
    def propose(self, findings: list[Finding], ctx: RuleContext) -> list[Recommendation]: ...
```
Rules are deterministic and their text is templated. Conflicting options for the same column are grouped as alternatives, always including "do nothing" with its consequence.

### 3.1 MVP op catalogue

| Op | Implementation | Risk | Destructive | Kind |
|---|---|---|---|---|
| `text.trim_whitespace` | pandas string ops | low | no | stateless_column |
| `text.normalise_case` | pandas string ops | low | no | stateless_column |
| `cast.to_numeric` | `pandas.to_numeric(errors="coerce")`; coercions reported | medium | no (unparseable values become NaN, so the preview shows them) | stateless_column |
| `cast.to_datetime` | `pandas.to_datetime` with an explicit format | medium | no | stateless_column |
| `categories.map` | Explicit user-confirmed mapping | medium | no | stateless_column |
| `impute.median` / `impute.mean` / `impute.most_frequent` | `sklearn.impute.SimpleImputer` | medium | no | fitted |
| `impute.add_missing_indicator` | `sklearn.impute.MissingIndicator` / `SimpleImputer(add_indicator=True)` | low | no | fitted |
| `encode.rare_label` | Feature-engine `RareLabelEncoder` | medium | no | fitted |
| `transform.yeo_johnson` | `sklearn.preprocessing.PowerTransformer` | medium | no | fitted |
| `outliers.flag` | Adds an indicator column (robust-z threshold) | low | no | fitted |
| `outliers.winsorize` | Feature-engine `Winsorizer` | **high** | **yes** (overwrites values) | fitted |
| `columns.drop` | Explicit list | **high** | **yes** | stateless_column |
| `columns.drop_constant` | Feature-engine `DropConstantFeatures` | **high** | **yes** | fitted |
| `columns.drop_correlated` | Feature-engine `DropCorrelatedFeatures` | **high** | **yes** | fitted |
| `rows.drop_exact_duplicates` | Deterministic hashing | **high** | **yes** | row_filter_rule |
| `rows.drop_missing_target` | Deterministic predicate | **high** | **yes** | row_filter_rule |
| `rows.drop_by_ids` | User-selected `_dm_row_id` list (e.g. reviewed label issues) | **high** | **yes** | row_filter_rule |
| `rows.flag_label_issues` | Adds a column from label-noise findings | low | no | stateless_column |

Biomedical ops are listed in [DOMAIN_MODULES.md §4.4](DOMAIN_MODULES.md).

**Risk rules**: removing rows or columns, overwriting observed values, altering the target, and batch correction are **high**. Fitted value-introducing ops and coercions are **medium**. Lossless cosmetic ops and added flags are **low**. **Destructive** = removes rows or columns, or overwrites observed values. Every destructive op is high-risk.

**Kinds and leakage handling**:
- `stateless_row` / `stateless_column`: deterministic per row or column, no learning across rows, so they are safe before splitting.
- `fitted`: learns parameters, so in experiments it is fitted on the training fold only.
- `row_filter_rule`: a deterministic predicate per row (or per duplicate set). **On commit** it applies to the whole version (an approved cleaning decision). **In experiments** it applies **only to training rows**, and test rows are never removed.
- `row_filter_fitted`: learns a threshold from data. In experiments it is fitted and applied on training rows only. There are none in the MVP catalogue, but the kind is reserved for V2 ops.

**Rule for every row-removal op in experiments: it may shrink the training set, never the test set.** Validation and test folds stay intact, so difficult examples are never silently dropped from evaluation.

---

## 4. Commit (execute)

1. Load the head version and the canonical spec. Re-verify source = head, the hashes, and the approvals (high-risk step approvals + pipeline confirmation if any medium step exists).
2. Apply steps in order: stateless ops directly, row filters as masks, and fitted ops through an sklearn `Pipeline`/`ColumnTransformer` with `set_output(transform="pandas")`, preserving `_dm_row_id`.
3. Produce a per-step report: rows in/out, columns added/removed, values changed per column, warnings.
4. Check post-conditions (e.g. no NaN in imputed columns).
5. Write the Parquet. Compute `file_sha256` and `content_hash`, the diff vs the parent, and HMAC'd fitted state. Set `fitted_on_full_data` if any fitted op ran.
6. Insert the version and the transformation run in one transaction after the object write. A reconciliation job removes orphaned objects older than 24 h.

**Impact preview** (prerequisite for approvals): the same code path as commit, run as a dry run that writes only an `impact_preview` artifact. It covers per-step counts, before/after summary statistics for changed columns, and up to 1,000 affected row IDs per step. The preview is bound to `spec_hash`. Above 1M rows it may be computed on a disclosed, seeded sample, and the approval dialog says so.

---

## 5. Experiments (evaluation; isolated)

### 5.1 Split selection
| Roles | Task | Splitter |
|---|---|---|
| group (or `bio.patient_id`) | classification | `sklearn.model_selection.StratifiedGroupKFold` |
| group | regression | `GroupKFold` |
| time | any | `TimeSeriesSplit` on time-sorted rows; group exclusivity also enforced if a group role exists |
| none | classification | `StratifiedKFold(shuffle=True, random_state=seed)` |
| none | regression | `KFold(shuffle=True, random_state=seed)` |

### 5.2 Evaluation cohort and frozen fold assignment (D2)
1. **Cohort**: rows of the source version with a non-missing target. Rows excluded for a missing target are counted and reported, and the cohort is fixed when the split is created.
2. **Split groups (grouping before splitting)**: rows that may represent the same patient or entity are placed in one group, so they can never sit on both sides of a split. Groups are the union (union-find) of:
   - the declared group role (`group` / `bio.patient_id`);
   - exact-duplicate clusters (same feature + target hash, excluding `_dm_row_id` and `id`-role columns);
   - normalised near-duplicate clusters (the same normalisation as `universal.duplicates.near_rows`).

   If any multi-row groups exist, the splitter is group-aware (`StratifiedGroupKFold` / `GroupKFold`), even when the user declared no group role. The split definition records the number of duplicate-derived groups, and a warning is shown.
3. The splitter runs **once**. The `_dm_row_id → fold` map (whole groups share a fold) is stored in the `fold_assignment` artifact. Time-series splits store training and test row IDs per fold.
4. All candidates and models use this map. Candidates never re-split.

### 5.3 Candidate evaluation
For each candidate and each fold *k*:
1. **Test rows** = every cohort row with fold = *k*. They are **never removed or filtered** by any candidate. Non-removing steps (stateless transforms, fitted transforms, column drops) are applied to them through the fitted pipeline, as they would be in production.
2. **Training rows** = cohort rows with fold ≠ *k*. Every row-removal step (`row_filter_rule` and `row_filter_fitted`) is applied **to the training rows only**, and the number removed is recorded per step and per fold.
3. Build `Pipeline([fitted steps…, model-specific minimal preprocessing, model])` and **fit on the retained training rows only**.
4. Predict on **all** test rows and compute metrics. Store OOF predictions keyed by `_dm_row_id`.
5. Record the effects: **training rows removed per fold and per step** (with total and mean), columns added or removed, and values changed.

Every candidate that contains a high-risk or destructive step is labelled in the results (`max_risk`, `has_destructive`, effects). **No version is written**; a test asserts no writes under `versions/`.

### 5.3.1 Comparability
- By construction every candidate is scored on the **same test cohort**: identical row IDs in every fold.
- After scoring, the runner verifies this per fold. If a candidate could not produce a prediction for every test row (e.g. a step failed on some test rows, or a transform left a row unscorable), the candidate is marked **`NOT_DIRECTLY_COMPARABLE`** with the reason and the affected fold and row counts. It is shown in a separate section, **excluded from the leaderboard ranking and from paired differences**, and no score improvement is claimed for it.
- **`ROWSET_DIFFERS`** stays as an informational warning when candidates retained different **training** rows (always the case when one candidate removes rows). It lists each candidate's training rows removed per fold, so the user sees what each score was trained on.

### 5.4 Models (baselines; no tuning)
| Key | Classifier | Regressor |
|---|---|---|
| `dummy` | `DummyClassifier(strategy="prior")` | `DummyRegressor(strategy="median")` |
| `linear` | `LogisticRegression` (+ `SimpleImputer`, `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`) | `Ridge` (same) |
| `hgb` | `HistGradientBoostingClassifier` (native NaN support; ordinal encoding for categoricals) | `HistGradientBoostingRegressor` |
| `xgb` | `xgboost.XGBClassifier` | `xgboost.XGBRegressor` |

The model-specific minimal preprocessing is fitted in-fold and disclosed, so "no preprocessing" stays honest. Hyper-parameters are library defaults or documented values. The limitations state that there is no tuning.

### 5.5 Metrics and comparison
- Classification: ROC AUC, average precision, balanced accuracy, F1 (macro for multiclass), log loss, Brier (binary).
- Regression: MAE, RMSE, R², median absolute error.
- Reported as mean ± sd across folds.
- **Paired comparison** of candidates *a* vs *b* is computed **only between directly comparable candidates** (§5.3.1), which share the identical test cohort. It is reported as a list of per-fold differences, the mean difference, `n_folds` and `test_rows`, together with each candidate's training rows removed.
- Limitations: fold scores are not independent, so no p-values in the MVP. A multiple-comparison caution appears with more than two non-baseline candidates.
- Sanity warnings: `SUSPICIOUSLY_HIGH_SCORE` (e.g. AUC > 0.99, R² > 0.99) with a link to the leakage audit, `FEW_SAMPLES_PER_CLASS_PER_FOLD`, `DUMMY_COMPETITIVE`, and `FITTED_ON_FULL_DATA_SOURCE` when the source version was produced by fitted ops.
- V2: bootstrap CIs (group-aware), calibration, locked holdout, random forest.

---

## 6. Leakage audit

| Check ID | Detects | Method | Requirements |
|---|---|---|---|
| `leakage.target.single_feature` | Features that alone predict the target near-perfectly | Per-feature out-of-fold shallow model using the frozen or recommended split, plus a functional-dependency test | target |
| `leakage.target.name_heuristic` | Post-outcome-sounding names | Patterns + user roles | target |
| `leakage.identifier` | ID columns carrying target signal | ID-like + single-feature predictiveness | target |
| `leakage.duplicates.cross_fold` | Identical or near-identical rows in different folds (catches user-overridden splits; DataMedic's own splits group duplicates per §5.2) | Row hashes vs fold map | split |
| `leakage.groups.cross_fold` | The same group in training and test folds | Overlap per fold | group role + split |
| `leakage.groups.undeclared` | Repeated entities with no group role | ID-like columns with repeats; repeated feature signatures | — |
| `leakage.temporal.ordering` | Random CV on time-ordered data | Time role + split strategy | time role + split |
| `leakage.temporal.post_outcome` | Features dated after the outcome | Timestamp comparison | outcome-time + feature-time roles |
| `leakage.pipeline.fit_on_full_data` | Experiments sourced from a version produced by fitted ops (`fitted_on_full_data`) | Lineage inspection | lineage |
| `leakage.pipeline.target_encoding` | Target-derived encodings outside folds | Spec inspection | lineage/spec |

If a split is needed but none was given or can be derived, split-dependent checks report `not_run`. Unassessable checks are **never** reported as "no leakage found". Each leakage finding links to a remediation recommendation (assign a group role, use a time split, exclude a column, re-run the experiment from an earlier version).

---

## 7. ML-readiness checklist

Implemented in `readiness/` exactly as PRD §6.12.1 (RD-01…RD-11). Inputs are version metadata (schema with `n_missing`/`n_unique`), roles, the latest non-stale diagnostics and leakage runs, and, for RD-11, the export validation result. Each item returns `pass | warn | fail | not_applicable` with evidence and a link to the relevant finding. Overall: any `fail` → `not_ready`; otherwise any `warn` → `ready_with_warnings`; otherwise `ready`.

## 8. Data Health Passport

`passport/` assembles an immutable JSON document and renders HTML with Jinja2 (autoescape on, self-contained, no external resources). Sections:
1. **Identity**: dataset name, version number, original file SHA-256, version `file_sha256` and `content_hash`, shape, created at/by.
2. **Lineage**: the full replay chain (ingest/reparse → transforms → reverts), each step with op, parameters, risk, **Destructive** label, approval (who and when, with the impact shown), and step report.
3. **Semantics**: domain, roles, task type, semantics hash.
4. **Diagnostics**: counts by status, severity and domain; every finding with severity ≥ medium in full; `not_run` and `error` checks listed with reasons.
5. **Leakage audit**: all leakage findings and their remediation status.
6. **Experiments** (if any): split, candidates (risk labels), leaderboard, paired differences, warnings.
7. **ML-readiness**: the checklist and overall status.
8. **Reproducibility**: engine version, library versions, seeds, export validation status (if generated with an export).
9. **Limitations**: the union of all finding, experiment and process limitations, plus the standing statements (heuristic PHI detection, no tuning, etc.).
10. **Staleness notes**: any excluded stale runs.

Domain modules contribute extra sections through `passport_sections`.

## 9. Code generation and export validation

- `pipeline.py` = `load_original(path)` (re-implements the version's base `parse_options` with the same reader calls, adds `_dm_row_id`) + the ordered steps from `replay_chain(version)` ([ARCHITECTURE.md §9.4](ARCHITECTURE.md)) + `save(df, path)`.
- Revert versions are replayed via their restored version's chain, so the code never contains "revert".
- Fitted ops are refit deterministically in code. There are no pickles in the bundle.
- Templates are per op. Parameters are emitted as `repr` of JSON primitives and column names as string literals, so no user string is ever interpolated as code.
- `requirements.txt` pins the library versions recorded on the version.
- **Validation** (D4): the worker copies the original into a temporary directory and runs `python pipeline.py --input … --output out.parquet` in a subprocess with `resource.setrlimit` CPU and memory limits and a timeout. The worker has no internet egress. Validation then compares `content_hash` → `validated`; on mismatch it compares column by column (exact for non-float, `numpy.allclose(rtol=1e-9, atol=1e-12, equal_nan=True)` for float) → `validated_within_tolerance` or `unvalidated` with a per-column diff.

## 10. Performance

- Parquet I/O via pyarrow, with selective column reads.
- Memory pre-check before loading. Expensive checks sample. Experiments refuse frames above the worker memory budget with `LIMIT_EXCEEDED`.
- Columns are capped at 2,000 in the MVP, so pairwise checks are restricted to top-variance columns beyond 200 (disclosed).
- Shared computations are cached per run.
