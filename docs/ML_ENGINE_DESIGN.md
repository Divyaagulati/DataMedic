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
| `OutlierDetector` | `IsolationForestDetector` (`sklearn.ensemble.IsolationForest`), plus **in-house `ECODDetector` and `COPODDetector`** (owner decision Phase 3: no PyOD in the MVP; see `engine/adapters/outliers.py` for the exact formulas and citations) | PyOD ensemble (e.g. `IForest`, `ECOD`, `COPOD`) | `universal.outliers.multivariate` |
| `DriftComparator` | `ScipyDistributionTests`: `scipy.stats.ks_2samp`, `scipy.stats.chi2_contingency` | Evidently `Report([DataDriftPreset()])` | Version comparison |
| `LabelNoiseDetector` | `CleanlabDetector` (`cleanlab.filter.find_label_issues`) | — | `universal.label_noise` |

A port returns DataMedic's own result types, so swapping adapters changes no callers. The check ID stays the same and `check_version` changes.

### 1.7 Descriptive profile conventions (`BuiltinProfiler` 1.0.0, Phase 2)
These conventions are verified against hand-calculated values in `tests/unit/test_profile_golden.py`.

| Statistic | Convention | When undefined |
|---|---|---|
| Missing % | Of all rows | — |
| Unique count, frequency % | Over present (non-missing) values | — |
| Mean, min, max | Finite values only; infinities counted separately (`n_infinite`) | No finite values → all null |
| Standard deviation | Sample (ddof = 1) | < 2 values → null |
| Quartiles, median | Linear interpolation at (n−1)p (Hyndman–Fan type 7, NumPy default) | — |
| Skewness | Adjusted Fisher–Pearson G1 | < 3 values or all equal → null + `skewness_note` |
| Kurtosis | Excess, bias-adjusted G2 | < 4 values or all equal → null |
| Histogram | 20 equal-width bins over [min, max], last bin closed; datetime bins in the column's own unit | One distinct value → none |
| Duplicate rows | Exact match on every column except `_dm_row_id`; missing equals missing | — |
| Constant / empty | Exactly one distinct present value / no present value | — |
| Pearson / Spearman | Pairwise-complete rows. Spearman re-ranks those rows with average ranks for ties. Numeric and boolean (0/1) columns only; constant, empty and identifier-like columns excluded; ≤ 50 columns; seeded sample of 200,000 rows above that | < 3 paired rows or zero variance → null (never 0) |
| Frequency tables | ≤ 50 distinct values → complete; categorical/text with ≤ 50 % distinct ratio → top 10; identifier-like and other high-cardinality columns → none | — |
| Identifier-like columns | Counts and uniqueness only: no mean, spread or histogram | — |

Class distribution and imbalance ratio (largest ÷ smallest class) are computed in the UI from the complete frequency table. Missing target values are excluded and stated.

---

## 2. Diagnostics

### 2.0 Implemented in Phase 3 (2026-11-09)
The live catalogue is `GET /api/v1/checks`: every check's id, version, requirements, method description and limitations, generated from the code. The structure:
- **Specs** (`engine/checks/spec.py`, standard library only) declare checks, roles and modules. The API imports them without pandas.
- **Implementations** are referenced as `"module:function"` and imported only by the worker.
- `engine/checks/framework.py` evaluates requirements (`not_assessed` + reason) and isolates errors. An exception becomes `error`/`CHECK_ERROR` with only the exception type; a `MemoryError` fails the job as `LIMIT_EXCEEDED`.
- Findings are validated: detected ⇒ evidence and a suggested action; every finding has limitations.

Universal checks implemented (17):
- Missingness: `missingness.column_rate`, `.pattern`, `.target_association`.
- Duplicates: `duplicates.exact_rows`, `.identifier`.
- Outliers: `outliers.univariate` (IQR 1.5/3, z > 3, modified z > 3.5) and `.multivariate` (consensus of Isolation Forest, ECOD and COPOD).
- Distribution and variance: `distribution.shape`, `variance.constant`, `cardinality.high`.
- Redundancy: `correlation.redundancy` (Pearson/Spearman, VIF) and `.categorical` (Cramér's V).
- Target: `target.imbalance`, `.quality`.
- Validity: `values.suspicious`.
- Leakage: `leakage.target_proxy`, `.identifier`.

Calibration notes:
- **Multivariate threshold:** a row is flagged only when all three detectors rank it in their top 1% **and** its Isolation Forest score is ≥ 0.65. That gave zero flags on clean Gaussian data and found all planted multi-feature outliers.
- **ECOD/COPOD limitation:** both are per-feature, so rows that only break a relationship between features are listed as "Isolation Forest only" evidence, not flagged.
- **Clean controls:** across five seeds, only low-severity outlier findings appear (PRD §9).

Not yet implemented from the table below:
- `duplicates.near_rows`, `duplicates.columns`, `duplicates.conflicting_labels`;
- `categories.inconsistent`, `categories.rare`;
- `datetime.parse_issues`;
- `label_noise` (Cleanlab);
- `privacy.pii_scan`;
- `encoding.text_issues`;
- `scale.units`.

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

Phase 4 owner decisions (2026-11-17, [PHASE4_PLAN.md](PHASE4_PLAN.md) §2.3) revise this table: ops are built on scikit-learn plus small in-house transformers (**no Feature-engine**, D2); each op also has a **stage** (`data`, `model` or `either`, D1); learned state is persisted as JSON only (D4). Rows marked "Feature-engine" below will be re-implemented in-house in Parts B–C. The implemented ops are documented in §3.2.

| Op | Implementation | Risk | Destructive | Kind |
|---|---|---|---|---|
| `text.trim_whitespace` | pandas string ops | low | no | stateless_column |
| `text.normalise_case` | pandas string ops | low | no | stateless_column |
| `cast.to_numeric` | `pandas.to_numeric(errors="coerce")`; coercions reported | medium | no (unparseable values become NaN, so the preview shows them) | stateless_column |
| `cast.to_datetime` | `pandas.to_datetime` with an explicit format | medium | no | stateless_column |
| `categories.map` | Explicit user-confirmed mapping | medium | no | stateless_column |
| `impute.mean` / `impute.median` / `impute.mode` | `sklearn.impute.SimpleImputer` (**implemented**, §3.2) | medium | no | fitted |
| `impute.constant` | Fixed user value (**implemented**) | medium | no | stateless_column |
| `impute.knn` / `impute.iterative` | `sklearn.impute.KNNImputer` / `IterativeImputer` (**implemented**) | medium | no | fitted |
| `missing.indicator` (was `impute.add_missing_indicator`) | `Series.isna()` on an explicit column list (**implemented**; stateless, so train and test always get the same columns) | low | no | stateless_column |
| `encode.rare_label` | Superseded by `encode.onehot` rare-category grouping (`max_categories`) | — | — | — |
| `encode.onehot` / `encode.ordinal` / `encode.frequency` | In-house, **model stage** (**implemented**, §3.3) | medium | no | fitted / stateless_column / fitted |
| `scale.standard` / `scale.minmax` / `scale.robust` / `scale.none` | scikit-learn scalers, **model stage** (**implemented**) | medium (none: low) | no | fitted |
| `transform.log` / `transform.log1p` / `transform.yeo_johnson` / `transform.quantile` | numpy / `PowerTransformer` λ / `QuantileTransformer` (**implemented**, §3.4) | medium (quantile: **high**) | no (quantile: **yes**, lossy) | stateless_column / fitted |
| `select.variance` / `select.mutual_information` / `select.univariate` / `select.drop_columns` | In-house + scikit-learn / SciPy, **model stage**, `report` by default (**implemented**, §3.4) | **high** | **yes** (with `drop`) | fitted / stateless_column |
| `select.correlation` | Proposal only (**implemented**) | low | no | fitted |
| `imbalance.class_weight` / `random_oversample` / `random_undersample` / `smote` | imbalanced-learn 0.14.2, **model stage, training rows only** (**implemented**, §3.4) | medium / medium / **high** / **high** | undersample: **yes** | training_only |
| `outliers.flag` | Adds `<column>__outlier`; rule iqr / zscore / mad / percentile (**implemented**) | low | no | fitted |
| `outliers.cap` (was `outliers.winsorize`) | Clips to learned bounds; `percentile` = winsorization (**implemented**, in-house) | **high** | **yes** (overwrites values) | fitted |
| `outliers.signed_log` | sign(x)·log(1+\|x\|) (**implemented**) | medium | no | stateless_column |
| `outliers.isolation_forest_flag` | `sklearn.ensemble.IsolationForest`, flag only (**implemented**) | low | no | fitted |
| `columns.drop` | Explicit list | **high** | **yes** | stateless_column |
| `columns.drop_constant` | Superseded by `select.variance` (model stage) | — | — | — |
| `columns.drop_correlated` | Superseded by `select.correlation` (proposal only; the user removes with `select.drop_columns`) | — | — | — |
| `rows.drop_exact_duplicates` | Deterministic hashing | **high** | **yes** | row_filter_rule |
| `rows.drop_missing_target` | Deterministic predicate | **high** | **yes** | row_filter_rule |
| `rows.drop_by_ids` | User-selected `_dm_row_id` list (e.g. reviewed label issues) | **high** | **yes** | row_filter_rule |
| `rows.flag_label_issues` | Adds a column from label-noise findings | low | no | stateless_column |

Biomedical ops are listed in [DOMAIN_MODULES.md §4.4](DOMAIN_MODULES.md).

**Risk rules**: removing rows or columns, overwriting observed values, altering the target, and batch correction are **high**. Fitted value-introducing ops and coercions are **medium**. Lossless cosmetic ops and added flags are **low**. **Destructive** = removes rows or columns, or overwrites observed values in a way that cannot be undone (e.g. capping). Invertible re-expressions (signed log, Yeo-Johnson, scaling) and model-stage representation changes (encodings, which are never committed) are not destructive. Every destructive op is high-risk. (Clarified in Phase 4 Part B.)

**Kinds and leakage handling**:
- `stateless_row` / `stateless_column`: deterministic per row or column, no learning across rows, so they are safe before splitting.
- `fitted`: learns parameters, so in experiments it is fitted on the training fold only.
- `row_filter_rule`: a deterministic predicate per row (or per duplicate set). **On commit** it applies to the whole version (an approved cleaning decision). **In experiments** it applies **only to training rows**, and test rows are never removed.
- `row_filter_fitted`: learns a threshold from data. In experiments it is fitted and applied on training rows only. There are none in the MVP catalogue, but the kind is reserved for V2 ops.

**Rule for every row-removal op in experiments: it may shrink the training set, never the test set.** Validation and test folds stay intact, so difficult examples are never silently dropped from evaluation.

**Stages (Phase 4, D1)**: `data` ops may be committed into a version; `model` ops (scaling, encoding, feature selection, resampling, class weights) run only inside a model's training-fold pipeline and are **never** committed. SMOTE and every other resampling op is `model`-only. `either` ops (imputation, some distribution transforms) can be committed: such versions are flagged `fitted_on_full_data` and carry a warning that they must not be used directly for unbiased model evaluation.

### 3.2 Implemented ops and engine (Phase 4 Part A, 2026-11-17)

Code: `engine/preprocess/`. The live catalogue is `GET /api/v1/ops`.

| Module | Role | Imports |
|---|---|---|
| `spec.py`, `catalog.py`, `validate.py` | Op specs, the catalogue (risk, destructive, kind, stage, params, texts), validation and step/spec hashing | Standard library only (the API uses them) |
| `runtime.py` | **The single implementation of every algorithm.** Embedded verbatim into generated code, so platform and code cannot drift | NumPy, pandas, scikit-learn only |
| `base.py`, `ops/impute.py` | `DMTransformer` (scikit-learn `TransformerMixin` + `BaseEstimator`) wrappers: limits, warnings, learned parameters | |
| `executor.py` | Builds a scikit-learn `Pipeline` from validated steps; `fit(rows)` learns, `transform(any rows)` never learns; per-step reports and post-conditions | |
| `report.py`, `table_schema.py` | Step reports (counts, dtypes, learned values with sensitive columns redacted); schema of a transformed table | |
| `codegen.py` | Standalone Python for steps: the runtime source + one function per step + `preprocess(fit_df, df)` | |

**Why each op selects its own columns instead of using `ColumnTransformer`** (verified in `tests/unit/test_sklearn_contract.py`): with `remainder="passthrough"` it moves the transformed columns first (reordering `_dm_row_id` and every other column), and `SimpleImputer` treats `None` in object columns as a value, so most-frequent imputation silently leaves gaps. The ops therefore convert every missing marker to NaN, select columns themselves and restore pandas dtypes. The statistics still come from scikit-learn's imputers, and a candidate is a real scikit-learn `Pipeline`.

**Rules for every op**:
- The input is never mutated.
- `_dm_row_id`, row count and row order are preserved.
- Observed values are never changed; only missing cells are written.
- Target, `id`, `group`, `time`, `weight`, `exclude` and other non-feature role columns (unknown roles included) are never transformed or used as predictors. `_dm_row_id` is reserved.
- Errors use stable codes and never contain data values; `EngineError.details` carries the step and column.

| Op | Columns | Learns (from fitting rows only) | Output dtype | Params | Fails with |
|---|---|---|---|---|---|
| `impute.mean` | numeric | mean (`SimpleImputer`) | float64 | — | `FIT_COLUMN_EMPTY`, `NON_FINITE_VALUES` |
| `impute.median` | numeric | median | float64 | — | same |
| `impute.mode` | numeric, categorical, boolean | most frequent value; ties → smallest (scikit-learn rule), reported as `MODE_TIE` | storage dtype kept (`Int64`, `Float64`, `boolean`, `string`, `category`, object); a number in an object column stays a number | — | same |
| `impute.constant` | numeric, categorical, boolean | nothing | storage dtype kept (a new category is added to `category` columns); a decimal value on an integer column → float64 (`DTYPE_CHANGE` warning) | `fill_value` (required; type must match the column; text ≤ 200 chars, not blank) | — |
| `impute.knn` | numeric | predictor means and standard deviations (for standardised distances) and the donor rows | float64 | `n_neighbors` 1–50 (default 5), `weights` uniform/distance, `predictors` | as above; `OP_LIMIT_EXCEEDED` above 50,000 fitting rows, 100 columns, or 2.5 billion neighbour comparisons (rows needing imputation × fitting rows, checked before any matrix is built) |
| `impute.iterative` | numeric | `IterativeImputer` (Bayesian ridge, `random_state` = seed, no posterior sampling); clip range | float64 | `max_iter` 1–50 (default 10), `clip_to_fit_range` (default true), `predictors` | as above; limits 100,000 rows and 50 columns |
| `missing.indicator` | any feature type | nothing | adds `<column>__missing` (bool) | — | `OUTPUT_COLUMN_CONFLICT` |

Details:
- **Predictors** (KNN, iterative) default to every numeric feature column at that point of the pipeline. They are resolved at validation time and stored in the step, so the hash and the generated code are explicit. A predictor with no observed value in the fitting rows is dropped (`PREDICTOR_EMPTY` warning).
- **KNN scaling**: the imputed column and its predictors are standardised with statistics from the fitting rows before distances are computed, and estimates are transformed back. A constant column uses scale 1. With fewer fitting rows than `n_neighbors`, scikit-learn uses the available donors (`FEW_FIT_ROWS` warning).
- **Iterative**: convergence is reported (`NOT_CONVERGED` warning, `converged`, `n_iter`). When clipping, a column that is constant in the fitting rows keeps its single value. scikit-learn requires max > min, so that bound is widened by one floating-point step.
- **Missing values**: `None`, `np.nan`, `pd.NA` and `NaT` are all missing, in every supported dtype (float64, `Float64`, `Int64`, object, `string`, `category`, `boolean`). A column with no observed value in the fitting rows fails with `FIT_COLUMN_EMPTY`. A column complete in the fitting rows but with gaps in other rows is imputed with the learned value (`tests/unit/test_preprocess_nullable.py`).
- **Memory and time**: row and column caps are checked before any matrix is built. KNN distance chunks are capped at 256 MB through `sklearn.config_context(working_memory=256)`, which is also used by generated code.
- **Unexpected exceptions** inside a step become `OP_FAILED` with only the exception type (library messages can quote cell values); `MemoryError` passes through to the worker's `LIMIT_EXCEEDED` handling. **Library warnings** raised inside a step are reported as `LIBRARY_WARNING` with their category only, so their text never reaches worker logs. Deprecation and future warnings, which concern code rather than data, are re-emitted.
- **Sensitive columns**: learned values are redacted in step reports. Generated code contains no learned values (it refits). A user-entered `fill_value` is configuration, not a learned value: it is stored in the step and appears in generated code, both owner-only. Callers must persist and return `TransformResult` reports, never a transformer's raw `learned_params()`.
- **Infinite values** are refused (`NON_FINITE_VALUES`), never guessed around. Datetime, text and identifier columns cannot be imputed; they can receive an indicator.
- **Validation** fills defaults and tracks the schema through the steps: added columns, dtype changes, and columns with nothing left to impute (`NO_MISSING_VALUES` warning). It refuses unknown ops or params, ill-typed params, unknown/duplicate/reserved/protected/wrong-type columns, and more than 50 steps. Only a valid result can be executed.
- **Hashes**: `step_hash` = SHA-256 of canonical JSON (op, op version, ordered columns, complete params); `spec_hash` = SHA-256 of the ordered step hashes.
- **Step report**: rows in/out/removed, columns added/removed, and per column: missing before/after, cells filled, observed values changed (always 0 for imputation), dtype before/after. Also learned values (`{"redacted": true}` for `sensitive` columns), warnings, number of fitting rows and duration.
- **Leakage proof** (`tests/unit/test_preprocess_leakage.py`):
  - a spy shows every step's `fit` sees only training-row IDs;
  - perturbing test rows changes neither learned values nor transformed training rows, and a negative control proves this check catches a deliberate leak;
  - test rows receive training statistics;
  - flipping the target never changes imputations.

---

### 3.3 Outliers, encoding and scaling (Phase 4 Part B, 2026-11-18)

Code: `engine/preprocess/ops/{outliers,encode,scale}.py`, algorithms in `runtime.py`. Op-specific validation now lives in `rules.py` (a check and a schema effect per op); shared validation types are in `state.py`.

**Detection versus treatment.** The detection rule is a parameter (`method`); `outliers.flag` (adds a column, never changes values) and `outliers.cap` (clips) are treatments. No op removes rows.

| Rule | Bounds learned from the fitting rows | Assumes | Parameters |
|---|---|---|---|
| `iqr` | [Q1 − k·IQR, Q3 + k·IQR] (quartiles by linear interpolation) | Roughly symmetric | `iqr_multiplier` 0.5–10 (1.5) |
| `zscore` | mean ∓ z·SD (sample SD, ddof = 1) | Roughly normal; the mean and SD are themselves pulled by outliers | `z_threshold` 1–10 (3) |
| `mad` | median ∓ t·1.4826·MAD | Symmetric core; robust to up to half the values being extreme | `mad_threshold` 1–10 (3.5, Iglewicz–Hoaglin) |
| `percentile` | [p_low, p_high] quantiles | Nothing about shape; always marks about the chosen share | `lower_percentile` 0–49 (1), `upper_percentile` 51–100 (99) |

Parameters that do not apply to the chosen rule are removed from the step; supplying one explicitly is an error (`PARAM_NOT_APPLICABLE`). Safety rules:
- **Small samples**: at least 8 observed fitting values per column, else `FIT_TOO_FEW_VALUES`. The percentile rule warns below 100 values (`SMALL_SAMPLE_PERCENTILE`).
- **No spread** (constant column, IQR = 0, MAD = 0 for zero-inflated columns, SD = 0, equal percentiles): no bounds exist, nothing is flagged or capped, warning `ZERO_SPREAD`.
- **Skew**: `iqr` and `zscore` warn `SKEWED_COLUMN` when |skewness| ≥ 1 in the fitting rows.
- **Missing values** are never outliers and are never filled. **Infinite values** are refused (`NON_FINITE_VALUES`).
- **Bounds are inclusive**: a value exactly on a bound is not an outlier.
- **Integer columns stay integer** under `outliers.cap`: bounds are rounded inward, or outward when no whole number lies between them. Nullable and object storage dtypes are kept.
- **Report**: learned bounds and the rule's statistics (Q1/Q3, mean/SD, median/scaled MAD, skewness, n), and effects per column (`below`, `above`, `affected_rows`, `affected_pct`) plus `rows_affected`. For `outliers.cap`, `observed_changed` counts overwritten values (the information loss).

`outliers.signed_log`: sign(x)·log(1+|x|). Stateless, defined for all real values (including −0.0 and subnormals), keeps order and sign, and never overflows (the largest float maps to ≈ 709.78). Mathematically invertible (sign(y)·(e^|y| − 1)); numerically the round trip is exact for |x| ≤ 1 in the audit set and within a relative error below 1e-13 overall (measured maximum 2.4e-14). Not a normalising transform (that is Part C).

`outliers.isolation_forest_flag`: `IsolationForest(n_estimators, contamination, random_state=seed)` fitted on fitting rows with a value in every selected column; it flags the `contamination` share of fitting rows (default 1%, range 0.001–0.5) and reuses that threshold for other rows. **scikit-learn's `"auto"` threshold is not offered**: in the Part B audit it flagged 13–60% of rows of clean normal, uniform and log-normal data at every size from 40 to 20,000 rows. The share is an assumption, not a measurement: the op flags about that share even when there are no real outliers (stated in the catalogue). Columns constant in the fitting rows are left out of the model (`CONSTANT_COLUMN_DROPPED`); if none varies, no model is fitted and nothing is flagged (`ZERO_SPREAD`). scikit-learn 1.9.1 accepts NaN, but the op does not rely on it: rows with a gap are not scored and never flagged (`rows_not_scored`, `ROWS_NOT_USED`). Needs ≥ 20 complete fitting rows; ≤ 200,000 fitting rows and ≤ 100 columns (checked before any matrix is built). Scoring, including the scoring inside `fit` that sets the threshold, runs in 256 MB working-memory chunks. The output column name is a validated parameter (`output_column`). Scale-free (axis-aligned splits), so no scaling is needed first. Deterministic for a given seed, data and scikit-learn version; identical results across operating systems and CPU architectures are expected but not guaranteed (floating-point differences).

**Encoding** (model stage only; never committed):

| Op | Learns | Unseen categories | Missing values | Output |
|---|---|---|---|---|
| `encode.onehot` | The `max_categories` (2–100, default 20) most frequent categories of the fitting rows (ties by text), and the output names | Never seen in the fitting rows: `error` → `UNKNOWN_CATEGORY` (also when an `__other__` column exists); `ignore` → `__other__` if it exists, else all zeros. Rare categories that were seen → `__other__` | All zeros (indistinguishable from an ignored unseen category; add an indicator first) | Replaces the column in place by `<column>=<category>` boolean columns in text order (+ `__other__`); every observed, seen value has exactly one true column |
| `encode.ordinal` | Nothing: the order is **required** (`order`, ≥ 2 distinct values); no default order is ever invented | Fitting rows: `ORDER_INCOMPLETE`. Other rows: `error` → `UNKNOWN_CATEGORY` (default) or `missing` | Stay missing | `Int64` codes 0…k−1 in place |
| `encode.frequency` | Share of each category among observed fitting values | 0.0 | Stay missing | float64 in place; ties reported (`FREQUENCY_TIES`) |

- **Collisions**: one-hot reserves the namespace `<column>=`. Validation refuses existing columns in it and later steps adding columns in it; the runtime refuses duplicate output names.
- **Output names are fixed at fit**, so every transform (any rows, none, any order) has the same schema.
- **Sensitive columns** are always one-hot named by position (`<column>=1`, …), so category values never appear in column names, reports or generated code (`SENSITIVE_POSITIONAL_NAMES`). Columns whose labels cannot be column names (control characters, blank, `__other__`, duplicates as text such as `"1"` and `1`, or names over 200 characters) are also named by position (`POSITIONAL_NAMES_FALLBACK`).
- Outputs are capped at 1,000 one-hot columns per step (columns × (max_categories + 1)). Target encoding is deliberately not offered (leakage-prone).
- Effects: rows with unseen categories per column (`UNSEEN_CATEGORIES` warning), output column counts.

**Numeric overflow guard (Parts A and B).** Arithmetic on values near the float limit (≈ 1.8e308) can overflow. Before the Part B audit this silently produced NaN (standard and robust scaling), zeros (min-max scaling), inf (median imputation of two huge values) or ±inf bounds (IQR, z-score, MAD). Now every learned statistic, bound and scaler parameter, and every value an op writes, must be finite; otherwise the step fails with `NUMERIC_OVERFLOW` (column in `details`). Standard scaling, for example, refuses columns with |x| beyond about 1e154, whose variance overflows. Percentile bounds are only refused when the bounds themselves are not finite.

**Scaling** (model stage only): `StandardScaler` (mean, population SD), `MinMaxScaler` (optional `clip`), `RobustScaler` (median, IQR 25–75), and `scale.none` (an explicit no-op). Learned from the fitting rows, NaN ignored when fitting and kept when transforming. A column constant in the fitting rows gets scale 1, so it is only shifted (`CONSTANT_COLUMN` warning). Values outside the fitting range are counted per column (`outside_fit_range_rows`), also when `clip` hides them. Output float64.

**Leakage proof** (`tests/unit/test_preprocess_leakage_b.py`): spy (fit sees training-row IDs only), perturbation of test rows (extreme values, new categories) changes no learned parameter and no transformed training row, negative control for every fitted op, Isolation Forest scores a row independently of the other transformed rows, caps and scales are applied unchanged to test rows, flipping the target changes nothing.

### 3.4 Distribution transforms, feature selection and class imbalance (Phase 4 Part C, 2026-11-18)

Code: `engine/preprocess/ops/{transform,select,imbalance}.py`, `engine/preprocess/split.py`; algorithms in `runtime.py` (shared with generated code). Dependency added: **imbalanced-learn 0.14.2** (MIT; requires `sklearn-compat` 0.1.6, which allows scikit-learn `<1.10`), pinned and covered by `tests/unit/test_imblearn_contract.py`. It is imported only inside the resampling functions, so generated code needs it only when a pipeline resamples.

**Stages.** Distribution transforms are `either` (they may be committed, D1). Feature selection and class-imbalance handling are **model stage only**: they run inside a model's training pipeline, learned from each training fold, and are never committed into a dataset version. The dataset always keeps every column and every row.

**Target access.** Supervised selectors and imbalance ops need the target. Validation resolves it from the declared `target` role and its task type (`ValidationContext.target`, `task_type`), never from the request: a `target` parameter in a request is `UNKNOWN_PARAM`. Resolved parameters (`target`, `task`, `role_kept`, `passthrough`, `discrete`) are part of the step hash and of generated code. Missing target or task type: `TARGET_REQUIRED` / `TASK_TYPE_REQUIRED`. The target, identifier, group, time, weight and excluded columns can never be selected by any op (`COLUMN_PROTECTED`), and the executor now checks after every step that no protected column changed (`POSTCONDITION_FAILED`). Supervised selectors also refuse the target as an input at runtime (`OP_INVALID`).

#### Distribution transforms

| Op | Learns | Domain | Inverse | Stage / risk |
|---|---|---|---|---|
| `transform.log` | Nothing (base `e`, `2`, `10`) | x > 0; zero or negative in fitting **or** applied rows → `VALUE_OUT_OF_DOMAIN` (never a silent NaN) | exp / power of base, exact up to rounding (relative error < 1e-13 measured) | either / medium |
| `transform.log1p` | Nothing | x > −1 | expm1, exact up to rounding | either / medium |
| `transform.yeo_johnson` | λ per column by maximum likelihood (scikit-learn `PowerTransformer`, not standardised) | all reals | exists; numerically approximate (< 1e-9 absolute in tests), unstable near the output bound for some λ | either / medium |
| `transform.quantile` | Quantiles per column (`QuantileTransformer`, `n_quantiles` 10–2000 reduced to the observed count and to the 100,000-row seeded subsample) | all reals | piecewise-linear approximation; clipped and tied values cannot be recovered | either / **high, destructive** |

- The Yeo-Johnson transform is applied by the runtime's own formula (`expm1`/`log1p`, matching scikit-learn within 1e-12 in the contract test). λ = 1 is an exact identity: a regression test covers the earlier `4.0 → 3.9999999999999996` rounding. Columns constant in the fitting rows keep λ = 1 (`CONSTANT_COLUMN`). Overflow at transform → `NUMERIC_OVERFLOW`.
- Quantile: scikit-learn maps a column constant in the fitting rows to 0 (or −5.2), which is misleading. The op maps that value to the middle (0.5 or 0), smaller values to the lower end and larger values to the upper end (`CONSTANT_COLUMN`). Values outside the fitting range are counted (`outside_fit_range_rows`, `OUTSIDE_FIT_RANGE`). The learned quantiles are interpolated fitting values and are **never reported**.
- **Before/after statistics** (`effects.distribution`, per column, over the rows transformed) are skewness, excess kurtosis, the tail ratio (q99 − q50)/(q50 − q01) and the number of distinct values. They show tail compression, distortion and merged values (`distinct_values_merged`, `VALUES_MERGED`: information loss). A transform that does not reduce |skewness| is reported as `SKEW_NOT_REDUCED`; no transform is described as an improvement. These statistics describe values, so they are redacted for sensitive columns.
- Minimum 8 observed fitting values for fitted transforms (`FIT_TOO_FEW_VALUES`). Caps: Yeo-Johnson ≤ 500,000 fitting rows and ≤ 100 columns; quantile ≤ 1,000,000 fitting rows (`OP_LIMIT_EXCEEDED`). Missing values stay missing; infinities are refused.

#### Feature selection (model stage)

Every selector has `action` = `report` (default: ranks and flags, removes nothing) or `drop` (removes failing columns from the model's input in that training fold), plus `keep` (columns never removed). **Columns with a domain role** (e.g. `bio.heart_rate`, `bio.age`, `bio.sex`) are resolved into `role_kept` and are never removed automatically (`DOMAIN_ROLE_KEPT`). Columns that cannot be assessed are kept. Each decision (`kept`, `would_remove`, `removed`, `kept_by_user`, `kept_domain_role`, `not_assessed`, `removed_by_user`) is reported in `learned.decisions` (not redacted, because a decision is not a value). Scores are under `learned.columns` (redacted for sensitive columns). In `drop` mode, validation removes the possibly-removed columns from the schema, so later steps cannot depend on them.

| Op | Criterion | Notes |
|---|---|---|
| `select.variance` | Population variance of observed values (true/false as 0/1) ≤ `threshold` (default 0: constant only) | Exactly constant columns get variance 0 (no float noise). A threshold above 0 with several columns warns `VARIANCE_SCALE_DEPENDENT`. No target needed |
| `select.correlation` | Pairs with \|r\| ≥ `threshold` (0.5–1, default 0.9), Pearson or Spearman | **Proposal only, never removes.** One proposed removal per pair (higher mean \|r\| to the others; kept and domain-role columns never proposed). Users act with `select.drop_columns`. Pairwise-complete correlations come from a chunked masked matrix product (matches pandas within 1e-10; pandas' own pairwise path was too slow at 200,000 × 200). Spearman ranks each column over its observed values. Pairs with < 10 common rows or a constant column are not assessed. ≤ 200 columns, ≤ 200,000 fitting rows |
| `select.mutual_information` | Top `k` by MI with the target (`mutual_info_classif` / `_regression` by task type, seeded) | Categorical and true/false columns are treated as discrete. Each column uses its own rows where it and the target are observed (`n_rows` reported). ≤ 100,000 fitting rows, ≤ 200 columns |
| `select.univariate` | Top `k` by p-value (ties by statistic) | `test` is **required** (no default). Classification: `anova_f`, `kruskal` (numeric columns), `chi2` (categorical/true-false). Regression: `f_regression`, `spearman` (numeric). A test/task mismatch is `TEST_TASK_MISMATCH`; a test/column type mismatch is `COLUMN_TYPE_UNSUPPORTED`. Benjamini–Hochberg adjusted p-values are reported |
| `select.drop_columns` | The columns the user lists | Stateless; the user's explicit decision |

- **Assumptions are checked, not assumed** (`ASSUMPTION_DOUBTFUL`, which names the robust alternative):
  - ANOVA: a class group with \|skew\| > 2 (n ≥ 8), or a variance ratio > 4;
  - chi-square: more than 20% of expected counts below 5, or any below 1;
  - f_regression: \|skew\| > 2 in the column or the target.

  Chi-square does not assess columns with more than 50 categories (`too_many_levels`).
- **Not assessed (kept):** fewer than 10 rows where both the column and the target are observed, a constant column, a single class, or an undefined statistic. A single class in the fitting rows fails the step (`TARGET_SINGLE_CLASS`). Rows with a missing target are not used for scores.
- **Interpretation:** a ranking is association in these rows, not causal, biological or clinical importance. A selected biomarker is not validated by being selected, and removing a column can remove scientifically meaningful information (catalogue limitations).
- **Feature lineage:** `TransformResult.lineage` lists every input and added column, its origin (`derived_from` for added columns), every step that transformed, assessed, added or removed it (with the decision) and whether it is still present.

#### Class imbalance (model stage, training rows only)

New op kind **`training_only`**: these ops change only the training table returned by `PreprocessExecutor.fit_resample(training_rows)`.
- `transform` (validation and test rows, previews) and `fit_transform` (the commit path) return the rows unchanged, so resampled data can never be committed. The executor checks this (`POSTCONDITION_FAILED`).
- `resample` refuses any rows other than the exact fitting rows (`RESAMPLE_NOT_FIT_ROWS`).
- A pipeline has at most one such step (`TOO_MANY_TRAINING_ONLY`), and it must be the last one (`TRAINING_ONLY_NOT_LAST`). So nothing is ever fitted on resampled rows, which gives the same semantics as an imbalanced-learn `Pipeline` without needing one.
- These ops select no columns (`uses_columns = false`; `COLUMNS_NOT_APPLICABLE`). They need a binary or multiclass target (`TASK_UNSUPPORTED` for regression), and every training row needs a label (`TARGET_MISSING`).

| Op | Training table | Risk |
|---|---|---|
| `imbalance.class_weight` | Adds `_dm_sample_weight` = n / (k · class count) per row ("balanced"); no row changes. Refused when a `weight` role exists (`WEIGHT_ROLE_PRESENT`) | medium |
| `imbalance.random_oversample` | imbalanced-learn `RandomOverSampler` (seeded). Original rows first, in order; duplicates appended as exact copies, including identifier and group values, so group-aware inner validation keeps them together | medium |
| `imbalance.random_undersample` | `RandomUnderSampler` (seeded); kept rows in original order; discarded count reported (`ROWS_DISCARDED`) | high, destructive |
| `imbalance.smote` | `SMOTE(k_neighbors, seeded)` on **every feature column** (all except the row id, target and protected columns). Features must be numeric (true/false and one-hot outputs refused, `FEATURE_NOT_NUMERIC`) and complete (`FEATURE_MISSING_VALUES`). Each oversampled class needs more than `k_neighbors` rows, checked at fit (`SMOTE_TOO_FEW_SAMPLES`; k is never reduced silently). Synthetic rows have missing identifier, group and time values; integer features become float | high |

- `sampling_strategy`: `not_majority` / `minority` (oversampling and SMOTE), `not_minority` / `majority` (undersampling). Binary and multiclass are both supported.
- Added rows have negative `_dm_row_id` values (−1, −2, …). The training table gets a fresh range index. Original rows keep their id and label (checked: `POSTCONDITION_FAILED`).
- **Report:**
  - `learned.columns[<target>]`: classes, counts before and after, and weights. Class labels are values, so all of this is redacted when the target is sensitive.
  - `effects`: `rows_before`, `rows_after`, `rows_added`, `rows_removed`.
- **Warnings:**
  - `SMALL_CLASS` (fewer than 10 rows) and `SYNTHETIC_ROWS`;
  - `GROUPED_OR_TEMPORAL_DATA` (SMOTE when id, group, patient or time roles are declared);
  - `TEMPORAL_DATA` (oversampling with a time role).
- **Caps:**
  - ≤ 500,000 training rows (SMOTE ≤ 200,000);
  - ≤ 1,000,000 rows after resampling, checked before resampling;
  - SMOTE ≤ 200 features; neighbour search in 256 MB chunks.
- **Limits stated in the catalogue:** balancing changes what the model is trained on; it does **not** improve data quality, and predicted probabilities are no longer calibrated to the real class frequencies. Synthetic rows are not independent real observations.

#### Frozen folds (`split.py`)

`make_splits(df, method, n_splits, seed, target, group, time, duplicate_columns)` returns folds of `_dm_row_id`s.
- **Methods:**
  - `kfold`, `stratified`: shuffled, seeded;
  - `group`, `stratified_group`: whole groups on one side;
  - `time`: forward chaining. Training always precedes testing, equal times are never split, and the earliest block is never tested (`EARLIEST_BLOCK_NOT_TESTED`).
- **Duplicates:** rows identical in `duplicate_columns` are kept on one side, merged with groups by union-find (`DUPLICATES_KEPT_TOGETHER`). For time folds, duplicates crossing a boundary are counted instead (`DUPLICATES_CROSS_TIME_SPLIT`).
- **Errors:** `SPLIT_INVALID`, `SPLIT_CLASS_TOO_SMALL`, `SPLIT_TOO_FEW_GROUPS`, `SPLIT_TARGET_MISSING`, `SPLIT_GROUP_MISSING`, `SPLIT_TIME_MISSING`.
- Random stratified splitting is not assumed: choosing the method is the caller's (Part D's) decision.

#### Generated code

Templates for all 13 ops re-learn every parameter from `fit_df`; no learned value is embedded (tested). The generated module gains `prepare_training(fit_df)`, which applies all steps and then the training-only step to the training rows. `preprocess(fit_df, df)` never resamples. Equivalence with the platform is tested for both outputs in an isolated `python -I` subprocess (15 cases, including two combined Part A+B+C pipelines).

#### Leakage proof

`tests/unit/test_preprocess_leakage_c.py`:
- **Spy:** every Part C step fits on training-row ids only.
- **Perturbation:** extreme test values, new categories and **flipped test labels** change no learned parameter, selection decision or training table.
- **Negative control:** fitting on all rows does change them.
- **Isolation:** test rows keep their ids, order, labels and class distribution.
- **In-fold:** two combined Part A+B+C pipelines (one with random oversampling, one with SMOTE) run under stratified, stratified-group and time folds:
  - no test id appears in a training table;
  - no patient crosses train and test;
  - time order holds;
  - perturbing a fold's test rows changes nothing learned in that fold.

#### Part C audit (2026-11-18)

An adversarial audit of all Part C areas changed the following. Each change has a regression test in `tests/unit/test_part_c_audit.py`.

- **Quantile ties at the extremes.** scikit-learn forces values equal to the fitting minimum or maximum to the end of the output: in a column that is 40% zeros, every zero became −5.2 with a normal output. The runtime now maps values with the fitted quantiles itself (`quantile_map`): forward and backward interpolation are averaged, so tied values get their middle rank (≈ 0.2 / −0.84 in that example). Results equal scikit-learn's wherever the extremes are not tied. A single minimum or maximum still maps to the end value (0/1, ±5.2), as documented.
- **Yeo-Johnson extrapolation** is reported: `effects.outside_fit_range_rows` plus an `OUTSIDE_FIT_RANGE` warning. The fitting range stays in memory only and is never reported.
- **Scale-free statistics at extreme magnitudes.** Correlation, mutual information and the univariate tests divide each column (and a regression target) by its largest absolute observed value first (`maxabs_scaled`). These statistics do not depend on scale. Before this change, values near 1e300 overflowed (perfect correlations reported as "not assessed", a `LIBRARY_WARNING`, ANOVA not assessed) and values near 1e-300 underflowed.
- **New selection warnings:**
  - `ALL_SELECTED_REMOVED`: every selected column was removed.
  - `SMALL_CLASS_COUNT`: a class has fewer than 5 fitting rows.
  - `POSSIBLE_TARGET_LEAKAGE`: mutual information reaches at least 90% of the target's entropy, so the column almost determines the target. This catches copies of the outcome.
- **Rankings are unstable between folds.** On 300 rows with 4 weak signals among 10 columns, the top-4 sets differed between 5 stratified folds (4 distinct sets for mutual information, 3 for ANOVA), often including pure-noise columns. Every selector's limitations now say so.
- **SMOTE on dependent rows needs an explicit opt-in.** Within a training fold, SMOTE interpolates between rows of different patients, groups or times; a test demonstrates this.
  - That is not test-set leakage: synthetic rows exist only in training tables.
  - It is a statistical problem: synthetic rows are not independent observations, patients with many rows are over-represented as interpolation anchors, and the effective sample size is overstated.
  - When identifier, patient, sample, group or time roles are declared, SMOTE now fails with `DEPENDENT_ROWS_NOT_ACKNOWLEDGED` unless `acknowledge_dependent_rows` is true. The warning then also appears in the step report.
- **Resampling memory cap.** Measured peaks are about 34 bytes per output cell for SMOTE and 23 for random oversampling. At its row and feature caps, SMOTE would have needed about 2.4 GB, above the 2,000 MB worker cap. New `RESAMPLE_MAX_OUTPUT_CELLS` = 30,000,000 (rows × columns after resampling) is checked before any sampler runs.
- **Correlation memory cap.** Measured peaks are about 25 bytes per cell for Pearson and 32 for Spearman. New `CORR_MAX_CELLS` = 20,000,000 (fitting rows × columns).
- **Report speed.** The per-step change count compared every cell in Python and dominated large runs (quantile on 1M rows: 15 s, of which about 14 s was this count). Boolean, integer and float columns are now compared with NumPy, with identical results (tested against the object comparison, including integers above 2^53). Other dtypes keep the exact object comparison.
- **Validation:**
  - A drop-mode selector's "may be removed" columns are kept aside (`maybe_removed`). SMOTE validation still sees them as features; before, it falsely reported `NO_FEATURES`.
  - Internal `_dm_` columns cannot be selected (`RESERVED_COLUMN`). Ingestion already renames such columns.
- **Executor:**
  - Refuses duplicate column names.
  - `fit_resample` refuses negative real row ids up front (they are reserved for added rows).
  - `_dm_sample_weight` gets a lineage entry (`derived_from` = target, `feature: false`), and every lineage entry has a `feature` flag.
  - `PreprocessExecutor.feature_columns(frame)` returns the model inputs: everything except the target, protected role columns and internal `_dm_` columns.
- **Splitter:**
  - `Splits.classes` and `Splits.fold_stats` give the training and test row counts and class counts per fold. Class labels are values: redact them for a sensitive target.
  - New warnings: `CLASS_MISSING_IN_FOLD`, and `UNEVEN_FOLDS` (largest test fold more than 3× the smallest).
  - `n_splits` and `seed` must be integers.
  - A numeric target cannot be stratified (`SPLIT_INVALID` with a clear reason).
  - Unorderable times (mixed time zones or types) fail with `SPLIT_TIME_INVALID`.
  - Unexpected library errors fail with `SPLIT_FAILED`, carrying the exception type only.

**Benchmarks** (this machine, single process, `fit_resample`; peak = traced allocations during the step):

| Workload | Time | Peak |
|---|---|---|
| Quantile, 1,000,000 × 2 | 1.0 s | 84 MB |
| Yeo-Johnson, 500,000 × 2 | 1.9 s | 42 MB |
| Mutual information, 100,000 × 5 | 4.2 s | 25 MB |
| Univariate ANOVA, 1,000,000 × 5 | 1.5 s | 72 MB |
| Correlation, 200,000 × 100, 5% missing | 2.4 s (Pearson), 8.2 s (Spearman) | 496 / 635 MB |
| SMOTE, 200,000 × 20 → 360k rows | 3.1 s | 250 MB |
| Random oversampling, 500,000 × 5 → 900k rows | 2.2 s | 159 MB |
| Combined A+B+C pipeline, 50,000 × 20 | 6.8 s | 57 MB |

Mutual information grows linearly with columns: about 3 minutes at 100,000 × 200. Timing runs and memory runs were separate, because tracing slows Python-level code.

**Part E enforcement gap (closed in Part D: `commit_transform` refuses model-stage steps, §5.0).** `PreprocessExecutor.fit_transform`, the method a commit worker would use, does not refuse model-stage steps. A model-stage selection step can remove columns there. No commit path exists yet (nothing outside the engine calls the executor), so nothing can be committed today. Resampled rows can never reach this path. `test_part_e_gap_fit_transform_accepts_model_stage_steps` pins the current behaviour, and Part E must turn it into a refusal.

### 3.5 Recommendation engine (Phase 4 Part D, 2026-11-24)

Code: `engine/preprocess/recommend/` (standard library only; the API runs it synchronously). API: `POST /versions/{vid}/recommendations`, `GET /versions/{vid}/recommendations`, `PATCH /recommendations/{id}`.

**Inputs.** The latest diagnostics run (it must be *current*: its semantics hash must equal the version's, otherwise `409`), the latest profile, the version schema, the declared roles and task type, and the dataset domain. The input hash covers all of these plus the rules and engine versions. The same inputs return the stored set unchanged (`200`); new inputs create a new insert-only set (`201`). A set whose inputs are no longer current is returned with `stale: true`.

**Rules never read column names.** Each rule is triggered by measured evidence (schema missing counts, profile skewness, sign and zero counts, frequency tables, Pearson correlations, detected findings) and by declared roles. A test renames every column and checks that the recommendations are unchanged.

| Rule | Evidence | Recommends | Alternatives (same decision) | Refuses or skips when |
|---|---|---|---|---|
| `missing.numeric` | numeric features with missing values (< 60%) | `impute.median` (never mean) | `impute.knn` if an incomplete column has a correlated neighbour (\|r\| ≥ 0.3); `impute.iterative` if \|r\| ≥ 0.5 | ≥ 60% missing (advisory, not imputed); no correlation profile → KNN/iterative `not_assessed`; KNN above 50,000 rows |
| `missing.categorical` | categorical/boolean with missing values | `impute.mode` (< 5% missing or boolean), otherwise `impute.constant` with an explicit "Missing" category | — | — |
| `missing.indicator` | missing-vs-target association finding, or ≥ 20% missing | `missing.indicator` | "do nothing" (impute only) | — |
| `distribution.skew` | profile \|skewness\| ≥ 1, ≥ 8 values, > 2 distinct | `transform.log` (all positive), `log1p` (zeros), `yeo_johnson` (negatives) | Yeo-Johnson; `transform.quantile` if \|skew\| ≥ 2; "preserve the distribution" | no profile → `not_assessed` |
| `outliers.extreme` | `universal.outliers.univariate` detected | `outliers.flag` (IQR × 3, the finding's own rule) | `outliers.cap` (destructive), `scale.robust`, "do nothing" | diagnostics missing → `not_assessed`; never row removal |
| `categories.cardinality` | > 20 categories | `encode.frequency` | grouped `encode.onehot` (20 kept) | ≥ 90% distinct → identifier-like advisory, no encoding |
| `categories.rare` | complete frequency table, ≥ 2 levels below 1% | `encode.onehot` grouping the rare levels | — | — |
| `target.imbalance` | `universal.target.imbalance` detected | `imbalance.class_weight` | random oversampling; SMOTE (all-numeric features, validated after the recommended imputation, ≥ 6 minority rows); undersampling (≥ 500 minority rows) | no target / regression → `not_assessed`; SMOTE refused for entity roles (patients, accounts, groups) or categorical features |
| `redundancy.correlation` | `universal.correlation.redundancy` detected | `select.correlation` (proposal only) | "keep every measurement" | never removes a column |
| `variance.constant` | constant numeric/boolean feature | `select.variance` in `report` mode | — | never removes a column |
| `selection.wide` | ≥ 20 numeric features and < 20 rows per feature | `select.univariate` in `report` mode (stability shown in the lab) | "keep all" | otherwise "not recommended" with the rows-per-feature reason |

Every drafted step is then:
1. checked against the **domain policy** (`recommend/domain.py`, keyed by roles and domain). Biomedical: capping is blocked for clinical domain-role columns and cautioned otherwise; quantile transforms are blocked ("discards the measurement scale"); selection and correlation carry "predictive selection is not biomarker validation"; imputation of clinical measurements is cautioned. Financial and industrial: capping extreme amounts or sensor readings is cautioned. Blocked columns move to "not recommended" with the reason;
2. **validated** with the step validator against the version schema and roles. An op whose roles, types or parameters do not fit becomes `not_assessed` with the validator's reason. Training-only ops are validated after the top-ranked imputation, as they would run.

Each recommendation carries:
- a deterministic id;
- op, version, parameters, columns and `applies_to`;
- stage, kind, risk and destructive flag (from the catalogue);
- what the op does, the reason and structured evidence (source, metric, value, finding id);
- the expected benefit, risks and information loss (level and detail);
- domain and task compatibility;
- the required user decisions;
- evidence strength (strong, moderate or weak, with an explanation);
- limitations (rule and op);
- its alternatives, its conflict group, and a no-guarantee statement.

Considered options that were refused are listed under `not_recommended`, unassessable questions under `not_assessed`, and problems no operation can fix (target proxies, identifier-like columns, mostly-empty columns, missing targets, duplicates) under `advisories`.

**Ranking** (`recommend/ranking.py`) is an ordering aid, not a predicted improvement. The score is the sum of named integer factors, each with its reason:

| Factor | Points |
|---|---|
| Evidence | strong +3, moderate +2, weak +1 |
| Task relevance | relevant +2, indirect +1 |
| Domain | compatible 0, caution −1 |
| Risk | low 0, medium −1, high −2 |
| Information loss | none/low 0, moderate −1, high −2 |
| Sample size | −1 below 100 rows (fitted ops) |
| Resources | −1 for KNN, iterative and SMOTE above 20,000 rows |
| Assumptions | −1 when one is doubtful (e.g. IQR fences on a strongly skewed column) |

Ties break by pipeline position, rule order and id, so the order is reproducible.

**Conflicts.** Options for the same decision are grouped with a trade-off text, every option's risk and information loss, a "do nothing" option with its consequence, and "there is no universal answer: compare them in the lab". The groups are:
- median vs KNN vs iterative imputation;
- imputation alone vs with indicators;
- flag vs cap vs robust scaling;
- log vs Yeo-Johnson vs quantile vs preserving the distribution;
- frequency vs grouped one-hot;
- class weights vs oversampling vs SMOTE vs undersampling;
- correlation review vs keeping every measurement;
- ranking vs keeping all.

**Candidates and split plan.** The engine proposes:
- the baseline;
- a "recommended core" (the top option of each decision plus every uncontested recommendation, assembled in pipeline order and validated incrementally);
- up to three alternatives, each changing a single decision.

The destructive options are never in the core. The split plan follows roles:
- a time role → forward-chaining time folds (with a warning that entities cannot also be kept together);
- an entity role → stratified-group or group folds;
- a classification target → stratified folds;
- otherwise k-fold.

Folds are reduced to the smallest class, the number of entities or the number of distinct times, and exact duplicates (features + target) are always kept together.

**Limitations.**
- Recommendations use whole-version descriptive statistics (as exploration does). They choose only *which* operations to compare; every learned parameter is fitted inside training folds.
- Rules are deterministic templates; semantic types are heuristic.
- Association is never presented as causation.

## 4. Commit (execute)

### 4.0 Implemented in Phase 4 Part E (2026-11-25)

Code:
- engine: `engine/preprocess/stages.py` (standard library: stage choice, plan, hashes), `commit.py` (materialise and preview), `compare.py` (version comparison) and `pipeline_code.py` (generated `pipeline.py`);
- service: `pipelines/` (`service.py` in the API; `jobs.py` with the worker handlers `impact_preview`, `commit`, `revert` on queue `execute` and `compare_versions` on `analysis`; `versioning.py` for lineage, roles and code).

**Stages.** Every pipeline step has an explicit chosen stage (`data` | `model`), part of its step hash:
- Catalogue `data` and `model` ops keep their stage. `either` ops default to `data` when added and can be switched; the UI shows the choice. They are never moved silently: a Lab candidate's `either` op after its first model-stage op stays model-stage, so the evaluated order is kept.
- Data-stage steps must precede every model-stage step (`STAGE_ORDER`).
- Only data-stage steps are materialised, through `PreprocessExecutor.commit_transform` (which refuses model-stage steps).
- Model-stage steps become the version's `model_pipeline_spec` (D1/E3): stored, shown and emitted as an in-fold function in `pipeline.py`, but never applied to the saved data. Resampled or synthetic rows can never reach a version, and `commit.py` refuses any result whose row ids differ from the source's.

**Impact preview** (job `impact_preview`). It runs the same `commit_transform` call as the commit on the source version, then replays the fitted steps one by one to describe each step:
- rows before and after; columns created and removed;
- per column: missing before and after, estimates filled, observed values overwritten (destructive steps) or re-expressed (invertible steps), and before/after summary statistics;
- rows affected (exact count) and information-loss indicators;
- up to `impact_preview_max_rows` (default 1,000, E1) affected rows per step, chosen as the lowest row ids, with before/after values for at most 20 changed columns and text truncated to 64 characters. Values of sensitive-role columns, and their statistics, are redacted. The list is flagged as truncated when it is.

The preview also records the content hash of the would-be version. It is stored as an `impact_preview` artifact and attached to the pipeline only if the pipeline's spec hash is unchanged when it completes. It is *current* only while its spec hash equals the pipeline's and the pipeline's source is still the dataset head.

**Approvals** (insert-only):
- every high-risk step (data or model stage) needs its own approval, bound to its `step_hash`, the `spec_hash`, the source version and the preview artifact. Destructive steps also need `acknowledge_destructive`;
- any medium-risk step requires one pipeline confirmation bound to `spec_hash`;
- the server stores the impact numbers and the statement it displayed. Client-supplied hashes are only compared with server-computed ones.

An approval is valid only while all of its bindings equal the current ones. Any edit, deletion, reordering or stage change changes the hashes, so earlier approvals and previews stop applying without being rewritten. The UI shows them as stale.

**Commit** (job `commit`, `Idempotency-Key` required). `commit_readiness` is the single authoritative check:
- the pipeline is not committed and is valid;
- the spec hash equals the request's and the recomputed one;
- the source is the dataset head and the preview is current;
- every required approval is valid;
- the version quota is not reached.

The API runs it under the dataset and pipeline row locks before queueing; a partial unique index allows one queued or running commit or revert per dataset. The worker runs it:
- again before reading data;
- then verifies the source file's SHA-256;
- runs `materialise` and requires the result's content hash to equal the previewed one (`PREVIEW_MISMATCH` otherwise);
- checks for cancellation and writes the Parquet object under a fresh key;
- finally runs the check a third time under the dataset row lock before inserting.

In one transaction it inserts:
- the `transform` version: `version_number` = max + 1, parent = head; `fitted_on_full_data` when any data step learns from data or the source was fitted; `model_pipeline_spec`; the schema with types carried from the source;
- the copied roles (lost roles reported);
- the `transformation_run` (spec snapshot with the roles used, data step hashes, approvals, step reports, seed, library versions);
- a `version_events` row and the version's preview artifact.

It then moves the head, marks the pipeline committed (its steps are frozen by a trigger), and queues profiling and diagnostics of the new version (E4). Their failure never affects the committed version.

**Failure and retry:**
- Any failure leaves no version, no run and the head unchanged.
- An object written before a failure is unreferenced; the hourly `reconcile_orphans` maintenance task deletes unreferenced version and artifact objects older than `orphan_grace_seconds` (24 h). It never touches `raw/` or a referenced object.
- A retried job whose run already exists ends as succeeded without doing anything.
- A durable commit is never reported as cancelled; cancellation is honoured only before the object is written.

**Revert** (job `revert`): a new `revert` version whose object is a server-side copy of the chosen earlier version's bytes. It has:
- the same content hash, schema, `fitted_on_full_data` and modelling specification;
- parent = head, `restored_from_version_id` = the target;
- the target version's roles copied;
- a `reverted` event, and follow-up profiling and diagnostics.

No version is deleted or changed and history stays linear. The version cap (`max_versions_per_dataset`, default 20, E2) applies to commits and reverts (`QUOTA_EXCEEDED`); nothing is deleted to make room. It is checked in the API and again under the dataset lock in the worker. Ingest's existing size and cell limits still apply to every version.

**Comparison** (job `compare_versions`, read-only; counts and statistics only):
- schema and row-count differences, and rows aligned by `_dm_row_id`;
- per column: values changed, filled and emptied; missing share; a distribution summary; IQR × 1.5 outlier counts;
- a test with effect size: KS D plus standardised mean difference (numeric), or chi-square with Cramér's V and a sparsity note (categorical);
- `not_assessed` with a reason for constant columns, fewer than 8 values, changed types, dates, or more than 50 categories;
- information-loss indicators, lineage between the versions, and limitations. The versions are not independent samples, so p-values are descriptive, and significance is not importance.
- Tests sample at most 200,000 values per side (disclosed) and at most 500 columns.

### 4.1 Original design

1. Load the head version and the canonical spec. Re-verify source = head, the hashes, and the approvals (high-risk step approvals + pipeline confirmation if any medium step exists).
2. Apply steps in order: stateless ops directly, row filters as masks, and fitted ops through an sklearn `Pipeline`/`ColumnTransformer` with `set_output(transform="pandas")`, preserving `_dm_row_id`.
3. Produce a per-step report: rows in/out, columns added/removed, values changed per column, warnings.
4. Check post-conditions (e.g. no NaN in imputed columns).
5. Write the Parquet. Compute `file_sha256` and `content_hash`, the diff vs the parent, and HMAC'd fitted state. Set `fitted_on_full_data` if any fitted op ran.
6. Insert the version and the transformation run in one transaction after the object write. A reconciliation job removes orphaned objects older than 24 h.

**Impact preview** (prerequisite for approvals): the same code path as commit, run as a dry run that writes only an `impact_preview` artifact. It covers per-step counts, before/after summary statistics for changed columns, and up to 1,000 affected row IDs per step. The preview is bound to `spec_hash`. Above 1M rows it may be computed on a disclosed, seeded sample, and the approval dialog says so.

---

## 5. Experiments (evaluation; isolated)

### 5.0 Implemented in Phase 4 Part D (2026-11-24)

Code:
- engine: `engine/preprocess/experiment/` (`runner.py`, `models.py`, `metrics.py`, `quality.py`, `stability.py`, `folds.py`), worker only;
- service and API: `experiments/` (`service.py` in the API, `jobs.py` for the worker handler `experiment`, queue `evaluate`, 3,600 s).

**Configuration (API).**
- Up to 6 candidates plus the automatic baseline.
- Models: `dummy`, `linear` and `hgb` for supervised tasks, `kmeans` for clustering (no XGBoost, per D3).
- 2–10 folds and a seed.
- Every candidate is validated with the step validator. Risk, stages and the destructive flag come from the catalogue.
- Group and time columns come from roles, never from the request. Overrides that ignore a declared entity or time role are allowed, with `GROUP_ROLE_IGNORED` / `TIME_ROLE_IGNORED`.
- Limits (settings): 200,000 rows, 5,000,000 cells, 300 model fits (candidates × folds × models). Exceeding one returns `409 QUOTA_EXCEEDED` before anything is queued.
- The stored configuration includes a snapshot of the roles, the source content hash, the split configuration, the seeds and each candidate's submitted steps and spec hash; its SHA-256 is `config_hash`. The worker rebuilds candidates from this snapshot and verifies every spec hash (`CONFIG_CHANGED` otherwise). Later role changes never alter an experiment, and `POST /experiments/{id}/rerun` reproduces it from the stored configuration.
- The `Idempotency-Key` header returns the original experiment.

**Frozen folds (D3).** The cohort is the rows with a target (all rows for clustering); the excluded count is reported. The worker looks up a split definition with the same configuration hash (method, folds, seed, target, group, time, duplicate columns, source content hash) on the version:
- if one exists, it loads the stored `fold_assignment` artifact (JSON: sorted `_dm_row_id`s and the test fold per row, `-1` = never tested) after verifying its SHA-256;
- otherwise it runs the Part C splitter once, stores the artifact and inserts the split definition in its own short transaction (a concurrent duplicate loses the unique constraint and reuses the winner).

Every candidate and every model use that one `Splits` object. The encoding is verified to reproduce the folds exactly before it is stored.

**Per candidate and fold** (`runner.py`):
1. `PreprocessExecutor.fit_resample(training rows)` fits every learned step on the training rows only. Resampling and class weights (the last, training-only step) change only that training table.
2. `transform(test rows)` learns nothing, resamples nothing and removes nothing. The runner checks that every test row id was scored, otherwise the fold fails as `NOT_DIRECTLY_COMPARABLE`.
3. Each model is fitted on `feature_columns()` of the training table (never `_dm_` columns, the target or protected roles), with `_dm_sample_weight` passed as `sample_weight`, and scored on the test rows.

Models use the same documented minimal in-fold preprocessing for every candidate:
- `linear` / `kmeans`: median imputation and standardisation; most-frequent imputation and one-hot encoding of the 20 most frequent categories;
- `hgb`: ordinal codes, missing values handled natively;
- dates and times are ignored and reported.

Class labels become integer codes, so models and metrics never see label values. One native thread per worker (`threadpoolctl`): thread-pool start-up made gradient boosting 4× slower on small folds.

**Failures are isolated.** A preprocessing failure fails that fold of that candidate; a model failure fails that model in that fold. The codes are the engine's own; unexpected errors carry only their type. A candidate with any failed fold is `failed` or `partial`, `comparable: false` with the reason, and excluded from comparisons. A `MemoryError` fails the job as `LIMIT_EXCEEDED`.

Results, the split reference and the summary are written in the job's final transaction, so a crash, failure or cancellation stores no partial results. Cancellation is checked before every fold through a separate session; progress and heartbeats are written there too. A completed experiment is never re-run (the handler returns `skipped`, and a trigger forbids changing a completed experiment).

**Metrics** (held-out rows only, per fold, then mean, standard deviation, minimum, maximum and per-fold values):
- Classification: accuracy, balanced accuracy, precision, recall, F1 (binary: the second label in sorted order is positive; multiclass: macro), ROC-AUC and PR-AUC (only when every class occurs among the test rows, otherwise `None` with a reason), log loss. Binary targets add the Brier score and expected calibration error (10 bins). Each model also has a summed confusion matrix and per-class precision, recall, F1 and support.
- Regression: MAE, RMSE, R² (`None` when the test target is constant) and median absolute error.
- Clustering (internal metrics, no ground truth): silhouette (seeded sample of ≤ 5,000 rows), Davies-Bouldin and Calinski-Harabasz, with explicit limitations.
- Fit and preprocessing times.

**Comparisons, without declaring a winner.** For every comparable candidate, model and metric against the baseline:
- the per-fold differences, their mean and standard deviation, and the folds better or worse (direction-aware);
- a descriptive verdict: `negligible` (below the practical threshold: 0.01 for bounded metrics, 1% of the baseline mean otherwise), `consistently_better` / `consistently_worse` (every fold), `within_fold_variation` (\|mean\| < sd), `mixed`, or `insufficient_folds`;
- the label "describes these folds only; not a significance test".

Warnings:
- `MULTIPLE_COMPARISONS` (> 2 candidates);
- `SUSPICIOUSLY_HIGH_SCORE` (> 0.99);
- `DUMMY_COMPETITIVE`;
- `FEW_SAMPLES_PER_CLASS_PER_FOLD` (< 5);
- `ROWSET_DIFFERS` (resampled training rows);
- `NOT_DIRECTLY_COMPARABLE`;
- `DATETIME_FEATURES_IGNORED`;
- every split warning.

**Data-quality changes** (counts only, test rows before → after the fitted pipeline, mean per fold):
- missing cells, duplicate rows, values beyond IQR × 1.5, mean \|skewness\|, model input columns, constant columns, categorical columns and category levels;
- cells imputed, observed values *overwritten* (destructive steps only) and values re-expressed (invertible transforms), and rows affected;
- columns added and removed, and training rows before and after resampling;
- information-loss indicators.

The UI states that fewer missing values or outliers is not automatically better.

**Selection stability** (`stability.py`). For every selection step and fold, the runner records the selected columns (forced keeps, from the user list or a domain role, are reported separately). From these it reports:
- selection frequency;
- mean pairwise Jaccard overlap;
- stable (≥ 0.8), unstable and rarely selected (≤ 0.2) columns;
- for supervised rankings, **noise probes**: the step is re-fitted on the same training rows with up to five permuted copies of its own numeric columns. The probe selection rate gives the frequency that noise alone reaches, and columns selected no more often are listed as indistinguishable from noise (`NOISE_SELECTED`).

`UNSTABLE_SELECTION` is reported below a mean Jaccard of 0.5. Limitations state that selection is association, not causal or scientific validation, and "predictive selection is not biomarker validation".

**What experiments never do.** They never write a dataset version or a version object (tests count versions and list the `versions/` prefix before and after). Resampled rows exist only in training tables in worker memory.

The commit path is now an explicit executor method, `commit_transform`, which refuses every model-stage step (`MODEL_STAGE_NOT_COMMITTABLE`) before reading data. This closes the Part C enforcement gap: Part E's commit worker must use it. `fit_transform` stays a general fit-and-apply helper (code-generation equivalence) and is documented as not being a commit path.

**Generated code.** Every suggested candidate renders with the existing Part A–C templates to code that reproduces the platform's held-out rows and training table in a `python -I` subprocess (`tests/unit/test_experiment_codegen.py`).

**Benchmarks** (reference machine, one process, one native thread; baseline plus 3 candidates including median imputation, log transform, univariate ranking with noise probes, robust scaling, class weights and random oversampling). Timing and memory were measured in separate runs, because tracing slows Python-level code:

| Rows × columns | Folds | Models | Fits | Time | Peak traced memory |
|---|---|---|---|---|---|
| 5,000 × 12 | 5 | dummy, linear, hgb | 60 | 34 s* | 7 MB |
| 20,000 × 22 | 5 | dummy, linear, hgb | 60 | 51 s | 43 MB |
| 100,000 × 12 | 3 | dummy, linear | 24 | 14 s | 105 MB |

\* Measured while other test suites were running; the 20,000-row run is the better guide.

Gradient boosting dominates the time (about 0.3 s per fit even on small folds). Before the thread limit it took about 1.2 s per fit; that reduced a 400-row smoke experiment from 117 s to 64 s under the profiler. Experiments at the 300-fit and 200,000-row caps were not measured (limited free memory on the reference machine). Linear extrapolation from the runs above suggests tens of minutes, within the 3,600 s job limit but not verified. A timeout fails the job cleanly, with no partial results.

**Not yet.** These remain for later:
- XGBoost;
- hyper-parameter tuning;
- out-of-fold prediction artifacts;
- subgroup metrics beyond per-class;
- bootstrap intervals;
- paired comparisons between non-baseline candidates.

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

`FITTED_ON_FULL_DATA_SOURCE` is emitted by the Experiment Lab since Phase 5 (owner decision P4), and the Lab
page shows a banner for such versions. Since the Phase 5 closure (2026-11-26) the Lab's HGB sets
`early_stopping=False` (fixed `max_iter=100`), as Phase 5 training does: the library default `"auto"`
switched on above 10,000 training rows and held out a random 10% of the *training* rows that ignored
declared groups, time order and duplicate clusters (test rows were never involved). Lab results on 10,000
training rows or fewer are unchanged.

**Terminology.** Evaluation modes `unbiased_direct` / `unbiased_replay` (internal identifiers, unchanged)
are shown as "Leakage-controlled evaluation" and "Leakage-controlled evaluation (replayed)": the
implemented split and preprocessing safeguards, not a guarantee of external validity, of the absence of
bias or of clinical reliability. Exploratory runs are "Exploratory only".

### 5.6 Supervised training (Phase 5, implemented 2026-11-25)

Code: `engine/ml/` (worker except `spec.py`, `eligibility.py`, `audit_spec.py`) and `training/` (API service,
worker handlers `train` and `reliability_audit` on queue `evaluate`).

**Models** (scikit-learn 1.9.1, one thread, seeded): `dummy` (prior / median), `linear` (logistic
regression, max_iter 2000 / ridge; Lab one-hot preprocessing), `random_forest` (100 trees, depth ≤ 12, in-fold
median / most-frequent imputation, ordinal categories), `hgb` (native missing values, ordinal categories,
`early_stopping=False`). Grids (≤ 4, simplest first; ties go to the earlier setting): logistic `C` ∈
{0.01, 0.1, 1, 10}; ridge `alpha` ∈ {100, 10, 1, 0.1}; forest `max_depth` ∈ {6, 12} × `min_samples_leaf` ∈ {10, 2};
HGB `max_leaf_nodes` ∈ {15, 31} × `learning_rate` ∈ {0.05, 0.1}. Defaults: C = 1, alpha = 1, depth 12 /
leaf 2, 31 leaves / 0.1. A contract test fits every setting and checks every parameter reached the
estimator.

**Task and target.** The confirmed target role decides the task; binary needs exactly 2 classes,
multiclass 3–50, regression a numeric target with ≥ 10 values; cohort ≥ 50 rows; every class ≥ the number of
folds. Predictors are every column without a non-feature role (target, id, group, time, weight, exclude,
domain identifiers) and not `_dm_`; date and time columns are ignored and reported. The `weight` role is not
used (disclosed).

**Eligibility** (§1 of PHASE5_PLAN §5.2; `eligibility.py`):
- `unbiased_direct`: the version is not `fitted_on_full_data`.
- `unbiased_replay`: the lineage (reverts followed to the version they restored) is walked to the latest
  ancestor not fitted on full data. Every later commit's data steps must re-validate, against its source
  schema and commit-time roles, to the recorded step hashes (fidelity). Safety: no row-changing step, no
  step on the current target, group/time roles must exist on the ancestor, same row count. The worker then
  refits the commits on all ancestor rows; the content hash must equal the version's (`exact`) or match
  within `allclose(rtol=1e-9, atol=1e-12)` with exact non-float columns
  (`equivalent_within_tolerance`); otherwise `REPLAY_NOT_ESTABLISHED`.
- `blocked`: otherwise (`LINEAGE_INCOMPLETE`, `REPLAY_HASH_MISMATCH`, `REPLAY_ROWS_DIFFER`,
  `REPLAY_UNSUPPORTED_STEP`, `TARGET_TRANSFORMED_IN_LINEAGE`, `ROLE_ON_DERIVED_COLUMN`). Creation returns
  `409 EVALUATION_BLOCKED` unless `allow_exploratory`, which trains on the version as stored with
  `evaluation_mode = exploratory`, kept on reruns.

**Per outer fold** (frozen split from `split_definitions`, built on the evaluated version): inner folds by
the same splitter on outer-training rows only (3 folds, seed `seed + 1000 + k`, groups / duplicate blocks /
time order kept), shared by every model and setting; preprocessing (replayed commits, each fitted on the
training rows and applied to all rows of the fold with a Parquet round trip, then the model-stage
specification via `fit_resample` / `transform`) refitted for every inner fold; selection on the inner mean of
ROC-AUC (binary, when every inner test fold has both classes; else balanced accuracy), balanced accuracy
(multiclass) or RMSE (regression). If inner folds cannot be built, or a class has fewer than 3 rows in an
inner test fold, that fold uses the defaults (reason recorded). Cohorts under 200 rows and runs above the fit
budget use defaults throughout (reason recorded). Then refit on all outer-training rows, predict the outer
test rows once, check every test row was scored, and record boundary evidence (rows each learned step was
fitted on, test completeness, no synthetic test rows).

**Metrics** (outer test rows, per fold, mean / sd / min / max; undefined → `None` + reason): binary
accuracy, balanced accuracy, precision, recall, F1 (positive class = second sorted label), ROC-AUC, PR-AUC,
log loss, Brier, ECE; multiclass adds weighted precision / recall / F1, one-vs-rest macro ROC-AUC and PR-AUC,
top-label ECE; regression MAE, MSE, RMSE, R², median AE. Pooled out-of-fold aggregates: ROC and PR curves
(≤ 200 points; per class for ≤ 10 classes), quantile-bin reliability curve (≥ 100 rows), confusion matrix and
per-class table, binned actual-vs-predicted and residual grids (40 × 40), residual histogram and quantiles
(axis values withheld for a sensitive target). Percentile bootstrap (1,000 resamples of split groups) of the
headline metrics; unavailable for time splits or fewer than 10 groups; labelled as conditional on the
evaluated predictions. Paired per-fold differences against the dummy (all metrics) and between models
(primary metric) with the Lab's descriptive verdicts. No winner is declared; the selection optimism of
choosing a model on the same folds is stated.

**Explainability** (outer test rows, after selection; never used for selection): grouped permutation
importance (5 seeded repeats, ≤ 5,000 rows per fold, ≤ 100 columns), aggregated per original column over
fold × repeat values with the number of folds in which the column was a model input; linear coefficient
norms per original column; random-forest impurity importance; error slices (quantile bins or top-10
categories of the 5 most important columns, ≥ 30 rows per slice, sensitive columns excluded). Not causal.

**Results** are written in the job's final transaction (no partial results); out-of-fold predictions are an
`oof_predictions` Parquet artifact (owner only, never served). Determinism: identical metrics, selections and
predictions for an identical configuration on the same platform (tested); cross-platform equality is not
claimed.

**Measured** (development machine, live stack, smoke test): see IMPLEMENTATION_ROADMAP Phase 5 verification.

---

## 6. Leakage audit

### 6.0 ML Reliability Auditor (Phase 5, implemented 2026-11-25)

The leakage audit is implemented as the reliability audit of a training run (`engine/ml/audit_spec.py`,
`engine/ml/audit.py`), stored as an `analysis_runs` row of kind `reliability_audit` with findings in the
existing schema. Every finding has `evidence.basis` = `confirmed`, `configuration` or `statistical`;
statistical detections are titled "Possible …". The 20 checks, with their thresholds, are listed by
`GET /api/v1/ml/models` (`audit_checks`) and in PHASE5_PLAN §6.2 (A1–A20). Implementation notes:
- A1 single-feature: out-of-fold depth-3 trees on the frozen folds (≤ 200 columns by variance, ≤ 50,000
  rows sampled); AUC is symmetrised (max(AUC, 1 − AUC)).
- A4 duplicates: exact (features + target), features-only and normalised (trimmed lower-case text,
  numbers to 6 significant digits) row hashes across every fold; near-duplicates beyond that normalisation
  are not found (stated).
- A8 post-outcome: always `not_assessed` with universal roles.
- Evaluation status: `blocked` (critical confirmed / configuration violation; for exploratory runs other than
  the acknowledged full-data fit), `exploratory`, `concerns` (high or critical statistical suspicion),
  `no_issues_detected_among_assessed` (with the list of checks not assessed). No numeric score.
- Clean synthetic controls produce no high or critical detection (tested). The original table below remains
  the design reference.

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

**Implemented in Part E (2026-11-25):** `GET /versions/{vid}/code` returns `pipeline.py` (owner only, `no-store`), rendered in the API without data libraries. Engine sources are read from their files, not imported. The module embeds, verbatim:
- the parsing code (`errors`, `table`, `io/readers`, `hashing`, `preview`, `io/parse`), with internal imports stripped; the renderer refuses any top-level name defined differently by two modules;
- the preprocessing runtime;
- per-op step templates, grouped per committed transformation in lineage order. A revert replays the chain of the version it restored, so undone steps are never replayed;
- the modelling specification as `model_preprocess` / `prepare_training`, never applied to the output.

`main` parses the original with the recorded format and extension, refuses to continue if the detected parse options differ from those recorded, applies each commit (fitted on all rows), passes every intermediate version through Parquet like the platform, and prints the content hash.

Tests run it in `python -I` subprocesses and require **exact content-hash equality** (no tolerance) for version 0, a two-commit lineage, committed versions through the API, and a reverted version. They also cover hostile column names. Committed data steps are re-validated against their recorded hashes; a mismatch refuses code generation. Exact hash equality holds on the platform that committed the version (the worker image). On another operating system, NumPy's maths library can differ in the last bit of some results (measured: `log` values differing by at most 2.2e-16 between Linux and Windows), so cross-platform equivalence is checked with `numpy.allclose(rtol=1e-9, atol=1e-12)` and exact equality for non-float columns, the design's documented fallback. Export bundles and sandboxed replay validation remain Phase 9.

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
