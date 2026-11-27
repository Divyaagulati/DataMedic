# DataMedic — Domain Modules

| | |
|---|---|
| Status | Revision 2, MVP-scoped |
| Related | [ML_ENGINE_DESIGN.md](ML_ENGINE_DESIGN.md) |

## 1. Universal vs domain-specific

| | Universal | Domain-specific |
|---|---|---|
| Applies to | Every dataset | Datasets whose `domain` matches |
| Knowledge | Statistics, types, universal roles | Domain semantics via domain roles |
| Lives in | `datamedic_engine.diagnostics` / `recommend` / `ops` | A module package, e.g. `datamedic_biomedical` |
| Finding `domain` | `universal` | Module key, e.g. `biomedical` |
| UI | "Universal" tab | Module tab (name and version shown) |
| Effect on universal results | — | **None.** It may add findings citing universal ones (e.g. "duplicate rows are repeat visits of the same patient"), but never edits or suppresses them. |

## 2. Module interface and registration

```python
class DomainModule(Protocol):
    key: str                                 # "biomedical"
    name: str
    version: str
    semantic_roles: list[RoleDefinition]     # key, description, cardinality, allowed types, implies (e.g. "group")
    role_suggesters: list[RoleSuggester]     # suggestions only; never auto-applied
    checks: list[Check]
    rules: list[RecommendationRule]
    ops: list[OpDefinition]                  # same contract incl. risk + destructive
    split_policy: SplitPolicy | None
    passport_sections: list[PassportSection]
```

- **MVP registration**: an explicit in-code registry, `datamedic_engine.domains.registry.register(module)`, called from one composition module (`datamedic_workers.bootstrap`). Core engine packages never import a module directly.
- **Future**: entry-point discovery (group `datamedic.domains`) replaces the composition list without changing the interface.
- A module whose optional dependencies are not installed in a worker registers with `available=False`. Its checks report `not_run` ("module dependencies not installed in this worker"), and its ops are hidden from the catalogue.

**Contract test kit** (`datamedic_engine.testing.domain_contract`), required for every module:
1. Every check passes finding-schema validation and the `not_run` honesty test.
2. Every op has a params JSON Schema, risk and destructive classification, `_dm_row_id` preservation and codegen equivalence.
3. Rules reference only registered checks and ops.
4. Golden datasets with expected findings exist.
5. Anything that removes samples, removes columns or rewrites measured values is high-risk **and** destructive.

**Extensibility proof**: a stub `fin` module (one role, one check, one rule, one op) lives in the test tree. CI asserts that it appears in `/domains`, `/checks` and `/ops` with no core code changes (Phase 4 onward).

## 3. Universal roles

`target` (with `task_type`), `id`, `group`, `time`, `weight`, `exclude`, `sensitive`.

## 4. Biomedical module (flagship)

### 4.1 Scope by tier
| Tier | Scope |
|---|---|
| **MVP** | **Clinical tabular**: one row per patient, visit or sample; clinical variables, labs, demographics, outcomes |
| **V2** | Omics matrices (samples × genes/proteins/metabolites) with Scanpy/anndata in a separate `worker-bio` image; per-patient aggregation ops |

### 4.2 Roles (MVP)
| Role key | Meaning | Implies |
|---|---|---|
| `bio.patient_id` | Subject identifier | `group` for splitting |
| `bio.sample_id` | Specimen identifier | `id` |
| `bio.visit_time` | Time of measurement | `time` |
| `bio.outcome_time` | Time the outcome was determined | Temporal leakage checks |
| `bio.batch` | Site, centre, plate, run or scanner | — |
| `bio.condition` | Biological condition or covariate | — |
| `bio.sex`, `bio.age`, `bio.dob` | Demographics | `sensitive` by default |
| `bio.lab_value` | Lab measurement | — |
| `bio.unit_column` | Units for a lab column (`unit_for: <column>`) | — |

### 4.3 Biomedical checks (MVP, clinical)

| Check ID | Detects | Method | Requirements | Limitations |
|---|---|---|---|---|
| `bio.patients.repeated_rows` | Multiple rows per patient | Rows-per-patient distribution | `bio.patient_id` | — |
| `bio.patients.inconsistent_static` | Static attributes varying within a patient | Distinct counts per patient | `bio.patient_id` + `bio.sex`/`bio.dob` | Error vs legitimate update is the user's call |
| `bio.leakage.patient_overlap` | Patient in both training and test folds | Overlap against the fold map | `bio.patient_id` + split | — |
| `bio.batch.target_confounding` | Batch associated with outcome | `scipy.stats.chi2_contingency` → Cramér's V (categorical target); `scipy.stats.kruskal` (continuous) | `bio.batch` + target | Association ≠ causation |
| `bio.batch.effect_strength` | Feature distributions differing by batch | Per-feature Kruskal–Wallis with Benjamini–Hochberg FDR (`scipy.stats.false_discovery_control`; verified to exist in the pinned SciPy, otherwise implemented in-house with tests); silhouette of batch labels on a PCA embedding (`sklearn.decomposition.PCA`, `sklearn.metrics.silhouette_score`) | `bio.batch` + ≥ 2 numeric features | Biological vs technical differences cannot be separated without design information |
| `bio.batch.design_balance` | Condition or target unevenly spread across batches | Contingency table | `bio.batch` + `bio.condition` or target | — |
| `bio.labs.plausibility` | Implausible lab values | Ranges from a **user-uploaded reference-range table** (analyte, unit, low, high, source) | `bio.lab_value` + matching reference | `not_run` per analyte with no reference. Ranges vary by lab, population and unit. |
| `bio.labs.unit_inconsistency` | Mixed units within one analyte | Unit-column heterogeneity; value clusters at known conversion ratios (from a small, cited conversion table shipped with the module) | `bio.lab_value` (+ `bio.unit_column`) | Ratio detection is suggestive |
| `bio.temporal.post_outcome_features` | Measurements after outcome time | Timestamp comparison | `bio.outcome_time` + `bio.visit_time` | `not_run` without timestamps |
| `bio.demographics.plausibility` | Age < 0 or > 120; DOB after visit | Rules | `bio.age` or `bio.dob` | — |
| `bio.privacy.phi_extended` | Clinical identifiers: MRN-like IDs, full dates, note-like free text | Extends the universal PII scan | — | Heuristic; does not establish HIPAA Safe Harbor |
| `bio.cohort.subgroup_balance` | Target imbalance across sex and age groups | Contingency + counts | target + demographics | Descriptive; fairness needs context |

### 4.4 Biomedical ops (MVP)

| Op | Risk | Destructive | Kind | Notes |
|---|---|---|---|---|
| `bio.labs.convert_units` | high | **yes** (rewrites measured values) | stateless_column | Uses a user-confirmed conversion factor per unit value; the impact preview shows before/after per unit |
| `bio.labs.flag_implausible` | low | no | stateless_column | Adds an indicator column from the reference ranges |
| `bio.labs.set_implausible_missing` | high | **yes** | stateless_column | Replaces implausible values with NaN |
| `bio.columns.drop_identifiers` | high | **yes** | stateless_column | Limitation: quasi-identifiers may still re-identify |

The patient-grouped split is not an op. It follows from `bio.patient_id` implying the `group` role (split policy).

### 4.5 Biomedical recommendation rules (examples)
| Trigger | Recommendation |
|---|---|
| `bio.patients.repeated_rows` failed and no group role | Assign `bio.patient_id` (implies group). Experiments then use `StratifiedGroupKFold`. |
| `bio.batch.target_confounding` high or critical | **No automatic correction.** Explain the confounding; suggest stratified reporting by batch and collecting design information. In V2, ComBat is only recommended with the condition or target as a covariate. |
| `bio.labs.unit_inconsistency` | `bio.labs.convert_units` (destructive), with the detected unit groups pre-filled for confirmation |
| `bio.labs.plausibility` failed | `bio.labs.flag_implausible` (preferred) or `bio.labs.set_implausible_missing` (destructive) |
| `bio.privacy.phi_extended` | Mark `sensitive` / `exclude`, or `bio.columns.drop_identifiers` (destructive) |

### 4.6 Biomedical golden datasets (synthetic)
- `bio_clinical_patient_leak`: 2,000 rows / 700 patients; random KFold inflates AUC, group KFold does not.
- `bio_clinical_batch_confounded`: site associated with outcome, plus a site-specific shift.
- `bio_labs_units`: creatinine mixed mg/dL and µmol/L (ratio ≈ 88.4).
- `bio_post_outcome`: a feature dated after the outcome.
- `bio_clean_control`: no seeded issues; at most low/info severity findings.

### 4.7 V2 omics extension (design kept for continuity)
- Separate `worker-bio` image with Scanpy and anndata. Checks: `bio.omics.qc_metrics` (`scanpy.pp.calculate_qc_metrics`), `bio.omics.low_quality_samples`, `bio.omics.sparsity`, `bio.omics.data_scale`. Ops: `bio.omics.normalize_total`, `bio.omics.log1p`, `bio.omics.select_hvg` (fitted), `bio.batch.combat`.
- **ComBat safety**: `scanpy.pp.combat(adata, key=…, covariates=[…])` fits across all provided samples. In experiments it is therefore **transductive**: the candidate is labelled "optimistic" with a `TRANSDUCTIVE_STEP` limitation and is never presented as leakage-free. ComBat is high-risk and destructive, is recommended only with the condition or target as a covariate, and is blocked when confounding is critical and no covariate is set.
- Per-patient aggregation (`bio.patients.aggregate_*`) needs new row-ID derivation rules and is deferred to V2.

## 4.8 Implemented in Phase 3 (2026-11-09)

Packages live in `datamedic/domains/<key>/`. Each has `__init__.py` (roles and check specs, standard library only) and `checks.py` (implementations). They are registered in `domains/bootstrap.py`, the only place that lists modules, and core packages never import them. A stub module registers without core changes (`tests/unit/test_check_framework.py`).

| Module | Roles | Checks |
|---|---|---|
| Biomedical | `bio.patient_id`, `bio.sample_id`, `bio.batch`, `bio.condition`, `bio.biomarker_id`, `bio.sex`, `bio.age`, `bio.heart_rate`, `bio.systolic_bp`, `bio.diastolic_bp`, `bio.temperature_c`, `bio.bmi` | `bio.patients.repeated_rows`, `bio.leakage.patient_level`, `bio.biomarkers.id_consistency`, `bio.dimensionality.hdlss`, `bio.batch.effect` (Kruskal–Wallis + BH FDR + ε²), `bio.batch.confounding` (Cramér's V / ε²), `bio.clinical.plausibility` |
| Financial | `fin.transaction_id`, `fin.account_id`, `fin.timestamp`, `fin.amount` | `fin.timestamps.consistency`, `fin.transactions.duplicates`, `fin.leakage.temporal` (target change across 5 time periods), `fin.amounts.unusual` |
| Business | `biz.customer_id`, `biz.segment` | `biz.customers.duplicates`, `biz.customers.suspicious_ids`, `biz.leakage.churn` (missingness pattern matches the outcome), `biz.segments.imbalance` |
| Industrial | `ind.timestamp`, `ind.asset_id`, `ind.sensor`, `ind.sensor_nonneg`, `ind.sensor_percent` | `ind.sensors.impossible` (declared physical limits, saturation), `ind.sampling.irregular`, `ind.sensors.shift` (KS early vs late + BH), `ind.sensors.drift` (Spearman with time + BH) |

Deviation from §4.3: `bio.clinical.plausibility` uses wide, fixed physiological bounds for explicitly declared vital-sign roles (units assumed and stated). A user-supplied reference-range table (`bio.labs.plausibility`) is not built yet.

Also not built yet:
- `bio.patients.inconsistent_static`, `bio.batch.design_balance` (partly covered: confounding also compares the batch with `bio.condition`);
- `bio.labs.unit_inconsistency`, `bio.temporal.post_outcome_features`;
- `bio.privacy.phi_extended`, `bio.cohort.subgroup_balance`;
- all ops, rules and split policies.

Time-based checks always sort by the declared time role, never by row order.

## 5. Future modules (interface proof only)

The financial, business and industrial modules in this section were built in Phase 3 (§4.8) with a first set of checks. The remaining ideas below are future work.

| Module | Roles | Example checks | Split policy |
|---|---|---|---|
| Financial `fin` | `fin.account_id` (group), `fin.as_of_time`, `fin.event_time`, `fin.currency` | Look-ahead bias, survivorship indicators, currency mixing, time gaps, regime shift, card-number detection (Luhn) | Walk-forward `TimeSeriesSplit` with `gap` |
| Business `biz` | `biz.customer_id`, `biz.snapshot_date`, `biz.churn_date` | Churn-label leakage, duplicate CRM entities, funnel consistency | Time- and group-aware |
| Industrial `ind` | `ind.sensor_id`, `ind.asset_id`, `ind.timestamp`, `ind.unit` | Stuck sensors, sampling irregularity, drift, unit changes, maintenance-event leakage | Time-ordered, asset-grouped |
