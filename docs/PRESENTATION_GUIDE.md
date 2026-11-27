# DataMedic — Presentation Guide

A 10–15 minute live demonstration of DataMedic on one real public dataset, using one dedicated
account whose results were all computed in advance by the application itself. Every number in this
guide was produced by DataMedic on 2026-11-27 and can be re-checked with `demo/verify_presentation.py`.
One-page version: [PRESENTATION_CHEATSHEET.md](PRESENTATION_CHEATSHEET.md).

---

## 1. Access

| | |
|---|---|
| App (local, works offline) | <http://localhost:3000> |
| Account | `datamedic.demo@example.com` (display name **DataMedic Presentation**) |
| Password | in `C:\Users\Delll\DataMedic-demo\credentials\presentation-account.env` (only your Windows user can read it; never in Git) |
| Dataset | **Diabetes 130-US Hospitals - 30-day readmission (UCI)** |
| Dashboard | **Hospital Readmission Intelligence** |

All demo artifacts belong to this account only. A second account sees none of them (verified: 404).

Before the talk (2 minutes):

```powershell
cd C:\Users\Delll\Downloads\DataMedic
docker compose up -d          # no --build needed; images are already built
# wait ~30 s, then check everything is in place:
backend\.venv\Scripts\python -I demo\verify_presentation.py --web http://localhost:3000 --api http://localhost:8000 `
  --credentials C:\Users\Delll\DataMedic-demo\credentials\presentation-account.env `
  --state C:\Users\Delll\DataMedic-demo\demo_state_local.json
# expected: "40/40 checks passed"
```

Log in once before you start, open the dataset, and keep the dashboard in a second tab.

---

## 2. The dataset and the real-world problem

**Problem.** Unplanned readmission within 30 days of discharge is costly for hospitals and harmful
for patients; in the US it is a penalised quality measure. Predicting it at discharge would let a
care team target follow-up. It is known to be hard: published models on this exact dataset reach
only modest ROC-AUC (around 0.6–0.7).

**Source.** *Diabetes 130-US Hospitals for Years 1999–2008*, UCI Machine Learning Repository,
<https://doi.org/10.24432/C5230J>, licence **CC BY 4.0**. Strack B. et al., *Impact of HbA1c
Measurement on Hospital Readmission Rates*, BioMed Research International, 2014. Public,
de-identified research data: 101,766 inpatient encounters of 71,518 diabetic patients in 130 US
hospitals. No confidential or private patient data is used.

**Presentation subset** (built by `demo/build_presentation_dataset.py`, deterministic):

| Step | What and why |
|---|---|
| Cohort | Excluded 2,423 encounters discharged to death or hospice (codes 11, 13, 14, 19, 20, 21): readmission is impossible for them (as in Strack et al.). |
| Sampling | **Patient-level**: patients shuffled with seed `20261127`; *all* eligible encounters of each selected patient kept until 4,000 encounters. Repeat visits stay intact; natural distributions, skew, missingness and imbalance are preserved. |
| Target | `readmitted_30d` = 1 if the original `readmitted` is `<30`, else 0 (`>30` and `NO`). The original `readmitted` column is **removed** (target-derived). |
| Missing markers | `?` → empty cell (race, weight, payer code, medical specialty, diagnoses). Admission type / discharge disposition / admission source codes replaced by their published labels from `IDS_mapping.csv`; labels meaning missing (`NULL`, `Not Available`, `Not Mapped`, `Unknown/Invalid`) → empty. |
| Lab results | `max_glu_serum`, `A1Cresult`: the documented `None` (test not performed) → `Not measured` (a real category, not missing). |
| Nothing invented | No values, outliers, timestamps, hospital IDs or patient attributes were added. |

Result: **4,000 encounters × 50 columns, 2,864 patients**, 652 with repeat encounters (up to 13;
mean 1.4 per patient), **30-day readmission 449 (11.2%)**, i.e. 7.9 : 1 imbalance. 250 patients
have encounters with different outcomes. Source archive SHA-256 `f82ac129…453112a`; subset SHA-256
`4c082a1c…3546e9` (`provenance.json` next to the CSV records every count).

**Prediction point: discharge.** All retained predictors (demographics, admission details, length
of stay, procedures, medications, diagnoses, lab results, prior-year visit counts) are recorded by
the end of the encounter. Excluded as predictors: the target, the patient number (group), the
encounter number (identifier).

**Roles declared** (Diagnostics → roles panel): `readmitted_30d` target (binary) · `patient_nbr` Patient ID
(groups rows; never a feature; protected) · `encounter_id` Row identifier (never a feature) ·
`gender` Sex (sensitive) · `race` Sensitive. Sensitive columns are protected everywhere: findings
never quote their values, Analytics can only count them, CSV exports drop them, and model error
slices withhold them. Age is a 10-year band, not a number, so it is a normal categorical column
(the numeric `bio.age` role would require inventing ages).

**Limitations to state.** US hospitals 1999–2008, diabetic encounters only; a 4,000-encounter
subset (wider uncertainty than the full data); no hospital/site identifier, no dates, no vital signs;
`weight` is 96% missing; diagnoses are raw ICD-9 codes. Results are research-use-only and not
clinical evidence.

---

## 3. Demonstration sequence (≈ 13 minutes)

Tabs along the top of the dataset: **Overview · Profile · Analytics Studio · Diagnostics ·
Biomedical · Experiment Lab · Models · History · Copilot · Passport & Export**.

### Step 1 — Overview and Profile (1.5 min)
*Screen:* Datasets → **Diabetes 130-US Hospitals…** → **Overview**, then **Profile**.
*Show:* 4,000 rows, version list (v0 original, v1 committed); in Profile the distributions,
missingness per column and type inference. (Roles are shown on the Diagnostics page, step 2.)
*Say:* "Every upload is parsed in a sandbox into an immutable version. Nothing is ever overwritten —
cleaning creates a new version with full lineage."

### Step 2 — Diagnostics (2 min)
*Screen:* **Diagnostics** (version v0 shows the raw data).
*Show and say (observed results):*
- **High:** `weight` 96.4% missing; **patient-level leakage risk** — "row-level splits would leak
  patients between training and test".
- **Medium:** `medical_specialty` 49.5% and `payer_code` 38.5% missing; extreme values in
  `num_medications` (50), `number_inpatient` (41), `number_diagnoses` (47); 7 constant medication
  columns (e.g. `examide`, `citoglipton`).
- Low/info: skewed utilisation counts, near-constant drugs, 11.2% minority class, placeholder text
  in diagnosis codes, correlation and duplicate checks **not detected**.
- **Not assessed** (point at these): batch effect, batch confounding, clinical plausibility,
  biomarker IDs — "the dataset has no site, vital-sign or biomarker columns, so DataMedic says *not
  assessed* instead of pretending they passed."
*Say:* "Every finding has evidence, severity and limitations. There is deliberately no single
quality score."

### Step 3 — Biomedical Intelligence (1.5 min)
*Screen:* **Biomedical**.
*Show:* cohort (2,864 patients, 652 repeated, up to 13 encounters), target distribution (3,551 / 449),
**patient-level comparisons** with effect sizes and Benjamini–Hochberg q-values.
*Observed:* readmitted patients had more prior-year inpatient visits (Hedges' g 0.60, q = 0.004),
more diagnoses (g 0.37, q < 0.001) and more outpatient visits (g 0.36, q < 0.001); length of stay
and lab procedures differ little (q > 0.18).
*Say:* "Comparisons are per patient, not per row, so repeat visitors don't count several times.
Groups under 10 are suppressed. This is descriptive and research-use-only: associations, not causes."

### Step 4 — Recommendations and Experiment Lab (2 min)
*Screen:* **Experiment Lab** → the experiment *Recommended preprocessing strategies (patient-grouped
5-fold)*.
*Show:* the 26 evidence-based recommendations (missing indicators, constant/mode imputation,
one-hot/frequency encoding, robust scaling, outlier flag/cap, log/Yeo–Johnson transforms, variance
filter for constant columns, class weights/oversampling) and the 5 candidates scored on **one frozen
patient-grouped split** (same folds for all).
*Observed (mean ± SD over 5 folds):*

| Candidate | Logistic ROC-AUC | Logistic recall | Brier |
|---|---|---|---|
| Baseline: no preprocessing | 0.579 ± 0.031 | 0.009 | 0.102 |
| Recommended core | 0.589 ± 0.029 | 0.485 | 0.229 |
| Core + random oversampling | 0.575 ± 0.028 | 0.436 | 0.230 |
| Core without missing indicators | 0.589 ± 0.029 | 0.488 | 0.229 |
| Core + robust scaling | 0.590 ± 0.028 | 0.481 | 0.229 |

*Say:* "The preprocessing mainly trades calibration for recall; ROC-AUC differences are inside
one standard deviation, and the Lab warns about multiple comparisons. So there is **no winner** —
DataMedic shows 'highest observed', never 'best'. Experiments never write versions."

### Step 5 — Controlled preprocessing (1.5 min) — **live action**
*Screen:* **History** → the committed pipeline *Explicit missingness (indicators + 'Missing'
category)*: impact preview, approval history, v0 → v1 comparison (adds `weight__missing`,
`payer_code__missing`, `medical_specialty__missing`), and **Code** (`pipeline.py`).
*Live:* open the draft **LIVE DEMO - cap extreme medication counts (not approved)**. Show the impact
preview (**Recompute preview** if asked). It is **high risk**: committing requires an explicit,
hash-bound approval. *Do not press* **Approve and commit** (that would create v2 and change the
demo state); explain instead: "The approval is tied to this exact preview; if anything changes,
it expires. Encoding, scaling, selection and resampling are model-stage and never committed —
they run inside training folds only."

### Step 6 — Models and reliability audit (2 min)
*Screen:* **Models** → *Readmission models: baseline vs advanced (patient-grouped, nested tuning)*.
*Observed (5-fold grouped by patient, nested tuning, leakage-controlled):*

| Model | ROC-AUC | PR-AUC (prevalence 0.112) | Brier | Recall @0.5 |
|---|---|---|---|---|
| Dummy baseline | 0.500 | 0.112 | 0.100 | 0.00 |
| Logistic regression | 0.623 ± 0.021 | 0.176 | 0.098 | 0.00 |
| Random forest | **0.631 ± 0.034** (highest observed) | 0.185 | 0.098 | 0.00 |
| Histogram gradient boosting | 0.594 ± 0.024 | 0.163 | 0.100 | 0.00 |

Show the ROC/PR curves, calibration, confusion matrix ([[3551, 0], [449, 0]] for the forest),
permutation importance and error slices. Then open *Same models with class weights…*:

| Model (class weights in folds) | ROC-AUC | Balanced acc. | Recall | ECE |
|---|---|---|---|---|
| Logistic regression | 0.619 ± 0.020 | 0.572 | 0.508 | 0.357 |
| Random forest | 0.631 ± 0.021 | 0.563 | 0.372 | 0.324 |
| Histogram gradient boosting | 0.576 ± 0.016 | 0.539 | 0.270 | 0.244 |

*Say:* "Models rank patients better than chance but weakly — consistent with the literature. At
the default threshold, the imbalanced models predict nobody as readmitted; class weights recover
recall at the cost of calibration, and the audit flags exactly that."
*Audit* (open **Reliability audit**): main run status *no issues detected among assessed checks*,
with two medium findings: overfitting (gradient boosting train/test ROC-AUC gap 0.35) and sample
size (~9 events per feature); temporal checks *not assessed* (no dates). Class-weighted run:
*concerns* (calibration). Patient leakage across folds: checked, not detected.
*Importance:* prior-year inpatient visits and number of diagnoses lead for logistic regression and
gradient boosting (permutation importance); the forest's impurity importance favours
high-cardinality diagnosis codes — a known bias of impurity importance.

### Step 7 — Analytics Studio dashboard (1.5 min) — **live action**
*Screen:* **Analytics Studio** → dashboard **Hospital Readmission Intelligence** (bound to v1).
Widgets: Total hospital encounters (4,000) · 30-day readmission rate (0.112) · Average length of
stay (4.31 days) · readmission outcome donut · average medications by age group (peak 17.1 at
60–70) · lab procedure histogram · **readmission rate by prior-year inpatient visits** (0: 8.9%,
1: 14.0%, 2: 17.0%, 3: 17.9%, 4: 25.9%, 5: 38.9% of 18 encounters) · readmission by HbA1c result
(9.7–13.0%: no clear gradient) · encounters by admission type and outcome · length of stay by age.
*Live:* **Add filter** → `admission_type` equals `Emergency` → every widget updates (2,179
encounters, 11.9% readmitted). Press **Save filters** only if you want to keep it; otherwise reload.
*Say:* "Queries are structured specifications, never SQL. Counts are encounters, not patients —
unique patients are on the Biomedical page because the patient ID is protected and can only be
counted. Groups under 10 rows are suppressed (age bands under 20)."

### Step 8 — Copilot (1.5 min) — **live action**
*Screen:* **Copilot**. Ask (verified to work):
1. *What are the most important data quality issues?*
2. *What is the average time in hospital by age group?* → highest 4.88 days for 90–100.
3. *Which variables have the most missing values?* → weight 96.38%, then discharge disposition.
4. *What are the major reliability concerns?*
5. *Which features were most informative in the trained models?*
6. (spare) *What are the most important insights about my dataset?* · *Which columns are sensitive?*
   · *Which admission type has the highest average time in hospital?* (Urgent, 4.60 days)
*Say:* "The Copilot is **rule-based and deterministic**: it turns the question into a validated
query and every number comes from the engine. No language model, no data leaves the server."
*Phrase with column words:* "hospital stay" is not recognised (it suggests "time in hospital");
"readmission rate" is ambiguous — say "average readmitted 30d by A1Cresult".

### Step 9 — Data Passport and exports (1 min) — **live action**
*Screen:* **Passport & Export**: quality, findings, recommendations, history with hashes,
experiments, models, audits, explainability, insights, dashboards, limitations. **Print / PDF**
for the report; JSON report and `pipeline.py` downloads; **Download CSV** (requires an explicit
acknowledgement; patient number, race and gender are dropped; spreadsheet formulas neutralised).
*Say:* "The passport only reads stored results — it never recomputes or changes anything."

---

## 4. Architecture (1 minute, if asked)

Browser → Next.js 16 web app → FastAPI API (session cookie, CSRF, per-request PostgreSQL
Row-Level Security) → PostgreSQL 16; long jobs go through Redis to Celery workers (ingest sandbox;
analysis: profiling, diagnostics, Lab, training, audits, insights, exports) that read/write a
private storage volume. Broker messages carry only a job ID. The analysis engine is pure Python
(pandas, NumPy, SciPy, scikit-learn, imbalanced-learn) behind ports; domain modules (biomedical,
financial, business, industrial) plug in through a registry. Production adds Caddy for HTTPS. Every
user-owned row has an owner and is protected twice: service-layer scoping and database RLS.

## 5. What this dataset cannot demonstrate meaningfully

- **Batch/site effects and confounding:** UCI publishes no hospital identifier → *not assessed*.
- **Temporal validation and post-outcome leakage:** no dates → *not assessed*; folds are grouped,
  not time-ordered.
- **Clinical plausibility checks:** need vital-sign roles (heart rate, BP, temperature, BMI) — absent.
- **Biomarker identifier checks / omics:** no biomarker columns.
- **Unique-patient KPI in Analytics Studio:** the patient ID is protected (count-only); use the
  Biomedical page.
- **Regression, multiclass, clustering:** the task here is binary.
- **SMOTE:** offered, but requires acknowledging that synthetic rows blend different patients;
  class weights are the appropriate choice here.
- **Revert:** supported (History), but not demonstrated live so the demo state stays consistent.

Note: the Models list also shows one **failed** class-weighted run. It failed because of a genuine
bug found during preparation (target-aware preprocessing steps were rejected on re-validation in the
worker); the bug was fixed, covered by a regression test, and the run was repeated successfully.
The failed record is kept because training history is never rewritten.

## 6. Likely examiner questions

**Is the evaluation really leakage-free?** It is *leakage-controlled*: folds group all encounters
of a patient, every learned step (imputation, encoding, scaling, weighting) is fitted on training
rows only, tuning uses inner folds inside the outer training rows, and the audit verifies these
boundaries per fold. It cannot see processing done before upload, and it is not external validation.

**Why is ROC-AUC only ~0.63?** Readmission depends on post-discharge factors not in the data
(social support, follow-up care). Published work on the full 100k data reports similar values;
we used 4,000 encounters. DataMedic shows the honest number with fold uncertainty and the dummy baseline.

**Why does accuracy look high (≈ 89%) but recall is zero?** 88.8% of encounters are not readmitted,
so predicting "no" for everyone is already 88.8% accurate. That is why the primary metric is
ROC-AUC and the audit compares with the dummy baseline.

**Did you clean the data before uploading?** Only documented, reversible normalisation: missing
markers to empty cells, code IDs to published labels, "None" to "Not measured", a binary target,
and the death/hospice cohort exclusion. Everything else happened inside DataMedic with lineage.

**Why are race and gender used as model inputs?** They are declared sensitive, so their values
are protected in outputs, but they remain inputs by default; excluding them is one role change
(`exclude`). Including protected attributes in clinical models is a fairness decision the user
must make explicitly; DataMedic does not make it silently.

**How do you prevent one user seeing another's data?** Service-layer owner scoping plus PostgreSQL
Row-Level Security on every table; another user's resource returns 404 (verified for this demo);
an automatically generated authorisation matrix tests every route.

**Is the Copilot an LLM?** No. It is deterministic and rule-based; it never sends data anywhere.

**Can it be used clinically?** No. Research use only; descriptive statistics, not clinical evidence.

## 7. Startup, offline fallback and re-provisioning

- **Offline:** after `docker compose up -d`, nothing needs the internet (images, fonts and the
  Mailpit mail catcher are local). Mailpit (<http://localhost:8025>) shows verification emails.
- **If the stack misbehaves:** `docker compose restart` (keeps all data). Never use `down -v`.
- **Persistence:** verified — all checks pass again after a full restart.
- **Cloud (Oracle or any other server):** deploy per `docs/DEPLOYMENT.md`, then run
  `demo/provision_presentation.py` against `https://<domain>` with a credentials file whose email
  you can receive (see `demo/README.md`). It rebuilds the same account and artifacts from the public
  dataset through the API — no development database or credentials are copied.
