# DataMedic — Presentation Cheat Sheet (one page)

**Start:** `docker compose up -d` → <http://localhost:3000> → log in `datamedic.demo@example.com`
(password: `C:\Users\Delll\DataMedic-demo\credentials\presentation-account.env`) → dataset
**Diabetes 130-US Hospitals - 30-day readmission (UCI)**. Check: `demo\verify_presentation.py` → 40/40.

**Data:** UCI Diabetes 130-US Hospitals 1999–2008 (doi 10.24432/C5230J, CC BY 4.0). Patient-level
sample, seed 20261127: **4,000 encounters, 2,864 patients (652 repeat), 11.2% readmitted < 30 days**.
Death/hospice discharges excluded. Prediction at discharge. Roles: target `readmitted_30d`, patient
`patient_nbr`, id `encounter_id`, sensitive `gender`, `race`.

| # | Tab | Show (observed) | One-liner |
|---|---|---|---|
| 1 | Overview / Profile | v0 original → v1 committed; distributions | "Immutable versions, full lineage." |
| 2 | Diagnostics | High: weight 96% missing, patient-leakage risk; constant drugs; outliers; 4 checks *not assessed* | "Evidence, severity, limitations — no fake passes." |
| 3 | Biomedical | 2,864 patients; prior inpatient visits g = 0.60 (q = 0.004), diagnoses g = 0.37 | "Per-patient, FDR-adjusted, research use only." |
| 4 | Experiment Lab | 26 recommendations; 5 candidates, same folds; logistic ROC-AUC 0.58–0.59, recall 0.01 → 0.49 | "No winner: differences within 1 SD." |
| 5 | History / pipeline | Committed missingness pipeline + code; **live:** open *LIVE DEMO – cap…* preview, high risk, needs approval (don't commit) | "Approvals are hash-bound to the preview." |
| 6 | Models | Forest 0.631 ± 0.034, logistic 0.623, GBM 0.594, dummy 0.500; class weights: recall 0.51, ECE 0.36 → audit *concerns* | "Leakage-controlled; honest, modest signal." |
| 7 | Analytics Studio | Dashboard *Hospital Readmission Intelligence*; readmission 8.9% (0 prior inpatient) → 25.9% (4); **live:** Add filter `admission_type` = Emergency (2,179; 11.9%) | "Structured queries, small groups suppressed." |
| 8 | Copilot | Ask the 5 questions below | "Rule-based; numbers come from the engine." |
| 9 | Passport & Export | **Print / PDF**, JSON, `pipeline.py`, **Download CSV** (acknowledge; sensitive dropped) | "Reads stored results only." |

**Copilot (verified):**
1. What are the most important data quality issues?
2. What is the average time in hospital by age group?
3. Which variables have the most missing values?
4. What are the major reliability concerns?
5. Which features were most informative in the trained models?

Spare: *Which columns are sensitive?* · *Which admission type has the highest average time in hospital?*
Avoid: "hospital stay", "readmission rate" (say "time in hospital", "average readmitted 30d").

**Key answers:** accuracy ≈ 89% means nothing (88.8% are "no") → use ROC-AUC/PR-AUC vs dummy ·
grouped folds keep each patient on one side · preprocessing fitted in training folds only · no
LLM, no data leaves the server · race/gender are inputs but protected; excluding them is one role
change · not assessable here: site/batch, temporal, clinical plausibility, unique-patient KPI.

**If stuck:** `docker compose restart` (never `down -v`). Failed run in Models = bug found and
fixed during preparation (history is never rewritten).
