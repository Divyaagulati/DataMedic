"""Build the DataMedic presentation dataset from the original UCI archive (reproducible).

Source: Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository,
https://doi.org/10.24432/C5230J (CC BY 4.0). Strack B. et al., "Impact of HbA1c Measurement on
Hospital Readmission Rates", BioMed Research International, 2014.

Usage:
  python -I demo/build_presentation_dataset.py <diabetes_zip> <output_dir> [--target-encounters 4000]

Every transformation is listed in the generated provenance JSON. No value is invented: missing
markers become empty cells, coded IDs are replaced by their published descriptions from
IDS_mapping.csv, and the documented "None" (test not performed) becomes "Not measured".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261127
# Discharge dispositions after which readmission is impossible (death or hospice); excluded as in
# Strack et al. (2014): 11 Expired, 13/14 Hospice, 19/20/21 Expired (home / medical facility / unknown).
EXCLUDED_DISCHARGE = {"11", "13", "14", "19", "20", "21"}
# Codes whose published description means "missing" (IDS_mapping.csv).
MISSING_DESCRIPTIONS = {"NULL", "Not Available", "Not Mapped", "Unknown/Invalid"}
ID_TABLES = {  # source column -> renamed label column
    "admission_type_id": "admission_type",
    "discharge_disposition_id": "discharge_disposition",
    "admission_source_id": "admission_source",
}


def read_mapping(text: str) -> dict[str, dict[str, str]]:
    """IDS_mapping.csv stacks three tables separated by blank lines, each with its own header."""
    tables: dict[str, dict[str, str]] = {}
    current: str | None = None
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(cell.strip() for cell in row):
            current = None
            continue
        if current is None:
            current = row[0].strip()
            tables[current] = {}
            continue
        tables[current][row[0].strip()] = ",".join(row[1:]).strip()
    return tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out_dir")
    ap.add_argument("--target-encounters", type=int, default=4000)
    a = ap.parse_args()
    archive = Path(a.archive)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    with zipfile.ZipFile(archive) as z:
        raw = pd.read_csv(z.open("diabetic_data.csv"), dtype=str, keep_default_na=False)
        mapping = read_mapping(z.read("IDS_mapping.csv").decode("utf-8"))
    log: list[str] = [f"read diabetic_data.csv: {len(raw)} encounters, {raw.patient_nbr.nunique()} patients, {raw.shape[1]} columns"]

    # 1. Cohort: exclude encounters ending in death or hospice (readmission impossible).
    eligible = raw[~raw.discharge_disposition_id.isin(EXCLUDED_DISCHARGE)].copy()
    log.append(f"excluded {len(raw) - len(eligible)} encounters discharged to death/hospice "
               f"(discharge_disposition_id in {sorted(EXCLUDED_DISCHARGE, key=int)})")

    # 2. Patient-level sampling: shuffle patients with a fixed seed and keep ALL eligible encounters of
    #    each selected patient until the target size is reached (repeat encounters preserved).
    patients = np.array(sorted(eligible.patient_nbr.unique(), key=int))
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    counts = eligible.patient_nbr.value_counts()
    cum = np.cumsum(counts.reindex(patients).to_numpy())
    n_take = int(np.searchsorted(cum, a.target_encounters) + 1)
    chosen = set(patients[:n_take])
    df = eligible[eligible.patient_nbr.isin(chosen)].copy()
    # Keep the source order (encounter_id ascending as published); no re-ordering by outcome.
    df = df.sort_values("encounter_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    log.append(f"patient-level sample (seed {SEED}): {len(chosen)} patients, {len(df)} encounters")

    # 3. Target: readmitted within 30 days of discharge (binary), from the original 3-level label.
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    original_label_counts = df["readmitted"].value_counts().to_dict()
    df = df.drop(columns=["readmitted"])
    log.append(f"target readmitted_30d = 1 if readmitted == '<30' else 0 (from {original_label_counts}); "
               "original 'readmitted' column removed (target-derived)")

    # 4. Missing-value markers -> empty cells (documented markers only).
    q = (df == "?").sum()
    q = {k: int(v) for k, v in q.items() if v}
    df = df.replace("?", pd.NA)
    log.append(f"'?' -> missing: {q}")
    unk = int((df.gender == "Unknown/Invalid").sum())
    df.loc[df.gender == "Unknown/Invalid", "gender"] = pd.NA
    log.append(f"gender 'Unknown/Invalid' -> missing: {unk}")

    # 5. Coded IDs -> published descriptions; descriptions that mean "missing" -> empty.
    for src_col, new_col in ID_TABLES.items():
        table = mapping[src_col]
        labels = df[src_col].map(table)
        unmapped = int(labels.isna().sum())
        missing = int(labels.isin(MISSING_DESCRIPTIONS).sum())
        labels = labels.where(~labels.isin(MISSING_DESCRIPTIONS))
        df.insert(df.columns.get_loc(src_col), new_col, labels)
        df = df.drop(columns=[src_col])
        log.append(f"{src_col} -> {new_col} via IDS_mapping.csv ({missing} coded as missing, {unmapped} unmapped)")

    # 6. Lab result "None" means the test was not performed (dataset documentation).
    for col in ("max_glu_serum", "A1Cresult"):
        n = int((df[col] == "None").sum())
        df[col] = df[col].replace("None", "Not measured")
        log.append(f"{col}: 'None' (test not performed) -> 'Not measured' ({n})")

    csv_path = out / "diabetes_readmission_presentation.csv"
    df.to_csv(csv_path, index=False, na_rep="", lineterminator="\n")
    out_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    enc_per_patient = df.patient_nbr.value_counts()
    prov = {
        "source": {
            "name": "Diabetes 130-US Hospitals for Years 1999-2008",
            "doi": "https://doi.org/10.24432/C5230J",
            "licence": "CC BY 4.0",
            "citation": "Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c Measurement on Hospital "
                        "Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. "
                        "BioMed Research International, 2014:781670.",
            "archive_sha256": sha,
        },
        "seed": SEED,
        "target_encounters": a.target_encounters,
        "output": {"file": csv_path.name, "sha256": out_sha, "rows": len(df), "columns": df.shape[1],
                   "patients": int(df.patient_nbr.nunique()),
                   "patients_with_repeat_encounters": int((enc_per_patient > 1).sum()),
                   "max_encounters_per_patient": int(enc_per_patient.max()),
                   "readmitted_30d_rate": round(float(df.readmitted_30d.mean()), 4)},
        "prediction_point": "discharge (all retained predictors are recorded by the end of the encounter)",
        "transformations": log,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov["output"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
