"""Biomedical (clinical tabular) module: roles and check specs. Standard library only."""

from datamedic.engine.checks.spec import (
    NUMERIC_TYPES,
    CheckSpec,
    DomainSpec,
    Requirements,
    RoleSpec,
)

KEY = "biomedical"
_M = "datamedic.domains.biomedical.checks"

# Physiologically implausible bounds for explicitly declared clinical variables (adult and
# paediatric). Wide on purpose: values outside are near-certainly unit or entry errors. Units
# are assumed as stated; ranges are not reference intervals.
PLAUSIBLE_RANGES: dict[str, tuple[float, float, str]] = {
    "bio.age": (0.0, 120.0, "years"),
    "bio.heart_rate": (20.0, 300.0, "beats per minute"),
    "bio.systolic_bp": (40.0, 300.0, "mmHg"),
    "bio.diastolic_bp": (20.0, 200.0, "mmHg"),
    "bio.temperature_c": (25.0, 45.0, "degrees Celsius"),
    "bio.bmi": (8.0, 100.0, "kg/m2"),
}

ROLES = (
    RoleSpec(
        "bio.patient_id",
        "Patient ID",
        "Identifies the patient or subject; several rows may belong to one patient.",
        KEY,
        max_per_version=1,
        is_feature=False,
        is_sensitive=True,
    ),
    RoleSpec(
        "bio.sample_id",
        "Sample ID",
        "Identifies a specimen or sample.",
        KEY,
        is_feature=False,
        is_sensitive=True,
    ),
    RoleSpec(
        "bio.batch",
        "Batch / site",
        "Site, centre, plate, run or scanner.",
        KEY,
        allowed_types=frozenset({"categorical", "numeric", "boolean", "text"}),
        is_feature=False,
    ),
    RoleSpec(
        "bio.condition", "Condition", "Biological condition or covariate (e.g. treatment arm).", KEY
    ),
    RoleSpec(
        "bio.biomarker_id",
        "Biomarker identifier",
        "Gene, protein or analyte identifiers (e.g. HGNC symbols).",
        KEY,
        allowed_types=frozenset({"categorical", "text", "identifier"}),
        is_feature=False,
    ),
    RoleSpec("bio.sex", "Sex", "Recorded sex.", KEY, is_sensitive=True),
    RoleSpec(
        "bio.age",
        "Age (years)",
        "Age in years.",
        KEY,
        allowed_types=NUMERIC_TYPES,
        max_per_version=1,
        is_sensitive=True,
    ),
    RoleSpec(
        "bio.heart_rate", "Heart rate (bpm)", "Beats per minute.", KEY, allowed_types=NUMERIC_TYPES
    ),
    RoleSpec(
        "bio.systolic_bp",
        "Systolic BP (mmHg)",
        "Systolic blood pressure.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
    RoleSpec(
        "bio.diastolic_bp",
        "Diastolic BP (mmHg)",
        "Diastolic blood pressure.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
    RoleSpec(
        "bio.temperature_c",
        "Body temperature (°C)",
        "Degrees Celsius.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
    RoleSpec("bio.bmi", "BMI (kg/m²)", "Body-mass index.", KEY, allowed_types=NUMERIC_TYPES),
)

PATIENT = frozenset({"bio.patient_id"})
BATCH = frozenset({"bio.batch"})


def _spec(
    name: str,
    category: str,
    title: str,
    description: str,
    func: str,
    *,
    requirements: Requirements | None = None,
    limitations: tuple[str, ...] = (),
    expensive: bool = False,
) -> CheckSpec:
    return CheckSpec(
        id=f"bio.{name}",
        version="1.0.0",
        domain=KEY,
        category=category,
        title=title,
        description=description,
        impl=f"{_M}:{func}",
        requirements=requirements or Requirements(),
        limitations=limitations,
        expensive=expensive,
    )


CHECKS = (
    _spec(
        "patients.repeated_rows",
        "duplicates",
        "Several rows per patient",
        "Distribution of rows per patient and exact duplicate rows within a patient.",
        "repeated_rows",
        requirements=Requirements(roles=PATIENT),
        limitations=("Repeated rows can be legitimate visits or samples.",),
    ),
    _spec(
        "leakage.patient_level",
        "leakage",
        "Patient-level leakage risk",
        "Patients with several rows would end up in both training and test data under a "
        "row-level split; also patients whose target differs between rows.",
        "patient_leakage",
        requirements=Requirements(roles=PATIENT | {"target"}),
        limitations=(
            "A risk, not a measured leak: it materialises only if evaluation splits "
            "rows rather than patients.",
        ),
    ),
    _spec(
        "biomarkers.id_consistency",
        "consistency",
        "Biomarker identifier consistency",
        "Identifiers that differ only in case or spacing (e.g. 'TP53' vs 'tp53 '), and "
        "mixtures of identifier systems (Ensembl IDs, numeric Entrez IDs, symbols).",
        "biomarker_ids",
        requirements=Requirements(roles=frozenset({"bio.biomarker_id"})),
        limitations=("Format-based: aliases and deprecated symbols are not resolved.",),
    ),
    _spec(
        "dimensionality.hdlss",
        "dimensionality",
        "Many features, few samples",
        "Numeric features compared with the number of rows (or patients). Severity: more "
        "features than samples high; more than one feature per 10 samples medium.",
        "high_dimensional",
        limitations=(
            "A rule of thumb; the effective sample size depends on the outcome and the model.",
        ),
    ),
    _spec(
        "batch.effect",
        "batch",
        "Feature differences between batches",
        "Per numeric feature, Kruskal-Wallis test across batches (batches with >= 3 rows), "
        "Benjamini-Hochberg FDR < 0.05; detected when >= 20% of features differ with "
        "epsilon-squared >= 0.06.",
        "batch_effect",
        requirements=Requirements(roles=BATCH, min_rows=10),
        expensive=True,
        limitations=(
            "Differences between batches can be biological (different patients "
            "per site) or technical; without a balanced design this check cannot "
            "separate them. Not a proof of a batch effect.",
        ),
    ),
    _spec(
        "batch.confounding",
        "confounding",
        "Batch associated with the outcome or condition",
        "Cramer's V (classification target, condition) or Kruskal-Wallis epsilon-squared "
        "(regression target) between batch and target/condition. Severity: >= 0.5 high, "
        ">= 0.3 medium, >= 0.1 low (p < 0.01).",
        "batch_confounding",
        requirements=Requirements(roles=BATCH, min_rows=10),
        limitations=(
            "Association is not causation. A confounded design cannot be fixed by "
            "batch correction alone.",
        ),
    ),
    _spec(
        "clinical.plausibility",
        "validity",
        "Physiologically implausible values",
        "Values outside wide physiological bounds for declared clinical variables: age "
        "0-120 years, heart rate 20-300 bpm, systolic BP 40-300 mmHg, diastolic BP 20-200 "
        "mmHg, temperature 25-45 °C, BMI 8-100.",
        "clinical_plausibility",
        requirements=Requirements(roles_any=frozenset(PLAUSIBLE_RANGES)),
        limitations=(
            "Units are assumed (see each role). The bounds catch impossible "
            "values, not abnormal ones.",
        ),
    ),
)

MODULE = DomainSpec(
    key=KEY,
    name="Biomedical (clinical)",
    version="1.0.0",
    description="Clinical tabular data: one row per patient, visit or sample.",
    roles=ROLES,
    checks=CHECKS,
)
