"""Reliability-audit check catalogue and the evaluation-status rule (standard library only).

Every check states its *basis*:

- ``confirmed``: a fact computed from data, folds or lineage (e.g. a group on both sides);
- ``configuration``: a fact about the chosen setup (e.g. a declared time role was ignored);
- ``statistical``: a suspicion from a threshold on an estimate. Statistical detections are
  titled "possible" / "suspected" and are never proof (a strong predictor is not proof of
  leakage; a very high score is not proof of leakage).

Unassessable checks report ``not_assessed`` with the specific reason; they never pass.
"""

from dataclasses import dataclass

AUDIT_VERSION = "1.1.0"
AUDIT_DOMAIN = "reliability"
BASES = ("confirmed", "configuration", "statistical")


@dataclass(frozen=True)
class AuditSpec:
    id: str
    category: str  # leakage | reliability
    basis: str
    title: str
    method: str
    limitations: tuple[str, ...]
    version: str = AUDIT_VERSION
    domain: str = AUDIT_DOMAIN


def _s(id_: str, category: str, basis: str, title: str, method: str, *limits: str) -> AuditSpec:
    return AuditSpec(id_, category, basis, title, method, tuple(limits))


AUDIT_CHECKS: tuple[AuditSpec, ...] = (
    _s(
        "leakage.target.single_feature",
        "leakage",
        "statistical",
        "Single features that almost predict the target",
        "Each original feature alone, as an out-of-fold depth-3 decision tree on the frozen "
        "outer folds: flagged when ROC-AUC (binary), balanced accuracy (multiclass) >= 0.98 or "
        "R² (regression) >= 0.95, or when one feature reaches 95% of the best model's score.",
        "A strong predictor is not proof of leakage: confirm the feature is available at "
        "prediction time and recorded before the outcome.",
        "At most 200 features (by variance beyond that) and 50,000 rows (seeded sample) are "
        "assessed; the coverage is reported.",
    ),
    _s(
        "leakage.target.derived",
        "leakage",
        "statistical",
        "Features that look derived from the target",
        "Near-deterministic relations: categories that each map to one class (>= 99% purity, "
        "far fewer categories than rows), numeric features constant within each class, or "
        "|Spearman| >= 0.999 with a numeric target.",
        "A deterministic relation can be genuine (e.g. a diagnosis code that defines the "
        "outcome); it is reported as a suspicion, not a confirmed leak.",
    ),
    _s(
        "leakage.identifier",
        "leakage",
        "statistical",
        "Identifier-like features carrying target signal",
        "Model inputs that look like identifiers (>= 95% distinct values, integers or codes) "
        "and predict the target alone (out-of-fold score) or are rank-associated with it "
        "(|Spearman| >= 0.3, p < 0.001), suggesting data sorted or numbered by outcome.",
        "Row order is not treated as time; an association can arise from how records were "
        "numbered.",
    ),
    _s(
        "leakage.duplicates.cross_fold",
        "leakage",
        "confirmed",
        "Duplicate rows across training and test folds",
        "Hashes of every row (features + target, features only, and a normalised form: text "
        "lower-cased and trimmed, numbers rounded to 6 significant digits) compared between "
        "the training and test rows of every outer fold.",
        "Near-duplicates are detected only up to the stated normalisation; rows that differ "
        "slightly in other ways are not found.",
    ),
    _s(
        "leakage.groups.cross_fold",
        "leakage",
        "confirmed",
        "The same entity in training and test folds",
        "Values of the declared group (entity) role present in both the training and test "
        "rows of any outer fold.",
        "Only declared entity roles are checked; undeclared repeated entities are assessed by "
        "'leakage.groups.undeclared'.",
    ),
    _s(
        "leakage.groups.undeclared",
        "leakage",
        "statistical",
        "Possibly repeated entities without a group role",
        "Without an entity role: an identifier column with repeated values (>= 1.2 rows per "
        "value), or a large share of rows with identical feature values (high); or a "
        "key-like feature (integer-valued or text, >= 20 distinct values and >= 5% of rows, "
        ">= 1.2 rows per value) (medium).",
        "Repeated values can be legitimate (e.g. a shared code); declare a group role when "
        "rows belong to the same entity.",
    ),
    _s(
        "leakage.temporal.ordering",
        "leakage",
        "configuration",
        "Time order of the evaluation",
        "With a declared time role: a non-time split is a configuration violation; with a time "
        "split, every training row must precede every test row in every fold.",
        "Without a time role, temporal leakage cannot be assessed.",
    ),
    _s(
        "leakage.temporal.post_outcome",
        "leakage",
        "configuration",
        "Features recorded after the outcome",
        "Needs outcome-time and feature-time roles, which only domain modules declare.",
        "Not assessable with universal roles in this version.",
    ),
    _s(
        "leakage.pipeline.fit_on_full_data",
        "leakage",
        "confirmed",
        "Preprocessing fitted on all rows",
        "Lineage inspection: an exploratory run on a version whose committed steps learned "
        "from all rows is a violation; a leakage-controlled replay evaluation records the eligible "
        "ancestor, the replayed step hashes and the fidelity proof.",
        "Covers committed DataMedic transformations only; processing done before upload is "
        "unknown to the platform.",
    ),
    _s(
        "leakage.pipeline.boundaries",
        "leakage",
        "confirmed",
        "Preprocessing, selection and resampling boundaries",
        "Runtime evidence per fold: every learned step was fitted on exactly the training "
        "rows, test rows were all scored and unchanged in number, no synthetic row reached the "
        "test rows, and sample weights were used for fitting only.",
        "Verifies the platform's own execution; it cannot see processing done before upload.",
    ),
    _s(
        "leakage.tuning.isolation",
        "leakage",
        "confirmed",
        "Model selection used training rows only",
        "Every inner fold lies inside the outer training rows, and settings were selected "
        "before the outer test rows were scored.",
        "Choosing among models on the outer results is a separate, disclosed optimism.",
    ),
    _s(
        "leakage.pipeline.target_encoding",
        "leakage",
        "configuration",
        "Target-derived preprocessing outside folds",
        "Inspection of the preprocessing specification and lineage for steps that read the "
        "target: allowed inside folds (training labels only); a violation when such a step was "
        "fitted on all rows of an exploratory source.",
        "Only DataMedic steps are inspected.",
    ),
    _s(
        "reliability.overfitting",
        "reliability",
        "statistical",
        "Gap between training and test performance",
        "Primary metric on the outer training rows versus the outer test rows per model: a gap "
        "above 0.15 (0.25 for random forests, whose training fit is near-perfect by design).",
        "Some gap is normal; a large gap suggests the model memorises training rows.",
    ),
    _s(
        "reliability.fold_variability",
        "reliability",
        "statistical",
        "Fold-to-fold variability",
        "Primary-metric standard deviation above 0.05, coefficient of variation above 0.2, or "
        "a fold range above 0.15.",
        "With few folds, the spread itself is uncertain.",
    ),
    _s(
        "reliability.class_imbalance",
        "reliability",
        "confirmed",
        "Class imbalance",
        "Share of the rarest class among cohort rows: below 10% (medium) or 2% (high); "
        "accuracy is compared with the majority-class rate.",
        "Imbalance is a data property, not a defect; it changes which metrics are informative.",
    ),
    _s(
        "reliability.sample_size",
        "reliability",
        "statistical",
        "Sample size",
        "Fewer than 30 test rows in a fold, fewer than 5 rows of some class in a test fold, "
        "fewer than 10 minority-class rows per feature (binary), or fewer than 10 rows per "
        "feature (regression).",
        "Rules of thumb; adequate size depends on the signal and the model.",
    ),
    _s(
        "reliability.suspicious_performance",
        "reliability",
        "statistical",
        "Suspiciously high performance",
        "Mean ROC-AUC, balanced accuracy or R² of 0.99 or more for a non-dummy model, "
        "together with any leakage detection above.",
        "A very high score is not proof of leakage; some problems are genuinely easy.",
    ),
    _s(
        "reliability.dummy_competitive",
        "reliability",
        "statistical",
        "Models no better than the dummy baseline",
        "The best non-dummy model's primary-metric difference from the dummy is negligible or "
        "within fold variation.",
        "The features may carry little signal for this target, or the models are unsuitable.",
    ),
    _s(
        "reliability.calibration",
        "reliability",
        "statistical",
        "Probability calibration",
        "Expected calibration error of pooled out-of-fold probabilities (binary: positive "
        "class; multiclass: top label): above 0.1 (medium) or 0.2 (high); Brier score versus "
        "the prevalence baseline.",
        "Resampling and class weights shift predicted probabilities by design; calibration is "
        "assessed on these folds only.",
    ),
    _s(
        "reliability.metadata",
        "reliability",
        "configuration",
        "Missing metadata limiting the assessment",
        "Lists missing or stale information that prevents checks: no entity role while "
        "repeated entities are suspected, date columns without a time role, diagnostics "
        "missing or older than the current roles.",
        "Declaring roles is the user's decision; DataMedic never assigns them automatically.",
    ),
)
AUDIT_IDS = tuple(s.id for s in AUDIT_CHECKS)

CRITICAL_VIOLATIONS = (
    "leakage.duplicates.cross_fold",
    "leakage.groups.cross_fold",
    "leakage.temporal.ordering",
    "leakage.pipeline.fit_on_full_data",
    "leakage.pipeline.boundaries",
    "leakage.tuning.isolation",
    "leakage.pipeline.target_encoding",
)
STATUS_RULES = [
    "blocked: a confirmed or configuration violation of critical severity was detected "
    "(for an exploratory run, other than the acknowledged full-data fit).",
    "exploratory: the run was explicitly exploratory (evaluated on a version fitted on all "
    "rows; reported as a critical violation); it is never reported as leakage-controlled.",
    "concerns: a statistical suspicion of high or critical severity was detected.",
    "no_issues_detected_among_assessed: none of the above; checks that could not be assessed "
    "are listed and are not counted as passed.",
]
STATUS_LIMITATION = (
    "This status summarises the checks that ran. It is not a certification of model "
    "validity; absence of a detection is not proof of absence."
)


def evaluation_status(findings: list[dict[str, object]], exploratory: bool) -> dict[str, object]:
    """Deterministic rule over audit findings (dicts with id/status/severity/basis)."""
    blocking = [
        f["check_id"]
        for f in findings
        if f["status"] == "detected"
        and f["severity"] == "critical"
        and f.get("basis") in ("confirmed", "configuration")
    ]
    concerns = [
        f["check_id"]
        for f in findings
        if f["status"] == "detected"
        and f["severity"] in ("high", "critical")
        and f.get("basis") == "statistical"
    ]
    # In an exploratory run the full-data fit is the acknowledged reason for the label; any
    # *other* critical violation still blocks.
    acknowledged = [b for b in blocking if exploratory and b == "leakage.pipeline.fit_on_full_data"]
    blocking = [b for b in blocking if b not in acknowledged]
    not_assessed = [f["check_id"] for f in findings if f["status"] == "not_assessed"]
    errors = [f["check_id"] for f in findings if f["status"] == "error"]
    if blocking:
        status = "blocked"
    elif exploratory:
        status = "exploratory"
    elif concerns:
        status = "concerns"
    else:
        status = "no_issues_detected_among_assessed"
    return {
        "status": status,
        "blocking": blocking,
        "concerns": concerns,
        "not_assessed": not_assessed,
        "errors": errors,
        "exploratory": exploratory,
        "rules": STATUS_RULES,
        "limitation": STATUS_LIMITATION,
    }


def catalogue() -> list[dict[str, object]]:
    return [
        {
            "id": s.id,
            "version": s.version,
            "category": s.category,
            "basis": s.basis,
            "title": s.title,
            "method": s.method,
            "limitations": list(s.limitations),
        }
        for s in AUDIT_CHECKS
    ]
