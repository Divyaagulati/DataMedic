"""Domain compatibility of recommended operations (standard library only).

Generic cleaning is not applied blindly to domain data. Each policy returns:
- ``status``: ``compatible`` or ``caution`` (shown with the reason, and lowers the rank);
- ``blocked``: columns the operation must not be recommended for (moved to "not
  recommended" with the reason).

Policies are keyed by declared roles and the dataset domain, never by column names.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from datamedic.engine.preprocess.recommend.facts import TIME_ROLES, Column, Facts


@dataclass
class DomainAssessment:
    status: str = "compatible"  # compatible | caution
    detail: str = "No domain-specific concern for this operation."
    blocked: dict[str, str] = field(default_factory=dict)  # column -> reason

    def caution(self, detail: str) -> None:
        self.status = "caution"
        self.detail = detail if self.detail.startswith("No domain") else f"{self.detail} {detail}"

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "detail": self.detail}


_BIO_CAP = (
    "Measurements with a clinical domain role keep their observed values: extreme laboratory "
    "or vital-sign values are often real and clinically important. Review them against "
    "reference ranges instead of capping."
)
_BIO_QUANTILE = (
    "A rank-based transformation discards the measurement scale that clinical and biological "
    "interpretation relies on (reference ranges, fold changes)."
)


def assess(facts: Facts, op: str, columns: Sequence[Column]) -> DomainAssessment:
    domain = facts.inp.domain
    out = DomainAssessment()
    family = op.split(".")[0]
    has_time = bool(facts.role_columns(TIME_ROLES))
    if domain == "biomedical":
        if op == "outliers.cap":
            for c in columns:
                if c.domain_roles:
                    out.blocked[c.name] = _BIO_CAP
            out.caution(
                "In biomedical data extreme values are often real (severe cases); capping can "
                "hide exactly the patients a model should learn about."
            )
        elif op == "transform.quantile":
            for c in columns:
                out.blocked[c.name] = _BIO_QUANTILE
        elif family == "select" and op != "select.correlation":
            out.caution(
                "Predictive selection is not biomarker validation: a column kept or dropped "
                "here says nothing about biological relevance. Columns with a domain role are "
                "always kept."
            )
        elif op == "select.correlation":
            out.caution(
                "Correlated biomarkers can reflect related biology rather than redundancy; "
                "both may matter scientifically. This step only proposes."
            )
        elif family == "impute" and any(c.domain_roles for c in columns):
            out.caution(
                "Missing clinical measurements are often not random (a test was not ordered "
                "because it was not indicated). Keep a missing-value indicator and review why "
                "values are missing."
            )
        elif op in ("transform.log", "transform.log1p"):
            out.detail = (
                "Log scales are standard for many laboratory measurements (ratios and fold "
                "changes stay interpretable)."
            )
        elif op == "imbalance.smote":
            out.caution(
                "Synthetic patients are not real observations; report this when publishing."
            )
    elif domain == "financial":
        if op == "outliers.cap":
            out.caution(
                "Extreme amounts can be the signal (fraud, large clients); capping can remove it."
            )
        elif family == "imbalance" and has_time:
            out.caution(
                "Financial labels are usually time-ordered; resampling happens only inside "
                "training folds of a time split."
            )
    elif domain == "industrial":
        if op == "outliers.cap":
            out.caution(
                "Extreme sensor readings may indicate faults or process changes; capping hides "
                "them."
            )
        elif family == "impute" and has_time:
            out.caution(
                "Sensor gaps in time-ordered data: imputing from other rows ignores time order "
                "(interpolation along time is not available yet)."
            )
    return out
