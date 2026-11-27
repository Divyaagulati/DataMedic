"""Universal roles and checks (apply to every dataset). Specs only: standard library."""

from datamedic.engine.checks.spec import (
    ANY_PRESENT,
    NUMERIC_TYPES,
    UNIVERSAL,
    CheckSpec,
    DomainSpec,
    Requirements,
    RoleSpec,
)

_M = "datamedic.engine.checks.universal"
CLASSIFICATION = frozenset({"binary", "multiclass"})
ANY_TASK = frozenset({"binary", "multiclass", "regression"})
TARGET = frozenset({"target"})

ROLES = (
    RoleSpec(
        "target",
        "Target",
        "The column to predict. Requires a task type (binary, multiclass or regression).",
        UNIVERSAL,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "id",
        "Row identifier",
        "Identifies a record; values should be unique.",
        UNIVERSAL,
        is_feature=False,
    ),
    RoleSpec(
        "group",
        "Group",
        "Rows sharing a value belong together (e.g. one entity).",
        UNIVERSAL,
        is_feature=False,
    ),
    RoleSpec(
        "time",
        "Time",
        "When the row was observed.",
        UNIVERSAL,
        allowed_types=frozenset({"datetime", "numeric"}),
        is_feature=False,
    ),
    RoleSpec(
        "weight",
        "Sample weight",
        "Per-row weight.",
        UNIVERSAL,
        allowed_types=NUMERIC_TYPES,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "exclude", "Exclude", "Ignored as a feature by all checks.", UNIVERSAL, is_feature=False
    ),
    RoleSpec(
        "sensitive",
        "Sensitive",
        "Personal or confidential: values are never quoted in findings.",
        UNIVERSAL,
        is_sensitive=True,
    ),
)


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
    module: str = "quality",
) -> CheckSpec:
    return CheckSpec(
        id=f"universal.{name}",
        version="1.0.0",
        domain=UNIVERSAL,
        category=category,
        title=title,
        description=description,
        impl=f"{_M}.{module}:{func}",
        requirements=requirements or Requirements(),
        limitations=limitations,
        expensive=expensive,
    )


CHECKS = (
    _spec(
        "missingness.column_rate",
        "missingness",
        "Missing values by column",
        "Share of missing values per column. Severity: >=50% high, >=20% medium, >=5% low, "
        "otherwise info. Entirely empty columns are high.",
        "missing_by_column",
        limitations=(
            "Only explicit missing values (and the parser's missing tokens) are "
            "counted; placeholders such as -999 are reported by the suspicious-values "
            "check.",
        ),
    ),
    _spec(
        "missingness.pattern",
        "missingness",
        "Missing-value patterns",
        "Columns that tend to be missing together (phi coefficient of missing indicators "
        ">= 0.5) and rows missing most of their values (>= 50% of columns).",
        "missing_patterns",
        limitations=(
            "Shows co-occurrence only; it cannot establish why values are missing "
            "(MCAR, MAR or MNAR).",
        ),
    ),
    _spec(
        "missingness.target_association",
        "missingness",
        "Missingness related to the target",
        "Whether rows with a missing value differ in target (Cramer's V of the missing "
        "indicator vs a classification target; Mann-Whitney test for regression). "
        "Informative missingness can leak information or bias models.",
        "missing_vs_target",
        requirements=Requirements(roles=TARGET, task_types=ANY_TASK, min_rows=20),
        limitations=(
            "Association only. A missing-value pattern linked to the outcome may be "
            "legitimate signal or a data-collection artefact.",
        ),
        module="relations",
    ),
    _spec(
        "duplicates.exact_rows",
        "duplicates",
        "Exact duplicate rows",
        "Rows identical in every column (the hidden row identifier is ignored). Severity: "
        ">=5% medium, otherwise low.",
        "exact_duplicates",
        limitations=(
            "Exact matching only: values differing by case, spacing or rounding are "
            "not duplicates here. Duplicates can be legitimate repeated "
            "measurements.",
        ),
    ),
    _spec(
        "duplicates.identifier",
        "duplicates",
        "Repeated identifiers",
        "Values that occur more than once in a column declared as a row identifier.",
        "identifier_duplicates",
        requirements=Requirements(roles=frozenset({"id"})),
        limitations=("Depends on the declared identifier really being meant to be unique.",),
    ),
    _spec(
        "outliers.univariate",
        "outliers",
        "Univariate outliers",
        "Per numeric column: Tukey fences (1.5 and 3 x IQR), z-score (|z| > 3) and robust "
        "modified z-score (|0.6745 (x - median) / MAD| > 3.5, Iglewicz and Hoaglin). Reported "
        "as detected when extreme values exist (beyond 3 x IQR or modified z > 3.5).",
        "univariate_outliers",
        limitations=(
            "An outlier is an unusual value, not necessarily an error. Columns with "
            "10 or fewer distinct values are skipped. Fences assume a roughly "
            "unimodal distribution; skewed columns produce more flags.",
        ),
        module="distribution",
    ),
    _spec(
        "outliers.multivariate",
        "outliers",
        "Unusual combinations of values",
        "Rows ranked in the top 1% by all three detectors (Isolation Forest, ECOD, COPOD) "
        "and with an Isolation Forest anomaly score >= 0.65. Needs >= 2 numeric columns "
        "and >= 100 rows; missing values are median-imputed for scoring.",
        "multivariate_outliers",
        requirements=Requirements(min_rows=100),
        limitations=(
            "Unsupervised screening: flagged rows are unusual, not wrong. The 1% "
            "share and the 0.65 threshold are assumptions. ECOD and COPOD are "
            "in-house implementations of the published methods.",
            "ECOD and COPOD score each feature separately, so a row that only breaks a "
            "relationship between features (each value ordinary) is not flagged; such rows "
            "are listed as 'Isolation Forest only' in the evidence for review.",
        ),
        expensive=True,
        module="distribution",
    ),
    _spec(
        "distribution.shape",
        "distribution",
        "Skewed or heavy-tailed distributions",
        "Numeric columns (more than 2 distinct values) with |skewness| >= 1, excess "
        "kurtosis >= 7 or >= 50% zeros.",
        "distribution_shape",
        limitations=(
            "Descriptive. Skewness is not an error; it can matter for models that "
            "assume symmetric or normal inputs.",
        ),
        module="distribution",
    ),
    _spec(
        "variance.constant",
        "variance",
        "Constant and near-constant columns",
        "Columns with one distinct value (constant) or whose most frequent value covers "
        ">= 95% of present values (near-constant).",
        "constant_columns",
        limitations=("Near-constant columns can still be informative for rare events.",),
    ),
    _spec(
        "cardinality.high",
        "cardinality",
        "High-cardinality columns",
        "Categorical or text features with more than 50 distinct values and more than 50% "
        "distinct values, and identifier-like columns used as features.",
        "high_cardinality",
        limitations=("Free text and identifiers look alike; the role you declare decides.",),
    ),
    _spec(
        "correlation.redundancy",
        "redundancy",
        "Strongly correlated numeric features",
        "Pairs with |Pearson| or |Spearman| >= 0.9 and variance inflation factors (VIF >= 10) "
        "from the inverse correlation matrix.",
        "numeric_redundancy",
        limitations=(
            "Correlation is not causation. VIF is computed on pairwise-complete "
            "correlations and is unstable under perfect collinearity.",
        ),
        module="relations",
    ),
    _spec(
        "correlation.categorical",
        "redundancy",
        "Strongly associated categorical features",
        "Pairs of categorical features with Cramer's V >= 0.8 (bias-uncorrected, scipy).",
        "categorical_association",
        limitations=(
            "Association is not causation. Cramer's V is inflated for small samples "
            "with many categories.",
        ),
        expensive=True,
        module="relations",
    ),
    _spec(
        "target.imbalance",
        "target",
        "Class imbalance",
        "Minority class share of a classification target. Severity: < 1% high, < 10% "
        "medium, < 20% low.",
        "class_imbalance",
        requirements=Requirements(roles=TARGET, task_types=CLASSIFICATION),
        limitations=(
            "Imbalance is a property of the data, not an error; it affects metrics "
            "and cross-validation.",
        ),
        module="relations",
    ),
    _spec(
        "target.quality",
        "target",
        "Target completeness and usability",
        "Missing target values, a single class, classes with fewer than 10 rows, and a "
        "regression target that is constant.",
        "target_quality",
        requirements=Requirements(roles=TARGET, task_types=ANY_TASK),
        limitations=("Rows with a missing target cannot be used for supervised training.",),
        module="relations",
    ),
    _spec(
        "values.suspicious",
        "validity",
        "Suspicious and invalid values",
        "Placeholder numbers (-999, -99, 999, 9999, 99999, -9999) that make up >= 1% of a "
        "column and lie apart from its other values; placeholder text ('?', '-', 'unknown', "
        "'missing', 'none'...); infinite values; dates in the future or before 1900.",
        "suspicious_values",
        limitations=("Placeholders are suggestions: -999 can be a real value.",),
    ),
    _spec(
        "leakage.target_proxy",
        "leakage",
        "Features that almost determine the target",
        "Single features that separate the target nearly perfectly: ROC AUC >= 0.98 for a "
        "binary target, |Spearman| >= 0.98 for regression, or categories that each map to one "
        "class (>= 99% purity, categories much fewer than rows). Critical when perfect.",
        "target_proxy",
        requirements=Requirements(roles=TARGET, task_types=ANY_TASK, min_rows=20),
        limitations=(
            "A strong predictor is not proof of leakage: confirm the feature is "
            "available at prediction time and recorded before the outcome.",
        ),
        module="relations",
    ),
    _spec(
        "leakage.identifier",
        "leakage",
        "Identifiers related to the target",
        "Identifier-like columns (declared 'id' or inferred) whose order is associated with "
        "the target (|Spearman| >= 0.3, p < 0.001), which suggests the data was sorted or "
        "collected by outcome.",
        "identifier_leakage",
        requirements=Requirements(roles=TARGET, task_types=ANY_TASK, min_rows=30),
        limitations=(
            "Row order is not treated as time. The association can arise from how "
            "records were numbered.",
        ),
        module="relations",
    ),
)

MODULE = DomainSpec(
    key=UNIVERSAL,
    name="Universal",
    version="1.0.0",
    description="Checks that apply to every tabular dataset.",
    roles=ROLES,
    checks=CHECKS,
)

__all__ = ["ANY_PRESENT", "CHECKS", "MODULE", "ROLES"]
