"""The preprocessing op catalogue (standard library only).

Risk, destructive flag, kind and stage are defined here and nowhere else: callers (API,
workers, UI) read them from the catalogue and never accept them from a client.
Every op is documented in docs/ML_ENGINE_DESIGN.md §3.2.
"""

from datamedic.engine.preprocess.spec import OpSpec, ParamSpec

_IMPUTE = "datamedic.engine.preprocess.ops.impute"
_OUTLIERS = "datamedic.engine.preprocess.ops.outliers"
_ENCODE = "datamedic.engine.preprocess.ops.encode"
_SCALE = "datamedic.engine.preprocess.ops.scale"
_TRANSFORM = "datamedic.engine.preprocess.ops.transform"
_SELECT = "datamedic.engine.preprocess.ops.select"
_IMBALANCE = "datamedic.engine.preprocess.ops.imbalance"

NUMERIC = frozenset({"numeric"})
NUMERIC_CATEGORICAL_BOOLEAN = frozenset({"numeric", "categorical", "boolean"})
ANY_FEATURE = frozenset({"numeric", "boolean", "datetime", "categorical", "text"})
CATEGORICAL = frozenset({"categorical"})
CATEGORICAL_BOOLEAN = frozenset({"categorical", "boolean"})
NUMERIC_BOOLEAN = frozenset({"numeric", "boolean"})

INDICATOR_SUFFIX = "__missing"
OUTLIER_SUFFIX = "__outlier"
ONEHOT_SEPARATOR = "="
ONEHOT_OTHER = "__other__"
SAMPLE_WEIGHT = "_dm_sample_weight"


class Limits:
    """Cost caps for expensive ops (PHASE4_PLAN §7). Exceeding one fails the step with
    ``OP_LIMIT_EXCEEDED`` and a reason; nothing is silently sampled."""

    KNN_MAX_FIT_ROWS = 50_000
    KNN_MAX_PREDICTORS = 100
    # Rows needing imputation x fitting rows: bounds the neighbour-distance work at transform.
    KNN_MAX_DISTANCE_PAIRS = 2_500_000_000
    ITERATIVE_MAX_FIT_ROWS = 100_000
    ITERATIVE_MAX_PREDICTORS = 50
    MAX_STEPS = 50
    MAX_CONSTANT_TEXT = 200
    MAX_TEXT = 200
    MAX_TEXT_LIST = 100
    # Outliers
    OUTLIER_MIN_VALUES = 8  # observed fitting values per column for a threshold rule
    IF_MIN_ROWS = 20  # complete fitting rows for Isolation Forest
    IF_MAX_FIT_ROWS = 200_000
    IF_MAX_COLUMNS = 100
    # Encoding
    ONEHOT_MAX_OUTPUT_COLUMNS = 1_000
    # Distribution transforms
    TRANSFORM_MIN_VALUES = 8  # observed fitting values per column for a fitted transform
    POWER_MAX_FIT_ROWS = 500_000
    POWER_MAX_COLUMNS = 100
    QUANTILE_MAX_FIT_ROWS = 1_000_000
    QUANTILE_SUBSAMPLE = 100_000  # rows used to estimate quantiles (seeded)
    # Feature selection
    SELECT_MIN_ROWS = 10  # observed (feature, target) pairs to score a column
    CORR_MAX_FIT_ROWS = 200_000
    CORR_MAX_COLUMNS = 200
    CORR_MIN_PAIRS = 10  # rows with both values observed for a pairwise correlation
    # Fitting rows x columns. Measured peak (Part C audit): about 25 bytes per cell for
    # Pearson and 32 for Spearman, so 20 million cells stay well inside the worker cap.
    CORR_MAX_CELLS = 20_000_000
    MI_MAX_FIT_ROWS = 100_000
    MI_MAX_COLUMNS = 200
    UNIVARIATE_MAX_FIT_ROWS = 1_000_000
    CHI2_MAX_LEVELS = 50  # categories per column for the chi-square test
    # Class imbalance (training rows only)
    RESAMPLE_MAX_FIT_ROWS = 500_000
    RESAMPLE_MAX_OUTPUT_ROWS = 1_000_000
    # Rows x columns of the resampled training table. Measured peaks (Part C audit): about
    # 34 bytes per output cell for SMOTE and 23 for random oversampling, so 30 million cells
    # stay near 1 GB, inside the 2,000 MB worker cap with room for the input table.
    RESAMPLE_MAX_OUTPUT_CELLS = 30_000_000
    SMOTE_MAX_FIT_ROWS = 200_000
    SMOTE_MAX_FEATURES = 200


_IMPUTE_LIMITS = (
    "Imputed values are estimates, not observations; downstream statistics treat them as real "
    "unless a missing-value indicator is kept.",
    "Assumes values are missing at random given the information used; this cannot be verified "
    "from the data alone.",
)

_PREDICTORS = ParamSpec(
    "predictors",
    "columns",
    "Numeric feature columns used to estimate missing values. Default: every numeric feature "
    "column (target, identifier, group, time, weight and excluded columns are never used).",
)

_OUTLIER_LIMITS = (
    "An outlier is unusual, not necessarily an error; review flagged values before acting.",
    f"Needs at least {Limits.OUTLIER_MIN_VALUES} observed values per column in the fitting "
    "rows. A column with no spread under the chosen rule (e.g. constant) is left unchanged and "
    "reported.",
    "IQR and z-score assume a roughly symmetric distribution; on skewed columns prefer the "
    "percentile rule or a transformation.",
    "Missing values are left as they are and never counted as outliers.",
)

_SCALE_LIMITS = (
    "A column constant in the fitting rows gets scale 1 (values are only shifted); reported.",
    "Missing values stay missing; scaling does not impute.",
    "Output is a decimal (float) column. Runs only inside model training; never saved into a "
    "dataset version.",
)


def _method(default: str) -> ParamSpec:
    return ParamSpec(
        "method",
        "choice",
        "Detection rule: 'iqr' (beyond Q1 - k*IQR or Q3 + k*IQR), 'zscore' (more than z "
        "standard deviations from the mean), 'mad' (robust z-score: median and median absolute "
        "deviation), 'percentile' (outside two percentiles).",
        default=default,
        choices=("iqr", "zscore", "mad", "percentile"),
    )


_METHOD_PARAMS = (
    ParamSpec(
        "iqr_multiplier",
        "float",
        "k for the IQR rule (1.5 = Tukey fences, 3 = far out).",
        default=1.5,
        minimum=0.5,
        maximum=10,
    ),
    ParamSpec(
        "z_threshold",
        "float",
        "Standard deviations from the mean.",
        default=3.0,
        minimum=1,
        maximum=10,
    ),
    ParamSpec(
        "mad_threshold",
        "float",
        "Robust z-score threshold (Iglewicz and Hoaglin suggest 3.5).",
        default=3.5,
        minimum=1,
        maximum=10,
    ),
    ParamSpec(
        "lower_percentile", "float", "Lower bound percentile.", default=1.0, minimum=0, maximum=49
    ),
    ParamSpec(
        "upper_percentile",
        "float",
        "Upper bound percentile.",
        default=99.0,
        minimum=51,
        maximum=100,
    ),
)

OPS: tuple[OpSpec, ...] = (
    OpSpec(
        op="impute.mean",
        version="1.0.0",
        family="missing",
        title="Mean imputation",
        description="Replaces missing values with the column mean learned from the fitting rows.",
        when_to_use="Roughly symmetric numeric columns without strong outliers and with a low "
        "missing share.",
        downsides=(
            "Pulled by skew and outliers: for skewed columns the mean is not a typical value.",
            "Shrinks the column's variance and weakens correlations with other columns.",
        ),
        limitations=_IMPUTE_LIMITS + ("Integer columns become decimal (float) columns.",),
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_IMPUTE}:MeanImputer",
    ),
    OpSpec(
        op="impute.median",
        version="1.0.0",
        family="missing",
        title="Median imputation",
        description="Replaces missing values with the column median learned from the fitting rows.",
        when_to_use="Skewed numeric columns or columns with outliers (counts, amounts, lab "
        "values).",
        downsides=(
            "Shrinks the column's variance and weakens correlations with other columns.",
            "Many identical filled values can create an artificial spike at the median.",
        ),
        limitations=_IMPUTE_LIMITS + ("Integer columns become decimal (float) columns.",),
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_IMPUTE}:MedianImputer",
    ),
    OpSpec(
        op="impute.mode",
        version="1.0.0",
        family="missing",
        title="Most-frequent-value imputation",
        description="Replaces missing values with the most frequent value learned from the "
        "fitting rows. Ties are broken by taking the smallest value (scikit-learn's rule) and are "
        "reported.",
        when_to_use="Categorical and boolean columns, or numeric columns with few distinct values.",
        downsides=(
            "Inflates the most common category and can hide that values were missing.",
            "Rarely appropriate for continuous numeric columns.",
        ),
        limitations=_IMPUTE_LIMITS,
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC_CATEGORICAL_BOOLEAN,
        impl=f"{_IMPUTE}:ModeImputer",
    ),
    OpSpec(
        op="impute.constant",
        version="1.0.0",
        family="missing",
        title="Constant-value imputation",
        description="Replaces missing values with a value you choose (for example 0, or "
        "'Unknown' as its own category). Learns nothing from the data.",
        when_to_use="When missing has a known meaning (e.g. 'not measured' = 0, or an explicit "
        "'Unknown' category).",
        downsides=(
            "A numeric constant outside the observed range distorts means and scales.",
            "A constant inside the range is indistinguishable from real values without an "
            "indicator.",
        ),
        limitations=("The fill value must match the column type.",),
        stage="either",
        kind="stateless_column",
        risk="medium",
        destructive=False,
        column_types=NUMERIC_CATEGORICAL_BOOLEAN,
        impl=f"{_IMPUTE}:ConstantImputer",
        params=(
            ParamSpec(
                "fill_value",
                "scalar",
                "Value written into missing cells: a number for numeric columns, text for "
                "categorical columns, true/false for boolean columns.",
                required=True,
            ),
        ),
    ),
    OpSpec(
        op="impute.knn",
        version="1.0.0",
        family="missing",
        title="K-nearest-neighbours imputation",
        description="Estimates each missing value from the k most similar fitting rows, "
        "measured on the predictor columns. Predictors are standardised with means and standard "
        "deviations learned from the fitting rows, so no single column dominates the distance.",
        when_to_use="Numeric columns related to other numeric columns, moderate data size, "
        "values plausibly similar for similar rows.",
        downsides=(
            "Slow on large data (distance to every fitting row).",
            "Sensitive to the choice of predictors and of k; noisy predictors degrade it.",
        ),
        limitations=_IMPUTE_LIMITS
        + (
            "Numeric predictors only; categorical columns are not used for distances.",
            f"At most {Limits.KNN_MAX_FIT_ROWS:,} fitting rows and "
            f"{Limits.KNN_MAX_PREDICTORS} predictors; rows needing imputation times fitting rows "
            f"at most {Limits.KNN_MAX_DISTANCE_PAIRS:,}.",
            "Integer columns become decimal (float) columns.",
        ),
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_IMPUTE}:KNNImputeOp",
        params=(
            ParamSpec(
                "n_neighbors",
                "int",
                "Number of neighbours averaged.",
                default=5,
                minimum=1,
                maximum=50,
            ),
            ParamSpec(
                "weights",
                "choice",
                "'uniform' averages neighbours equally; 'distance' weights closer neighbours more.",
                default="uniform",
                choices=("uniform", "distance"),
            ),
            _PREDICTORS,
        ),
    ),
    OpSpec(
        op="impute.iterative",
        version="1.0.0",
        family="missing",
        title="Iterative (model-based) imputation",
        description="Models each column with missing values as a regression on the predictor "
        "columns (scikit-learn IterativeImputer with Bayesian ridge regression), refining the "
        "estimates over several rounds. Seeded and deterministic. Imputed values are clipped to "
        "the range observed in the fitting rows by default.",
        when_to_use="Numeric columns with strong linear relationships to other numeric columns.",
        downsides=(
            "Can overfit relationships and produce over-confident, too-smooth values.",
            "Slower than simple imputation; may not converge (reported).",
        ),
        limitations=_IMPUTE_LIMITS
        + (
            "scikit-learn marks IterativeImputer as experimental; its behaviour may change "
            "between versions (the version is recorded).",
            "Linear relationships only (Bayesian ridge).",
            f"At most {Limits.ITERATIVE_MAX_FIT_ROWS:,} fitting rows and "
            f"{Limits.ITERATIVE_MAX_PREDICTORS} predictors.",
            "Integer columns become decimal (float) columns.",
        ),
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_IMPUTE}:IterativeImputeOp",
        params=(
            ParamSpec(
                "max_iter",
                "int",
                "Maximum number of refinement rounds.",
                default=10,
                minimum=1,
                maximum=50,
            ),
            ParamSpec(
                "clip_to_fit_range",
                "bool",
                "Keep imputed values within the minimum and maximum observed in the fitting rows.",
                default=True,
            ),
            _PREDICTORS,
        ),
    ),
    OpSpec(
        op="missing.indicator",
        version="1.0.0",
        family="missing",
        title="Missing-value indicator",
        description="Adds a true/false column '<column>__missing' marking where the value was "
        "missing. Stateless: the same columns are added for every row set.",
        when_to_use="Before imputing, when missingness itself may carry information (e.g. a "
        "test not ordered).",
        downsides=(
            "Adds one column per selected column.",
            "Can leak information if missingness was caused by the outcome.",
        ),
        limitations=("Records only whether a value was missing, not why.",),
        stage="either",
        kind="stateless_column",
        risk="low",
        destructive=False,
        column_types=ANY_FEATURE,
        impl=f"{_IMPUTE}:MissingIndicatorOp",
        adds_columns=True,
    ),
    # ---- Outliers: detection rules are parameters; flag and cap are treatments ----------------
    OpSpec(
        op="outliers.flag",
        version="1.0.0",
        family="outliers",
        title="Flag outliers",
        description="Adds a true/false column '<column>__outlier' marking values outside bounds "
        "learned from the fitting rows. Values are not changed. Detection rules: IQR fences, "
        "z-score, MAD (robust z-score) or percentile bounds.",
        when_to_use="To mark unusual values for review or as a model feature, without deciding "
        "that they are errors.",
        downsides=(
            "An outlier is unusual, not necessarily wrong: rare but real cases are flagged too.",
            "Adds one column per selected column.",
        ),
        limitations=_OUTLIER_LIMITS,
        stage="either",
        kind="fitted",
        risk="low",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_OUTLIERS}:OutlierFlagOp",
        params=(_method("iqr"), *_METHOD_PARAMS),
        adds_columns=True,
    ),
    OpSpec(
        op="outliers.cap",
        version="1.0.0",
        family="outliers",
        title="Cap outliers (winsorize)",
        description="Replaces values below the lower bound with the lower bound and above the "
        "upper bound with the upper bound. Bounds are learned from the fitting rows with the "
        "chosen rule; the 'percentile' rule is classic winsorization. Rows are never removed.",
        when_to_use="When extreme values would dominate a model or statistic and their exact "
        "magnitude is not essential.",
        downsides=(
            "Overwrites observed values: the original magnitudes are lost (destructive).",
            "Real extreme cases become indistinguishable from values at the bound.",
            "Shrinks variance and can hide genuine effects in the tails.",
        ),
        limitations=_OUTLIER_LIMITS
        + ("Integer columns stay integer: bounds are rounded inward to whole numbers.",),
        stage="either",
        kind="fitted",
        risk="high",
        destructive=True,
        column_types=NUMERIC,
        impl=f"{_OUTLIERS}:OutlierCapOp",
        params=(_method("percentile"), *_METHOD_PARAMS),
    ),
    OpSpec(
        op="outliers.signed_log",
        version="1.0.0",
        family="outliers",
        title="Signed log (robust transformation)",
        description="Replaces x with sign(x) * log(1 + |x|). Compresses extreme values while "
        "keeping order and sign; defined for zero and negative values; learns nothing from the "
        "data. Mathematically invertible (x = sign(y) * (exp(|y|) - 1)); numerically the "
        "round trip is exact up to floating-point rounding (relative error below 1e-13).",
        when_to_use="Heavy-tailed columns with zeros or negative values (amounts, balances, "
        "differences) used by scale-sensitive models.",
        downsides=(
            "Changes the unit of the column: results are no longer in the original scale.",
            "Values between -1 and 1 are barely changed; the effect depends on the unit.",
        ),
        limitations=(
            "Not a normalising transformation: for that, see the distribution transforms "
            "(Yeo-Johnson, quantile).",
            "Output is a decimal (float) column.",
        ),
        stage="either",
        kind="stateless_column",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_OUTLIERS}:SignedLogOp",
    ),
    OpSpec(
        op="outliers.isolation_forest_flag",
        version="1.0.0",
        family="outliers",
        title="Flag multivariate outliers (Isolation Forest)",
        description="Fits scikit-learn's IsolationForest on the fitting rows that have a value "
        "in every selected column and flags the 'contamination' share of rows it isolates most "
        "quickly (the threshold is set on the fitting rows and reused for other rows). Seeded "
        "and deterministic. Values are not changed.",
        when_to_use="Rows that are unusual in combination (e.g. a plausible height with an "
        "implausible weight) rather than in a single column.",
        downsides=(
            "It always flags about the chosen share of fitting rows, even when the data has no "
            "real outliers: the share is an assumption, not a measurement.",
            "Harder to explain than a single-column rule; no per-column reason is given.",
        ),
        limitations=(
            "Rows with a missing value in any selected column are not scored and are never "
            "flagged (counted in the report).",
            f"Needs at least {Limits.IF_MIN_ROWS} complete fitting rows; at most "
            f"{Limits.IF_MAX_FIT_ROWS:,} fitting rows and {Limits.IF_MAX_COLUMNS} columns.",
            "An outlier is unusual, not necessarily wrong.",
            "Columns constant in the fitting rows are left out of the model (reported); if "
            "none varies, no row is flagged.",
            "scikit-learn's 'auto' threshold is not offered: on clean data it flags 13-60% of "
            "rows depending on the distribution.",
        ),
        stage="either",
        kind="fitted",
        risk="low",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_OUTLIERS}:IsolationForestFlagOp",
        params=(
            ParamSpec(
                "n_estimators", "int", "Number of trees.", default=100, minimum=10, maximum=500
            ),
            ParamSpec(
                "contamination",
                "float",
                "Share of fitting rows to flag (0.001 to 0.5).",
                default=0.01,
                minimum=0.001,
                maximum=0.5,
            ),
            ParamSpec(
                "output_column",
                "text",
                "Name of the added true/false column.",
                default="isolation_forest_outlier",
            ),
        ),
        adds_columns=True,
        max_columns=Limits.IF_MAX_COLUMNS,
    ),
    # ---- Encoding (model stage only) ---------------------------------------------------------
    OpSpec(
        op="encode.onehot",
        version="1.0.0",
        family="encoding",
        title="One-hot encoding",
        description="Replaces a categorical column by one true/false column per category seen "
        "in the fitting rows, named '<column>=<category>' and ordered by category. Implies no "
        "order between categories. Categories beyond 'max_categories' (the least frequent) are "
        "grouped into '<column>=__other__'.",
        when_to_use="Nominal categories (no natural order) with a moderate number of levels, "
        "for linear and distance-based models.",
        downsides=(
            "Adds one column per category; many levels make the data wide and sparse.",
            "Grouping rare categories loses the distinction between them.",
        ),
        limitations=(
            "Categories never seen in the fitting rows: an error with 'error'; with 'ignore', "
            "the '__other__' column when rare categories were grouped, otherwise all zeros.",
            "A missing value gives all zeros, like an unseen category with 'ignore'; add a "
            "missing-value indicator first to tell them apart.",
            "Sensitive columns are named by position ('<column>=1', ...), never by value. So "
            "are columns whose labels cannot be column names (control characters, blank, "
            "'__other__', duplicates as text, longer than 200 characters).",
            "Runs only inside model training; never saved into a dataset version.",
        ),
        stage="model",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=CATEGORICAL_BOOLEAN,
        impl=f"{_ENCODE}:OneHotOp",
        params=(
            ParamSpec(
                "max_categories",
                "int",
                "Most categories kept per column (the most frequent in the fitting rows).",
                default=20,
                minimum=2,
                maximum=100,
            ),
            ParamSpec(
                "handle_unknown",
                "choice",
                "What to do with a category not seen in the fitting rows.",
                default="ignore",
                choices=("ignore", "error"),
            ),
            ParamSpec(
                "name_by",
                "choice",
                "Name output columns by category value or by position.",
                default="value",
                choices=("value", "position"),
            ),
        ),
        adds_columns=True,
        max_columns=100,
    ),
    OpSpec(
        op="encode.ordinal",
        version="1.0.0",
        family="encoding",
        title="Ordinal encoding (explicit order)",
        description="Replaces categories by their position (0, 1, 2, ...) in an order you "
        "provide, e.g. low < medium < high. There is no default order: an order is never "
        "invented for nominal categories.",
        when_to_use="Categories with a real, known order (grades, stages, sizes).",
        downsides=(
            "Treats steps between levels as equal (low→medium = medium→high).",
            "Wrong for nominal categories: it would invent an order.",
        ),
        limitations=(
            "Every category in the fitting rows must appear in the order; otherwise the step "
            "fails.",
            "Categories not in the order in other rows fail the step ('error') or become "
            "missing ('missing').",
            "Runs only inside model training; never saved into a dataset version.",
        ),
        stage="model",
        kind="stateless_column",
        risk="medium",
        destructive=False,
        column_types=CATEGORICAL,
        impl=f"{_ENCODE}:OrdinalOp",
        params=(
            ParamSpec("order", "text_list", "Categories from lowest to highest.", required=True),
            ParamSpec(
                "handle_unknown",
                "choice",
                "What to do with a category not in the order.",
                default="error",
                choices=("error", "missing"),
            ),
        ),
        max_columns=1,
    ),
    OpSpec(
        op="encode.frequency",
        version="1.0.0",
        family="encoding",
        title="Frequency encoding",
        description="Replaces each category by its share of the fitting rows (0 to 1). A "
        "category not seen in the fitting rows gets 0. Implies no order by meaning, only by "
        "frequency.",
        when_to_use="High-cardinality nominal columns for tree models, where one-hot would be "
        "too wide.",
        downsides=(
            "Categories with the same frequency become indistinguishable (reported).",
            "Frequency may be unrelated to the outcome.",
        ),
        limitations=(
            "Missing values stay missing.",
            "Runs only inside model training; never saved into a dataset version.",
        ),
        stage="model",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=CATEGORICAL,
        impl=f"{_ENCODE}:FrequencyOp",
    ),
    # ---- Scaling (model stage only) ----------------------------------------------------------
    OpSpec(
        op="scale.standard",
        version="1.0.0",
        family="scaling",
        title="Standardisation (z-score scaling)",
        description="Subtracts the mean and divides by the standard deviation learned from the "
        "fitting rows (scikit-learn StandardScaler).",
        when_to_use="Linear models, regularised models, neural networks, PCA and distance-based "
        "methods (k-NN, SVM). Tree models do not need scaling.",
        downsides=("Sensitive to outliers, which inflate the standard deviation.",),
        limitations=_SCALE_LIMITS,
        stage="model",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_SCALE}:StandardScaleOp",
    ),
    OpSpec(
        op="scale.minmax",
        version="1.0.0",
        family="scaling",
        title="Min-max scaling",
        description="Maps the minimum and maximum of the fitting rows to 0 and 1 "
        "(scikit-learn MinMaxScaler). Values outside the fitting range fall outside [0, 1] "
        "unless 'clip' is on.",
        when_to_use="Models that need bounded inputs (some neural networks) and columns "
        "without strong outliers.",
        downsides=(
            "One extreme value compresses all others into a narrow band.",
            "Clipping hides values outside the fitting range.",
        ),
        limitations=_SCALE_LIMITS,
        stage="model",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_SCALE}:MinMaxScaleOp",
        params=(ParamSpec("clip", "bool", "Clip results to [0, 1].", default=False),),
    ),
    OpSpec(
        op="scale.robust",
        version="1.0.0",
        family="scaling",
        title="Robust scaling",
        description="Subtracts the median and divides by the interquartile range learned from "
        "the fitting rows (scikit-learn RobustScaler). Outliers barely affect the scaling.",
        when_to_use="Scale-sensitive models on columns with outliers or skew.",
        downsides=("Outliers keep their extreme position after scaling (they are not removed).",),
        limitations=_SCALE_LIMITS,
        stage="model",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_SCALE}:RobustScaleOp",
    ),
    OpSpec(
        op="scale.none",
        version="1.0.0",
        family="scaling",
        title="No scaling",
        description="Leaves the columns unchanged. Records the decision explicitly, e.g. for "
        "tree models, which do not depend on scale.",
        when_to_use="Tree-based models (gradient boosting, random forests) or when units carry "
        "meaning.",
        downsides=("Scale-sensitive models are dominated by columns with large values.",),
        limitations=("Does nothing; listed so that 'no scaling' is a visible choice.",),
        stage="model",
        kind="stateless_column",
        risk="low",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_SCALE}:NoScaleOp",
    ),
)

_TRANSFORM_LIMITS = (
    "Changes the unit and shape of the column: coefficients and thresholds are no longer in "
    "the original scale.",
    "A transformation does not necessarily help: the report shows skewness and tail shape "
    "before and after, and warns when skewness did not decrease.",
    "Missing values stay missing.",
)

_ACTION = ParamSpec(
    "action",
    "choice",
    "'report' ranks or flags columns and removes nothing; 'drop' removes the columns that fail "
    "the criterion from the model's input (learned in each training fold).",
    default="report",
    choices=("report", "drop"),
)
_KEEP = ParamSpec(
    "keep",
    "columns",
    "Columns this step never removes, whatever their score (e.g. biomarkers you need). "
    "Columns with a domain role are always kept.",
    default=[],
)
_SELECT_LIMITS = (
    "Runs only inside model training, learned from each training fold; never saved into a "
    "dataset version, and the dataset keeps every column.",
    "A ranking reflects association in these rows, not causal, biological or clinical "
    "importance. A selected biomarker is not validated by being selected.",
    "Columns with a domain role (e.g. biomarkers, age, sex) and columns listed in 'keep' are "
    "never removed automatically; target, identifier, group, time and weight columns can never "
    "be selected.",
    "Removing a column can remove biologically or scientifically meaningful information; "
    "compare with 'report' mode before relying on it.",
    "Rankings and removals are learned separately in each training fold and are often "
    "unstable: with weak signals, different folds can keep different columns, including "
    "pure noise. Treat a selection that changes between folds as uncertain.",
)
_SUPERVISED_LIMITS = (
    "Needs a target with a task type. Uses the target values of the training rows only; "
    "validation and test labels are never read.",
    f"A column with fewer than {Limits.SELECT_MIN_ROWS} rows where both it and the target are "
    "observed, or with no variation, is not assessed and is kept (reported).",
    "Scores each column on its own: interactions and redundancy between columns are ignored.",
)
_IMBALANCE_LIMITS = (
    "Changes only the training rows used to fit a model, inside each training fold. "
    "Validation and test rows, and every saved dataset version, are never resampled.",
    "Balancing classes changes what the model is trained on; it does not improve data quality. "
    "Predicted probabilities are no longer calibrated to the real class frequencies.",
    "Classification targets only (binary or multiclass). Must be the last step of a pipeline; "
    "at most one per pipeline.",
)
_SAMPLING = ParamSpec(
    "sampling_strategy",
    "choice",
    "'not_majority' raises every class except the largest to the size of the largest; "
    "'minority' raises only the smallest class.",
    default="not_majority",
    choices=("not_majority", "minority"),
)

OPS_C: tuple[OpSpec, ...] = (
    # ---- Distribution transforms -------------------------------------------------------------
    OpSpec(
        op="transform.log",
        version="1.0.0",
        family="transform",
        title="Logarithm",
        description="Replaces x with log(x) (natural, base 2 or base 10). Defined only for "
        "strictly positive values: a zero or negative value in the fitting rows or in any row "
        "it is applied to fails the step (it is never turned into a missing value). Learns "
        "nothing. Inverse: exp (or a power of the base), exact up to floating-point rounding.",
        when_to_use="Strictly positive, right-skewed columns whose effects are multiplicative "
        "(concentrations, incomes, durations).",
        downsides=(
            "Fails on zero and negative values; use log1p (x > -1) or Yeo-Johnson instead.",
            "Expands values near zero: tiny positive values become large negative numbers.",
        ),
        limitations=_TRANSFORM_LIMITS + ("Output is a decimal (float) column.",),
        stage="either",
        kind="stateless_column",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_TRANSFORM}:LogOp",
        params=(
            ParamSpec(
                "base",
                "choice",
                "Logarithm base: 'e' (natural), '2' or '10'.",
                default="e",
                choices=("e", "2", "10"),
            ),
        ),
    ),
    OpSpec(
        op="transform.log1p",
        version="1.0.0",
        family="transform",
        title="Log of one plus x (log1p)",
        description="Replaces x with log(1 + x). Defined for x > -1: a value of -1 or below "
        "fails the step. Zero maps to zero. Learns nothing. Inverse: exp(y) - 1, exact up to "
        "floating-point rounding.",
        when_to_use="Non-negative counts and amounts with zeros (visits, purchases, read "
        "counts) that are right-skewed.",
        downsides=(
            "The result depends on the unit: log(1 + x) barely changes values much smaller "
            "than 1 and compresses large ones.",
            "Values between -1 and 0 become negative and very large in magnitude near -1.",
        ),
        limitations=_TRANSFORM_LIMITS + ("Output is a decimal (float) column.",),
        stage="either",
        kind="stateless_column",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_TRANSFORM}:Log1pOp",
    ),
    OpSpec(
        op="transform.yeo_johnson",
        version="1.0.0",
        family="transform",
        title="Yeo-Johnson power transformation",
        description="Learns a power parameter (lambda) per column from the fitting rows by "
        "maximum likelihood (scikit-learn PowerTransformer, not standardised) and applies it to "
        "every row. Works with positive, zero and negative values. A column constant in the "
        "fitting rows keeps lambda = 1 (unchanged). Inverse: exists but is numerically "
        "approximate, and unstable near the bound of the output range for some lambdas.",
        when_to_use="Skewed numeric columns with zeros or negative values, before models that "
        "assume roughly symmetric inputs. Combine with standardisation if needed.",
        downsides=(
            "The lambda is fitted to make the column look normal; it cannot fix multimodal "
            "columns or heavy ties and may make outliers more extreme.",
            "Not standardised: add scale.standard afterwards when a model needs unit variance.",
        ),
        limitations=_TRANSFORM_LIMITS
        + (
            f"Needs at least {Limits.TRANSFORM_MIN_VALUES} observed values per column in the "
            f"fitting rows; at most {Limits.POWER_MAX_FIT_ROWS:,} fitting rows and "
            f"{Limits.POWER_MAX_COLUMNS} columns.",
            "Values far outside the fitting range can overflow; the step then fails with "
            "NUMERIC_OVERFLOW instead of writing infinite values.",
        ),
        stage="either",
        kind="fitted",
        risk="medium",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_TRANSFORM}:YeoJohnsonOp",
        max_columns=Limits.POWER_MAX_COLUMNS,
    ),
    OpSpec(
        op="transform.quantile",
        version="1.0.0",
        family="transform",
        title="Quantile transformation",
        description="Maps each value to its position in the distribution of the fitting rows "
        "(scikit-learn QuantileTransformer), then to a uniform [0, 1] or a normal output. "
        "Quantiles are estimated from at most "
        f"{Limits.QUANTILE_SUBSAMPLE:,} fitting rows (seeded). Values outside the fitting range "
        "are clipped to the ends of the output; with a normal output the ends are about "
        "-5.2 and 5.2.",
        when_to_use="Strongly skewed or heavy-tailed columns where only the rank matters, "
        "e.g. for distance-based or linear models; robust to outliers.",
        downsides=(
            "Destroys distances between values: only order is kept, and gaps become uniform.",
            "Values outside the fitting range all map to the same end value (information "
            "loss, counted in the report); tied values stay tied.",
            "Not exactly invertible: the inverse is a piecewise-linear approximation and "
            "cannot recover clipped values.",
        ),
        limitations=_TRANSFORM_LIMITS
        + (
            f"Needs at least {Limits.TRANSFORM_MIN_VALUES} observed values per column; at most "
            f"{Limits.QUANTILE_MAX_FIT_ROWS:,} fitting rows. 'n_quantiles' is reduced to the "
            "number of observed fitting values when larger (reported).",
            "A column constant in the fitting rows maps that value to the middle of the output "
            "(0.5 or 0), smaller values to the lower end and larger values to the upper end "
            "(reported).",
            "The learned quantiles are interpolated fitting values, so they are never shown in "
            "reports.",
        ),
        stage="either",
        kind="fitted",
        risk="high",
        destructive=True,
        column_types=NUMERIC,
        impl=f"{_TRANSFORM}:QuantileOp",
        params=(
            ParamSpec(
                "n_quantiles",
                "int",
                "Number of quantiles estimated (resolution of the mapping).",
                default=1000,
                minimum=10,
                maximum=2000,
            ),
            ParamSpec(
                "output_distribution",
                "choice",
                "'uniform' maps to [0, 1]; 'normal' maps to a standard normal shape.",
                default="uniform",
                choices=("uniform", "normal"),
            ),
        ),
    ),
    # ---- Feature selection (model stage only) ------------------------------------------------
    OpSpec(
        op="select.variance",
        version="1.0.0",
        family="selection",
        title="Low-variance filter",
        description="Computes each column's variance over its observed values in the fitting "
        "rows (true/false columns as 0/1). With 'drop', columns whose variance is at or below "
        "the threshold are removed from the model's input; the default threshold 0 removes "
        "only columns that are constant in the fitting rows. Does not use the target.",
        when_to_use="Removing columns that cannot help any model because they do not vary in "
        "the training data.",
        downsides=(
            "Variance depends on the unit: a threshold above 0 compares columns measured in "
            "different units (reported). Scale first or keep the threshold at 0.",
            "A rarely varying column (e.g. a rare mutation or exposure) can still be highly "
            "informative.",
        ),
        limitations=_SELECT_LIMITS
        + ("A column with no observed value in the fitting rows fails the step.",),
        stage="model",
        kind="fitted",
        risk="high",
        destructive=True,
        column_types=NUMERIC_BOOLEAN,
        impl=f"{_SELECT}:VarianceSelectOp",
        params=(
            ParamSpec(
                "threshold",
                "float",
                "Columns with variance at or below this value fail the filter.",
                default=0.0,
                minimum=0,
                maximum=1e12,
            ),
            _ACTION,
            _KEEP,
        ),
    ),
    OpSpec(
        op="select.correlation",
        version="1.0.0",
        family="selection",
        title="Highly correlated pairs (proposal only)",
        description="Finds pairs of columns whose absolute correlation in the fitting rows is "
        "at or above the threshold (Pearson or Spearman, pairwise over rows where both are "
        "observed) and proposes one column per pair for removal: the one more correlated with "
        "the other selected columns on average. Removes nothing: to act on a proposal, add a "
        "'Remove columns' step for the columns you choose.",
        when_to_use="Spotting redundant measurements (e.g. the same quantity in two units) "
        "before linear models, where they make coefficients unstable.",
        downsides=(
            "Correlated columns can each carry distinct, meaningful information (e.g. related "
            "biomarkers); correlation alone does not mean one is redundant.",
            "Tree models handle correlated columns well; removing one rarely helps them.",
        ),
        limitations=(
            "Proposals are learned from the fitting rows and can differ between training folds.",
            f"Pairs with fewer than {Limits.CORR_MIN_PAIRS} rows where both values are observed, "
            "or involving a constant column, are reported as not assessed.",
            "Pearson measures linear association; Spearman measures monotonic association. "
            "Neither detects other dependence.",
            f"At most {Limits.CORR_MAX_COLUMNS} columns, {Limits.CORR_MAX_FIT_ROWS:,} "
            f"fitting rows and {Limits.CORR_MAX_CELLS:,} cells (rows x columns).",
            "Columns with a domain role and columns listed in 'keep' are never proposed.",
        ),
        stage="model",
        kind="fitted",
        risk="low",
        destructive=False,
        column_types=NUMERIC_BOOLEAN,
        impl=f"{_SELECT}:CorrelationSelectOp",
        params=(
            ParamSpec(
                "threshold",
                "float",
                "Absolute correlation at or above which a pair is reported.",
                default=0.9,
                minimum=0.5,
                maximum=1.0,
            ),
            ParamSpec(
                "method",
                "choice",
                "'pearson' (linear) or 'spearman' (rank, monotonic).",
                default="pearson",
                choices=("pearson", "spearman"),
            ),
            _KEEP,
        ),
        max_columns=Limits.CORR_MAX_COLUMNS,
    ),
    OpSpec(
        op="select.mutual_information",
        version="1.0.0",
        family="selection",
        title="Mutual-information ranking",
        description="Estimates the mutual information between each column and the target in "
        "the fitting rows (scikit-learn mutual_info_classif or mutual_info_regression, by task "
        "type; k-nearest-neighbour estimator, seeded). Categorical and true/false columns are "
        "treated as discrete. Ranks the columns; with 'drop', keeps the k highest and removes "
        "the rest from the model's input.",
        when_to_use="Ranking candidate features by any kind of dependence on the target, "
        "including non-linear dependence.",
        downsides=(
            "Estimates are noisy on small samples and can rank unrelated columns above zero.",
            "Ignores redundancy: two copies of the same informative column both rank high.",
        ),
        limitations=_SELECT_LIMITS
        + _SUPERVISED_LIMITS
        + (
            "Each column uses the rows where it and the target are observed, so columns may "
            "be scored on different rows (counts reported).",
            f"At most {Limits.MI_MAX_FIT_ROWS:,} fitting rows and {Limits.MI_MAX_COLUMNS} columns.",
        ),
        stage="model",
        kind="fitted",
        risk="high",
        destructive=True,
        column_types=NUMERIC_CATEGORICAL_BOOLEAN,
        impl=f"{_SELECT}:MutualInfoSelectOp",
        params=(
            ParamSpec(
                "k",
                "int",
                "Number of highest-ranked columns kept with 'drop'.",
                default=10,
                minimum=1,
                maximum=500,
            ),
            ParamSpec(
                "n_neighbors",
                "int",
                "Neighbours used by the estimator for continuous columns.",
                default=3,
                minimum=1,
                maximum=20,
            ),
            _ACTION,
            _KEEP,
        ),
        max_columns=Limits.MI_MAX_COLUMNS,
        needs_target=True,
    ),
    OpSpec(
        op="select.univariate",
        version="1.0.0",
        family="selection",
        title="Univariate statistical test ranking",
        description="Tests each column's association with the target in the fitting rows "
        "with the test you choose, and ranks columns by p-value (Benjamini-Hochberg adjusted "
        "p-values are reported). With 'drop', keeps the k best-ranked and removes the rest from "
        "the model's input. Tests: 'anova_f' and 'kruskal' (numeric column, class target), "
        "'chi2' (categorical or true/false column, class target), 'f_regression' and "
        "'spearman' (numeric column, numeric target). A test that does not match the column "
        "type or task type is refused.",
        when_to_use="Screening many features against a target with a simple, explainable "
        "test whose assumptions you have checked.",
        downsides=(
            "A p-value measures evidence of association in this sample, not its size or "
            "importance; with many rows tiny effects are 'significant'.",
            "Tests one column at a time; misses interactions and redundancy.",
        ),
        limitations=_SELECT_LIMITS
        + _SUPERVISED_LIMITS
        + (
            "Assumptions are checked and reported, not assumed: ANOVA (normal-ish groups, "
            "similar variances), chi-square (expected counts of at least 5 in most cells), "
            "f_regression (linear relationship). A doubtful assumption produces a warning that "
            "names the more robust test.",
            f"Chi-square assesses at most {Limits.CHI2_MAX_LEVELS} categories per column.",
            f"At most {Limits.UNIVARIATE_MAX_FIT_ROWS:,} fitting rows.",
        ),
        stage="model",
        kind="fitted",
        risk="high",
        destructive=True,
        column_types=NUMERIC_CATEGORICAL_BOOLEAN,
        impl=f"{_SELECT}:UnivariateSelectOp",
        params=(
            ParamSpec(
                "test",
                "choice",
                "Statistical test (no default: choose one whose assumptions fit your data).",
                required=True,
                choices=("anova_f", "kruskal", "chi2", "f_regression", "spearman"),
            ),
            ParamSpec(
                "k",
                "int",
                "Number of best-ranked columns kept with 'drop'.",
                default=10,
                minimum=1,
                maximum=500,
            ),
            _ACTION,
            _KEEP,
        ),
        needs_target=True,
    ),
    OpSpec(
        op="select.drop_columns",
        version="1.0.0",
        family="selection",
        title="Remove columns (your decision)",
        description="Removes the columns you select from the model's input. Learns nothing: "
        "use it to act on a correlation proposal or on domain knowledge. The dataset keeps the "
        "columns.",
        when_to_use="After reviewing a ranking or a correlation proposal, to exclude columns "
        "you have decided not to use.",
        downsides=("Removed columns cannot inform the model.",),
        limitations=(
            "Runs only inside model training; never saved into a dataset version.",
            "Target, identifier, group, time and weight columns cannot be selected.",
        ),
        stage="model",
        kind="stateless_column",
        risk="high",
        destructive=True,
        column_types=ANY_FEATURE,
        impl=f"{_SELECT}:DropColumnsOp",
    ),
    # ---- Class imbalance (model stage, training rows only) ------------------------------------
    OpSpec(
        op="imbalance.class_weight",
        version="1.0.0",
        family="imbalance",
        title="Class weights (balanced)",
        description="Learns one weight per class from the training rows, n_rows / (n_classes "
        "x class count), and adds them as a per-row weight column '_dm_sample_weight' to the "
        "training table only. No row is added or removed and no value changes.",
        when_to_use="Imbalanced classification with a model that accepts sample weights; "
        "usually the least invasive option.",
        downsides=(
            "Rare-class rows get large weights; a few mislabeled rare rows can dominate.",
            "Shifts the model's decision threshold; predicted probabilities are not calibrated "
            "to the real class frequencies.",
        ),
        limitations=_IMBALANCE_LIMITS
        + ("Refused when the dataset already has a sample-weight column (no silent combination).",),
        stage="model",
        kind="training_only",
        risk="medium",
        destructive=False,
        column_types=ANY_FEATURE,
        impl=f"{_IMBALANCE}:ClassWeightOp",
        uses_columns=False,
        needs_target=True,
    ),
    OpSpec(
        op="imbalance.random_oversample",
        version="1.0.0",
        family="imbalance",
        title="Random oversampling",
        description="Duplicates randomly chosen training rows of smaller classes "
        "(imbalanced-learn RandomOverSampler, seeded) until the chosen classes match the "
        "largest. Duplicates are exact copies of real rows, including their identifier and "
        "group values, so group-aware validation keeps them together.",
        when_to_use="Imbalanced classification with models that cannot use sample weights.",
        downsides=(
            "Duplicated rows add no new information and encourage overfitting to them.",
            "Counts and statistics of the training table no longer describe the real data.",
        ),
        limitations=_IMBALANCE_LIMITS
        + (
            "Added rows get a negative '_dm_row_id' (-1, -2, ...); they exist only in the "
            "training table.",
            f"At most {Limits.RESAMPLE_MAX_FIT_ROWS:,} training rows, "
            f"{Limits.RESAMPLE_MAX_OUTPUT_ROWS:,} rows after resampling and "
            f"{Limits.RESAMPLE_MAX_OUTPUT_CELLS:,} cells (rows x columns) after resampling.",
        ),
        stage="model",
        kind="training_only",
        risk="medium",
        destructive=False,
        column_types=ANY_FEATURE,
        impl=f"{_IMBALANCE}:RandomOversampleOp",
        params=(_SAMPLING,),
        uses_columns=False,
        needs_target=True,
    ),
    OpSpec(
        op="imbalance.random_undersample",
        version="1.0.0",
        family="imbalance",
        title="Random undersampling",
        description="Removes randomly chosen training rows of larger classes "
        "(imbalanced-learn RandomUnderSampler, seeded) until the chosen classes match the "
        "smallest. Validation and test rows are never removed.",
        when_to_use="Large, imbalanced training sets where discarding majority rows is affordable.",
        downsides=(
            "Discards real training rows and the information they carry (counted in the report).",
            "With a very small minority class, the training table becomes very small.",
        ),
        limitations=_IMBALANCE_LIMITS
        + (f"At most {Limits.RESAMPLE_MAX_FIT_ROWS:,} training rows.",),
        stage="model",
        kind="training_only",
        risk="high",
        destructive=True,
        column_types=ANY_FEATURE,
        impl=f"{_IMBALANCE}:RandomUndersampleOp",
        params=(
            ParamSpec(
                "sampling_strategy",
                "choice",
                "'not_minority' reduces every class except the smallest to its size; "
                "'majority' reduces only the largest class.",
                default="not_minority",
                choices=("not_minority", "majority"),
            ),
        ),
        uses_columns=False,
        needs_target=True,
    ),
    OpSpec(
        op="imbalance.smote",
        version="1.0.0",
        family="imbalance",
        title="SMOTE (synthetic oversampling)",
        description="Creates synthetic training rows for smaller classes by interpolating "
        "between a row and one of its k nearest neighbours of the same class "
        "(imbalanced-learn SMOTE, seeded). Uses every feature column, which must all be "
        "numeric with no missing values. Synthetic rows have no identifier, group or time "
        "value (missing) and exist only in the training table.",
        when_to_use="Imbalanced classification on numeric features of independent rows, when "
        "weights or random oversampling underperform.",
        downsides=(
            "Synthetic rows are not real observations: they can fall in impossible regions "
            "(e.g. between two clusters) and blur class boundaries.",
            "With grouped, repeated-measures, patient-level or time-ordered data, synthetic "
            "rows mix information from different patients, groups or times and do not "
            "represent independent real samples. When identifier, patient, group or time "
            "roles are declared, SMOTE is refused unless 'acknowledge_dependent_rows' is true.",
        ),
        limitations=_IMBALANCE_LIMITS
        + (
            "Every feature column must be numeric (encode categorical columns first; "
            "true/false and one-hot columns are refused) and have no missing value in the "
            "training rows (impute first). Integer columns become decimal columns.",
            "Each class that is oversampled needs more training rows than 'k_neighbors'; "
            "otherwise the step fails (k is never reduced silently).",
            "Synthetic rows get a negative '_dm_row_id' (-1, -2, ...).",
            f"At most {Limits.SMOTE_MAX_FIT_ROWS:,} training rows, "
            f"{Limits.SMOTE_MAX_FEATURES} feature columns, "
            f"{Limits.RESAMPLE_MAX_OUTPUT_ROWS:,} rows and "
            f"{Limits.RESAMPLE_MAX_OUTPUT_CELLS:,} cells (rows x columns) after resampling.",
        ),
        stage="model",
        kind="training_only",
        risk="high",
        destructive=False,
        column_types=NUMERIC,
        impl=f"{_IMBALANCE}:SmoteOp",
        params=(
            _SAMPLING,
            ParamSpec(
                "k_neighbors",
                "int",
                "Nearest neighbours of the same class used for interpolation.",
                default=5,
                minimum=1,
                maximum=50,
            ),
            ParamSpec(
                "acknowledge_dependent_rows",
                "bool",
                "Required (true) when identifier, patient, group or time columns are declared: "
                "you accept that synthetic rows blend different patients, groups or times.",
                default=False,
            ),
        ),
        uses_columns=False,
        needs_target=True,
    ),
)

OPS = OPS + OPS_C

_BY_OP = {spec.op: spec for spec in OPS}
if len(_BY_OP) != len(OPS):  # pragma: no cover - guarded at import
    raise ValueError("duplicate op names in catalogue")


def get_op(op: str) -> OpSpec | None:
    return _BY_OP.get(op)


def all_ops() -> tuple[OpSpec, ...]:
    return OPS
