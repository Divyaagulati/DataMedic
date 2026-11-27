"""Op-specific validation rules and schema effects (standard library only).

Each op may register:
- a **check**: extra errors/warnings and parameter normalisation (e.g. resolving default
  predictors, dropping parameters that do not apply to the chosen method);
- an **effect**: how a valid step changes the schema seen by later steps (added, removed or
  retyped columns).

Generic checks (op exists, parameter types, column existence/roles/types) stay in
``validate.py``. Messages never contain data values; column names travel in ``column``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.catalog import (
    INDICATOR_SUFFIX,
    ONEHOT_SEPARATOR,
    OUTLIER_SUFFIX,
    Limits,
)
from datamedic.engine.preprocess.spec import OpSpec
from datamedic.engine.preprocess.state import ColumnState, Issue, ValidationContext, is_number

RESERVED_PREFIX = "_dm_"


@dataclass
class RuleContext:
    index: int
    spec: OpSpec
    selected: list[str]
    params: dict[str, Any]  # complete (defaults filled); rules may normalise it
    explicit: frozenset[str]  # parameter names the caller supplied
    columns: dict[str, ColumnState]  # schema before this step (mutated by effects)
    reserved_prefixes: set[str]  # name prefixes claimed by earlier steps (one-hot outputs)
    vctx: ValidationContext
    # Columns an earlier drop-mode selector may remove: later steps cannot select them, but
    # they may still be model inputs (e.g. SMOTE features), so their state is kept here.
    maybe_removed: dict[str, ColumnState] = field(default_factory=dict)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    def err(
        self, code: str, message: str, *, column: str | None = None, param: str | None = None
    ) -> None:
        self.errors.append(Issue(code, message, self.index, column, param))

    def warn(
        self, code: str, message: str, *, column: str | None = None, param: str | None = None
    ) -> None:
        self.warnings.append(Issue(code, message, self.index, column, param))

    def valid_columns(self) -> list[str]:
        return [c for c in dict.fromkeys(self.selected) if c in self.columns]

    def name_taken(self, name: str) -> bool:
        return name in self.columns or any(name.startswith(p) for p in self.reserved_prefixes)

    def check_new_names(self, names: list[tuple[str, str | None]]) -> None:
        """``names``: (new column name, source column for the message)."""
        seen: set[str] = set()
        for name, source in names:
            if self.name_taken(name) or name in seen:
                self.err(
                    "OUTPUT_COLUMN_CONFLICT",
                    "The output column name already exists.",
                    column=source,
                )
            seen.add(name)

    def drop_params(self, keep: set[str]) -> None:
        """Remove parameters that do not apply; supplying one explicitly is an error."""
        for name in list(self.params):
            if name not in keep:
                if name in self.explicit:
                    self.err(
                        "PARAM_NOT_APPLICABLE",
                        f"'{name}' does not apply to the chosen method.",
                        param=name,
                    )
                del self.params[name]


Check = Callable[[RuleContext], None]
Effect = Callable[[RuleContext], None]


# ---- Missing values (Part A) ------------------------------------------------------------------

FLOAT_OUTPUT_OPS = frozenset({"impute.mean", "impute.median", "impute.knn", "impute.iterative"})
PREDICTOR_CAPS = {
    "impute.knn": Limits.KNN_MAX_PREDICTORS,
    "impute.iterative": Limits.ITERATIVE_MAX_PREDICTORS,
}
FIT_ROW_CAPS = {
    "impute.knn": Limits.KNN_MAX_FIT_ROWS,
    "impute.iterative": Limits.ITERATIVE_MAX_FIT_ROWS,
    "outliers.isolation_forest_flag": Limits.IF_MAX_FIT_ROWS,
    "transform.yeo_johnson": Limits.POWER_MAX_FIT_ROWS,
    "transform.quantile": Limits.QUANTILE_MAX_FIT_ROWS,
    "select.correlation": Limits.CORR_MAX_FIT_ROWS,
    "select.mutual_information": Limits.MI_MAX_FIT_ROWS,
    "select.univariate": Limits.UNIVARIATE_MAX_FIT_ROWS,
    "imbalance.class_weight": Limits.RESAMPLE_MAX_FIT_ROWS,
    "imbalance.random_oversample": Limits.RESAMPLE_MAX_FIT_ROWS,
    "imbalance.random_undersample": Limits.RESAMPLE_MAX_FIT_ROWS,
    "imbalance.smote": Limits.SMOTE_MAX_FIT_ROWS,
}


def _no_missing_warning(rc: RuleContext) -> None:
    for name in rc.valid_columns():
        state = rc.columns[name]
        if (
            state.n_missing == 0
            and name not in rc.vctx.protected
            and state.inferred_type in rc.spec.column_types
        ):
            rc.warn(
                "NO_MISSING_VALUES",
                "This column has no missing values here; the step changes nothing now but "
                "still applies to new data.",
                column=name,
            )


def _fill_value_error(value: Any, column: ColumnState) -> str | None:
    t = column.inferred_type
    if t == "numeric":
        return None if is_number(value) else "must be a number for a numeric column"
    if t == "boolean":
        return None if isinstance(value, bool) else "must be true or false for a boolean column"
    if t == "categorical":
        if not isinstance(value, str):
            return "must be text for a categorical column"
        if not value.strip():
            return "must not be empty or blank"
        if len(value) > Limits.MAX_CONSTANT_TEXT:
            return f"must be at most {Limits.MAX_CONSTANT_TEXT} characters"
        return None
    return "is not supported for this column type"


def check_impute(rc: RuleContext) -> None:
    _no_missing_warning(rc)


def check_constant(rc: RuleContext) -> None:
    _no_missing_warning(rc)
    if "fill_value" not in rc.params:
        return
    value = rc.params["fill_value"]
    for name in rc.valid_columns():
        if name in rc.vctx.protected:
            continue
        problem = _fill_value_error(value, rc.columns[name])
        if problem:
            rc.err("PARAM_INVALID", f"'fill_value' {problem}.", column=name, param="fill_value")
    if (
        isinstance(value, float)
        and not value.is_integer()
        and any(rc.columns[c].dtype == "integer" for c in rc.valid_columns())
    ):
        rc.warn(
            "DTYPE_CHANGE",
            "A decimal fill value turns an integer column into a decimal column.",
            param="fill_value",
        )


def check_predictors(rc: RuleContext) -> None:
    _no_missing_warning(rc)
    requested = rc.params.get("predictors")
    selected = set(rc.selected)
    if requested is None:
        chosen = [
            c.name
            for c in rc.columns.values()
            if c.inferred_type == "numeric"
            and c.name not in rc.vctx.protected
            and c.name != ROW_ID
            and c.name not in selected
        ]
    else:
        chosen = []
        for name in requested:
            state = rc.columns.get(name)
            if name == ROW_ID:
                rc.err(
                    "RESERVED_COLUMN",
                    "The internal row identifier cannot be a predictor.",
                    column=name,
                    param="predictors",
                )
            elif state is None:
                rc.err(
                    "COLUMN_NOT_FOUND",
                    "Predictor column not found.",
                    column=name,
                    param="predictors",
                )
            elif name in rc.vctx.protected:
                rc.err(
                    "COLUMN_PROTECTED",
                    "Columns with a non-feature role (such as the target) are never used as "
                    "predictors.",
                    column=name,
                    param="predictors",
                )
            elif state.inferred_type != "numeric":
                rc.err(
                    "COLUMN_TYPE_UNSUPPORTED",
                    "Predictors must be numeric columns.",
                    column=name,
                    param="predictors",
                )
            elif name not in selected:
                chosen.append(name)
        order = list(rc.columns)
        chosen.sort(key=lambda n: order.index(n) if n in order else len(order))
    rc.params["predictors"] = chosen
    cap = PREDICTOR_CAPS[rc.spec.op]
    if len(dict.fromkeys(rc.selected)) + len(chosen) > cap:
        rc.err(
            "OP_LIMIT_EXCEEDED",
            f"This operation uses at most {cap} columns in total (imputed columns plus "
            "predictors). Choose fewer predictors.",
            param="predictors",
        )
    if not chosen and len(dict.fromkeys(rc.selected)) == 1:
        rc.warn(
            "NO_PREDICTORS",
            "No other numeric column is available, so the estimate uses only this column's "
            "own observed values.",
            param="predictors",
        )


def effect_impute(rc: RuleContext) -> None:
    value = rc.params.get("fill_value")
    for name in rc.selected:
        state = rc.columns[name]
        dtype = state.dtype
        if rc.spec.op in FLOAT_OUTPUT_OPS or (
            rc.spec.op == "impute.constant"
            and isinstance(value, float)
            and not float(value).is_integer()
        ):
            dtype = "float"
        rc.columns[name] = replace(state, dtype=dtype, n_missing=0)


def check_indicator(rc: RuleContext) -> None:
    rc.check_new_names([(c + INDICATOR_SUFFIX, c) for c in rc.selected])


def effect_indicator(rc: RuleContext) -> None:
    for name in rc.selected:
        out = name + INDICATOR_SUFFIX
        rc.columns[out] = ColumnState(out, "boolean", "boolean", 0)


# ---- Outliers (Part B) ------------------------------------------------------------------------

METHOD_PARAMS = {
    "iqr": {"iqr_multiplier"},
    "zscore": {"z_threshold"},
    "mad": {"mad_threshold"},
    "percentile": {"lower_percentile", "upper_percentile"},
}


def _check_method(rc: RuleContext) -> None:
    method = rc.params.get("method")
    if method not in METHOD_PARAMS:
        return  # the generic parameter check already reported it
    rc.drop_params({"method", *METHOD_PARAMS[method]})
    if method == "percentile":
        low, high = rc.params.get("lower_percentile"), rc.params.get("upper_percentile")
        if (
            isinstance(low, int | float)
            and isinstance(high, int | float)
            and is_number(low)
            and is_number(high)
            and low >= high
        ):
            rc.err(
                "PARAM_INVALID",
                "'lower_percentile' must be below 'upper_percentile'.",
                param="lower_percentile",
            )


def check_outlier_flag(rc: RuleContext) -> None:
    _check_method(rc)
    rc.check_new_names([(c + OUTLIER_SUFFIX, c) for c in rc.selected])


def effect_outlier_flag(rc: RuleContext) -> None:
    for name in rc.selected:
        out = name + OUTLIER_SUFFIX
        rc.columns[out] = ColumnState(out, "boolean", "boolean", 0)


def check_outlier_cap(rc: RuleContext) -> None:
    _check_method(rc)


def effect_float(rc: RuleContext) -> None:
    for name in rc.selected:
        rc.columns[name] = replace(rc.columns[name], dtype="float")


def _check_text_name(rc: RuleContext, value: Any, param: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        rc.err("PARAM_INVALID", f"'{param}' must be a non-empty name.", param=param)
        return False
    if value.startswith(RESERVED_PREFIX):
        rc.err("PARAM_INVALID", f"'{param}' must not start with '{RESERVED_PREFIX}'.", param=param)
        return False
    return True


def check_isolation_forest(rc: RuleContext) -> None:
    name = rc.params.get("output_column")
    if _check_text_name(rc, name, "output_column"):
        rc.check_new_names([(str(name), None)])
    if len(dict.fromkeys(rc.selected)) > Limits.IF_MAX_COLUMNS:
        rc.err("OP_LIMIT_EXCEEDED", f"This operation uses at most {Limits.IF_MAX_COLUMNS} columns.")
    if len(dict.fromkeys(rc.selected)) == 1:
        rc.warn(
            "SINGLE_COLUMN",
            "With one column, a univariate rule (IQR, MAD, percentile) is simpler to explain "
            "and usually as effective.",
        )


def effect_isolation_forest(rc: RuleContext) -> None:
    name = str(rc.params["output_column"])
    rc.columns[name] = ColumnState(name, "boolean", "boolean", 0)


# ---- Encoding (Part B) ------------------------------------------------------------------------


def check_onehot(rc: RuleContext) -> None:
    sensitive = [c for c in rc.selected if c in rc.vctx.sensitive]
    if sensitive and rc.params.get("name_by") == "value":
        rc.params["name_by"] = "position"
        for column in sensitive:
            rc.warn(
                "SENSITIVE_POSITIONAL_NAMES",
                "Output columns of a sensitive column are named by position, not by category "
                "value, so category values never appear in column names or reports.",
                column=column,
            )
    for column in rc.selected:
        prefix = column + ONEHOT_SEPARATOR
        clash = [n for n in rc.columns if n.startswith(prefix)] + [
            p for p in rc.reserved_prefixes if p.startswith(prefix) or prefix.startswith(p)
        ]
        if clash:
            rc.err(
                "OUTPUT_COLUMN_CONFLICT",
                f"Columns named '<column>{ONEHOT_SEPARATOR}...' already exist, so one-hot "
                "output names could collide.",
                column=column,
            )
    max_categories = rc.params.get("max_categories")
    if isinstance(max_categories, int) and not isinstance(max_categories, bool):
        outputs = len(dict.fromkeys(rc.selected)) * (max_categories + 1)
        if outputs > Limits.ONEHOT_MAX_OUTPUT_COLUMNS:
            rc.err(
                "OP_LIMIT_EXCEEDED",
                f"This step could create up to {outputs:,} columns; the limit is "
                f"{Limits.ONEHOT_MAX_OUTPUT_COLUMNS:,}. Select fewer columns or lower "
                "'max_categories'.",
                param="max_categories",
            )


def effect_onehot(rc: RuleContext) -> None:
    for name in rc.selected:
        del rc.columns[name]  # replaced by columns learned at fit time
        rc.reserved_prefixes.add(name + ONEHOT_SEPARATOR)


def check_ordinal(rc: RuleContext) -> None:
    order = rc.params.get("order")
    if isinstance(order, list) and len(order) < 2:
        rc.err("PARAM_INVALID", "'order' must list at least two categories.", param="order")


def effect_ordinal(rc: RuleContext) -> None:
    for name in rc.selected:
        rc.columns[name] = replace(
            rc.columns[name], inferred_type="numeric", dtype="integer", n_missing=None
        )


def effect_frequency(rc: RuleContext) -> None:
    for name in rc.selected:
        rc.columns[name] = replace(rc.columns[name], inferred_type="numeric", dtype="float")


# ---- Distribution transforms (Part C) ---------------------------------------------------------
# All four keep the column and make it a float column: ``effect_float``.


# ---- Feature selection (Part C) ---------------------------------------------------------------

CLASSIFICATION_TASKS = frozenset({"binary", "multiclass"})
CLASSIFICATION_TESTS = frozenset({"anova_f", "kruskal", "chi2"})
REGRESSION_TESTS = frozenset({"f_regression", "spearman"})
GROUPING_ROLES = frozenset({"group", "time", "id", "bio.patient_id", "bio.sample_id"})


def _resolve_keep(rc: RuleContext) -> None:
    """``keep`` must list selected columns; columns with a domain role are always kept
    (resolved into ``role_kept`` so the step hash and generated code carry it)."""
    keep = rc.params.get("keep") or []
    selected = set(rc.selected)
    for name in keep:
        if name not in selected:
            rc.err(
                "PARAM_INVALID",
                "'keep' lists a column that is not selected in this step.",
                column=name,
                param="keep",
            )
    rc.params["keep"] = list(keep)
    role_kept = [c for c in rc.valid_columns() if c in rc.vctx.domain_roles]
    rc.params["role_kept"] = role_kept
    if rc.params.get("action", "report") == "drop" or rc.spec.op == "select.correlation":
        for column in role_kept:
            rc.warn(
                "DOMAIN_ROLE_KEPT",
                "This column has a domain role ("
                + ", ".join(rc.vctx.domain_roles[column])
                + "); selection never removes it automatically.",
                column=column,
            )


def _resolve_target(rc: RuleContext, *, classification_only: bool = False) -> str | None:
    """Resolve the target and task type from the declared roles (never from the request)."""
    target, task = rc.vctx.target, rc.vctx.task_type
    if target is None:
        rc.err(
            "TARGET_REQUIRED",
            "This operation needs a target: declare exactly one target column with a task type.",
        )
        return None
    if task is None:
        rc.err("TASK_TYPE_REQUIRED", "Declare the target's task type first.", column=target)
        return None
    state = rc.columns.get(target)
    if state is None:
        rc.err("COLUMN_NOT_FOUND", "The target column is not in the table.", column=target)
        return None
    if classification_only and task not in CLASSIFICATION_TASKS:
        rc.err(
            "TASK_UNSUPPORTED",
            "Class imbalance handling applies to classification targets (binary or multiclass) "
            "only.",
            column=target,
        )
        return None
    if task == "regression" and state.inferred_type != "numeric":
        rc.err(
            "TARGET_TYPE_UNSUPPORTED",
            "A regression target must be a numeric column.",
            column=target,
        )
        return None
    rc.params["target"] = target
    rc.params["task"] = task
    return target


def check_variance(rc: RuleContext) -> None:
    _resolve_keep(rc)
    threshold = rc.params.get("threshold")
    if (
        isinstance(threshold, int | float)
        and is_number(threshold)
        and threshold > 0
        and len(rc.valid_columns()) > 1
    ):
        rc.warn(
            "VARIANCE_SCALE_DEPENDENT",
            "Variance depends on each column's unit, so one threshold above 0 treats columns "
            "in different units differently. Scale first or use threshold 0.",
            param="threshold",
        )


def check_correlation(rc: RuleContext) -> None:
    _resolve_keep(rc)
    if len(dict.fromkeys(rc.selected)) < 2:
        rc.err("TOO_FEW_COLUMNS", "Select at least two columns to compare.")


def check_mutual_information(rc: RuleContext) -> None:
    _resolve_keep(rc)
    _resolve_target(rc)
    rc.params["discrete"] = [
        rc.columns[c].inferred_type in ("categorical", "boolean") if c in rc.columns else False
        for c in rc.selected
    ]


def check_univariate(rc: RuleContext) -> None:
    _resolve_keep(rc)
    target = _resolve_target(rc)
    test = rc.params.get("test")
    if target is None or test is None:
        return
    task = rc.params["task"]
    if task == "regression" and test in CLASSIFICATION_TESTS:
        rc.err(
            "TEST_TASK_MISMATCH",
            "This test needs a classification target; use 'f_regression' or 'spearman'.",
            param="test",
        )
    if task in CLASSIFICATION_TASKS and test in REGRESSION_TESTS:
        rc.err(
            "TEST_TASK_MISMATCH",
            "This test needs a numeric (regression) target; use 'anova_f', 'kruskal' or 'chi2'.",
            param="test",
        )
    for name in rc.valid_columns():
        if name in rc.vctx.protected:
            continue
        kind = rc.columns[name].inferred_type
        if test == "chi2" and kind not in ("categorical", "boolean"):
            rc.err(
                "COLUMN_TYPE_UNSUPPORTED",
                "The chi-square test needs categorical or true/false columns.",
                column=name,
                param="test",
            )
        elif test != "chi2" and kind != "numeric":
            rc.err(
                "COLUMN_TYPE_UNSUPPORTED",
                "This test needs numeric columns; use 'chi2' for categorical columns.",
                column=name,
                param="test",
            )


def effect_select(rc: RuleContext) -> None:
    """With 'drop', any selected column that is not kept may be removed, so later steps
    cannot rely on it."""
    if rc.params.get("action") != "drop":
        return
    kept = set(rc.params.get("keep") or []) | set(rc.params.get("role_kept") or [])
    for name in rc.selected:
        if name not in kept and name in rc.columns:
            rc.maybe_removed[name] = rc.columns.pop(name)


def effect_drop_columns(rc: RuleContext) -> None:
    for name in rc.selected:
        rc.columns.pop(name, None)


# ---- Class imbalance (Part C) -----------------------------------------------------------------


def check_imbalance(rc: RuleContext) -> None:
    target = _resolve_target(rc, classification_only=True)
    if target is None:
        return
    passthrough = [c for c in rc.columns if c in rc.vctx.protected and c != target]
    rc.params["passthrough"] = passthrough
    roles = {r for c in passthrough for r in rc.vctx.protected[c]}
    if rc.spec.op == "imbalance.class_weight" and "weight" in roles:
        rc.err(
            "WEIGHT_ROLE_PRESENT",
            "The dataset already has a sample-weight column; class weights are not combined "
            "with it silently. Remove the weight role or choose another method.",
        )
    dependent = bool(roles & GROUPING_ROLES)
    if rc.spec.op == "imbalance.smote":
        rc.params["dependent_rows"] = dependent
    if rc.spec.op == "imbalance.smote" and dependent:
        if rc.params.get("acknowledge_dependent_rows") is not True:
            rc.err(
                "DEPENDENT_ROWS_NOT_ACKNOWLEDGED",
                "Identifier, patient, group or time columns are declared, so rows are not "
                "independent: SMOTE would blend different patients, groups or times into "
                "synthetic rows. Prefer class weights; to use SMOTE anyway, set "
                "'acknowledge_dependent_rows' to true.",
                param="acknowledge_dependent_rows",
            )
        else:
            rc.warn(
                "GROUPED_OR_TEMPORAL_DATA",
                "Identifier, group, patient or time columns are declared. Synthetic rows "
                "interpolate between rows of different patients, groups or times and are not "
                "independent real samples; validate with group- or time-aware splits.",
            )
    if rc.spec.op == "imbalance.random_oversample" and "time" in roles:
        rc.warn(
            "TEMPORAL_DATA",
            "A time column is declared; duplicated rows repeat past observations. Validate with "
            "time-ordered splits.",
        )
    if rc.spec.op != "imbalance.smote":
        return
    candidates = {**rc.columns, **rc.maybe_removed}
    features = [c for c in candidates if c not in rc.vctx.protected]
    if rc.reserved_prefixes:
        rc.err(
            "FEATURE_NOT_NUMERIC",
            "One-hot outputs are true/false columns, which SMOTE cannot interpolate. Use "
            "random oversampling or class weights.",
        )
    if not features and not rc.reserved_prefixes:
        rc.err("NO_FEATURES", "SMOTE needs at least one feature column.")
    if len(features) > Limits.SMOTE_MAX_FEATURES:
        rc.err(
            "OP_LIMIT_EXCEEDED",
            f"SMOTE uses at most {Limits.SMOTE_MAX_FEATURES} feature columns.",
        )
    for name in features:
        state = candidates[name]
        if state.inferred_type != "numeric":
            rc.err(
                "FEATURE_NOT_NUMERIC",
                "SMOTE interpolates between rows, so every feature column must be numeric. "
                "Encode or remove this column first, or use random oversampling.",
                column=name,
            )
        elif state.n_missing:
            rc.err(
                "FEATURE_MISSING_VALUES",
                "SMOTE cannot interpolate missing values; impute this column first.",
                column=name,
            )


# ---- Registry ---------------------------------------------------------------------------------

RULES: dict[str, tuple[Check | None, Effect | None]] = {
    "impute.mean": (check_impute, effect_impute),
    "impute.median": (check_impute, effect_impute),
    "impute.mode": (check_impute, effect_impute),
    "impute.constant": (check_constant, effect_impute),
    "impute.knn": (check_predictors, effect_impute),
    "impute.iterative": (check_predictors, effect_impute),
    "missing.indicator": (check_indicator, effect_indicator),
    "outliers.flag": (check_outlier_flag, effect_outlier_flag),
    "outliers.cap": (check_outlier_cap, None),
    "outliers.signed_log": (None, effect_float),
    "outliers.isolation_forest_flag": (check_isolation_forest, effect_isolation_forest),
    "encode.onehot": (check_onehot, effect_onehot),
    "encode.ordinal": (check_ordinal, effect_ordinal),
    "encode.frequency": (None, effect_frequency),
    "scale.standard": (None, effect_float),
    "scale.minmax": (None, effect_float),
    "scale.robust": (None, effect_float),
    "scale.none": (None, None),
    "transform.log": (None, effect_float),
    "transform.log1p": (None, effect_float),
    "transform.yeo_johnson": (None, effect_float),
    "transform.quantile": (None, effect_float),
    "select.variance": (check_variance, effect_select),
    "select.correlation": (check_correlation, None),
    "select.mutual_information": (check_mutual_information, effect_select),
    "select.univariate": (check_univariate, effect_select),
    "select.drop_columns": (None, effect_drop_columns),
    "imbalance.class_weight": (check_imbalance, None),
    "imbalance.random_oversample": (check_imbalance, None),
    "imbalance.random_undersample": (check_imbalance, None),
    "imbalance.smote": (check_imbalance, None),
}
