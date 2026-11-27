"""Industrial (sensor) module: roles and check specs. Standard library only."""

from datamedic.engine.checks.spec import (
    NUMERIC_TYPES,
    TIME_TYPES,
    CheckSpec,
    DomainSpec,
    Requirements,
    RoleSpec,
)

KEY = "industrial"
_M = "datamedic.domains.industrial.checks"

SENSOR_ROLES = frozenset({"ind.sensor", "ind.sensor_nonneg", "ind.sensor_percent"})

ROLES = (
    RoleSpec(
        "ind.timestamp",
        "Measurement time",
        "When the measurement was taken.",
        KEY,
        allowed_types=TIME_TYPES,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "ind.asset_id",
        "Asset / machine ID",
        "Identifies the machine or asset.",
        KEY,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "ind.sensor",
        "Sensor reading",
        "A numeric sensor reading.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
    RoleSpec(
        "ind.sensor_nonneg",
        "Sensor reading (never negative)",
        "A reading that cannot be negative, e.g. flow, absolute pressure, concentration.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
    RoleSpec(
        "ind.sensor_percent",
        "Sensor reading (0-100 %)",
        "A percentage reading, e.g. valve opening or humidity.",
        KEY,
        allowed_types=NUMERIC_TYPES,
    ),
)

TIME = frozenset({"ind.timestamp"})


def _spec(
    name: str,
    category: str,
    title: str,
    description: str,
    func: str,
    *,
    requirements: Requirements | None = None,
    limitations: tuple[str, ...] = (),
) -> CheckSpec:
    return CheckSpec(
        id=f"ind.{name}",
        version="1.0.0",
        domain=KEY,
        category=category,
        title=title,
        description=description,
        impl=f"{_M}:{func}",
        requirements=requirements or Requirements(),
        limitations=limitations,
    )


CHECKS = (
    _spec(
        "sensors.impossible",
        "validity",
        "Impossible or saturated sensor values",
        "Non-finite readings; negative readings for never-negative sensors; readings "
        "outside 0-100 for percentage sensors; and saturation (>= 5% of readings exactly at "
        "the minimum or maximum, which suggests clipping).",
        "impossible_values",
        requirements=Requirements(roles_any=SENSOR_ROLES),
        limitations=(
            "Physical limits come only from the declared sensor type; bounds such "
            "as a calibrated range are not known.",
        ),
    ),
    _spec(
        "sampling.irregular",
        "time",
        "Irregular measurement intervals",
        "Intervals between consecutive measurements in time order (per asset when declared): "
        "duplicate timestamps, gaps over 5x the median interval, and a coefficient of "
        "variation of intervals above 0.5.",
        "irregular_sampling",
        requirements=Requirements(roles=TIME, min_rows=10),
        limitations=(
            "Sorted by the declared time, never by row order. Planned downtime looks like a gap.",
        ),
    ),
    _spec(
        "sensors.shift",
        "drift",
        "Sensor distribution shift",
        "Per sensor, readings in the earliest third vs the latest third of time "
        "(two-sample Kolmogorov-Smirnov, Benjamini-Hochberg q < 0.01 and a median shift of "
        ">= 0.5 IQR).",
        "distribution_shift",
        requirements=Requirements(roles=TIME, roles_any=SENSOR_ROLES, min_rows=30),
        limitations=("A shift can be a real change in operation, not sensor failure.",),
    ),
    _spec(
        "sensors.drift",
        "drift",
        "Gradual sensor drift",
        "Monotonic trend of each sensor over time (|Spearman| >= 0.5 with q < 0.01).",
        "sensor_drift",
        requirements=Requirements(roles=TIME, roles_any=SENSOR_ROLES, min_rows=30),
        limitations=(
            "A trend can reflect a real process change, wear or seasonality, not "
            "only calibration drift.",
        ),
    ),
)

MODULE = DomainSpec(
    key=KEY,
    name="Industrial (sensors)",
    version="1.0.0",
    description="Time-stamped sensor or machine data.",
    roles=ROLES,
    checks=CHECKS,
)
