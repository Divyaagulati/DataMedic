"""Financial (transactions) module: roles and check specs. Standard library only."""

from datamedic.engine.checks.spec import (
    NUMERIC_TYPES,
    TIME_TYPES,
    CheckSpec,
    DomainSpec,
    Requirements,
    RoleSpec,
)

KEY = "financial"
_M = "datamedic.domains.financial.checks"

ROLES = (
    RoleSpec(
        "fin.transaction_id",
        "Transaction ID",
        "Identifies one transaction.",
        KEY,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "fin.account_id",
        "Account ID",
        "Identifies the account or customer.",
        KEY,
        max_per_version=1,
        is_feature=False,
        is_sensitive=True,
    ),
    RoleSpec(
        "fin.timestamp",
        "Transaction time",
        "When the transaction happened.",
        KEY,
        allowed_types=TIME_TYPES,
        max_per_version=1,
        is_feature=False,
    ),
    RoleSpec(
        "fin.amount",
        "Amount",
        "Monetary amount.",
        KEY,
        allowed_types=NUMERIC_TYPES,
        max_per_version=1,
    ),
)

TIME = frozenset({"fin.timestamp"})


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
        id=f"fin.{name}",
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
        "timestamps.consistency",
        "time",
        "Timestamp inconsistencies",
        "Transaction times in the future or before 1970, missing times, and one transaction "
        "ID recorded with different times.",
        "timestamp_consistency",
        requirements=Requirements(roles=TIME),
        limitations=("Row order is never used as time order. Time zones are taken as parsed.",),
    ),
    _spec(
        "transactions.duplicates",
        "duplicates",
        "Duplicate transactions",
        "Repeated transaction IDs; without an ID, the same account, amount and time "
        "recorded more than once.",
        "duplicate_transactions",
        requirements=Requirements(roles_any=frozenset({"fin.transaction_id", "fin.account_id"})),
        limitations=("Identical account, amount and time can be legitimate (split payments).",),
    ),
    _spec(
        "leakage.temporal",
        "leakage",
        "Temporal leakage risk",
        "Whether the target rate (or mean) changes over time: rows are ordered by the "
        "declared transaction time and split into 5 equal periods; chi-square "
        "(classification) or Kruskal-Wallis (regression), p < 0.01.",
        "temporal_leakage",
        requirements=Requirements(roles=TIME | {"target"}, min_rows=50),
        limitations=(
            "Detects non-stationarity of the outcome; it cannot detect features "
            "computed with future information.",
        ),
    ),
    _spec(
        "amounts.unusual",
        "outliers",
        "Unusual amounts",
        "Non-finite amounts, zero amounts, negative amounts, extreme amounts (modified "
        "z-score of log(1 + |amount|) > 3.5) and heavy concentration on round numbers.",
        "unusual_amounts",
        requirements=Requirements(roles=frozenset({"fin.amount"})),
        limitations=(
            "Negative amounts may be refunds or reversals; extreme amounts can be genuine.",
        ),
    ),
)

MODULE = DomainSpec(
    key=KEY,
    name="Financial (transactions)",
    version="1.0.0",
    description="Transaction-level financial data.",
    roles=ROLES,
    checks=CHECKS,
)
