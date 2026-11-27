"""Business (customer) module: roles and check specs. Standard library only."""

from datamedic.engine.checks.spec import CheckSpec, DomainSpec, Requirements, RoleSpec

KEY = "business"
_M = "datamedic.domains.business.checks"

ROLES = (
    RoleSpec(
        "biz.customer_id",
        "Customer ID",
        "Identifies the customer.",
        KEY,
        max_per_version=1,
        is_feature=False,
        is_sensitive=True,
    ),
    RoleSpec(
        "biz.segment",
        "Segment",
        "Customer segment, region, plan or channel.",
        KEY,
        allowed_types=frozenset({"categorical", "boolean", "text", "numeric"}),
    ),
)

CUSTOMER = frozenset({"biz.customer_id"})


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
        id=f"biz.{name}",
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
        "customers.duplicates",
        "duplicates",
        "Duplicate customer records",
        "Customer IDs occurring more than once, and IDs that collide after removing case, "
        "spaces and leading zeros.",
        "duplicate_customers",
        requirements=Requirements(roles=CUSTOMER),
        limitations=("Several rows per customer can be intended (one row per order).",),
    ),
    _spec(
        "customers.suspicious_ids",
        "validity",
        "Suspicious customer identifiers",
        "Placeholder IDs ('0', '-1', 'test', 'unknown'...), surrounding spaces, and a mix of "
        "ID formats (letters/digits pattern) where one format covers < 90% of IDs.",
        "suspicious_ids",
        requirements=Requirements(roles=CUSTOMER),
        limitations=("Formats can change legitimately between systems.",),
    ),
    _spec(
        "leakage.churn",
        "leakage",
        "Fields recorded only after churn",
        "Features whose missing-value pattern matches a binary target (agreement >= 95%), "
        "the signature of fields filled in only after the outcome (e.g. a cancellation "
        "date).",
        "post_outcome_fields",
        requirements=Requirements(
            roles=frozenset({"target"}), task_types=frozenset({"binary"}), min_rows=20
        ),
        limitations=(
            "A matching pattern is strong evidence but not proof; confirm when "
            "each field is recorded.",
        ),
    ),
    _spec(
        "segments.imbalance",
        "balance",
        "Segment imbalance",
        "Segments with fewer than 30 rows or under 1% of rows, a segment holding more than "
        "90% of rows, and (with a target) the spread of the target rate across segments.",
        "segment_imbalance",
        requirements=Requirements(roles=frozenset({"biz.segment"})),
        limitations=("Small segments are not errors, but results for them are unreliable.",),
    ),
)

MODULE = DomainSpec(
    key=KEY,
    name="Business (customers)",
    version="1.0.0",
    description="Customer-level business data such as CRM or churn datasets.",
    roles=ROLES,
    checks=CHECKS,
)
