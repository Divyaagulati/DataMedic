"""Composition root: the only place that lists domain modules. Core packages never import
them (CLAUDE.md invariant 11); adding a module means adding one line here."""

from functools import lru_cache

from datamedic.domains import biomedical, business, financial, industrial
from datamedic.engine.checks.universal import MODULE as UNIVERSAL_MODULE
from datamedic.engine.registry import Registry

DOMAIN_MODULES = (biomedical.MODULE, financial.MODULE, business.MODULE, industrial.MODULE)


@lru_cache
def build_registry() -> Registry:
    registry = Registry(universal=UNIVERSAL_MODULE)
    for module in DOMAIN_MODULES:
        registry.register(module)
    return registry
