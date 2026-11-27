"""Domain modules (DOMAIN_MODULES.md). Each package exposes ``MODULE`` (a DomainSpec, standard
library only) and keeps its check implementations in a ``checks`` submodule that is imported
only by the worker. ``bootstrap.build_registry()`` is the composition root."""
