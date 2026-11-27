"""Domain registry (DOMAIN_MODULES §2). The engine never imports a domain module; a
composition root (``datamedic.domains.bootstrap``) registers them here."""

from dataclasses import dataclass, field

from datamedic.engine.checks.spec import GENERIC, UNIVERSAL, CheckSpec, DomainSpec, RoleSpec


@dataclass
class Registry:
    universal: DomainSpec
    modules: dict[str, DomainSpec] = field(default_factory=dict)

    def register(self, module: DomainSpec) -> None:
        if module.key in (GENERIC, UNIVERSAL) or module.key in self.modules:
            raise ValueError(f"domain key already registered: {module.key}")
        ids = {c.id for c in self.all_checks()}
        roles = {r.key for r in self.all_roles()}
        for check in module.checks:
            if check.domain != module.key or check.id in ids:
                raise ValueError(f"invalid or duplicate check id: {check.id}")
        for role in module.roles:
            if role.domain != module.key or role.key in roles:
                raise ValueError(f"invalid or duplicate role: {role.key}")
        self.modules[module.key] = module

    def domain_keys(self) -> list[str]:
        return [GENERIC, *self.modules]

    def module(self, domain: str) -> DomainSpec | None:
        return self.modules.get(domain)

    def roles_for(self, domain: str) -> list[RoleSpec]:
        module = self.modules.get(domain)
        return [*self.universal.roles, *(module.roles if module else ())]

    def checks_for(self, domain: str) -> list[CheckSpec]:
        module = self.modules.get(domain)
        return [*self.universal.checks, *(module.checks if module else ())]

    def all_roles(self) -> list[RoleSpec]:
        return [*self.universal.roles, *(r for m in self.modules.values() for r in m.roles)]

    def all_checks(self) -> list[CheckSpec]:
        return [*self.universal.checks, *(c for m in self.modules.values() for c in m.checks)]

    def role(self, key: str) -> RoleSpec | None:
        return next((r for r in self.all_roles() if r.key == key), None)
