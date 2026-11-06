# CLAUDE.md — DataMedic

Guidance for AI coding agents and contributors working in this repository.

## Status
Planning documents are at **revision 2 (MVP-scoped)**. Do not write application code until the owner explicitly approves starting Phase 1. The design lives in `docs/`:

- `docs/PRD.md`: requirements R1–R11, MVP / Version 2 / Future tiers, ML-readiness checklist, quotas
- `docs/ARCHITECTURE.md`: containers, modules, lifecycles, versioning, jobs, dependencies, licenses, errors, testing, extension points
- `docs/DATABASE_SCHEMA.md`, `docs/API_SPECIFICATION.md`
- `docs/PRIVACY_AND_SECURITY.md`: auth, isolation, threat model
- `docs/ML_ENGINE_DESIGN.md`, `docs/DOMAIN_MODULES.md`
- `docs/IMPLEMENTATION_ROADMAP.md`: phases and compatibility checks

Only build what the current phase requires. Do not pull V2 or Future items into the MVP. If code and docs disagree, stop and raise it, and update the docs in the same change as any design decision.

## Non-negotiable invariants
1. **User isolation**: every user-owned row has `owner_id`. Every query is owner-scoped in the service layer *and* protected by Postgres RLS (`app.user_id`, set per transaction by the session hook). Another user's resource returns 404. Every route is covered by the auto-generated AuthZ matrix; never exclude a route.
2. **No dataset content leaves the platform**: no LLM, analytics or third-party service receives cell values, labels, rows or column names. The MVP has no LLM integration at all. Never log data values or column names.
3. **Broker messages carry only `job_id`**. Celery uses the JSON serializer only. Job parameters live in `jobs.params`.
4. **Originals are immutable**: nothing writes or deletes under `raw/` except the initial upload and the purge job.
5. **Versions are immutable and linear (MVP)**: a new version's parent is the current head. Revert creates a new version by server-side copy. Never modify a version object or row.
6. **`_dm_row_id` is sacred**: every op preserves it on surviving rows. Folds, row references and OOF predictions are keyed by it.
7. **Approvals gate commits only**: experiments may evaluate high-risk and destructive steps freely (they never write versions). Committing a high-risk step requires a valid hash-bound approval tied to a current impact preview, verified in the API **and** the worker. Risk, destructive and kind come from the op registry, never from the client.
8. **Honest diagnostics**: evidence, severity and limitations on every finding. Unmet requirements produce `not_run` with a reason. Disclose sampling. Never report "no issue found" for a check that did not run.
9. **Separation of concerns**: diagnostics, recommendations and experiments never write versions. Only the commit and revert paths do.
10. **Leakage safety**: in experiments, fitted ops and all row-removal ops use training rows only; test folds stay intact; duplicates and declared groups are grouped before splitting; folds are frozen by `_dm_row_id`; candidates not scored on the shared test cohort are flagged `NOT_DIRECTLY_COMPARABLE` and never ranked.
11. **Domain modularity**: core packages never import domain modules. Modules register through `domains.registry` and never alter universal findings.
12. **Job status lives only in `jobs`**. Run tables hold inputs and results.

## Library usage rules
- **Do not invent library APIs.** Confirm every third-party function, class and parameter in the official documentation for the pinned version in the lockfile. If unsure, write an adapter plus a contract test first.
- Third-party analytics are reached only through engine ports and adapters (`ports/`, `adapters/`), ops and concrete checks.
- MVP excludes YData Profiling (fg-data-profiling), PyOD and Evidently. They are V2 adapters behind the `Profiler`, `OutlierDetector` and `DriftComparator` ports. When added: use `fg-data-profiling` (import `data_profiling`) with sample sections disabled, and the current Evidently API (`from evidently import Report, Dataset, DataDefinition`).
- Storage: S3 API via boto3 only. Browsers never receive storage URLs in the MVP.
- No GPL, AGPL or SSPL dependencies without a recorded license decision (the CI license gate enforces this).
- Never `pickle`/`joblib.load` anything not created by the platform and HMAC-verified. Exports are pickle-free.
- Generated code comes only from per-op templates. Emit values as `repr` of JSON primitives; never interpolate user strings as code.

## Pinned baseline (confirmed in the Phase 1 compatibility report)
Python 3.12 · pandas 2.3.x · NumPy 2.x · scikit-learn 1.9.x · Feature-engine 1.9.x · XGBoost 3.4.x · Cleanlab 2.9.x (Apache-2.0) · SciPy · statsmodels.

## Code conventions
- Python: ruff, mypy (strict for `engine` and `contracts`), Pydantic v2, SQLAlchemy 2.x, Alembic expand/contract.
- `datamedic_engine` is pure: no FastAPI, SQLAlchemy, Celery or boto3 imports.
- Errors: typed `DomainError` subclasses with stable codes ([ARCHITECTURE.md §13](docs/ARCHITECTURE.md)); RFC 9457 problem+json.
- Frontend: TypeScript strict; generated API client only; shadcn/ui; no `dangerouslySetInnerHTML`; polling for jobs.
- Tests accompany every check (golden + `not_run`), op (row-ID preservation + codegen equivalence), rule and route (AuthZ matrix).
- Fixtures are synthetic. **Never commit real patient or personal data.**

## Repository note
This directory has its own git repository (initialised in Phase 1). The user's home directory (`C:/Users/Delll`) is also a git repository; never commit to it.

## Commands
Backend (in `backend/`): `uv sync`, `uv run pytest` (integration tests need `TEST_ADMIN_DATABASE_URL`, a PostgreSQL superuser URL), `uv run ruff check src migrations tests`, `uv run ruff format --check src migrations tests`, `uv run mypy`, `uv run alembic upgrade head`.
Frontend (in `apps/web/`): `npm test`, `npm run typecheck`, `npm run lint`, `npm run build`, `npm run gen:api` (after `uv run python scripts/export_openapi.py ../apps/web/src/lib/api/openapi.json` in `backend/`).
Stack: `docker compose up --build` (see README). Next.js 16 differs from older versions: read `apps/web/node_modules/next/dist/docs/` before changing frontend code.
New API routes must keep `tests/integration/test_isolation_matrix.py` passing (extend `KNOWN_PARAMS` for new path parameters).
