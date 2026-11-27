# DataMedic — Development Guide

Guidance for contributors working in this repository: architecture invariants, security rules, library usage and conventions.

## Status
Planning documents are at **revision 2 (MVP-scoped)**. Phases 1–4 are complete. Phase 5 was approved by the owner on 2026-11-26 (with the conservative privacy policy: a column sensitive on any version of a dataset stays protected in historical and future results). The owner then approved one combined Phase 6/7 + deployment milestone (MVP completion): Biomedical Intelligence Lite, Data Passport, exports, the built-in Copilot, UI polish and production deployment (`docs/DEPLOYMENT.md`). It was accepted on 2026-11-27 together with Phases 1–7. Phase 8 (Analytics Studio, the final feature milestone) was browser-verified by the owner and closed on 2026-11-27 after two fixes (Copilot insights routing; widget edit restoring sort/limit). The product is ready for public deployment (docs/DEPLOYMENT.md); do not add features, deploy to external infrastructure or purchase resources without explicit owner authorisation. It was deployed on Microsoft Azure on 2026-11-27 (docs/DEPLOYMENT.md §6, `deploy/server_setup.sh`); the reproducible presentation demo lives in `demo/` (docs/PRESENTATION_GUIDE.md). Owner milestone numbers differ from the original roadmap phases (see the roadmap's "Phase numbering"). The design lives in `docs/`:

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
- Phase 4 decisions: preprocessing ops use scikit-learn plus in-house transformers (no Feature-engine, D2); persisted pipeline artifacts are JSON only, never pickle/joblib (D4); `model`-stage ops (scaling, encoding, feature selection, resampling) are never committed into a version (D1). See `docs/PHASE4_PLAN.md`.
- A commit (new version) may use only `PreprocessExecutor.commit_transform`, which refuses model-stage steps; `fit_transform` is not a commit path. Experiments (`experiments/`, `engine/preprocess/experiment/`) never write versions, and every candidate uses the stored frozen split. Recommendation rules (`engine/preprocess/recommend/`, standard library only) read measured evidence and roles, never column names.
- Part E: `pipelines.service.commit_readiness` is the one authoritative commit check (API before queueing, worker before reading data and again under the dataset lock). Only data-stage steps are materialised; model-stage steps are stored as `model_pipeline_spec`. Approvals are valid only while their hashes, preview and source equal the current ones; never rewrite approvals, transformation runs or version events. Version code generation must not import data libraries (engine sources are read as text).
- Preprocessing algorithms live only in `engine/preprocess/runtime.py`, which code generation embeds verbatim: it may import only the standard library, NumPy, pandas and scikit-learn.
- Phase 5: training never writes versions. A version fitted on all rows is evaluated only through `engine/ml/eligibility.py` replay (fidelity proven by `replay.establish` in the worker) or as a labelled `exploratory` run; never weaken group/time splits for tuning; inner folds come only from outer training rows; no trained model is persisted (no pickle/joblib); audit findings state their `basis` and never call a statistical suspicion confirmed; no numeric reliability score.
- Phase 5 closure: training results are served through `training/privacy.py` (training-time, current and lineage-derived sensitivity; filter copies, never rewrite stored rows); never enable HistGradientBoosting internal early stopping (its random validation split ignores groups/time); user-facing text says "leakage-controlled", never "unbiased" (internal ids `unbiased_direct`/`unbiased_replay` stay); list headlines are "highest observed", never a winner.
- Generated code comes only from per-op templates. Emit values as `repr` of JSON primitives; never interpolate user strings as code.

- MVP completion: the biomedical summary (`engine/bio/summary.py`) is descriptive and research-use-only (no clinical, causal or discovery claims); groups under 10 units are suppressed; missing roles give `not_assessed`. The Data Passport (`reports/passport.py`) and Copilot (`reports/copilot.py`) only read stored results, never recompute, write or score; the Copilot is rule-based with no external model (an optional LLM must be explicitly enabled, server-side only, and never receive rows, column names, identifiers or sensitive values). Exports run as worker jobs, drop `_dm_` and sensitive columns unless separately acknowledged, neutralise spreadsheet formulas, and are downloaded only through `/exports/{job_id}/download`.

- Phase 8 (Analytics Studio): queries are structured specifications only (`engine/analytics/query.py`), never SQL or expressions; validate types without coercion; keep null semantics, limits and privacy rules (protected columns count-only; privacy-mode suppression and binned scatter); interactive queries run in the API under `analytics/data.py` bounds, whole-dataset insights in the worker. Dashboards store specifications only and never switch version silently. The Copilot's NL layer (`analytics/nl.py`) is deterministic; every number comes from the engine. Hugging Face models are deferred (ARCHITECTURE §18).

## Pinned baseline (confirmed in the Phase 1 compatibility report)
Installed (see `backend/uv.lock`): Python 3.12 · pandas 2.3.3 · NumPy 2.5 · pyarrow · scikit-learn 1.9.1 · imbalanced-learn 0.14.2 · SciPy 1.18 · threadpoolctl. Deferred, not installed: XGBoost, Cleanlab, statsmodels, SHAP, YData Profiling, PyOD, Evidently.

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
Integration tests need `TEST_ADMIN_DATABASE_URL` for a disposable PostgreSQL instance (README "Disposable test database"); never skip them to avoid configuring one.
New API routes must keep `tests/integration/test_isolation_matrix.py` passing (extend `KNOWN_PARAMS` for new path parameters).
