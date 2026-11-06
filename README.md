# DataMedic

Privacy-first, domain-aware data intelligence and ML reliability platform.
Design documents live in [docs/](docs/); contributor and agent rules in [CLAUDE.md](CLAUDE.md).

**Current status: Phase 1 (foundation).** Accounts, secure sessions, per-user isolation with
PostgreSQL Row-Level Security, private storage, streamed uploads with content validation,
background jobs and health checks. Parsing, diagnostics, preprocessing and ML arrive in later
phases ([docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)).

## Repository layout

```
docker-compose.yml        local stack: postgres, redis, mailpit, migrate, api, worker, beat, web
.env.example              every setting (copy to .env)
backend/                  FastAPI API + Celery workers (Python 3.12, uv)
  src/datamedic/
    api/                  app factory, middleware, dependencies, health
    auth/                 signup, email verification, login, sessions, password reset
    datasets/             dataset API, streamed upload, file validation, job handlers
    jobs/                 job service, publisher (broker messages carry only job_id), API
    workers/              Celery app, job runner (claim_job + RLS), maintenance tasks
    db/                   models, RLS session hook, role bootstrap
    storage/              private storage interface + local filesystem adapter
    security/             Argon2id passwords, tokens
  migrations/             Alembic (0001: tables, RLS policies, triggers, grants, claim_job)
  tests/unit, tests/integration
apps/web/                 Next.js 16 + TypeScript + Tailwind + shadcn/ui
  src/app/                routes: /, /login, /signup, /verify-email, /reset-password,
                          /datasets, /datasets/[datasetId], /settings/security
  src/lib/api/            generated OpenAPI types + typed client
```

---

## Option A — Run everything with Docker Desktop (recommended)

Requirements: Windows 10/11 with [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) (WSL 2 backend).

In **PowerShell**, from the repository folder:

```powershell
Copy-Item .env.example .env
notepad .env        # replace every "change-me" password, then save
docker compose up --build
```

Then open:

| URL | What |
|---|---|
| http://localhost:3000 | Web app |
| http://localhost:8000/api/docs | API documentation (development only) |
| http://localhost:8025 | Mailpit: verification and password-reset emails land here |

First run: create an account at http://localhost:3000/signup, open the verification email in
Mailpit, click the link, sign in, and upload a CSV/XLSX/XLS/Parquet/JSON file (max 200 MB).

Stop with `Ctrl+C`; remove all data with `docker compose down -v`.

> The Docker setup was written in Phase 1 but **could not be executed on the development
> machine** (Docker was not installed). Treat the first `docker compose up` as a check.

---

## Option B — Run without Docker (Windows)

Use this if Docker is not available. Background jobs run in-process
(`DATAMEDIC_JOBS_EAGER=true`, development only), so Redis is not needed.

### 1. Install prerequisites

```powershell
# uv (Python package manager): https://docs.astral.sh/uv/getting-started/installation/
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12

# Node.js 24 LTS: https://nodejs.org  (check: node --version)
# PostgreSQL 16 for Windows: https://www.postgresql.org/download/windows/
#   During install, set a password for the "postgres" superuser and keep port 5432.
```

### 2. Configure

```powershell
Copy-Item .env.example .env
notepad .env
```

In `.env`: set the passwords, put the `postgres` password into `DATAMEDIC_BOOTSTRAP_ADMIN_URL`
and the other passwords into the three role URLs, and set:

```
DATAMEDIC_JOBS_EAGER=true
DATAMEDIC_EMAIL_BACKEND=console
```

(`console` prints verification links in the API terminal instead of sending email.)

### 3. Backend

```powershell
cd backend
uv sync --python 3.12
uv run python -m datamedic.db.bootstrap      # creates roles dm_migrator/dm_app/dm_maintenance + database
uv run alembic upgrade head                  # creates tables, RLS policies, triggers
uv run uvicorn datamedic.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Check: http://127.0.0.1:8000/readyz should return `{"status":"ok", ...}`.

### 4. Frontend (second PowerShell window)

```powershell
cd apps\web
npm ci
npm run dev
```

Open http://localhost:3000. The verification link appears in the backend terminal.

### Optional: a real worker without Docker

Redis does not run natively on Windows; use WSL or Docker for it. With Redis on
`localhost:6379` and `DATAMEDIC_JOBS_EAGER=false`:

```powershell
cd backend
uv run celery -A datamedic.workers.tasks worker -Q ingest,maintenance --pool=solo --loglevel INFO
uv run celery -A datamedic.workers.tasks beat --loglevel INFO
```

(`--pool=solo` is required on Windows; containers use the default prefork pool.)

---

## Tests and checks

Backend (from `backend/`):

```powershell
uv run pytest tests/unit                       # no database needed
$env:TEST_ADMIN_DATABASE_URL = "postgresql://postgres:<password>@127.0.0.1:5432/postgres"
uv run pytest                                  # unit + integration (creates and drops a temp database)
uv run ruff check src migrations tests
uv run ruff format --check src migrations tests
uv run mypy
```

Integration tests need a PostgreSQL superuser URL because they create the `dm_*` roles and a
throwaway database. With Docker: `docker compose up -d db` and use
`postgresql://postgres:<POSTGRES_PASSWORD>@127.0.0.1:5432/postgres` after adding
`ports: ["127.0.0.1:5432:5432"]` to the `db` service.

Frontend (from `apps/web/`):

```powershell
npm test            # Vitest
npm run typecheck   # tsc
npm run lint        # ESLint
npm run build       # production build
```

Regenerate the typed API client after backend API changes:

```powershell
cd backend; uv run python scripts/export_openapi.py ../apps/web/src/lib/api/openapi.json
cd ../apps/web; npm run gen:api
```

## Security model (summary)

- Identity comes only from the server-side session cookie (HttpOnly, SameSite=Lax; `__Host-`
  prefix when Secure). Client-supplied user/owner IDs are rejected (`extra="forbid"`) or ignored.
- Every dataset/job/file operation is owner-scoped in the service layer **and** enforced by
  PostgreSQL Row-Level Security; other users' resources return 404.
- Unsafe requests require the `X-CSRF-Token` header; foreign `Origin` headers are rejected.
- Files are stored privately under opaque keys, write-once; there are no public file URLs.
- Uploads are size-checked while streaming and validated by magic bytes; XLSX archives are
  inspected for macros, external links, unsafe paths and zip-bomb ratios.
- Broker messages contain only `{"job_id": ...}`; logs are scrubbed of dataset content.
- Secrets live only in environment variables (`.env`, never committed).
