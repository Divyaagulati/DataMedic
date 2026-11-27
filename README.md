# DataMedic

**A privacy-first, domain-aware data intelligence and ML reliability platform.**

DataMedic takes a tabular dataset from upload to a trustworthy, reproducible model evaluation:
it profiles and diagnoses the data, lets you explore it interactively, tests preprocessing ideas
without leakage, applies approved transformations as new immutable versions, trains baseline
models with leakage-controlled evaluation, audits their reliability, and exports a consolidated
Data Passport.

> **Live demo:** <https://20-196-218-255.sslip.io> — deployed on Microsoft Azure (Ubuntu 24.04,
> Docker Compose, Caddy with Let's Encrypt HTTPS); see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
> This is a demonstration MVP: use synthetic or non-sensitive data. Biomedical features are for
> research use only; DataMedic is not a medical device. Verification emails may arrive in the
> spam folder (shared-hosting domain).

## Why DataMedic

Most analytics and AutoML tools make it easy to produce a number and hard to know whether to
trust it. Common failure modes are silent: patients split across training and test folds,
preprocessing fitted on all rows, identifiers used as predictors, imputation that hides
informative missingness, tiny groups that identify people, and charts that mislead. DataMedic is
built around the opposite defaults:

- **Honest results.** A check that could not run is *not assessed*, never "passed". Statistical
  suspicions are labelled as such. Exploratory evaluations are flagged as optimistic. There is no
  single magic reliability score.
- **Leakage control by construction.** Frozen, group- and time-aware folds; every learned step
  fitted on training rows only; committed full-data steps replayed inside folds or blocked.
- **Privacy first.** Per-user isolation enforced by PostgreSQL Row-Level Security, private
  storage, protection of sensitive columns across versions and results, small-group suppression,
  and no dataset content sent to external services.
- **Reproducibility.** Immutable versions with content hashes, hash-bound approvals, recorded seeds
  and library versions, and generated Python code for every version.

## Key features

| Area | What it does |
|---|---|
| **Upload & versions** | CSV/TSV, XLSX/XLS, Parquet and JSON parsed in a resource-limited sandbox into immutable versions; previews; background profiling. |
| **Diagnostics** | Universal checks plus biomedical, financial, business and industrial modules driven by declared column roles, with evidence, severity, limitations and suggested actions. |
| **Analytics Studio** | No-code chart builder (bar, horizontal/grouped/stacked bar, line, area, scatter, bubble, histogram, box, pie/donut, correlation and missingness heatmaps, tables) with filters, recommendations and misleading-chart guards; KPI cards; pivot tables with totals; saved dashboards bound to immutable versions; automated evidence-based insights; visual version comparison; PNG/SVG/CSV/JSON exports. |
| **Biomedical Intelligence Lite** | Patients and repeated observations, target balance, biomarker distributions and outliers, patient-level comparisons with effect sizes and Benjamini–Hochberg q-values, site/batch effects, confounding warnings, patient-leakage checks. Research use only; groups under 10 suppressed. |
| **Intelligent preprocessing & Experiment Lab** | Operation catalogue with risk and stage flags, evidence-based recommendations, side-by-side comparison of candidate pipelines on one frozen split, impact previews, hash-bound approvals, commits into new versions, revert, before/after comparison and `pipeline.py`. |
| **AutoML & Reliability Auditor** | Binary, multiclass and regression baselines (dummy, linear, random forest, gradient boosting) with bounded nested selection, fold dispersion and bootstrap intervals, ROC/PR/calibration/residual charts, permutation importance, error slices, and a 20-check reliability audit. No trained model is persisted. |
| **Analytical Copilot** | A **rule-based**, read-only assistant: analytical questions ("average amount by region", "histogram of age", "monthly trend") become validated queries whose numbers come from the query engine, with the interpreted query and a chart; report questions are answered from stored results with references. No language model, no API key. |
| **Data Passport & exports** | One consolidated report (quality, findings, recommendations, provenance and hashes, experiments, models, audits, explainability, insights, dashboards, limitations) with print-to-PDF, plus a JSON report, `pipeline.py` and an acknowledged, formula-safe CSV export. |

## Architecture

```mermaid
flowchart LR
  B[Browser] -- HTTPS --> C[Caddy<br/>reverse proxy, TLS]
  C -- "/" --> W[Next.js 16 web app]
  C -- "/api, /healthz" --> A[FastAPI API<br/>auth, RLS session, analytics queries]
  A --> P[(PostgreSQL 16<br/>Row-Level Security)]
  A -- job_id only --> R[(Redis broker)]
  R --> WI[Celery worker: ingest<br/>parse sandbox]
  R --> WA[Celery worker: analysis<br/>profiling, diagnostics, Lab,<br/>training, audits, insights, exports]
  BT[Celery beat<br/>maintenance] --> R
  WI --> S[(Private storage volume)]
  WA --> S
  A --> S
  WI --> P
  WA --> P
```

Only Caddy is exposed in production; the database, Redis, API, workers and web server stay on an
internal network. Broker messages carry only a job id. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Technology stack

- **Backend:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, Celery, Redis, PostgreSQL 16 (RLS), pandas, NumPy, PyArrow, SciPy, scikit-learn, imbalanced-learn; uv for dependency locking.
- **Frontend:** Next.js 16, React 19, TypeScript (strict), Tailwind CSS 4, shadcn/ui, TanStack Query, Recharts, an OpenAPI-generated typed client.
- **Infrastructure:** Docker Compose (development and production), Caddy (automatic HTTPS), Mailpit (local email).
- **Quality:** pytest (unit + PostgreSQL integration, RLS and AuthZ matrix), Vitest, ruff, mypy (strict), ESLint, TypeScript.

## Project structure

```
docker-compose.yml        local stack: postgres, redis, mailpit, migrate, api, worker-ingest, worker, beat, web
docker-compose.prod.yml   production stack behind Caddy (see docs/DEPLOYMENT.md)
deploy/                   Caddyfile, .env.production.example, backup.sh, server_setup.sh (one-command VM setup)
demo/                     presentation dataset builder, API-only provisioning and verification scripts
.env.example              every development setting (copy to .env)
DEVELOPMENT_GUIDE.md      architecture invariants, security and library rules for contributors
backend/                  FastAPI API + Celery workers (Python 3.12, uv)
  src/datamedic/
    api/ auth/ jobs/ workers/ db/ storage/ security/   platform: app, sessions, jobs, RLS, private storage
    datasets/ versions/   upload, parse sandbox, immutable versions, previews, profiles
    engine/               pure analysis engine: checks, preprocessing, ML, biomedical, analytics
    domains/              biomedical, financial, business, industrial modules
    analyses/             roles and diagnostics
    recommendations/ experiments/ pipelines/ preprocessing/   preprocessing and Experiment Lab
    training/             training runs, reliability audits, sensitive-column protection
    biomedical/           Biomedical Intelligence Lite
    analytics/            Analytics Studio API, dashboards, insights, natural-language parser
    reports/ exports/     Data Passport, Copilot, exports
  migrations/             Alembic 0001–0009 (expand-only)
  tests/unit, tests/integration, tests/fixtures (synthetic only)
apps/web/                 Next.js app (routes under src/app, components under src/components)
docs/                     PRD, architecture, API, database, privacy, ML engine, domains, roadmap, deployment
```

## Verified test results (2026-11-27)

| Suite | Result |
|---|---|
| Backend full regression (Phase 5 closure) | 1,523 passed, 4 skipped; one pre-existing flaky test fixed and re-run green |
| Backend Phase 8 regression (unit suite + analytics, MVP, isolation-matrix, database-security and related integration suites) | 1,447 passed, 4 skipped, 0 failed |
| Final closure targeted tests | backend 44 unit + 13 integration passed; frontend 105 passed (18 files) |
| Static checks | ruff, format, strict mypy, TypeScript, ESLint, production builds — all pass |
| Live end-to-end smoke (dev stack) | 15/15 Analytics Studio checks; 22/22 MVP checks |
| Production rehearsal (`docker-compose.prod.yml`, HTTPS via Caddy, clean database) | 22/22 and 14/14 |
| Presentation demo verification (`demo/verify_presentation.py`), local and Azure | 41/41 each (incl. cross-account isolation); 40/40 again after a full restart / VM reboot |
| Azure production checks | valid HTTPS certificate, HTTP→HTTPS redirect, 5 security headers, ports 5432/6379/8000/3000 closed |
| Regression tests for the final fixes | training with class weights / oversampling (run and rerun); auth incl. resend cooldown (14 passed); email headers; frontend 105 passed |

The 4 skips are pre-existing parametrisations (`tests/unit/test_check_framework.py:91`).

## Demonstration dataset (reproducible)

The live and local demos use a public, de-identified research dataset: **UCI Diabetes 130-US
Hospitals for Years 1999–2008** (<https://doi.org/10.24432/C5230J>, CC BY 4.0). Task: predict
**readmission within 30 days**, at discharge. `demo/build_presentation_dataset.py`
rebuilds the subset from the original archive: death/hospice discharges excluded, **patient-level**
sampling (seed `20261127`) to 4,000 encounters of 2,864 patients (11.2% readmitted), documented
missing-value normalisation and a binary target; every step and both SHA-256 hashes go into
`provenance.json`. The data itself is not committed.

```bash
python -I demo/build_presentation_dataset.py diabetes_130_us_hospitals.zip <out_dir>
python -I demo/provision_presentation.py --web https://<domain> --credentials <account.env>   --dataset <out_dir>/diabetes_readmission_presentation.csv --state <out_dir>/state.json
python -I demo/verify_presentation.py --web https://<domain> --credentials <account.env> --state <out_dir>/state.json
```

Provisioning uses only the public API (normal sign-up and email verification, no database access)
and prepares diagnostics, the biomedical summary, recommendations, an Experiment Lab comparison, a
committed version, two training runs with reliability audits, insights, the dashboard *Hospital
Readmission Intelligence* and exports. Observed results (patient-grouped 5-fold, nested tuning):
ROC-AUC random forest 0.631 ± 0.034, logistic regression 0.623 ± 0.021, gradient boosting
0.594 ± 0.024, dummy 0.500 — a modest, honestly reported signal. Walkthrough:
[docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md), one page:
[docs/PRESENTATION_CHEATSHEET.md](docs/PRESENTATION_CHEATSHEET.md), details: [demo/README.md](demo/README.md).

## Privacy and security safeguards

- Identity from a server-side session cookie only (`__Host-`, Secure, HttpOnly, SameSite); CSRF tokens on unsafe requests; Argon2id passwords; email verification before uploads.
- Owner scoping in the service layer **and** PostgreSQL Row-Level Security; another user's resource returns 404; an auto-generated AuthZ matrix covers every route.
- Private, write-once storage with opaque keys; no public file URLs; downloads only through authorised endpoints with `Cache-Control: no-store`.
- Sensitive columns (sensitive on any version, or derived from one) are protected in results, analytics (count-only), the Copilot, the passport and exports; small groups (< 10) are suppressed in privacy mode; scatter plots are binned there.
- No SQL or code accepted from users (structured query specifications only); spreadsheet formula injection neutralised in exports.
- No external AI or analytics services receive dataset content; logs carry ids and codes, not values or column names.
- Full details: [docs/PRIVACY_AND_SECURITY.md](docs/PRIVACY_AND_SECURITY.md).

## Scientific limitations

- Results describe the uploaded data only; associations are not causal; no external validation exists.
- "Leakage-controlled" covers the implemented split and preprocessing safeguards, not external validity, absence of bias or clinical reliability. Choosing the highest of several scores is optimistic.
- Statistical tests assume independence unless an entity role is declared; temporal tests can overstate evidence under autocorrelation.
- Small-group suppression does not protect against differencing across many queries; no differential privacy is claimed.
- The Copilot understands a fixed set of question patterns; it is not a generative model.
- Analytics are limited to versions of up to 200,000 rows / 5 million cells on a single server.
- Single-VM deployment without high availability; the arm64 build is prepared but has only been run on x86-64.
- Account emails use a shared sslip.io hostname and a personal SMTP sender, so they may be filtered as spam; a custom domain with SPF/DKIM is recommended for wider use.

## Future enhancements

Optional, privacy-safe external or local language models for the Copilot (evaluated against a
labelled question set), public dashboard sharing with access controls, server-side PDF rendering,
a nonce-based Content Security Policy, horizontal scaling, and advanced omics analyses. See
[docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md).

## Production deployment

DataMedic deploys to a single Ubuntu server with Docker Compose, Caddy and automatic HTTPS.
The live instance runs on **Microsoft Azure** (Standard_B2s_v2: 2 vCPU, 8 GB RAM, 64 GB SSD,
Korea Central) at <https://20-196-218-255.sslip.io>. Step-by-step instructions, sizing, hosting
options, the Azure CLI commands used, backups, restore, rollback and cost control:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

```bash
# on the server, in a checkout of this repository
cp deploy/.env.production.example .env.production   # fill in every value; never commit it
./deploy/server_setup.sh   # swap, Docker, firewall, image builds, migrations, health check, nightly backups
```

Contributor guide (architecture invariants, security and library rules): [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).

---

## Local installation

### Option A — Run everything with Docker Desktop (recommended)

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

After pulling new code, run `docker compose up --build` again: the `migrate` service applies new
database migrations automatically, and existing data is kept.

---

### Option B — Run without Docker (Windows)

Use this if Docker is not available. Background jobs run in-process
(`DATAMEDIC_JOBS_EAGER=true`, development only), so Redis is not needed.

#### 1. Install prerequisites

```powershell
# uv (Python package manager): https://docs.astral.sh/uv/getting-started/installation/
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12

# Node.js 24 LTS: https://nodejs.org  (check: node --version)
# PostgreSQL 16 for Windows: https://www.postgresql.org/download/windows/
#   During install, set a password for the "postgres" superuser and keep port 5432.
```

#### 2. Configure

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

#### 3. Backend

```powershell
cd backend
uv sync --python 3.12
uv run python -m datamedic.db.bootstrap      # creates roles dm_migrator/dm_app/dm_maintenance + database
uv run alembic upgrade head                  # creates tables, RLS policies, triggers
uv run uvicorn datamedic.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Check: http://127.0.0.1:8000/readyz should return `{"status":"ok", ...}`.

#### 4. Frontend (second PowerShell window)

```powershell
cd apps\web
npm ci
npm run dev
```

Open http://localhost:3000. The verification link appears in the backend terminal.

#### Optional: a real worker without Docker

Redis does not run natively on Windows; use WSL or Docker for it. With Redis on
`localhost:6379` and `DATAMEDIC_JOBS_EAGER=false`:

```powershell
cd backend
uv run celery -A datamedic.workers.tasks worker -Q ingest,analysis,evaluate,execute,maintenance --pool=solo --loglevel INFO
uv run celery -A datamedic.workers.tasks beat --loglevel INFO
```

(`--pool=solo` is required on Windows; containers use the default prefork pool.)

---

## Resource limits

Docker Compose is sized for a Docker VM with about 8 GB of memory.

| Service | Runs | Concurrency | Memory cap |
|---|---|---|---|
| `worker-ingest` | upload validation + parsing (queue `ingest`) | 1 | each parse runs in a sandbox process capped at `DATAMEDIC_PARSE_MEMORY_LIMIT_MB` (default 2500); container 3 GB |
| `worker` | profiling, diagnostics, experiments, impact previews, commits, reverts, version comparisons, maintenance (queues `analysis`, `evaluate`, `execute`, `maintenance`) | 1 | each worker process capped at `DATAMEDIC_WORKER_MEMORY_LIMIT_MB` (default 2000, `RLIMIT_DATA`); container 2.5 GB |

A job that reaches its cap fails with `LIMIT_EXCEEDED` (not a crash), and the dataset stays
recoverable. Measured at the 20-million-cell limit: parsing about 50 s, profiling about 1 GB.

To change the caps, set the variables in `.env` and run `docker compose up -d` again:

- **More memory available:** raise both caps, or raise `--concurrency` for `worker` in `docker-compose.yml`. Keep the sum of (concurrency × cap) plus about 1.5 GB for the other services below the Docker VM's memory.
- **Less memory:** lower the caps. Large files then fail with `LIMIT_EXCEEDED` instead of exhausting the VM.

Stuck work is recovered automatically every minute by the `beat` service:
- Jobs whose heartbeat is older than their time limit plus 5 minutes are failed (`JOB_STALE`).
- A dataset left in "validating" with no active ingest job becomes "Parse failed", and **Retry parsing** works.
- An upload interrupted for over an hour (`DATAMEDIC_UPLOAD_STALE_AFTER_SECONDS`) is reopened for upload.

## Tests and checks

Backend (from `backend/`):

```powershell
uv run pytest tests/unit                       # no database needed
$env:TEST_ADMIN_DATABASE_URL = "postgresql://dm_test_admin:<password>@127.0.0.1:55432/postgres"
uv run pytest                                  # unit + integration (creates and drops a temp database)
uv run ruff check src migrations tests
uv run ruff format --check src migrations tests
uv run mypy
```

### Disposable test database

Integration tests (RLS, isolation matrix, jobs) must not be skipped. They need
`TEST_ADMIN_DATABASE_URL`: an administrator of a **dedicated, disposable** PostgreSQL instance.
Never point it at the application's database or at production credentials.

Required permissions: the test harness creates the `dm_migrator`, `dm_app` and
`dm_maintenance` roles and a database named `datamedic_test_<random>` (dropped afterwards).
`dm_maintenance` has `BYPASSRLS`, and PostgreSQL 16 lets only a **superuser** create such a role,
so the test administrator must be a superuser **of the disposable instance only**. The
application roles themselves are created `NOSUPERUSER`, and RLS is tested exactly as in production.

A throwaway instance with Docker (data in memory, reachable only from this machine):

```powershell
$pw = [guid]::NewGuid().ToString("N")
docker run -d --name datamedic-testdb -e POSTGRES_USER=dm_test_admin -e POSTGRES_PASSWORD=$pw `
  -p 127.0.0.1:55432:5432 --tmpfs /var/lib/postgresql/data postgres:16.15-alpine
$env:TEST_ADMIN_DATABASE_URL = "postgresql://dm_test_admin:$pw@127.0.0.1:55432/postgres"
# ... run the tests ...
docker rm -f datamedic-testdb   # removes the instance and all its data
```

Keep the URL in a local environment variable only; do not commit it or put it in `.env`.

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
- Sensitive columns (sensitive on any version of a dataset, or derived from one) are protected in
  training results, biomedical summaries, the passport and exports; exports require explicit
  acknowledgement and are served only through an authorised download (`Cache-Control: no-store`).
- Production (docs/DEPLOYMENT.md): only Caddy is exposed (HTTPS); secure `__Host-` cookies;
  authenticated TLS SMTP is mandatory; database and Redis are never published.
