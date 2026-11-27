# DataMedic — Production deployment (single VM)

The MVP runs the same Docker Compose architecture as development on one Linux VM:
Caddy (HTTPS) → Next.js web and FastAPI API → PostgreSQL (RLS), Redis, Celery workers (ingest;
analysis/evaluate/execute/maintenance), Celery beat, and a private storage volume.

```
Internet ──443/80──► Caddy ──/api/*, /healthz──► api:8000 ──► db (PostgreSQL 16, RLS)
                         └──────── everything else ──► web:3000      redis (password)
                                       workers (ingest, analysis) ──► storage volume (private)
```

Only Caddy publishes ports. PostgreSQL, Redis, the API, workers and web stay on the internal
Docker network. The browser talks to one origin (`https://<domain>`), so CORS is limited to that
origin and session cookies are `__Host-`, `Secure`, `HttpOnly`, `SameSite`.

Files: `docker-compose.prod.yml`, `deploy/Caddyfile`, `deploy/.env.production.example`,
`deploy/backup.sh`.

## 1. Resource requirements (measured on the local stack)

| Component | Memory cap | Notes |
|---|---|---|
| analysis worker (profiling, diagnostics, Lab, training, audits, exports) | 2.3 GB (process cap 1.8 GB) | one job at a time |
| ingest worker (parse sandbox) | 2 GB (sandbox cap 1.5 GB) | one upload at a time |
| api (incl. Analytics Studio cache: ≤ 2 versions, ≤ 2 concurrent queries) | 1.25 GB cap, ~300–600 MB typical | Phase 8 |
| web, db, redis, beat, caddy | ~2 GB caps, ~0.7 GB typical | |

- Measured in the production rehearsal (2026-11-27, idle after the smoke tests): api 146 MiB, worker 292 MiB, ingest worker 178 MiB, web 87 MiB, db 50 MiB, beat 79 MiB, caddy 16 MiB.
- **Minimum:** 2 vCPU, 4 GB RAM + 4 GB swap, 30 GB disk (small synthetic datasets only).
- **Recommended:** 4 vCPU, 8 GB RAM, 40–80 GB disk.
- Training a 4-model nested run on 1,500 rows took ~77 s on one core; time scales with rows and folds.
- Images: backend ~1 GB, web ~0.5 GB. Base images are multi-arch and the Python dependencies publish aarch64 wheels, but an arm64 build has not been verified yet (built and verified on x86-64).

## 2. Hosting options

| Option | Fit | Cost |
|---|---|---|
| **Oracle Cloud Always Free, Ampere A1** (up to 4 OCPU / 24 GB, 200 GB) | Best free fit (arm64). Capacity in popular regions can be temporarily unavailable. | Free (card needed for identity) |
| **Hetzner Cloud CAX21** (4 vCPU arm64 / 8 GB / 80 GB) or CX32 (x86) | Recommended paid; fast to provision | ~€7–8 / month |
| DigitalOcean / Linode 8 GB droplet | Works | ~$48 / month |
| Google Cloud e2-micro, AWS free tier (1 GB RAM) | **Too small** for the ML workers | — |
| Render / Railway / Fly free tiers | No good fit: needs several always-on services, workers and persistent volumes | — |

No domain purchase is required: `203-0-113-10.sslip.io` (your IP with dashes) resolves to the
server and works with Let's Encrypt. Use your own domain if you have one.

## 3. First deployment (exact steps)

On the VM (Ubuntu 24.04 LTS):

```bash
# 1. Docker Engine + compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 2. Firewall: SSH, HTTP, HTTPS only (cloud firewall/security list must allow 80 and 443 too)
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw --force enable

# 3. Swap (recommended on 4–8 GB machines)
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile \
  && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. Code (copy the repository: git clone from your remote, or scp a release archive)
git clone <your-repository-url> datamedic && cd datamedic

# 5. Secrets: fill in every value; generate secrets with `openssl rand -hex 24`
cp deploy/.env.production.example .env.production && chmod 600 .env.production && nano .env.production

# 6. Build and start (migrations run automatically in the `migrate` service on a clean database)
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
docker compose -f docker-compose.prod.yml --env-file .env.production ps
curl -fsS https://$DOMAIN/healthz && curl -fsS https://$DOMAIN/readyz
```

If `--build` stops with "additional privileges requested" (Compose's bake builder), build the two
images directly and start without building:

```bash
set -a; . ./.env.production; set +a
docker build -t datamedic-backend:${DATAMEDIC_RELEASE:-latest} backend
docker build -t datamedic-web:${DATAMEDIC_RELEASE:-latest}   --build-arg NEXT_PUBLIC_API_BASE_URL=https://$DATAMEDIC_DOMAIN apps/web
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --no-build
```

SMTP is required: users must verify their email before uploading. Any provider with STARTTLS or
SSL works (for example Brevo, Mailjet, Resend or SendGrid free tiers, or a Gmail app password).
Production refuses plaintext SMTP and console email.

### 3a. First checks after deployment (no scripts needed)

1. Open `https://<domain>`: the landing page loads over HTTPS; `http://` redirects.
2. Sign up, open the verification email, sign in.
3. Upload a small **synthetic** CSV; wait for "ready"; open Profile and Diagnostics.
4. Analytics Studio: build a bar chart (mean of a number by a category), add a filter, save it to a new dashboard, reopen the dashboard; generate insights; ask the Copilot "What is the average <number> by <category>?".
5. Models: train the dummy and linear models; open the reliability audit.
6. Passport & Export: print the passport; download the CSV (acknowledge) and `pipeline.py`.
7. Sign up a second account and confirm it cannot open the first account's dataset URL (404).
8. `docker compose -f docker-compose.prod.yml --env-file .env.production exec db psql -U postgres -d datamedic -c "select version_num from alembic_version"` shows `0009`.

## 4. Operations

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.production"
$C logs -f --tail 100 api worker        # logs carry ids and codes, never data values
$C restart worker                       # restart one service
./deploy/backup.sh                      # PostgreSQL dump + storage archive into ./backups/<UTC time>
```

Schedule backups with cron, for example `0 3 * * * cd ~/datamedic && ./deploy/backup.sh`, and copy
`./backups` off the server.

**Restore** (into a stopped stack):

```bash
$C stop api worker worker-ingest beat web
gunzip -c backups/<stamp>/postgres.sql.gz | $C exec -T db psql -U postgres
$C run --rm --no-deps -T --entrypoint tar api -C /data -xzf - < backups/<stamp>/storage.tar.gz
$C up -d
```

**Update and rollback.** Tag each release so the previous images stay available:

```bash
git pull && DATAMEDIC_RELEASE=$(git rev-parse --short HEAD) $C up -d --build   # deploy
DATAMEDIC_RELEASE=<previous-tag> $C up -d --no-build                             # roll back
```

Migrations are expand-only, so rolling back the application keeps working on the newer schema.
Take a backup before every update.

## 5. Security checklist (verify after deployment)

- `https://<domain>` serves a valid certificate; `http://` redirects to HTTPS (Caddy default).
- From outside: `nc -zv <ip> 5432 6379 8000 3000` all fail (only 80/443 open).
- Response headers include HSTS, CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer` and `Cache-Control: no-store` on API data. The page CSP is set by
  Caddy (`deploy/Caddyfile`) and allows `'unsafe-inline'` scripts because Next.js inlines its
  bootstrap scripts; a nonce-based CSP is deferred (the app renders no raw HTML).
- Sign up, verify by email, upload a **synthetic** CSV; a second account cannot open the first
  account's dataset URL (404).
- `.env.production` is mode 600 and not in git. No development data or credentials are copied.
- Until a security review of the deployed instance is complete, restrict use to synthetic or
  non-sensitive demonstration data (the UI footer and landing page say so).
