# DataMedic — Privacy and Security

| | |
|---|---|
| Status | Revision 2, MVP-scoped |
| Design target | OWASP ASVS 4.x Level 2. HIPAA and GDPR principles are design inputs, not a certification claim. |

## 1. Data classification

| Class | Examples | Where it may exist |
|---|---|---|
| **C3 Dataset content** | Cell values, rows, category labels, original files, versions, previews, impact-preview rows, fitted state, out-of-fold predictions, export bundles | Object storage (internal network; encrypted at rest in production), worker memory, the owner's browser. In Postgres only as capped evidence examples (owner-only under RLS). **Never** in logs, traces, metrics, error messages, **broker messages** or any external service. |
| **C2 Dataset metadata** | Column names, dtypes, shapes, filenames | Postgres, UI. Column names may be sensitive (e.g. `HIV_status`), so they are treated like C3 for logging and are never sent externally. |
| **C1 Aggregates** | Severities, rates, coefficients, metric scores | Postgres, UI, passport |
| **C0 Account and operational** | Email, password hash, sessions, audit events | Postgres |

## 2. Privacy controls

| Principle | Control |
|---|---|
| Owner-only access (R1) | §4 authorisation, RLS on `owner_id`, 404 for other users' resources, storage not reachable from browsers |
| **No raw data to external LLMs (R2)** | **MVP: there is no LLM integration and no outbound data path.** The V2 design is in §3. |
| Minimisation | 100-row preview; capped evidence examples, omitted for `sensitive` columns; no profiling report containing sample rows (the V2 YData Profiling adapter must disable sample and duplicate sections) |
| Original integrity (R3) | Write-once `raw/` objects; DB trigger; no code path writes `raw/` except purge; SHA-256 recomputed and compared whenever the original is read (ingest, reparse, export validation) |
| **Job-message privacy** | Celery messages contain only `{"job_id": …}`. JSON serializer only (`task_serializer="json"`, `accept_content=["json"]`). Job parameters live in `jobs.params` in Postgres under RLS. A test inspects published messages. |
| Transparency | Audit log visible to the user, deletion receipts, the passport's limitations section |
| Erasure | Dataset and account deletion purged within 24 h. Backup ageing (30 days) disclosed. |
| No third-party trackers | No analytics, session replay or external fonts/scripts on authenticated pages. Enforced by CSP. |

### 2.1 PHI/PII detection
`universal.privacy.pii_scan` (all datasets) and `bio.privacy.phi_extended` (biomedical) run at ingest. They use heuristic detectors on names and values: email, phone, national-ID formats, IP addresses, full dates of birth, name-like text, postal codes, and MRN-like identifiers. Limitations always state that **detection is heuristic and cannot guarantee absence**. DataMedic does not claim to de-identify data. Dropping identifier columns is a high-risk, destructive op whose limitation notes that re-identification through quasi-identifiers remains possible.

## 3. LLM explanations (V2 design; not in the MVP)

The MVP provides deterministic template explanations only. When V2 adds opt-in LLM explanations, all of the following are mandatory:
1. Off by default per user, explicitly enabled and audited.
2. A dedicated `llm-gateway` container is the **only** component with internet egress (network policy).
3. Payloads are typed Pydantic models with `extra="forbid"` and **no free-text fields that can carry data**: check IDs, severities, numeric evidence and aliased column names (`col_1…`) only.
4. A pre-send guard checks payloads against hashes of the dataset's distinct string values (computed in the worker; only hashes are passed).
5. The exact payload is logged in `llm_egress_log` and visible to the user, and a payload preview is shown before the first use.
6. A CI canary test fails the build if a unique token injected into a test dataset appears in any outbound payload.

## 4. Authentication and authorisation

### 4.1 Authentication
- **Passwords**: Argon2id (argon2-cffi), with parameters tuned to about 250 ms. Minimum length 12, a check against a breached-password list using a k-anonymity range query or a local list (decided in Phase 2), and no composition rules.
- **Email verification** is required before uploading.
- **Sessions**: server-side rows, and the cookie holds a 256-bit random token (only its SHA-256 is stored). Cookie attributes: `__Host-`, `Secure`, `HttpOnly`, `SameSite=Lax`. Idle timeout 7 d, absolute 30 d, rotated on login and password change. The user can list and revoke sessions.
- **CSRF**: synchroniser token header + SameSite + `Origin` check on unsafe methods.
- **Brute force**: **progressive delays**, not hard lockout, so attackers cannot lock victims out. The delay grows per account and per IP with failed attempts, plus per-IP rate limits (stored in Valkey). Error messages are generic.
- **Password reset**: single-use hashed token, valid 30 minutes, revokes all sessions on use.
- **Email delivery**: SMTP abstraction. A mail catcher in dev; the production provider is chosen before deployment (PRD Q4).
- V2: TOTP MFA. Future: OIDC SSO.

### 4.2 Authorisation and user isolation
- **Model**: every resource has `owner_id`. One function `authorize(user, resource, action)` holds the rule (MVP: `resource.owner_id == user.id`), so Future team roles change one place.
- **Enforcement layers**:
  1. **Service queries**: subjects are loaded with `WHERE id = :id AND owner_id = :uid`. Not found returns 404.
  2. **Postgres RLS**: `app.user_id` is set per transaction by a session hook. A missing value returns zero rows ([DATABASE_SCHEMA.md §5](DATABASE_SCHEMA.md)).
  3. **Workers**: `claim_job(job_id)` returns `owner_id`. The worker sets `app.user_id` and re-loads every subject under RLS. A forged or replayed message can only act on that job's own owner and subject, and `claim_job` refuses jobs that are already running or finished.
  4. **Storage**: keys are `…/{owner_id}/{dataset_id}/{uuid}`. Browsers never get storage URLs; the API streams after authorisation.
- **Approvals**: only the owner can approve. Approvals bind `approved_by`, `approved_hash` and the impact shown.
- **Experiments vs commits**: experiments may evaluate high-risk steps automatically because they write only metrics and artifacts. **Only the commit path writes versions**, and it requires valid approvals, checked in the API and again in the worker.

### 4.3 Quotas as a security control
Plan limits (PRD §6.13) bound storage, upload size, cells, columns, versions, running and queued jobs, and experiment size, which caps the resources one account can consume. Admission control runs in the API before job creation. Per-queue time limits and worker memory limits contain runaway jobs.

## 5. Storage security and deletion

| Prefix | Contents | Policy |
|---|---|---|
| `raw/{owner}/{dataset}/{uuid}` | Original bytes | Write-once. The API writes once during upload; the purge path is the only delete. |
| `versions/{owner}/{dataset}/{version}.parquet` | Version snapshots | Write-once; revert uses server-side copy |
| `artifacts/{owner}/…` | Previews, row refs, impact previews, fold assignments, fitted state (HMAC), OOF predictions, passports | |
| `exports/{owner}/{export}.zip` | Bundles | 30-day expiry |

- **Network**: storage, broker and DB are on an internal network. Storage has no public endpoint.
- **Encryption**: TLS between services in production; server-side encryption at rest on the storage service; encrypted DB volumes.
- **Credentials**: API storage credential (put/get on `raw/` and `exports/` and `artifacts/` previews and passports, no delete); worker credential (read/write data prefixes, no delete on `raw/`); purge credential (delete). This is only possible if the backend supports prefix-scoped policies. The dev backend's capability is checked in Phase 1; if it is unsupported, the separation is enforced in code and documented as a gap.
- **Deletion flow**:
  1. `DELETE` sets `deleted_at`. The dataset immediately disappears from every query, and downloads stop because all downloads go through the API.
  2. The purge job (`dm_maintenance`) deletes all objects under the dataset's prefixes.
  3. It hard-deletes rows in FK order.
  4. It writes the tombstone and receipt.
  5. It verifies that listing the prefixes returns zero objects (otherwise the job fails and alerts).
- **Backups**: encrypted, 30-day retention, disclosed.

## 6. Threat model (STRIDE)

| # | Threat | Mitigation | Test |
|---|---|---|---|
| T1 | IDOR: reading another user's dataset by ID | Owner-scoped queries, RLS, 404 | **AuthZ matrix** (every route × owner/other/anon), RLS tests |
| T2 | Data leaking via broker messages | Messages hold `job_id` only; JSON serializer | Message-inspection test |
| T3 | Malicious file (zip bomb, XML bomb, macro, parser exploit) | Magic bytes, ZIP inspection, defusedxml, macro rejection, subprocess with rlimits, non-root worker | `fixtures/malicious` |
| T4 | Oversized or slow uploads exhausting API resources | `Content-Length` required, 200 MB cap, byte counting, proxy body limit, request timeouts, per-user rate limits | Tests |
| T5 | CSV formula injection in exports | Detection, warning, opt-in escape | Fixture |
| T6 | Pickle deserialisation of tampered fitted state | Platform-created only, HMAC verified, never user-supplied, pickle-free exports | Tamper test |
| T7 | Code injection via generated `pipeline.py` (hostile column names or params) | Template-only codegen; values emitted as `repr` of primitives; validated params; validation subprocess with rlimits, a temporary working directory, inputs copied in, and a worker with no internet egress | Hostile-name tests |
| T8 | Committing a high-risk step without approval (bypass, stale, race) | Server-derived risk and destructive flags, hash-bound approvals tied to the impact preview, worker re-verification, one mutation job per dataset | Bypass tests |
| T9 | Tampering with originals or historical versions | Write-once storage, triggers, hash checks on read | Tests |
| T10 | XSS → session theft | HttpOnly cookies, nonce-based CSP, React escaping, lint ban on `dangerouslySetInnerHTML`. The passport HTML is generated with Jinja2 autoescape, served as a download (`attachment`), and rendered in-app from JSON rather than raw HTML. | CSP and lint tests |
| T11 | CSRF | Token + SameSite + Origin | Test |
| T12 | Credential stuffing | Progressive delays, rate limits, breached-password check (MFA in V2) | Test |
| T13 | Resource exhaustion through jobs | Quotas, admission control, time and memory limits, sampling budgets | Load tests |
| T14 | Data in logs | Scrubber deny-list; values never in error messages | Log tests |
| T15 | Supply chain | Hashed lockfiles, pip-audit and npm audit, SBOM, license gate, base images pinned by digest | CI |
| T16 | Unmaintained infrastructure (MinIO community archived) | Managed storage in production; any dev-only backend never exposed | Deployment checklist |
| T17 | Operator access | Least-privilege DB roles, audited maintenance role, no production data in non-production | Review |
| T18 | Repudiation of approvals or deletions | Append-only `approvals` and `audit_events` | — |

## 7. HTTP security headers

HSTS (production), a `Content-Security-Policy` with `frame-ancestors 'none'` and `connect-src 'self'` (same-origin API) set by the production reverse proxy (`deploy/Caddyfile`; `script-src` currently allows `'unsafe-inline'` for Next.js bootstrap scripts — a nonce-based CSP is planned, see DEPLOYMENT.md §5; correction recorded 2026-11-26), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, a restrictive `Permissions-Policy`, and `Cache-Control: no-store` on data-bearing responses. CORS allows only the web origin, with credentials. Nonce-based CSP makes Next.js pages render dynamically, which is an accepted performance trade-off.

## 7a. Phase 5 safeguards (training and reliability audit)

- Every new row is owner-scoped with RLS; every new route is in the isolation matrix (404 across users).
- Responses carry aggregates only: bounded curves, binned grids, fold statistics, importance per column,
  error slices with at least 30 rows and never for sensitive-role columns. Class names of a sensitive target
  are replaced by "class 1, 2, …"; regression charts withhold axis values for a sensitive target.
- **Sensitivity across role changes (closure 2026-11-26, `training/privacy.py`).** A column is protected
  when it was sensitive in the run's training-time roles, is sensitive now on any version of the dataset,
  or was derived from such a column by committed steps. Protection is applied when a run is created
  (the engine never slices it) and again whenever results are served (error slices withheld; protected
  target: class names and residual axis values withheld). It is enforced in the API, not by the frontend;
  stored results and provenance are never rewritten, and the response discloses what was withheld.
- Out-of-fold predictions are a platform-written Parquet artifact, owner only and never served; no trained
  model, pickle or joblib file is created.
- Error messages from libraries are reduced to engine codes or exception types; logs carry ids, codes and
  counts only.
- Broker messages carry only the job id; parameters live in `jobs.params` and the run configuration.
- No new dependency and no external service.

## 8. Verification summary (MVP)

| Requirement | Release-blocking tests |
|---|---|
| User isolation | AuthZ matrix, RLS fail-closed, cross-user worker test |
| No external data flow | Dependency review (no LLM or analytics SDKs), broker message test, log scrubber test |
| Original immutability | No-write-path test, trigger test, hash-on-read test |
| Approval enforcement | API and worker bypass tests |
| Malicious inputs | Malicious fixture suite |

## 7b. MVP completion safeguards (2026-11-26)

- Owner policy (approved 2026-11-26): a column marked sensitive on any version of a dataset, or derived from one by committed steps, stays protected across historical and future training results, the biomedical summary (labels replaced, no per-category values), the Data Passport and exports.
- Biomedical summaries are aggregates only; groups with fewer than 10 units are suppressed.
- CSV exports need an explicit row-level acknowledgement; sensitive columns (including `bio.patient_id`, a sensitive role) are dropped unless a second acknowledgement includes them; text cells starting with `= + - @`, tab or CR are prefixed with an apostrophe; files are downloaded only through the authorised endpoint with sanitised filenames and `no-store`.
- The Copilot is rule-based and in-process: no external model, no API key, nothing leaves the platform. An external LLM, if ever added, must be opt-in, server-side, and receive no rows, column names, identifiers or sensitive values.
- Production: Caddy is the only exposed service (HTTPS, HSTS); `__Host-` Secure cookies; Redis requires a password; plaintext SMTP is refused in production; secrets live only in `.env.production` (mode 600, never committed). Until the deployed instance passes the checklist in docs/DEPLOYMENT.md §5, it is for synthetic or non-sensitive demonstration data only.

## 7c. Analytics Studio safeguards (Phase 8, 2026-11-27)

- No SQL, expressions, paths, URLs or code are accepted: queries are validated structured specifications (ARCHITECTURE.md §18).
- Protected columns can only be counted or tested for missingness: never grouped, filtered by value, plotted, aggregated, shown in tooltips or returned by the Copilot.
- Privacy mode (biomedical datasets or any protected column): cells under 10 rows suppressed, secondary suppression, whole-result suppression for filtered cohorts under 10, binned scatter plots instead of points.
- Exports are aggregate-only (no scatter points), formula-neutralised and served with sanitised filenames and `no-store`; dashboard JSON contains configuration only. No public dashboard sharing.
- The Copilot's language layer is deterministic and local; nothing is sent to external services.
- Residual risk (documented, not solved): repeated overlapping queries could narrow small groups by differencing. No differential privacy or anonymisation is claimed.
