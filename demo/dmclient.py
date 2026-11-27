"""Minimal DataMedic API client for the presentation provisioning script (standard library only).

Uses only public, supported API routes: cookie session, CSRF header, Origin header, job polling.
Never prints credentials.
"""

from __future__ import annotations

import http.cookiejar
import json
import ssl
import time
import urllib.error
import urllib.request
import uuid


class ApiError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, payload: object) -> None:
        detail = payload.get("detail") or payload.get("title") if isinstance(payload, dict) else str(payload)[:300]
        code = payload.get("code") if isinstance(payload, dict) else None
        errs = payload.get("errors", [])[:3] if isinstance(payload, dict) else []
        extra = "; ".join(f"{e.get('pointer')}: {e.get('message')}" for e in errs if isinstance(e, dict))
        super().__init__(f"{method} {path} -> {status} {code or ''} {detail} {extra}".strip())
        self.status, self.payload, self.code = status, payload, code


class Client:
    def __init__(self, web: str, api: str | None = None, insecure: bool = False) -> None:
        self.web = web.rstrip("/")
        self.base = (api or web).rstrip("/")
        self.api = self.base + "/api/v1"
        ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()  # noqa: S323
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar),
                                                  urllib.request.HTTPSHandler(context=ctx))
        self.csrf: str | None = None

    def call(self, method: str, path: str, body: object = None, raw: bytes | None = None,
             ok: tuple[int, ...] = (200, 201, 202, 204), headers: dict | None = None,
             text: bool = False) -> object:
        url = path if path.startswith("http") else self.api + path
        h = {"Origin": self.web} if url.startswith(self.base) else {}
        h.update(headers or {})
        data = None
        if raw is not None:
            data, h["Content-Type"] = raw, "application/octet-stream"
        elif body is not None:
            data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        if self.csrf and method != "GET":
            h["X-CSRF-Token"] = self.csrf
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with self.opener.open(req, timeout=600) as r:
                content, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            content, status = e.read(), e.code
        if text and status in ok:
            return content.decode("utf-8", "replace")
        try:
            payload = json.loads(content) if content else {}
        except ValueError:
            payload = content.decode("utf-8", "replace")
        if status not in ok:
            raise ApiError(method, path, status, payload)
        return payload

    def idem(self) -> dict:
        return {"Idempotency-Key": str(uuid.uuid4())}

    def login(self, email: str, password: str) -> bool:
        try:
            out = self.call("POST", "/auth/login", {"email": email, "password": password})
        except ApiError as e:
            if e.status in (400, 401, 403, 422):
                return False
            raise
        self.csrf = out["csrf_token"]
        return True

    def wait_job(self, job_id: str, limit_s: int = 3600, label: str = "") -> dict:
        start = time.time()
        while time.time() - start < limit_s:
            job = self.call("GET", f"/jobs/{job_id}")
            if job["status"] in ("succeeded", "failed", "cancelled"):
                if job["status"] != "succeeded":
                    raise RuntimeError(f"job {label or job_id} {job['status']}: {job.get('error')}")
                return job
            time.sleep(2)
        raise TimeoutError(f"job {label or job_id} still running after {limit_s}s")
