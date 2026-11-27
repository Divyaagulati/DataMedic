"""Provision the DataMedic presentation account and all demo artifacts through the public API.

Usage (backend virtualenv, which has pandas for nothing here — standard library only):
  python -I demo/provision_presentation.py --web http://localhost:3000 --api http://localhost:8000 \
      --credentials <presentation-account.env> --dataset <diabetes_readmission_presentation.csv> \
      --state <demo_state.json> [--mailpit http://localhost:8025] [--stages all|account,upload,...]

Cloud: omit --mailpit; the script signs up, then waits until you click the verification link that
DataMedic emails (use an address you control in the credentials file).

Idempotent: completed stages are recorded in the state file and skipped on re-runs.
Only supported routes are used; no database access, no authentication bypass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmclient import ApiError, Client  # noqa: E402

STAGES: list[str] = []


def stage(fn):  # noqa: ANN001, ANN201
    STAGES.append(fn.__name__)
    return fn


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Provisioner:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.c = Client(a.web, a.api, a.insecure)
        self.cred = read_env(Path(a.credentials))
        self.state_path = Path(a.state)
        self.state: dict = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ account
    @stage
    def account(self) -> None:
        email, pw = self.cred["DEMO_EMAIL"], self.cred["DEMO_PASSWORD"]
        if self.c.login(email, pw):
            log("presentation account exists and is verified; logged in")
            return
        self.c.call("POST", "/auth/signup", {"email": email, "password": pw,
                                            "display_name": self.cred.get("DEMO_DISPLAY_NAME")})
        log("signed up; waiting for email verification")
        if self.a.mailpit:
            token = None
            for _ in range(60):
                msgs = self.c.call("GET", self.a.mailpit.rstrip("/") + "/api/v1/messages")
                for m in msgs.get("messages", []):
                    if any(t.get("Address") == email for t in m.get("To", [])):
                        full = self.c.call("GET", self.a.mailpit.rstrip("/") + f"/api/v1/message/{m['ID']}")
                        found = re.search(r"token=([A-Za-z0-9_\-]+)", full.get("Text", ""))
                        token = found.group(1) if found else token
                if token:
                    break
                time.sleep(1)
            if not token:
                raise SystemExit("verification email not found in Mailpit")
            self.c.call("POST", "/auth/verify-email", {"token": token})
        else:
            log("open the verification email sent to the presentation address and click the link")
        for _ in range(720):
            if self.c.login(email, pw):
                me = self.c.call("GET", "/auth/me")
                if me["user"]["email_verified"]:
                    log("verified and logged in")
                    return
            time.sleep(5)
        raise SystemExit("account not verified within an hour")

    # ------------------------------------------------------------------ data
    @stage
    def upload(self) -> None:
        content = Path(self.a.dataset).read_bytes()
        name = "Diabetes 130-US Hospitals - 30-day readmission (UCI)"
        existing = [d for d in self.c.call("GET", "/datasets").get("items", []) if d["name"] == name]
        if existing:
            ds = existing[0]
            log(f"dataset already uploaded ({ds['id']})")
        else:
            created = self.c.call("POST", "/datasets", {"name": name, "filename": "diabetes_readmission_presentation.csv",
                                                       "size_bytes": len(content)})
            ds = created["dataset"]
            up = self.c.call("PUT", f"/datasets/{ds['id']}/content", raw=content)
            self.c.wait_job(up["job"]["id"], label="upload")
        for _ in range(600):
            ds = self.c.call("GET", f"/datasets/{ds['id']}")
            if ds["status"] in ("ready", "rejected"):
                break
            time.sleep(2)
        if ds["status"] != "ready":
            raise SystemExit(f"dataset status {ds['status']}")
        self.c.call("PATCH", f"/datasets/{ds['id']}", {"domain": "biomedical"})
        self.state.update(dataset_id=ds["id"], v1=ds["head_version"]["id"])
        log(f"dataset ready: {ds['id']} version {self.state['v1']} ({ds['head_version'].get('row_count')} rows)")

    ROLES = [
        {"column": "readmitted_30d", "role": "target", "task_type": "binary"},
        {"column": "patient_nbr", "role": "bio.patient_id"},
        {"column": "encounter_id", "role": "id"},
        {"column": "gender", "role": "bio.sex"},
        {"column": "race", "role": "sensitive"},
    ]

    def set_roles(self, vid: str) -> None:
        self.c.call("PUT", f"/versions/{vid}/roles", {"roles": self.ROLES})

    def diagnostics(self, vid: str) -> None:
        try:
            a = self.c.call("POST", f"/versions/{vid}/analyses")
            job_id = a["job"]["id"]
        except ApiError as e:
            if e.status != 409:
                raise
            job_id = self.c.call("GET", f"/versions/{vid}/analyses/latest")["job"]["id"]
        self.c.wait_job(job_id, label="diagnostics")
        # a run started before the roles existed is superseded: make sure the latest run saw the roles
        latest = self.c.call("GET", f"/versions/{vid}/analyses/latest")
        log(f"diagnostics run {latest['run']['id'] if latest.get('run') else None} complete")

    @stage
    def roles_and_diagnostics(self) -> None:
        vid = self.state["v1"]
        self.set_roles(vid)
        log("roles declared: target, patient, encounter id, sex (sensitive), race (sensitive)")
        try:
            prof = self.c.call("POST", f"/versions/{vid}/profile")
            if isinstance(prof, dict) and prof.get("job"):
                self.c.wait_job(prof["job"]["id"], label="profile")
        except ApiError as e:
            if e.status != 409:
                raise
        self.diagnostics(vid)

    @stage
    def biomedical(self) -> None:
        vid = self.state["v1"]
        b = self.c.call("POST", f"/versions/{vid}/biomedical")
        self.c.wait_job(b["job"]["id"], label="biomedical")
        log("biomedical summary computed")

    @stage
    def recommendations(self) -> None:
        vid = self.state["v1"]
        r = self.c.call("POST", f"/versions/{vid}/recommendations")
        if isinstance(r, dict) and r.get("job"):
            self.c.wait_job(r["job"]["id"], label="recommendations")
        recs = self.c.call("GET", f"/versions/{vid}/recommendations")
        self.state["recommendation_set_id"] = (recs.get("set") or {}).get("id") or recs.get("id")
        log(f"recommendations ready ({len(recs.get('items', []))} items)")

    # ------------------------------------------------------------------ Experiment Lab
    @stage
    def experiment(self) -> None:
        vid = self.state["v1"]
        rs = self.c.call("GET", f"/versions/{vid}/recommendations")["set"]
        # Same as the Lab UI (lib/experiments.ts draftsFromSuggestions): recommendation keys -> ids,
        # the empty baseline is added by the server; Lab defaults: dummy, linear, hgb; 5 folds.
        key_to_id = {r["key"]: r["id"] for r in rs["recommendations"]}
        cands = [{"label": c["label"], "steps": c["steps"],
                  "recommendation_ids": [key_to_id[k] for k in c.get("recommendation_ids", []) if k in key_to_id]}
                 for c in rs["candidates"] if c["steps"]][:6]
        body = {"name": "Recommended preprocessing strategies (patient-grouped 5-fold)",
                "candidates": cands, "recommendation_set_id": rs["id"], "models": ["dummy", "linear", "hgb"],
                "seed": 0, "split": {"method": None, "n_splits": 5, "seed": 0}}
        ex = self.c.call("POST", f"/versions/{vid}/experiments", body)
        self.c.wait_job(ex["job"]["id"], limit_s=5400, label="experiment")
        self.state["experiment_id"] = ex["experiment"]["id"]
        log(f"experiment {self.state['experiment_id']} complete ({len(cands)} candidates)")

    # ------------------------------------------------------------------ controlled preprocessing
    def approve_and_commit(self, pid: str) -> dict:
        pp = self.c.call("GET", f"/pipelines/{pid}")
        for step in pp["steps"]:
            if step["risk"] == "high":
                self.c.call("POST", f"/pipelines/{pid}/steps/{step['id']}/approve",
                            {"step_hash": step["step_hash"], "spec_hash": pp["spec_hash"],
                             "acknowledge_destructive": step["destructive"]})
        if any(step["risk"] == "medium" for step in pp["steps"]):
            self.c.call("POST", f"/pipelines/{pid}/confirm", {"spec_hash": pp["spec_hash"]})
        cm = self.c.call("POST", f"/pipelines/{pid}/commit", {"spec_hash": pp["spec_hash"]}, headers=self.c.idem())
        self.c.wait_job(cm["job"]["id"], limit_s=1800, label="commit")
        return cm

    def preview(self, pid: str) -> None:
        prev = self.c.call("POST", f"/pipelines/{pid}/impact-preview")
        self.c.wait_job(prev["job"]["id"], limit_s=1800, label="impact preview")

    @stage
    def commit_pipeline(self) -> None:
        ds = self.state["dataset_id"]
        steps = [
            {"op": "missing.indicator", "columns": ["weight", "payer_code", "medical_specialty"]},
            {"op": "impute.constant", "params": {"fill_value": "Missing"},
             "columns": ["admission_type", "admission_source", "payer_code", "medical_specialty"]},
        ]
        name = "Explicit missingness (indicators + 'Missing' category)"
        existing = [p for p in self.c.call("GET", f"/datasets/{ds}/pipelines").get("items", [])
                    if p["name"] == name and p.get("committed_version_id")]
        if existing:
            p = existing[0]
            log("pipeline already committed; reusing it")
        else:
            p = self.c.call("POST", f"/datasets/{ds}/pipelines", {"name": name, "steps": steps})
            self.preview(p["id"])
            self.approve_and_commit(p["id"])
        head = self.c.call("GET", f"/datasets/{ds}")["head_version"]
        self.state.update(pipeline_committed=p["id"], v2=head["id"])
        roles = self.c.call("GET", f"/versions/{head['id']}/roles")["roles"]
        if {r["column"] for r in roles} != {r["column"] for r in self.ROLES}:
            self.set_roles(head["id"])
        self.diagnostics(head["id"])
        log(f"committed v{head['version_number']} ({head['id']}); roles and diagnostics in place")

    @stage
    def live_demo_draft(self) -> None:
        """High-risk draft left UNAPPROVED for the live demo: preview exists, approval is required."""
        ds = self.state["dataset_id"]
        p = self.c.call("POST", f"/datasets/{ds}/pipelines", {
            "name": "LIVE DEMO - cap extreme medication counts (not approved)",
            "steps": [{"op": "outliers.cap", "params": {"method": "percentile", "lower_percentile": 1.0,
                                                         "upper_percentile": 99.0},
                       "columns": ["num_medications"]}]})
        self.preview(p["id"])
        self.state["pipeline_live_demo"] = p["id"]
        log(f"live-demo draft pipeline {p['id']} previewed, left unapproved")

    @stage
    def version_comparison(self) -> None:
        v1, v2 = self.state["v1"], self.state["v2"]
        r = self.c.call("POST", f"/versions/{v1}/compare/{v2}")
        if isinstance(r, dict) and r.get("job"):
            self.c.wait_job(r["job"]["id"], label="compare")
        code = self.c.call("GET", f"/versions/{v2}/code", text=True)
        log(f"version comparison ready; pipeline.py has {len(str(code).splitlines())} lines")

    @stage
    def biomedical_v2(self) -> None:
        """The committed version is the one the dashboard, models, Copilot and passport use."""
        vid = self.state["v2"]
        b = self.c.call("POST", f"/versions/{vid}/biomedical")
        self.c.wait_job(b["job"]["id"], label="biomedical v2")
        log("biomedical summary computed for the committed version")

    # ------------------------------------------------------------------ machine learning
    def train(self, key: str, body: dict) -> None:
        vid = self.state["v2"]
        tr = self.c.call("POST", f"/versions/{vid}/training-runs", body)
        rid = tr["training_run"]["id"]
        self.c.wait_job(tr["job"]["id"], limit_s=7200, label=key)
        for _ in range(600):
            au = self.c.call("GET", f"/training-runs/{rid}/audit")
            if au.get("run"):
                break
            time.sleep(3)
        else:
            raise SystemExit(f"audit for {key} not stored")
        self.state[key] = rid
        log(f"training run {key} {rid} complete, audit stored")

    @stage
    def training(self) -> None:
        self.train("training_main", {
            "name": "Readmission models: baseline vs advanced (patient-grouped, nested tuning)",
            "models": ["dummy", "linear", "random_forest", "hgb"], "tuning": {"mode": "nested"},
            "split": {"method": None, "n_splits": 5, "seed": 0}, "seed": 0})

    @stage
    def training_class_weight(self) -> None:
        self.train("training_class_weight", {
            "name": "Same models with class weights (imbalance handling inside training folds)",
            "models": ["dummy", "linear", "random_forest", "hgb"], "tuning": {"mode": "nested"},
            "preprocess_steps": [{"op": "imbalance.class_weight", "columns": [], "params": {}}],
            "split": {"method": None, "n_splits": 5, "seed": 0}, "seed": 0})

    # ------------------------------------------------------------------ analytics
    @stage
    def insights(self) -> None:
        vid = self.state["v2"]
        j = self.c.call("POST", f"/versions/{vid}/insights")
        self.c.wait_job(j["job"]["id"], limit_s=1800, label="insights")
        ins = self.c.call("GET", f"/versions/{vid}/insights")
        log(f"insights stored ({len(ins['summary']['insights'])})")

    DASHBOARD = "Hospital Readmission Intelligence"

    @staticmethod
    def widgets() -> list[dict]:
        def agg(measures, group_by=(), **kw):  # noqa: ANN001, ANN003, ANN202
            return {"kind": "aggregate", "group_by": [dict(g) for g in group_by], "measures": measures, **kw}
        rate = [{"agg": "mean", "column": "readmitted_30d"}]
        return [
            {"kind": "kpi", "size": "small", "title": "Total hospital encounters",
             "spec": {"query": agg([{"agg": "count"}]), "display": {}}},
            {"kind": "kpi", "size": "small", "title": "30-day readmission rate (share of encounters)",
             "spec": {"query": agg(rate), "display": {}}},
            {"kind": "kpi", "size": "small", "title": "Average length of stay (days)",
             "spec": {"query": agg([{"agg": "mean", "column": "time_in_hospital"}]), "display": {}}},
            {"kind": "chart", "size": "medium",
             "title": "Readmission outcome: encounters by readmitted_30d (1 = readmitted within 30 days)",
             "spec": {"query": agg([{"agg": "count"}], [{"column": "readmitted_30d"}]),
                      "display": {"chart_type": "donut"}}},
            {"kind": "chart", "size": "medium", "title": "Average number of medications by age group",
             "spec": {"query": agg([{"agg": "mean", "column": "num_medications"}], [{"column": "age"}],
                                   sort={"by": "group", "dir": "asc"}),
                      "display": {"chart_type": "bar"}}},
            {"kind": "chart", "size": "medium", "title": "Distribution of laboratory procedures per encounter",
             "spec": {"query": {"kind": "histogram", "columns": ["num_lab_procedures"], "bins": 20},
                      "display": {"chart_type": "histogram"}}},
            {"kind": "chart", "size": "medium",
             "title": "30-day readmission rate by inpatient visits in the previous year (0-6 visits)",
             "spec": {"query": agg(rate, [{"column": "number_inpatient"}], sort={"by": "group", "dir": "asc"},
                                   filters=[{"column": "number_inpatient", "op": "lte", "value": 6}]),
                      "display": {"chart_type": "bar"}}},
            {"kind": "chart", "size": "medium", "title": "30-day readmission rate by HbA1c test result",
             "spec": {"query": agg(rate, [{"column": "A1Cresult"}], sort={"by": "measure", "dir": "desc"}),
                      "display": {"chart_type": "hbar"}}},
            {"kind": "chart", "size": "large", "title": "Encounters by admission type and 30-day readmission",
             "spec": {"query": agg([{"agg": "count"}], [{"column": "admission_type"}, {"column": "readmitted_30d"}]),
                      "display": {"chart_type": "stacked_bar"}}},
            {"kind": "chart", "size": "medium", "title": "Length of stay (days) by age group",
             "spec": {"query": {"kind": "box", "columns": ["time_in_hospital"], "group_by": [{"column": "age"}]},
                      "display": {"chart_type": "box"}}},
        ]

    @stage
    def dashboard(self) -> None:
        ds, vid = self.state["dataset_id"], self.state["v2"]
        for d in self.c.call("GET", f"/datasets/{ds}/dashboards").get("items", []):
            if d["name"] == self.DASHBOARD:
                self.state["dashboard_id"] = d["id"]
                log("dashboard already exists; left unchanged")
                return
        d = self.c.call("POST", f"/datasets/{ds}/dashboards", {
            "version_id": vid, "name": self.DASHBOARD,
            "description": "UCI Diabetes 130-US Hospitals (1999-2008), presentation subset of 4,000 encounters. "
                           "Counts are encounters, not patients. Research use only."})
        for w in self.widgets():
            self.c.call("POST", f"/dashboards/{d['id']}/widgets", w)
        again = self.c.call("GET", f"/dashboards/{d['id']}")
        self.state["dashboard_id"] = d["id"]
        log(f"dashboard {d['id']} with {len(again['widgets'])} widgets")

    # ------------------------------------------------------------------ exports
    @stage
    def exports(self) -> None:
        vid = self.state["v2"]
        ex = self.c.call("POST", f"/versions/{vid}/exports", {"acknowledged": True})
        job = self.c.wait_job(ex["job"]["id"], limit_s=1800, label="csv export")
        text = self.c.call("GET", f"/exports/{job['id']}/download", text=True)
        header = str(text).splitlines()[0].split(",")
        leaked = {"patient_nbr", "race", "gender"} & set(header)
        if leaked:
            raise SystemExit(f"sensitive columns in CSV export: {sorted(leaked)}")
        self.state["csv_export_job"] = job["id"]
        log(f"CSV export ok ({len(header)} columns, sensitive columns excluded)")

    # ------------------------------------------------------------------ runner
    def run(self, wanted: list[str]) -> None:
        done = set(self.state.get("done", []))
        self.account()  # always establish the session
        for name in STAGES:
            if name == "account" or (wanted != ["all"] and name not in wanted):
                continue
            if name in done and name not in wanted:
                log(f"skip {name} (done)")
                continue
            log(f"== stage {name}")
            getattr(self, name)()
            done.add(name)
            self.state["done"] = sorted(done)
            self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", required=True)
    ap.add_argument("--api")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--mailpit")
    ap.add_argument("--credentials", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--state", required=True)
    ap.add_argument("--stages", default="all")
    a = ap.parse_args()
    try:
        Provisioner(a).run(a.stages.split(","))
    except ApiError as e:
        log(f"API error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
