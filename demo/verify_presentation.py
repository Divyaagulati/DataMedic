"""Verify the presentation account and every prepared artifact (read-only, supported API only).

Usage:
  python -I demo/verify_presentation.py --web http://localhost:3000 --api http://localhost:8000 \
      --credentials <presentation-account.env> --state <demo_state.json> [--mailpit URL] [--isolation]

--isolation signs up one throwaway account (local stack: verified through Mailpit) and checks it
cannot see any presentation resource. Prints PASS/FAIL lines; exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmclient import ApiError, Client  # noqa: E402

results: list[bool] = []


def check(name: str, ok: object, detail: str = "") -> None:
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""), flush=True)


def status(c: Client, path: str) -> int:
    try:
        c.call("GET", path, ok=(200,))
        return 200
    except ApiError as e:
        return e.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", required=True)
    ap.add_argument("--api")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--credentials", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--mailpit")
    ap.add_argument("--isolation", action="store_true")
    a = ap.parse_args()
    cred = dict(line.split("=", 1) for line in Path(a.credentials).read_text(encoding="utf-8").splitlines() if "=" in line)
    st = json.loads(Path(a.state).read_text(encoding="utf-8"))
    ds, v1, v2 = st["dataset_id"], st["v1"], st["v2"]
    c = Client(a.web, a.api, a.insecure)

    check("login as the presentation account", c.login(cred["DEMO_EMAIL"], cred["DEMO_PASSWORD"]))
    me = c.call("GET", "/auth/me")["user"]
    check("display name and verified email", me.get("display_name") == "DataMedic Presentation" and me["email_verified"])

    d = c.call("GET", f"/datasets/{ds}")
    check("dataset ready, biomedical domain, head is the committed version",
          d["status"] == "ready" and d.get("domain") == "biomedical" and d["head_version"]["id"] == v2)
    versions = c.call("GET", f"/datasets/{ds}/versions")["items"]
    check("linear lineage v0 -> v1, original kept", [v["id"] for v in versions] == [v1, v2]
          and versions[1]["parent_version_id"] == v1 and not versions[1]["fitted_on_full_data"])
    ver = c.call("GET", f"/versions/{v1}")
    check("original version 4,000 x 50", ver["n_rows"] == 4000 and ver["n_cols"] == 50)

    for vid, label in ((v1, "v0"), (v2, "v1")):
        prof = c.call("GET", f"/versions/{vid}/profile")
        check(f"profile stored ({label})", bool(prof.get("profile")))
        an = c.call("GET", f"/versions/{vid}/analyses/latest")
        run = an.get("run") or {}
        check(f"diagnostics current with roles ({label})", bool(run.get("findings")) and not run.get("stale"),
              f"{len(run.get('findings', []))} findings")
        bio = c.call("GET", f"/versions/{vid}/biomedical")
        cohort = (bio.get("summary") or {}).get("cohort", {})
        check(f"biomedical summary ({label})", cohort.get("patients", {}).get("unique") == 2864 and not bio.get("stale"),
              f"unique patients {cohort.get('patients', {}).get('unique')}")
    roles = {r["column"]: r["role"] for r in c.call("GET", f"/versions/{v2}/roles")["roles"]}
    check("roles: target, patient, encounter id, sensitive sex and race",
          roles == {"readmitted_30d": "target", "patient_nbr": "bio.patient_id", "encounter_id": "id",
                    "gender": "bio.sex", "race": "sensitive"}, json.dumps(roles))

    recs = c.call("GET", f"/versions/{v1}/recommendations")["set"]
    ops = {r["op"].split(".")[0] for r in recs["recommendations"]}
    check("recommendations cover missing, encoding, scaling, outliers, transforms, selection, imbalance",
          {"impute", "missing", "encode", "scale", "outliers", "transform", "select", "imbalance"} <= ops,
          f"{len(recs['recommendations'])} recommendations")
    ex = c.call("GET", f"/experiments/{st['experiment_id']}")
    cands = ex["candidates"]
    check("experiment: baseline + 4 candidates completed on one frozen grouped split",
          len(cands) == 5 and all(x["result"]["status"] == "completed" and x["result"]["comparable"] for x in cands)
          and ex["split"]["method"] == "stratified_group" and ex["split"]["group_column"] == "patient_nbr")

    live = c.call("GET", f"/pipelines/{st['pipeline_live_demo']}")
    high = [s for s in live["steps"] if s["risk"] == "high"]
    check("live-demo pipeline: high-risk step, preview available, not approved, not committed",
          high and not live.get("committed_version_id") and live.get("state") != "committed"
          and status(c, f"/pipelines/{st['pipeline_live_demo']}/impact-preview") == 200, live.get("state"))
    code = c.call("GET", f"/versions/{v2}/code", text=True)
    check("pipeline.py for the committed version", "import" in str(code) and "missing" in str(code))
    cmp_ = c.call("GET", f"/versions/{v1}/compare/{v2}")
    check("version comparison v0 -> v1", isinstance(cmp_, dict) and bool(cmp_.get("columns_added")),
          str(cmp_.get("columns_added")))

    for key, cw in (("training_main", False), ("training_class_weight", True)):
        run = c.call("GET", f"/training-runs/{st[key]}")
        evals = {e["model"]: e for e in run["evaluations"]}
        ok = (run["evaluation_mode"] == "unbiased_direct" and run["split"]["method"] == "stratified_group"
              and run["split"]["group_column"] == "patient_nbr"
              and set(evals) == {"dummy", "linear", "random_forest", "hgb"}
              and all(e["status"] == "completed" for e in evals.values()))
        check(f"training run {key}: 4 models, patient-grouped, leakage-controlled", ok)
        lin = evals["linear"]
        check(f"  {key}: curves, calibration, confusion, importance, error slices stored",
              all(k in lin["details"] for k in ("curves", "calibration", "confusion", "error_slices"))
              and lin["importance"]["available"])
        feats = set(run["summary"].get("features") or run.get("config", {}).get("features") or [])
        check(f"  {key}: identifiers and target are not features",
              not ({"patient_nbr", "encounter_id", "readmitted_30d"} & feats), f"{len(feats)} features")
        au = c.call("GET", f"/training-runs/{st[key]}/audit")
        check(f"  {key}: reliability audit (20 checks)", au.get("run") and len(au["findings"]) == 20,
              au["run"]["summary"]["evaluation_status"]["status"] if au.get("run") else "")
        if cw:
            check("  class weights inside folds", [s["op"] for s in run["preprocess_steps"]] == ["imbalance.class_weight"])

    ins = c.call("GET", f"/versions/{v2}/insights")
    check("automated insights stored", len(ins["summary"]["insights"]) > 0, f"{len(ins['summary']['insights'])}")
    dash = c.call("GET", f"/dashboards/{st['dashboard_id']}")
    check("dashboard 'Hospital Readmission Intelligence' with 10 widgets bound to v1",
          dash["name"] == "Hospital Readmission Intelligence" and len(dash["widgets"]) == 10 and dash["version_id"] == v2)
    bad = []
    for w in dash["widgets"]:
        try:
            c.call("POST", f"/versions/{v2}/analytics/query", {"spec": w["spec"]["query"]})
        except ApiError as e:
            bad.append(f"{w['title']}: {e}")
    check("every dashboard widget query runs", not bad, "; ".join(bad))
    check("dashboard JSON export", c.call("GET", f"/dashboards/{st['dashboard_id']}/export").get("format") == "datamedic.dashboard/1")

    p = c.call("GET", f"/versions/{v2}/passport")
    check("data passport: diagnostics, models and audits available, no single score",
          p["diagnostics"]["status"] == "available" and p["models"]["status"] == "available" and "score" not in p)
    text = c.call("GET", f"/exports/{st['csv_export_job']}/download", text=True)
    header = str(text).splitlines()[0].split(",")
    check("CSV export downloadable, sensitive columns excluded",
          not ({"patient_nbr", "race", "gender"} & set(header)) and len(header) == 50)

    questions = {
        "What are the most important data quality issues?": "diagnostics",
        "What is the average time in hospital by age group?": "analytics",
        "Which variables have the most missing values?": "analytics",
        "What are the major reliability concerns?": "reliability",
        "Which features were most informative in the trained models?": "models",
        "What are the most important insights about my dataset?": "insights",
    }
    for q, topic in questions.items():
        ans = c.call("POST", f"/versions/{v2}/copilot", {"question": q})
        check(f"Copilot: {q}", ans.get("topic") == topic and ans.get("answer"), ans.get("topic", ""))

    web_paths = ["", "/profile", "/diagnostics", "/biomedical", "/lab", "/models", f"/models/{st['training_main']}",
                 "/analytics", f"/analytics/dashboards/{st['dashboard_id']}", "/copilot", "/passport", "/history",
                 f"/pipelines/{st['pipeline_live_demo']}"]
    codes = {pth: status(c, f"{c.web}/datasets/{ds}{pth}") for pth in web_paths}
    check("every presentation page loads", all(v == 200 for v in codes.values()), json.dumps(codes))

    anon = Client(a.web, a.api, a.insecure)
    check("anonymous access refused", status(anon, f"/datasets/{ds}") == 401)
    if a.isolation:
        o = Client(a.web, a.api, a.insecure)
        email, pw = f"isolation-check-{uuid.uuid4().hex[:8]}@example.org", "Iso-" + uuid.uuid4().hex
        o.call("POST", "/auth/signup", {"email": email, "password": pw})
        token = None
        for _ in range(30):
            for m in o.call("GET", a.mailpit.rstrip("/") + "/api/v1/messages").get("messages", []):
                if any(t.get("Address") == email for t in m.get("To", [])):
                    full = o.call("GET", a.mailpit.rstrip("/") + f"/api/v1/message/{m['ID']}")
                    found = re.search(r"token=([A-Za-z0-9_\-]+)", full.get("Text", ""))
                    token = found.group(1) if found else token
            if token:
                break
            time.sleep(1)
        o.call("POST", "/auth/verify-email", {"token": token})
        o.login(email, pw)
        paths = [f"/datasets/{ds}", f"/versions/{v2}/passport", f"/training-runs/{st['training_main']}",
                 f"/dashboards/{st['dashboard_id']}", f"/experiments/{st['experiment_id']}",
                 f"/exports/{st['csv_export_job']}/download"]
        check("another account gets 404 for every presentation resource",
              all(status(o, pth) == 404 for pth in paths)
              and not o.call("GET", "/datasets").get("items"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
