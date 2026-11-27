"""DataMedic Copilot: a built-in, rule-based assistant (no external model, no API key).

It answers questions about one dataset version from its Data Passport (stored, owner-authorised
results only). It classifies the question by keywords into a topic, composes an answer from the
relevant passport sections, and cites them. It never modifies data, commits transformations,
trains models or executes code, and it never sends anything to an external service.
"""

import re
from collections.abc import Callable
from typing import Any

ASSISTANT = {
    "name": "DataMedic Copilot",
    "kind": "built-in rule-based assistant",
    "note": "Answers are composed from this version's stored analyses by fixed rules. No "
    "language model and no external service is used; nothing is changed by asking.",
}
SUGGESTIONS = [
    "Is this dataset ready for modelling?",
    "What are the most important data-quality issues?",
    "Which preprocessing steps are recommended?",
    "How well do the models perform?",
    "Can I trust the model results?",
    "Which features matter most?",
    "Is there patient-level leakage or a batch effect?",
    "What should I do next?",
    "What are the most important insights about my dataset?",
]
TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("privacy", ("sensitive", "privacy", "private", "redact", "protected", "personal")),
    ("biomedical", ("patient", "biomarker", "batch", "site", "cohort", "clinical", "confound")),
    (
        "reliability",
        ("trust", "reliab", "leak", "audit", "overfit", "bias", "valid", "exploratory"),
    ),
    (
        "insights",
        (
            "insight",
            "interesting",
            "notable",
            "stand out",
            "stands out",
            "key finding",
            "pattern",
            "about my data",
            "about this data",
            "summar",
        ),
    ),
    # Feature importance only for explicit model/feature questions: a bare "important"
    # ("the most important insights", "important quality issues") is not a model question.
    (
        "explain",
        (
            "importance",
            "important feature",
            "important variable",
            "important column",
            "important predictor",
            "feature",
            "features matter",
            "driver",
            "contribut",
            "explain the model",
        ),
    ),
    ("models", ("model", "perform", "score", "auc", "accuracy", "r2", "rmse", "train", "predict")),
    ("experiments", ("experiment", "lab", "compare", "candidate", "baseline")),
    ("recommendations", ("recommend", "preprocess", "clean", "impute", "fix", "transform", "step")),
    ("history", ("history", "version", "provenance", "hash", "reproduc", "lineage", "approval")),
    ("next", ("next", "should i", "what now", "todo", "to do", "ready")),
    (
        "diagnostics",
        (
            "issue",
            "problem",
            "diagnos",
            "finding",
            "warning",
            "error",
            "quality",
            "missing",
            "duplicate",
            "outlier",
        ),
    ),
]


def _ref(section: str, label: str, href: str) -> dict[str, str]:
    return {"section": section, "label": label, "href": href}


class _Answer:
    def __init__(self, p: dict[str, Any], base: str) -> None:
        self.p, self.base = p, base
        self.lines: list[str] = []
        self.refs: list[dict[str, str]] = []

    def say(self, text: str) -> None:
        self.lines.append(text)

    def cite(self, section: str, label: str, path: str) -> None:
        r = _ref(section, label, f"{self.base}{path}")
        if r not in self.refs:
            self.refs.append(r)


def _available(section: dict[str, Any]) -> bool:
    return section.get("status") == "available"


def _quality(a: _Answer) -> None:
    q = a.p["quality"]
    if not _available(q):
        a.say(f"Data quality: {q['reason']} Open the Profile tab to run profiling.")
        a.cite("quality", "Profile", "/profile")
        return
    s = q["summary"] or {}
    a.say(
        f"The version has {s.get('n_rows', '?'):,} rows and {s.get('n_cols', '?')} columns; "
        f"{s.get('missing_cells_pct', 0)}% of cells are missing, {s.get('duplicate_rows', 0)} "
        f"rows are exact duplicates and {len(s.get('constant_columns') or [])} column(s) are "
        "constant."
    )
    worst = sorted(
        [c for c in q["columns"] if (c.get("missing_pct") or 0) > 0],
        key=lambda c: -(c.get("missing_pct") or 0),
    )[:3]
    if worst:
        a.say(
            "Most missing: " + ", ".join(f"{c['name']} ({c['missing_pct']}%)" for c in worst) + "."
        )
    a.cite("quality", "Profile", "/profile")


def _diagnostics(a: _Answer) -> None:
    d = a.p["diagnostics"]
    if not _available(d):
        a.say(f"Diagnostics: {d['reason']} Run them on the Diagnostics tab.")
        a.cite("diagnostics", "Diagnostics", "/diagnostics")
        return
    det = d["detected"]
    a.say(
        f"Diagnostics detected {len(det)} issue(s); {len(d['not_assessed'])} check(s) could not "
        "be assessed (missing roles or data), which is not the same as passing."
    )
    for f in det[:5]:
        basis = " (statistical suspicion)" if f.get("basis") == "statistical" else ""
        a.say(f"• [{f['severity']}] {f['title']}{basis}")
    if d.get("stale"):
        a.say("Roles or the domain changed after this run; re-run diagnostics for current results.")
    a.cite("diagnostics", "Diagnostics", "/diagnostics")


def _recommendations(a: _Answer) -> None:
    r = a.p["recommendations"]
    if not _available(r):
        a.say(f"Recommendations: {r['reason']} Generate them in the Experiment Lab.")
        a.cite("recommendations", "Experiment Lab", "/lab")
        return
    open_items = [i for i in r["items"] if i["status"] == "open"]
    a.say(f"{len(open_items)} open preprocessing recommendation(s):")
    for i in open_items[:6]:
        cols = ", ".join(i["columns"][:3]) if i["columns"] else "all rows"
        risk = f", {i['risk']} risk" + (", destructive" if i["destructive"] else "")
        a.say(f"• {i.get('title') or i['op']} on {cols} ({i['stage']} stage{risk})")
    a.say(
        "Recommendations are hypotheses: compare them in the Experiment Lab on frozen folds "
        "before committing anything; high-risk steps need an approval."
    )
    a.cite("recommendations", "Experiment Lab", "/lab")


def _experiments(a: _Answer) -> None:
    e = a.p["experiments"]
    if not _available(e):
        a.say(f"Experiments: {e['reason']}")
        a.cite("experiments", "Experiment Lab", "/lab")
        return
    latest = e["items"][0]
    a.say(
        f"Latest experiment '{latest['name']}' compared {len(latest['candidates'])} candidate(s) "
        f"on {latest.get('primary_metric') or 'the primary metric'}."
    )
    for c in latest["comparisons"][:5]:
        a.say(f"• {c['candidate']} vs {c['baseline']} ({c['model']}): {c['interpretation']}")
    a.say("These describe the same folds only; they are not significance tests.")
    a.cite("experiments", "Experiment Lab", "/lab")


def _models(a: _Answer) -> None:
    m = a.p["models"]
    if not _available(m):
        a.say(f"Models: {m['reason']} Train them on the Models tab.")
        a.cite("models", "Models", "/models")
        return
    for run in m["items"][:3]:
        h = run["headline"]
        if h["state"] == "available":
            arrow = "higher is better" if h["higher_is_better"] else "lower is better"
            base = (
                f"; dummy baseline {h['baseline_mean']}"
                if h.get("baseline_mean") is not None
                else ""
            )
            a.say(
                f"Run '{run['name']}' ({run['evaluation_label']}): highest observed {h['label']} "
                f"{h['mean']} ± {h['sd']} ({h['model']}, {arrow}{base})."
            )
        else:
            a.say(f"Run '{run['name']}': {h['state']} — {h.get('reason') or ''}")
        if h.get("exploratory"):
            a.say("  This run is exploratory: its scores are likely optimistic.")
        a.cite("models", f"Run {run['name']}", f"/models/{run['training_run_id']}")
    a.say(
        "The highest of several scores on the same folds is optimistic; no model is declared the "
        "winner and no external validation exists."
    )


def _reliability(a: _Answer) -> None:
    m = a.p["models"]
    if not _available(m):
        a.say("No model has been trained, so model reliability has not been audited yet.")
        a.cite("models", "Models", "/models")
        return
    for run in m["items"][:3]:
        au = run.get("audit")
        if not au:
            a.say(f"Run '{run['name']}': the reliability audit has not finished.")
            continue
        a.say(f"Run '{run['name']}': audit status '{au['status']}'. {au.get('limitation') or ''}")
        for f in au["detected"][:4]:
            kind = "statistical suspicion" if f.get("basis") == "statistical" else "fact"
            a.say(f"• [{f['severity']}, {kind}] {f['title']}")
        if au["not_assessed"]:
            a.say(f"  {len(au['not_assessed'])} check(s) were not assessed (missing metadata).")
        a.cite("reliability", f"Audit of {run['name']}", f"/models/{run['training_run_id']}")


def _explain(a: _Answer) -> None:
    m = a.p["models"]
    if not _available(m):
        a.say("Importance is computed when models are trained; none are trained yet.")
        return
    run = m["items"][0]
    best = (run["headline"] or {}).get("model")
    model = next((x for x in run["models"] if x["model"] == best), None) or next(
        (x for x in run["models"] if x["top_importance"]), None
    )
    if model is None or not model["top_importance"]:
        a.say("No permutation importance is stored for the latest run.")
        return
    a.say(
        f"Most important inputs for {model['model']} in '{run['name']}' (permutation importance "
        "on held-out rows): "
        + ", ".join(f"{t['column']} ({t['mean']:.3g})" for t in model["top_importance"])
        + "."
    )
    a.say("Importance is not causal; correlated inputs share or mask importance.")
    a.cite("explain", f"Explainability of {run['name']}", f"/models/{run['training_run_id']}")


def _biomedical(a: _Answer) -> None:
    d = a.p["domain_findings"]
    bio = d.get("biomedical_summary")
    if not bio:
        a.say(
            "No biomedical summary is stored for this version. Open the Biomedical tab and run "
            "it (declare patient, target and batch roles first for a complete assessment)."
        )
        a.cite("biomedical", "Biomedical", "/biomedical")
        return
    pat = bio["cohort"].get("patients", {})
    if pat.get("status") == "assessed":
        a.say(
            f"{pat['unique']} unique patients; {pat['repeated_patients']} have repeated "
            f"observations (median {pat['observations_per_patient']['median']} per patient)."
        )
    else:
        a.say(pat.get("reason", "Patients were not assessed."))
    for w in bio.get("confounding", [])[:4]:
        a.say(f"• [{w['severity']}] {w['message']}")
    if bio["batches"].get("status") != "assessed":
        a.say(bio["batches"]["reason"])
    for w in bio.get("small_cohort", []):
        a.say(f"• {w['message']}")
    a.say(bio["disclaimer"])
    a.cite("biomedical", "Biomedical", "/biomedical")


def _privacy(a: _Answer) -> None:
    cols = a.p.get("protected_columns", [])
    a.say(
        f"{len(cols)} column(s) are protected as sensitive (on any version of this dataset, or "
        "derived from one): "
        + (", ".join(cols) if cols else "none")
        + ". Their category-level error slices are withheld, CSV exports drop them unless you "
        "explicitly include them with an acknowledgement, and nothing is sent to external "
        "services."
    )
    a.cite("privacy", "Data Passport", "/passport")


def _history(a: _Answer) -> None:
    h = a.p["history"]
    v = a.p["version"]
    a.say(
        f"Version {v['number']} ({v['kind']}), content hash {v['content_hash'][:16]}…; "
        f"{len(h['versions'])} version(s) in its lineage."
    )
    for item in h["versions"]:
        t = item.get("transformation")
        if t:
            a.say(
                f"• v{item['version_number']}: {len(t['steps'])} committed step(s) "
                f"({', '.join(s['op'] for s in t['steps'][:4])}), {t['approvals']} approval(s)."
            )
    if v["fitted_on_full_data"]:
        a.say(
            "This version's steps learned from all rows: evaluate models through Models (which "
            "replays them inside folds), not directly."
        )
    a.say("Download pipeline.py from Exports to reproduce this version from the original file.")
    a.cite("history", "History", "/history")


def _next(a: _Answer) -> None:
    steps: list[str] = []
    p = a.p
    if not _available(p["quality"]):
        steps.append("Profile the version (Profile tab).")
    if not p["roles"].get("roles", {}).get("target"):
        steps.append(
            "Declare the target and its task type, plus patient/group, time and "
            "sensitive roles (Diagnostics → Roles)."
        )
    if not _available(p["diagnostics"]):
        steps.append("Run diagnostics.")
    elif p["diagnostics"].get("stale"):
        steps.append("Re-run diagnostics: roles changed since the last run.")
    elif any(f["severity"] in ("critical", "high") for f in p["diagnostics"]["detected"]):
        steps.append("Review the critical and high diagnostics findings.")
    if not _available(p["recommendations"]):
        steps.append(
            "Generate preprocessing recommendations and compare them in the Experiment Lab."
        )
    if not _available(p["models"]):
        steps.append("Train baseline models in Models (grouped by patient if rows repeat).")
    else:
        run = p["models"]["items"][0]
        if run["evaluation_mode"] == "exploratory":
            steps.append(
                "Re-evaluate on a version that is not fitted on all rows (exploratory "
                "results are optimistic)."
            )
        au = run.get("audit")
        if au and au["status"] in ("blocked", "concerns"):
            steps.append(
                f"Resolve the reliability audit's '{au['status']}' findings before relying "
                "on scores."
            )
    if p["dataset"]["domain"] == "biomedical" and not p["domain_findings"].get(
        "biomedical_summary"
    ):
        steps.append("Run the biomedical summary (Biomedical tab).")
    steps.append("Export the Data Passport and pipeline.py for your records.")
    a.say("Suggested next steps:")
    for i, s in enumerate(steps, 1):
        a.say(f"{i}. {s}")
    a.cite("passport", "Data Passport", "/passport")


def _insights(a: _Answer) -> None:
    """Stored automated insights (Analytics Studio); never recomputed here."""
    ins = (a.p.get("analytics") or {}).get("insights") or {}
    if ins.get("status") != "available":
        a.say(
            "No automated insights are stored for this version yet. Open Analytics Studio → "
            "Insights and generate them: they scan distributions, missingness, outliers, "
            "correlations, group differences, trends and target associations."
        )
        a.cite("insights", "Analytics Studio", "/analytics")
        return
    labels = {
        "descriptive": "observed",
        "association": "statistical association",
        "suspicion": "suspicion",
        "configuration": "configuration fact",
    }
    top = ins.get("top") or []
    if not top:
        a.say("The automated insights found nothing notable among the assessed checks.")
    else:
        a.say(f"The {min(len(top), 5)} most notable automated insights (ranked by importance):")
        for i in top[:5]:
            a.say(f"• [{labels.get(str(i.get('basis')), i.get('basis'))}] {i['title']}")
    counts = ins.get("counts") or {}
    if counts:
        a.say(
            "Totals: "
            + ", ".join(f"{v} {labels.get(k, k)}" for k, v in counts.items() if v)
            + ". Associations are not causal; see each insight's method and limitations."
        )
    if ins.get("not_assessed"):
        a.say(f"{len(ins['not_assessed'])} area(s) were not assessed (missing roles or data).")
    a.cite("insights", "Insights", "/analytics")


HANDLERS: dict[str, list[Callable[[_Answer], None]]] = {
    "insights": [_insights],
    "privacy": [_privacy],
    "biomedical": [_biomedical],
    "reliability": [_reliability],
    "explain": [_explain],
    "models": [_models],
    "experiments": [_experiments],
    "recommendations": [_recommendations],
    "history": [_history],
    "next": [_next],
    "diagnostics": [_quality, _diagnostics],
    "overview": [_quality, _diagnostics, _next],
}


def classify(question: str) -> str:
    q = re.sub(r"\s+", " ", question.lower())
    best, hits = "overview", 0
    for topic, words in TOPICS:
        n = sum(1 for w in words if w in q)
        if n > hits:
            best, hits = topic, n
    return best


def answer(passport: dict[str, Any], question: str, base: str) -> dict[str, Any]:
    topic = classify(question)
    a = _Answer(passport, base)
    for handler in HANDLERS[topic]:
        handler(a)
    follow = [s for s in SUGGESTIONS if classify(s) != topic][:4]
    return {
        "assistant": ASSISTANT,
        "topic": topic,
        "answer": a.lines,
        "references": a.refs,
        "suggestions": follow,
    }
