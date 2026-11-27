"""Feature-selection stability across folds.

For every selection step of a candidate, each fold records which columns the statistic
selected (``kept``) and which it would remove or removed. Columns kept only because the user
listed them or because of a domain role are *forced*, not selected, and are reported
separately. From the folds:

- **selection frequency** per column: share of the folds that assessed it in which it was
  selected;
- **stability**: mean pairwise Jaccard similarity of the selected sets (1 = identical in every
  fold);
- **classes**: ``stable_selected`` (frequency >= 0.8), ``stable_not_selected`` (<= 0.2),
  ``unstable`` (in between);
- **noise probes** (supervised rankings only): in each fold the step is re-fitted on the same
  training rows with up to five extra columns that are random permutations of the step's own
  numeric columns. A probe carries no information about the target by construction, so a
  column selected no more often than the probes cannot be distinguished from noise.

Selection is predictive association in these rows. It is not causal, scientific or clinical
validation (for biomedical data: predictive selection is not biomarker validation).
"""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.preprocess.base import DMTransformer

STABLE = 0.8
UNSTABLE_LOW = 0.2
MAX_PROBES = 5
PROBE_PREFIX = "__probe_"
PROBED_OPS = ("select.mutual_information", "select.univariate")
SELECTED = ("kept",)
NOT_SELECTED = ("would_remove", "removed")
FORCED = ("kept_by_user", "kept_domain_role")

LIMITATIONS = [
    "Selection reflects association with the target in each training fold, not causation.",
    "A frequently selected column is not scientifically or clinically validated by being "
    "selected; for biomedical data, predictive selection is not biomarker validation.",
    "Few folds give coarse frequencies (with 5 folds: 0, 0.2, ..., 1).",
]


def fold_selection(decisions: dict[str, str]) -> dict[str, list[str]]:
    return {
        "selected": sorted(c for c, d in decisions.items() if d in SELECTED),
        "not_selected": sorted(c for c, d in decisions.items() if d in NOT_SELECTED),
        "forced": sorted(c for c, d in decisions.items() if d in FORCED),
    }


def probe_selection(
    transformer: DMTransformer, train_input: pd.DataFrame, seed: int
) -> dict[str, int] | None:
    """Re-fit a copy of ``transformer``'s step on ``train_input`` (the training rows as the
    step sees them) with permuted copies of its numeric columns added. Returns how many
    probes were selected, or None when the op is not probed or has no numeric column."""
    if transformer.op not in PROBED_OPS:
        return None
    sources = [
        c
        for c in transformer.columns
        if pd.api.types.is_numeric_dtype(train_input[c])
        and not pd.api.types.is_bool_dtype(train_input[c])
    ][:MAX_PROBES]
    if not sources:
        return None
    rng = np.random.default_rng(seed)
    frame = train_input.copy()
    names = []
    for i, c in enumerate(sources):
        name = f"{PROBE_PREFIX}{i}"
        frame[name] = frame[c].to_numpy()[rng.permutation(len(frame))]
        names.append(name)
    params = dict(transformer.params or {})
    if "discrete" in params:
        flags = dict(zip(transformer.columns, params["discrete"], strict=True))
        params["discrete"] = list(params["discrete"]) + [flags[c] for c in sources]
    probe = type(transformer)(
        columns=tuple(transformer.columns) + tuple(names), params=params, seed=transformer.seed
    )
    probe.fit(frame)
    decisions = dict(probe.decisions())
    return {
        "probes": len(names),
        "probes_selected": sum(decisions.get(n) in SELECTED for n in names),
    }


def summarise(step: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    """``folds``: per fold ``{selected, not_selected, forced, probe}`` for this step."""
    assessed: dict[str, int] = {}
    chosen: dict[str, int] = {}
    forced: set[str] = set()
    sets = []
    for f in folds:
        for c in f["selected"] + f["not_selected"]:
            assessed[c] = assessed.get(c, 0) + 1
        for c in f["selected"]:
            chosen[c] = chosen.get(c, 0) + 1
        forced.update(f["forced"])
        sets.append(set(f["selected"]))
    frequency = {c: round(chosen.get(c, 0) / n, 4) for c, n in sorted(assessed.items())}
    jaccard = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(sets, 2)]
    probes = [f["probe"] for f in folds if f.get("probe")]
    probe_rate = None
    if probes:
        total = sum(p["probes"] for p in probes)
        probe_rate = round(sum(p["probes_selected"] for p in probes) / total, 4) if total else None
    stable = sorted(c for c, v in frequency.items() if v >= STABLE)
    never = sorted(c for c, v in frequency.items() if v <= UNSTABLE_LOW)
    unstable = sorted(c for c, v in frequency.items() if UNSTABLE_LOW < v < STABLE)
    warnings = []
    if jaccard and float(np.mean(jaccard)) < 0.5:
        warnings.append(
            {
                "code": "UNSTABLE_SELECTION",
                "message": "Different folds selected "
                "substantially different columns (mean Jaccard similarity below 0.5). "
                "Treat the selection as uncertain.",
            }
        )
    noise_like: list[str] = []
    if probe_rate is not None and probe_rate > 0:
        noise_like = sorted(c for c, v in frequency.items() if v <= probe_rate)
        warnings.append(
            {
                "code": "NOISE_SELECTED",
                "message": f"Random noise columns were "
                f"selected in {probe_rate:.0%} of probe draws. Columns selected no more "
                "often than noise cannot be distinguished from noise.",
            }
        )
    return {
        "step": step["index"],
        "op": step["op"],
        "action": step.get("action"),
        "folds": len(folds),
        "per_fold": [{"selected": f["selected"], "forced": f["forced"]} for f in folds],
        "frequency": frequency,
        "mean_jaccard": round(float(np.mean(jaccard)), 4) if jaccard else None,
        "stable_selected": stable,
        "stable_not_selected": never,
        "unstable": unstable,
        "forced": sorted(forced),
        "probe_selection_rate": probe_rate,
        "indistinguishable_from_noise": noise_like,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }
