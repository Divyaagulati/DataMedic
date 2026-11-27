import { describe, expect, it } from "vitest";

import type { FindingView } from "@/lib/diagnostics";

import {
  DEFAULT_TRAINING,
  fmt,
  fmtStats,
  groupFindings,
  headlineView,
  linePath,
  MODE_INFO,
  trainingPayload,
} from "./training";

function finding(status: FindingView["status"], basis: string): FindingView {
  return {
    id: `${status}-${basis}`, check_id: "x", check_version: "1", domain: "reliability", category: "leakage",
    status, severity: "high", title: "t", explanation: "e", evidence: { basis }, columns: [],
    affected_rows: null, affected_pct: null, recommendation: null, limitations: ["l"],
    not_assessed_reason: status === "not_assessed" ? "r" : null, error_code: null, sampled: false, sample_info: null,
  } as FindingView;
}

describe("training helpers", () => {
  it("builds the request body; null steps keep the version's modelling specification", () => {
    const body = trainingPayload({ ...DEFAULT_TRAINING, name: "  ", allowExploratory: true }, null);
    expect(body.preprocess_steps).toBeNull();
    expect(body.name).toBeNull();
    expect(body.tuning).toEqual({ mode: "nested" });
    expect(body.split).toEqual({ method: null, n_splits: 5, seed: 0 });
    expect(body.allow_exploratory).toBe(true);
  });

  it("never renders an undefined metric as zero", () => {
    expect(fmt(null)).toBe("—");
    expect(fmt(undefined)).toBe("—");
    expect(fmt(Number.NaN)).toBe("—");
    expect(fmt(0)).toBe("0.000");
    expect(fmtStats({ mean: null, sd: null, min: null, max: null, n_folds: 0, values: [] })).toBe("—");
    expect(fmtStats({ mean: 0.81234, sd: 0.02, min: 0.8, max: 0.83, n_folds: 3, values: [] })).toBe("0.812 ± 0.020");
  });

  it("separates facts, suspicions, passed and unassessed checks", () => {
    const g = groupFindings([
      finding("detected", "confirmed"),
      finding("detected", "configuration"),
      finding("detected", "statistical"),
      finding("not_detected", "confirmed"),
      finding("not_assessed", "statistical"),
    ]);
    expect(g.violations).toHaveLength(2);
    expect(g.suspicions).toHaveLength(1);
    expect(g.passed).toHaveLength(1);
    expect(g.notAssessed).toHaveLength(1);
  });

  it("draws unit-square curves with y up", () => {
    expect(linePath([[0, 0], [1, 1]], 100)).toBe("M0.0,100.0 L100.0,0.0");
  });

  it("renders the headline with its direction, never as a winner, and qualifies exploratory runs", () => {
    const base = {
      state: "available", metric: "roc_auc", label: "ROC-AUC", higher_is_better: true, model: "linear", mean: 0.8512,
      sd: 0.02, baseline_mean: 0.5, n_folds: 3, total_folds: 3, exploratory: false,
      note: "Highest observed mean ... not a declared winner.", exploratory_note: null, reason: null,
    };
    const v = headlineView(base);
    expect(v.text).toBe("ROC-AUC ↑ 0.851 ± 0.020 · highest observed: Linear");
    expect(v.title).toMatch(/not a declared winner/);
    expect(v.title).toMatch(/Dummy baseline: 0.500/);
    expect(v.text).not.toMatch(/best|winner/i);
    expect(v.qualifier).toBeNull();
    const rmse = headlineView({ ...base, metric: "rmse", label: "RMSE", higher_is_better: false, mean: 1.5, sd: null });
    expect(rmse.text).toBe("RMSE ↓ 1.500 · highest observed: Linear");
    expect(rmse.title).toMatch(/Lower is better/);
    const partial = headlineView({ ...base, n_folds: 2 });
    expect(partial.text).toMatch(/\(2 of 3 folds\)/);
    const exp = headlineView({ ...base, exploratory: true, exploratory_note: "Exploratory: likely optimistic." });
    expect(exp.qualifier).toBe("Exploratory: likely optimistic");
    expect(exp.title).toMatch(/Exploratory: likely optimistic\./);
  });

  it("shows honest states instead of a dash-only headline", () => {
    const h = (state: string, reason: string | null) => headlineView({ state, mean: null, reason });
    expect(h("pending", "The run has not finished.").text).toBe("Pending");
    expect(h("cancelled", "The run was cancelled; no results were stored.").text).toBe("Cancelled");
    expect(h("failed", "x").text).toBe("No results");
    const u = h("unavailable", "ROC-AUC could not be computed for any compared model.");
    expect(u.text).toBe("Unavailable");
    expect(u.title).toMatch(/could not be computed/);
    expect(headlineView(null).text).toBe("—");
  });

  it("labels evaluation modes without claiming unbiasedness", () => {
    expect(MODE_INFO.unbiased_direct.label).toBe("Leakage-controlled evaluation");
    expect(MODE_INFO.unbiased_replay.label).toBe("Leakage-controlled evaluation (replayed)");
    expect(MODE_INFO.exploratory.label).toBe("Exploratory only");
    expect(MODE_INFO.blocked.label).toBe("Blocked");
    for (const info of Object.values(MODE_INFO)) {
      expect(`${info.label} ${info.help}`).not.toMatch(/\bunbiased\b/i);
    }
    expect(MODE_INFO.unbiased_direct.help).toMatch(/does not guarantee external validity, the absence of bias or clinical reliability/);
  });
});

