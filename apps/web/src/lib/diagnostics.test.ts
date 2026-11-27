import { describe, expect, it } from "vitest";

import {
  countBy,
  evidenceLabel,
  evidenceScalar,
  filterFindings,
  type FindingView,
} from "./diagnostics";

function finding(overrides: Partial<FindingView>): FindingView {
  return {
    id: crypto.randomUUID(),
    check_id: "universal.x",
    check_version: "1.0.0",
    domain: "universal",
    category: "missingness",
    status: "detected",
    severity: "low",
    title: "t",
    explanation: "e",
    evidence: { n: 1 },
    columns: [],
    affected_rows: null,
    affected_pct: null,
    recommendation: "r",
    limitations: ["l"],
    not_assessed_reason: null,
    error_code: null,
    sampled: false,
    sample_info: null,
    ...overrides,
  };
}

const ALL = { severity: "all", category: "all", domain: "all" } as const;

describe("filterFindings", () => {
  const items = [
    finding({ check_id: "b", severity: "low" }),
    finding({ check_id: "a", severity: "critical" }),
    finding({ check_id: "c", severity: "high", category: "leakage", domain: "biomedical" }),
    finding({ check_id: "d", status: "not_assessed", severity: "info" }),
  ];

  it("keeps one status and sorts by severity, then check id", () => {
    expect(filterFindings(items, { status: "detected", ...ALL }).map((f) => f.check_id)).toEqual([
      "a",
      "c",
      "b",
    ]);
    expect(filterFindings(items, { status: "not_assessed", ...ALL })).toHaveLength(1);
  });

  it("applies severity, category and domain filters together", () => {
    expect(
      filterFindings(items, { status: "detected", severity: "high", category: "leakage", domain: "biomedical" }),
    ).toHaveLength(1);
    expect(filterFindings(items, { status: "detected", ...ALL, domain: "financial" })).toEqual([]);
  });

  it("counts by a key", () => {
    expect(countBy(items, (f) => f.status).get("detected")).toBe(3);
  });
});

describe("evidence formatting", () => {
  it("labels keys readably", () => {
    expect(evidenceLabel("rows_in_duplicate_groups")).toBe("Rows in duplicate groups");
    expect(evidenceLabel("missing_pct")).toBe("Missing %");
    expect(evidenceLabel("roc_auc")).toBe("ROC AUC");
  });

  it("formats scalars", () => {
    expect(evidenceScalar(1234)).toBe("1,234");
    expect(evidenceScalar(0.123456)).toBe("0.1235");
    expect(evidenceScalar(0.0000012)).toBe("1.20e-6");
    expect(evidenceScalar(null)).toBe("—");
    expect(evidenceScalar(true)).toBe("yes");
  });
});
