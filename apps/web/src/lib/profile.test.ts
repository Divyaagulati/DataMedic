import { describe, expect, it } from "vitest";

import {
  binLabel,
  classBalance,
  correlationColor,
  formatNumber,
  formatPct,
  formatStat,
  missingRanking,
  skewLabel,
  type ColumnProfile,
} from "./profile";

function column(overrides: Partial<ColumnProfile>): ColumnProfile {
  return {
    name: "y",
    inferred_type: "categorical",
    dtype: "string",
    n_missing: 0,
    missing_pct: 0,
    n_present: 10,
    n_unique: 2,
    unique_pct: 20,
    is_empty: false,
    is_constant: false,
    top_value_pct: 70,
    numeric: null,
    datetime: null,
    text: null,
    histogram: null,
    frequencies: {
      values: [
        { value: "no", count: 7, pct: 70 },
        { value: "yes", count: 3, pct: 30 },
      ],
      complete: true,
      other_count: 0,
      reason: null,
    },
    ...overrides,
  };
}

describe("classBalance", () => {
  it("computes the imbalance ratio", () => {
    const result = classBalance(column({}));
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.majority.label).toBe("no");
      expect(result.minority.label).toBe("yes");
      expect(result.ratio).toBeCloseTo(7 / 3);
    }
  });

  it("labels boolean and numeric classes", () => {
    const result = classBalance(
      column({
        frequencies: {
          values: [
            { value: true, count: 2, pct: 50 },
            { value: 0, count: 2, pct: 50 },
          ],
          complete: true,
          other_count: 0,
          reason: null,
        },
      }),
    );
    expect(result.ok && result.classes.map((c) => c.label)).toEqual(["true", "0"]);
  });

  it.each([
    [{ frequencies: null, is_empty: true }, "no values"],
    [{ inferred_type: "identifier" as const }, "identifier"],
    [
      { n_unique: 300, frequencies: { values: [], complete: false, other_count: 300, reason: "x" } },
      "too many",
    ],
    [
      {
        frequencies: { values: [{ value: "a", count: 5, pct: 100 }], complete: true, other_count: 0 },
      },
      "Only one class",
    ],
  ])("explains why a distribution is unavailable (%#)", (overrides, text) => {
    const result = classBalance(column(overrides as Partial<ColumnProfile>));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain(text);
  });
});

describe("formatting", () => {
  it("formats numbers", () => {
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(1284)).toBe("1,284");
    expect(formatNumber(12900)).toBe("12.9K");
    expect(formatNumber(3.14159)).toBe("3.142");
  });

  it("formats percentages without rounding partial shares to 0% or 100%", () => {
    expect(formatPct(0)).toBe("0%");
    expect(formatPct(0.04)).toBe("<0.1%");
    expect(formatPct(4.25)).toBe("4.3%");
    expect(formatPct(42.6)).toBe("42.6%");
    expect(formatPct(99.96)).toBe(">99.9%");
    expect(formatPct(100)).toBe("100%");
  });

  it("formats statistics exactly, never compacted", () => {
    expect(formatStat(123456.78)).toBe("123,457");
    expect(formatStat(2.1380899)).toBe("2.13809");
    expect(formatStat(null)).toBe("—");
  });

  it("describes skewness", () => {
    expect(skewLabel(0.2)).toBe("approximately symmetric");
    expect(skewLabel(-0.7)).toBe("moderately skewed (left)");
    expect(skewLabel(2.1)).toBe("highly skewed (right)");
    expect(skewLabel(null)).toBe("not available");
  });

  it("labels histogram bins", () => {
    expect(binLabel("numeric", 1, 2.5)).toBe("1 – 2.5");
    expect(binLabel("datetime", "2024-01-01T00:00:00", "2024-02-01T00:00:00")).toBe(
      "2024-01-01 – 2024-02-01",
    );
  });
});

describe("correlationColor", () => {
  it("uses the neutral midpoint at zero and the poles at the extremes", () => {
    expect(correlationColor(0)).toContain("var(--viz-div-pos) 0%");
    expect(correlationColor(-1)).toContain("var(--viz-div-neg) 100%");
    expect(correlationColor(0.5)).toContain("var(--viz-div-pos) 50%");
    expect(correlationColor(null)).toBe("var(--viz-surface)");
  });
});

describe("missingRanking", () => {
  it("lists only columns with missing values, worst first", () => {
    const ranked = missingRanking([
      column({ name: "a", n_missing: 1, missing_pct: 10 }),
      column({ name: "b" }),
      column({ name: "c", n_missing: 5, missing_pct: 50 }),
    ]);
    expect(ranked.map((c) => c.name)).toEqual(["c", "a"]);
  });
});
