import { describe, expect, it } from "vitest";

import {
  defaultParams,
  groupByFamily,
  indicatorColumn,
  KIND_INFO,
  missingRequiredParams,
  opsForColumnType,
  RISK_INFO,
  STAGE_INFO,
  type Op,
} from "./preprocess";

function op(overrides: Partial<Op>): Op {
  return {
    op: "impute.median",
    version: "1.0.0",
    family: "missing",
    title: "Median imputation",
    description: "d",
    when_to_use: "w",
    downsides: ["x"],
    limitations: ["y"],
    stage: "either",
    kind: "fitted",
    risk: "medium",
    destructive: false,
    column_types: ["numeric"],
    adds_columns: false,
    max_columns: 500,
    uses_columns: true,
    needs_target: false,
    params: [],
    ...overrides,
  };
}

describe("preprocess helpers", () => {
  it("filters ops by column type", () => {
    const ops = [
      op({ op: "impute.median" }),
      op({ op: "impute.mode", column_types: ["boolean", "categorical", "numeric"] }),
    ];
    expect(opsForColumnType(ops, "categorical").map((o) => o.op)).toEqual(["impute.mode"]);
    expect(opsForColumnType(ops, "numeric")).toHaveLength(2);
    expect(opsForColumnType(ops, "datetime")).toEqual([]);
  });

  it("groups by family in order", () => {
    const groups = groupByFamily([op({ op: "a" }), op({ op: "b", family: "outliers" }), op({ op: "c" })]);
    expect([...groups.keys()]).toEqual(["missing", "outliers"]);
    expect(groups.get("missing")!.map((o) => o.op)).toEqual(["a", "c"]);
  });

  it("fills defaults and reports missing required params", () => {
    const knn = op({
      op: "impute.knn",
      params: [
        { name: "n_neighbors", type: "int", description: "", default: 5, required: false, minimum: 1, maximum: 50, choices: null },
        { name: "predictors", type: "columns", description: "", default: null, required: false, minimum: null, maximum: null, choices: null },
      ],
    });
    expect(defaultParams(knn)).toEqual({ n_neighbors: 5 });
    const constant = op({
      op: "impute.constant",
      params: [
        { name: "fill_value", type: "scalar", description: "", default: null, required: true, minimum: null, maximum: null, choices: null },
      ],
    });
    expect(missingRequiredParams(constant, {})).toEqual(["fill_value"]);
    expect(missingRequiredParams(constant, { fill_value: 0 })).toEqual([]);
  });

  it("has an explanation for every classification", () => {
    expect(Object.keys(RISK_INFO)).toEqual(["low", "medium", "high"]);
    expect(Object.keys(STAGE_INFO)).toEqual(["data", "model", "either"]);
    expect(STAGE_INFO.either.help).toMatch(/leakage-controlled model evaluation/);
    expect(KIND_INFO.fitted).toMatch(/training rows only/);
    expect(KIND_INFO.training_only).toMatch(/test rows and saved versions are never changed/);
  });

  it("mirrors the backend indicator name", () => {
    expect(indicatorColumn("age")).toBe("age__missing");
  });
});
