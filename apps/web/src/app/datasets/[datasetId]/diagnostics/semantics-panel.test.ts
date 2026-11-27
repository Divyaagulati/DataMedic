import { describe, expect, it } from "vitest";

import { roleErrorMessage } from "./semantics-panel";

describe("roleErrorMessage", () => {
  it("names the column of each invalid role", () => {
    const problem = {
      code: "VALIDATION_FAILED",
      status: 422,
      title: "Validation failed",
      errors: [
        { pointer: "/roles/1", code: "TASK_TYPE_REQUIRED", message: "Choose binary, multiclass or regression." },
      ],
    };
    const submitted = [
      { column: "age", role: "bio.age" },
      { column: "outcome", role: "target" },
    ];
    expect(roleErrorMessage(problem, submitted)).toBe(
      "outcome: Choose binary, multiclass or regression.",
    );
  });

  it("falls back to the general message", () => {
    expect(roleErrorMessage({ code: "CONFLICT", status: 409, title: "x", detail: "Busy." }, [])).toBe("Busy.");
  });
});
