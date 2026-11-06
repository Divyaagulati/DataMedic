import { describe, expect, it } from "vitest";

import { isProblem, problemMessage } from "./problem";

describe("problemMessage", () => {
  it("prefers field-level messages", () => {
    expect(
      problemMessage({
        title: "Validation failed",
        status: 422,
        code: "VALIDATION_FAILED",
        errors: [{ pointer: "/password", code: "PASSWORD_POLICY", message: "too short" }],
      }),
    ).toBe("too short");
  });

  it("uses the server detail, then a friendly default", () => {
    expect(problemMessage({ title: "x", status: 404, code: "NOT_FOUND", detail: "Gone." })).toBe(
      "Gone.",
    );
    expect(problemMessage({ title: "x", status: 429, code: "RATE_LIMITED" })).toMatch(/wait/i);
  });

  it("falls back for network errors", () => {
    expect(problemMessage(new TypeError("Failed to fetch"))).toMatch(/could not reach/i);
    expect(isProblem({ status: 500 })).toBe(false);
  });
});
