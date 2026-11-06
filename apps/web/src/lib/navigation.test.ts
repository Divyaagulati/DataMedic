import { describe, expect, it } from "vitest";

import { formatBytes } from "./format";
import { safeNext } from "./navigation";

describe("safeNext", () => {
  it.each([
    [null, "/datasets"],
    ["/datasets/abc", "/datasets/abc"],
    ["https://evil.example", "/datasets"],
    ["//evil.example", "/datasets"],
    ["/\\evil.example", "/datasets"],
    ["javascript:alert(1)", "/datasets"],
  ])("safeNext(%s) -> %s", (input, expected) => {
    expect(safeNext(input)).toBe(expected);
  });
});

describe("formatBytes", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(200 * 1024 * 1024)).toBe("200 MB");
  });
});
