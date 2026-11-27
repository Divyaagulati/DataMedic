import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "@/app/page";
import { setCsrfToken } from "@/lib/api/client";

import { askCopilot, downloadFile, fmtNum } from "./mvp";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("MVP helpers", () => {
  it("formats numbers without inventing values", () => {
    expect(fmtNum(0.12345)).toBe("0.123");
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(Number.NaN)).toBe("—");
  });

  it("asks the built-in Copilot with the CSRF token and cookies", async () => {
    setCsrfToken("tok");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ assistant: { kind: "built-in rule-based assistant" }, answer: ["x"], references: [], suggestions: [], topic: "next" })),
    );
    const out = await askCopilot("v1", "What next?");
    expect(out.assistant.kind).toBe("built-in rule-based assistant");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/v1\/versions\/v1\/copilot$/);
    expect(init.credentials).toBe("include");
    expect((init.headers as Record<string, string>)["X-CSRF-Token"]).toBe("tok");
  });

  it("downloads through the API with the server filename and revokes the object URL", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("a,b\n1,2\n", { headers: { "content-disposition": 'attachment; filename="Cohort_v0.csv"' } }),
    );
    const create = vi.fn(() => "blob:x");
    const revoke = vi.fn();
    Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    await downloadFile("/api/v1/exports/j/download", "export.csv");
    expect(click).toHaveBeenCalled();
    expect(revoke).toHaveBeenCalledWith("blob:x");
  });

  it("fails loudly when a download is refused", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("", { status: 404 }));
    await expect(downloadFile("/x", "y")).rejects.toThrow(/404/);
  });
});

describe("Landing page", () => {
  it("communicates the complete workflow and the demonstration scope", () => {
    render(<Home />);
    for (const step of ["Upload", "Understand", "Diagnose", "Experiment", "Transform", "Train", "Audit", "Export"]) {
      expect(screen.getByText(step)).toBeTruthy();
    }
    expect(screen.getByText(/privacy-first, domain-aware data intelligence and ML reliability platform/)).toBeTruthy();
    expect(screen.getByText(/use synthetic or non-sensitive data/i)).toBeTruthy();
  });
});
