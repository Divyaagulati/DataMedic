import { afterEach, describe, expect, it, vi } from "vitest";

describe("api client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  async function setup() {
    const calls: Request[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (req: Request) => {
        calls.push(req);
        return new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    const mod = await import("./client");
    return { ...mod, calls };
  }

  it("sends the CSRF token on unsafe methods only, with credentials", async () => {
    const { api, setCsrfToken, calls } = await setup();
    setCsrfToken("csrf-123");
    await api.GET("/api/v1/datasets");
    await api.POST("/api/v1/auth/logout");
    expect(calls[0].headers.get("X-CSRF-Token")).toBeNull();
    expect(calls[1].headers.get("X-CSRF-Token")).toBe("csrf-123");
    expect(calls[1].credentials).toBe("include");
  });

  it("never sends a client-chosen user id", async () => {
    const { api, calls } = await setup();
    await api.GET("/api/v1/datasets");
    expect(calls[0].url).not.toMatch(/user_id|owner_id/);
  });
});
