import createClient, { type Middleware } from "openapi-fetch";

import type { components, paths } from "./schema";

/** Base URL of the DataMedic API. Browsers talk to the API directly (with cookies). */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Schemas = components["schemas"];
export type Dataset = Schemas["DatasetOut"];
export type Job = Schemas["JobOut"];
export type User = Schemas["UserOut"];

const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

let csrfToken: string | null = null;

/** The CSRF token comes from /auth/login or /auth/me and is kept in memory only. */
export function setCsrfToken(token: string | null): void {
  csrfToken = token;
}

export function getCsrfToken(): string | null {
  return csrfToken;
}

const csrfMiddleware: Middleware = {
  onRequest({ request }) {
    if (csrfToken && UNSAFE_METHODS.has(request.method)) {
      request.headers.set("X-CSRF-Token", csrfToken);
    }
    return request;
  },
};

export const api = createClient<paths>({ baseUrl: API_BASE_URL, credentials: "include" });
api.use(csrfMiddleware);

/** For TanStack Query: return the data, or throw the problem so the query enters error state. */
export function unwrap<T>(result: { data?: T; error?: unknown }): T {
  if (result.data !== undefined) return result.data;
  throw result.error ?? new Error("Empty response");
}
