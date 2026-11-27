"use client";

import { useQuery } from "@tanstack/react-query";

import { API_BASE_URL, api, getCsrfToken, unwrap } from "@/lib/api/client";

const POLL_MS = 2000;
const ACTIVE_JOB = new Set(["queued", "running", "cancel_requested"]);

export type Dict = Record<string, unknown>;

/** Biomedical workspace of a version; polls while its summary job runs. */
export function useBiomedical(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "biomedical"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/biomedical", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

/** The Data Passport (structured JSON built from stored results). */
export function usePassport(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "passport"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/api/v1/versions/${versionId}/passport`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error(`The passport could not be loaded (${res.status}).`);
      return (await res.json()) as Dict;
    },
    enabled: Boolean(versionId),
  });
}

/** Download an authorised API file through fetch (cookies) and save it locally. The browser
 * never receives a storage URL; the object URL is revoked immediately after use. */
export async function downloadFile(path: string, fallbackName: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}${path}`, { credentials: "include" });
  if (!res.ok) throw new Error(`Download failed (${res.status}).`);
  const disposition = res.headers.get("content-disposition") ?? "";
  const match = /filename="([^"]+)"/.exec(disposition);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = match?.[1] ?? fallbackName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export type CopilotAnswer = {
  assistant: { name: string; kind: string; note: string };
  topic: string;
  answer: string[];
  references: { section: string; label: string; href: string }[];
  suggestions: string[];
  analysis?: Record<string, unknown>;
};

export async function askCopilot(versionId: string, question: string): Promise<CopilotAnswer> {
  const res = await fetch(`${API_BASE_URL}/api/v1/versions/${versionId}/copilot`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": getCsrfToken() ?? "" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`The assistant could not answer (${res.status}).`);
  return (await res.json()) as CopilotAnswer;
}

export const COPILOT_SUGGESTIONS = [
  "Is this dataset ready for modelling?",
  "What are the most important data-quality issues?",
  "Which preprocessing steps are recommended?",
  "How well do the models perform?",
  "Can I trust the model results?",
  "Which features matter most?",
  "Is there patient-level leakage or a batch effect?",
  "What should I do next?",
];

export const SEVERITY_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };

export function fmtNum(v: unknown, digits = 3): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}
