"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type Schemas } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { TYPE_LABELS, type SemanticType } from "@/lib/profile";

type RoleSpec = Schemas["RoleSpecOut"];
type ColumnSchema = Schemas["ColumnSchemaOut"];
type TaskType = "binary" | "multiclass" | "regression";
type Draft = Record<string, { role: string; task_type: TaskType | null }>;

const SELECT = "h-8 rounded-md border bg-background px-2 text-sm";

/** Domain choice and one role per column (the API allows several; one covers common use). */
export function SemanticsPanel({
  datasetId,
  versionId,
  domain,
  columns,
  domains,
  roles,
  onNotice,
}: {
  datasetId: string;
  versionId: string;
  domain: string;
  columns: ColumnSchema[];
  domains: Schemas["DomainOut"][];
  roles: Schemas["RolesOut"];
  /** Success messages live in the parent: this panel remounts after every successful save. */
  onNotice: (message: string) => void;
}) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The parent keys this component by the roles' semantics hash, so the draft re-initialises
  // whenever the saved roles change.
  const [draft, setDraft] = useState<Draft>(() => {
    const out: Draft = {};
    for (const r of roles.roles) out[r.column] = { role: r.role, task_type: r.task_type ?? null };
    return out;
  });

  const availableKeys = new Set(roles.available_roles.map((r) => r.key));
  // Roles saved under another domain (or no longer matching the column type) stay stored but
  // are not offered here; show them explicitly instead of letting the select hide them.
  const unavailable = roles.roles.filter((r) => !availableKeys.has(r.role));

  const multiRole = new Set(
    roles.roles.map((r) => r.column).filter((c, i, all) => all.indexOf(c) !== i),
  );

  async function changeDomain(next: string) {
    setBusy(true);
    setError(null);
    const { error: problem, response } = await api.PATCH("/api/v1/datasets/{dataset_id}", {
      params: { path: { dataset_id: datasetId } },
      body: { domain: next },
    });
    if (response.status !== 200) setError(problemMessage(problem));
    else onNotice("Domain changed. Existing diagnostics are now out of date: run them again.");
    await queryClient.invalidateQueries({ queryKey: ["datasets", datasetId] });
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId] });
    setBusy(false);
  }

  async function saveRoles() {
    setBusy(true);
    setError(null);
    const submitted = Object.entries(draft)
      .filter(([, v]) => v.role)
      .map(([column, v]) => ({
        column,
        role: v.role,
        task_type: v.role === "target" ? v.task_type : null,
      }));
    const { error: problem, response } = await api.PUT("/api/v1/versions/{version_id}/roles", {
      params: { path: { version_id: versionId } },
      body: { roles: submitted },
    });
    if (response.status !== 200) setError(roleErrorMessage(problem, submitted));
    else onNotice("Roles saved. Existing diagnostics are now out of date: run them again.");
    await queryClient.invalidateQueries({ queryKey: ["versions", versionId] });
    setBusy(false);
  }

  function rolesFor(column: ColumnSchema): RoleSpec[] {
    return roles.available_roles.filter((r) => r.allowed_types.includes(column.inferred_type));
  }

  const domainName = domains.find((d) => d.key === domain)?.name ?? domain;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Domain and column roles</CardTitle>
        <CardDescription>
          Domain checks run only when the columns they need are declared; otherwise they are
          reported as “not assessed”. Roles are never guessed or applied automatically.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FormError message={error} />
        {unavailable.length ? (
          <FormNotice
            message={`${unavailable.length} saved role(s) are not valid for this domain and are ignored by its checks: ${unavailable
              .map((r) => `${r.column} (${r.role})`)
              .join(", ")}. Choose another role or “Feature (no role)” for these columns, then save.`}
          />
        ) : null}
        <label className="flex max-w-md flex-col gap-1 text-sm">
          <span className="text-muted-foreground">Domain</span>
          <select
            className={SELECT}
            value={domain}
            disabled={busy}
            onChange={(e) => void changeDomain(e.target.value)}
          >
            {domains.map((d) => (
              <option key={d.key} value={d.key}>
                {d.name}
                {d.check_count ? ` (${d.check_count} extra checks)` : ""}
              </option>
            ))}
          </select>
        </label>
        <div className="max-h-[28rem] overflow-auto rounded-md border">
          <table className="w-full text-sm">
            <caption className="sr-only">Role of each column for {domainName}</caption>
            <thead className="sticky top-0 bg-muted text-xs">
              <tr>
                <th scope="col" className="px-3 py-2 text-left font-medium">Column</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Detected type</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Role</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Task type</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((column) => {
                const value = draft[column.name] ?? { role: "", task_type: null };
                return (
                  <tr key={column.name} className="border-t">
                    <td className="max-w-[14rem] truncate px-3 py-1.5" title={column.name}>
                      {column.name}
                    </td>
                    <td className="px-3 py-1.5 text-muted-foreground">
                      {TYPE_LABELS[column.inferred_type as SemanticType]}
                    </td>
                    <td className="px-3 py-1.5">
                      <select
                        aria-label={`Role of ${column.name}`}
                        className={SELECT}
                        value={value.role}
                        disabled={multiRole.has(column.name)}
                        title={multiRole.has(column.name) ? "Several roles set through the API" : undefined}
                        onChange={(e) =>
                          setDraft({
                            ...draft,
                            [column.name]: { role: e.target.value, task_type: value.task_type },
                          })
                        }
                      >
                        <option value="">Feature (no role)</option>
                        {value.role && !rolesFor(column).some((r) => r.key === value.role) ? (
                          <option value={value.role}>{value.role} (not valid here)</option>
                        ) : null}
                        {rolesFor(column).map((r) => (
                          <option key={r.key} value={r.key} title={r.description}>
                            {r.name}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-3 py-1.5">
                      {value.role === "target" ? (
                        <select
                          aria-label={`Task type of ${column.name}`}
                          className={SELECT}
                          value={value.task_type ?? ""}
                          onChange={(e) =>
                            setDraft({
                              ...draft,
                              [column.name]: {
                                role: "target",
                                task_type: (e.target.value || null) as TaskType | null,
                              },
                            })
                          }
                        >
                          <option value="">Choose…</option>
                          <option value="binary">Binary</option>
                          <option value="multiclass">Multiclass</option>
                          <option value="regression">Regression</option>
                        </select>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Button onClick={() => void saveRoles()} disabled={busy}>
          Save roles
        </Button>
      </CardContent>
    </Card>
  );
}

/** Prefix each per-role error with its column (the API points at /roles/<index>). */
export function roleErrorMessage(
  problem: unknown,
  submitted: { column: string; role: string }[],
): string {
  const errors = (problem as { errors?: { pointer: string; message?: string }[] } | undefined)
    ?.errors;
  if (!errors?.length) return problemMessage(problem);
  return errors
    .map((e) => {
      const index = Number(e.pointer.split("/").pop());
      const item = Number.isInteger(index) ? submitted[index] : undefined;
      return item ? `${item.column}: ${e.message ?? "invalid role"}` : (e.message ?? "");
    })
    .filter(Boolean)
    .join(" ");
}
