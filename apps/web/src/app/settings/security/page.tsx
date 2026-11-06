"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useAuth } from "@/components/auth-provider";
import { FormError } from "@/components/form-error";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, unwrap } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";

export default function SecurityPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [revokeError, setRevokeError] = useState<string | null>(null);
  const sessions = useQuery({
    queryKey: ["auth", "sessions"],
    queryFn: async () => unwrap(await api.GET("/api/v1/auth/sessions")).items,
    enabled: Boolean(user),
  });

  async function revoke(id: string) {
    const { error: problem, response } = await api.DELETE("/api/v1/auth/sessions/{session_id}", {
      params: { path: { session_id: id } },
    });
    if (response.status === 204) {
      setRevokeError(null);
      await queryClient.invalidateQueries({ queryKey: ["auth", "sessions"] });
    } else {
      setRevokeError(problemMessage(problem));
    }
  }

  const error = revokeError ?? (sessions.error ? problemMessage(sessions.error) : null);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Security</h1>
      <FormError message={error} />
      <Card>
        <CardHeader>
          <CardTitle>Active sessions</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="divide-y text-sm">
            {(sessions.data ?? []).map((s) => (
              <li key={s.id} className="flex items-center justify-between gap-4 py-3">
                <div>
                  <div className="font-medium">{s.user_agent ?? "Unknown device"}</div>
                  <div className="text-muted-foreground">
                    Last active {new Date(s.last_seen_at).toLocaleString()}
                  </div>
                </div>
                {s.current ? (
                  <Badge variant="secondary">This device</Badge>
                ) : (
                  <Button variant="outline" size="sm" onClick={() => revoke(s.id)}>
                    Sign out
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
