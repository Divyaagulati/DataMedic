"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { useAuth } from "@/components/auth-provider";
import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";

export function VerifyEmail() {
  const token = useSearchParams().get("token");
  const { refresh } = useAuth();
  const [state, setState] = useState<"idle" | "busy" | "done">("idle");
  const [error, setError] = useState<string | null>(null);

  // Verification is an explicit click (not automatic on page load), so link scanners in
  // email clients cannot consume the single-use token.
  async function verify() {
    if (!token) return;
    setState("busy");
    setError(null);
    try {
      const { error: problem, response } = await api.POST("/api/v1/auth/verify-email", {
        body: { token },
      });
      if (response.status === 204) {
        setState("done");
        await refresh();
      } else {
        setError(problemMessage(problem));
        setState("idle");
      }
    } catch (e) {
      setError(problemMessage(e));
      setState("idle");
    }
  }

  return (
    <Card className="mx-auto max-w-md">
      <CardHeader>
        <CardTitle>Verify your email</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!token ? (
          <FormError message="This verification link is incomplete." />
        ) : state === "done" ? (
          <>
            <FormNotice message="Your email address is verified." />
            <Button asChild className="w-full">
              <Link href="/datasets">Continue</Link>
            </Button>
          </>
        ) : (
          <>
            <FormError message={error} />
            <Button className="w-full" onClick={verify} disabled={state === "busy"}>
              {state === "busy" ? "Verifying…" : "Verify email address"}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}
