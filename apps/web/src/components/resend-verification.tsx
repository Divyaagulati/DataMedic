"use client";

import { useState } from "react";

import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";

/** Unverified users can request a new verification link (the server allows one a minute). */
export function ResendVerification() {
  const [state, setState] = useState<"idle" | "sending" | "sent">("idle");
  const [error, setError] = useState<string | null>(null);

  async function resend() {
    setState("sending");
    setError(null);
    const { error: problem } = await api.POST("/api/v1/auth/verify-email/resend");
    if (problem) {
      setError(problemMessage(problem));
      setState("idle");
      return;
    }
    setState("sent");
  }

  return (
    <div className="space-y-2">
      <FormNotice message="Verify your email address (check your inbox and spam or junk folder) before uploading data." />
      <Button variant="outline" size="sm" onClick={() => void resend()} disabled={state !== "idle"}>
        {state === "sending" ? "Sending…" : state === "sent" ? "Verification email sent" : "Resend verification email"}
      </Button>
      {state === "sent" ? (
        <p className="text-xs text-muted-foreground">
          A new link was sent; it is valid for 24 hours. It can take a few minutes to arrive.
        </p>
      ) : null}
      <FormError message={error} />
    </div>
  );
}
