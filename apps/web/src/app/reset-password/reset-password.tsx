"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";

export function ResetPassword() {
  const token = useSearchParams().get("token");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onRequest(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const email = String(new FormData(event.currentTarget).get("email"));
    setBusy(true);
    setError(null);
    try {
      await api.POST("/api/v1/auth/password-reset/request", { body: { email } });
      setNotice("If an account exists for that address, a reset link has been sent.");
    } catch (e) {
      setError(problemMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function onConfirm(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const newPassword = String(new FormData(event.currentTarget).get("password"));
    setBusy(true);
    setError(null);
    try {
      const { error: problem, response } = await api.POST(
        "/api/v1/auth/password-reset/confirm",
        { body: { token: token ?? "", new_password: newPassword } },
      );
      if (response.status === 204) setNotice("Password updated. All sessions were signed out.");
      else setError(problemMessage(problem));
    } catch (e) {
      setError(problemMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="mx-auto max-w-md">
      <CardHeader>
        <CardTitle>{token ? "Choose a new password" : "Reset your password"}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <FormError message={error} />
        <FormNotice message={notice} />
        {notice ? (
          <Button asChild className="w-full">
            <Link href="/login">Go to sign in</Link>
          </Button>
        ) : token ? (
          <form onSubmit={onConfirm} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="password">New password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="new-password"
                minLength={12}
                required
              />
            </div>
            <Button type="submit" className="w-full" disabled={busy}>
              Update password
            </Button>
          </form>
        ) : (
          <form onSubmit={onRequest} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" name="email" type="email" autoComplete="email" required />
            </div>
            <Button type="submit" className="w-full" disabled={busy}>
              Send reset link
            </Button>
          </form>
        )}
      </CardContent>
    </Card>
  );
}
