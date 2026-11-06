"use client";

import Link from "next/link";
import { useState } from "react";

import { FormError, FormNotice } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";

export default function SignupPage() {
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setError(null);
    try {
      const { error: problem, response } = await api.POST("/api/v1/auth/signup", {
        body: {
          email: String(form.get("email")),
          password: String(form.get("password")),
          display_name: String(form.get("display_name") || "") || null,
        },
      });
      if (response.status === 202) setDone(true);
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
        <CardTitle>Create your account</CardTitle>
      </CardHeader>
      <CardContent>
        {done ? (
          <div className="space-y-4">
            <FormNotice message="Check your email for a verification link, then sign in." />
            <Button asChild className="w-full">
              <Link href="/login">Go to sign in</Link>
            </Button>
          </div>
        ) : (
          <form onSubmit={onSubmit} className="space-y-4">
            <FormError message={error} />
            <div className="space-y-2">
              <Label htmlFor="display_name">Name (optional)</Label>
              <Input id="display_name" name="display_name" maxLength={100} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" name="email" type="email" autoComplete="email" required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="new-password"
                minLength={12}
                required
              />
              <p className="text-xs text-muted-foreground">At least 12 characters.</p>
            </div>
            <Button type="submit" className="w-full" disabled={busy}>
              {busy ? "Creating…" : "Create account"}
            </Button>
          </form>
        )}
      </CardContent>
    </Card>
  );
}
