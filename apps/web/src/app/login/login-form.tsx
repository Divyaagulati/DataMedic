"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { useAuth } from "@/components/auth-provider";
import { FormError } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { safeNext } from "@/lib/navigation";

export function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { setSession } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setError(null);
    try {
      const { data, error: problem } = await api.POST("/api/v1/auth/login", {
        body: { email: String(form.get("email")), password: String(form.get("password")) },
      });
      if (data) {
        setSession(data.user, data.csrf_token);
        router.push(safeNext(params.get("next")));
      } else {
        setError(problemMessage(problem));
      }
    } catch (e) {
      setError(problemMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="mx-auto mt-6 max-w-md border-t-4 border-t-[var(--brand-teal)] shadow-lg">
      <CardHeader>
        <p className="text-xs font-semibold uppercase tracking-widest text-primary">DataMedic</p>
        <CardTitle className="text-xl">Sign in</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <FormError message={error} />
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
              autoComplete="current-password"
              required
            />
          </div>
          <Button type="submit" className="w-full" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </Button>
          <div className="flex justify-between text-sm text-muted-foreground">
            <Link href="/signup" className="underline">
              Create account
            </Link>
            <Link href="/reset-password" className="underline">
              Forgot password?
            </Link>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
