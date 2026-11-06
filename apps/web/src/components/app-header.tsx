"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";

export function AppHeader() {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  return (
    <header className="border-b">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link href="/" className="font-semibold tracking-tight">
          DataMedic
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          {loading ? null : user ? (
            <>
              <Button asChild variant="ghost" size="sm">
                <Link href="/datasets">Datasets</Link>
              </Button>
              <Button asChild variant="ghost" size="sm">
                <Link href="/settings/security">Security</Link>
              </Button>
              <span className="hidden text-muted-foreground sm:inline">{user.email}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  await logout();
                  router.push("/login");
                }}
              >
                Sign out
              </Button>
            </>
          ) : (
            <>
              <Button asChild variant="ghost" size="sm">
                <Link href="/login">Sign in</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/signup">Create account</Link>
              </Button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
