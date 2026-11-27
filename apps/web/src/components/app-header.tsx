"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";

export function AppHeader() {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  return (
    <header className="sticky top-0 z-20 border-b bg-[var(--brand-navy)] text-white">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span
            aria-hidden
            className="grid h-7 w-7 place-items-center rounded-md bg-[var(--brand-teal)] text-sm font-bold text-[var(--brand-navy)]"
          >
            D
          </span>
          DataMedic
        </Link>
        <nav className="flex items-center gap-2 text-sm [&_a]:text-white/90 [&_button]:text-white/90">
          {loading ? null : user ? (
            <>
              <Button asChild variant="ghost" size="sm" className="hover:bg-white/10 hover:text-white">
                <Link href="/datasets">Datasets</Link>
              </Button>
              <Button asChild variant="ghost" size="sm" className="hover:bg-white/10 hover:text-white">
                <Link href="/settings/security">Security</Link>
              </Button>
              <span className="hidden text-white/60 sm:inline">{user.email}</span>
              <Button
                variant="outline"
                size="sm"
                className="border-white/30 bg-transparent hover:bg-white/10"
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
              <Button asChild variant="ghost" size="sm" className="hover:bg-white/10 hover:text-white">
                <Link href="/login">Sign in</Link>
              </Button>
              <Button asChild size="sm" className="bg-[var(--brand-teal)] text-[var(--brand-navy)] hover:bg-[var(--brand-teal)]/90">
                <Link href="/signup">Create account</Link>
              </Button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
