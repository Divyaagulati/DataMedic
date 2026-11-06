import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <section className="space-y-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">DataMedic</h1>
      <p className="max-w-2xl text-muted-foreground">
        Privacy-first, domain-aware data diagnostics and ML reliability. Your datasets stay
        private to your account and are never sent to external AI services.
      </p>
      <div className="flex gap-3">
        <Button asChild>
          <Link href="/signup">Create account</Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/login">Sign in</Link>
        </Button>
      </div>
    </section>
  );
}
