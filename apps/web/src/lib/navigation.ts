/** Only allow same-site relative paths after login (prevents open redirects). */
export function safeNext(next: string | null): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) {
    return "/datasets";
  }
  return next;
}
