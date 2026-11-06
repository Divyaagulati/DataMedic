import { NextResponse, type NextRequest } from "next/server";

// Optimistic check only (UX): send visitors without a session cookie to the login page.
// The API is the sole authority — every API call is authenticated and authorised server-side.
const SESSION_COOKIES = ["__Host-dm_session", "dm_session"];

export function proxy(request: NextRequest) {
  const hasSession = SESSION_COOKIES.some((name) => request.cookies.has(name));
  if (!hasSession) {
    const url = new URL("/login", request.url);
    url.searchParams.set("next", request.nextUrl.pathname);
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/datasets/:path*", "/settings/:path*"],
};
