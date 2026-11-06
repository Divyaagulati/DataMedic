"use client";

import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

import { api, setCsrfToken, type User } from "@/lib/api/client";

interface AuthState {
  user: User | null;
  loading: boolean;
  refresh: () => Promise<void>;
  setSession: (user: User, csrfToken: string) => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);
const ME_KEY = ["auth", "me"] as const;

async function fetchMe(): Promise<User | null> {
  const { data } = await api.GET("/api/v1/auth/me");
  setCsrfToken(data?.csrf_token ?? null);
  return data?.user ?? null;
}

function AuthStateProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const me = useQuery({ queryKey: ME_KEY, queryFn: fetchMe, staleTime: 60_000 });

  const refresh = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey: ME_KEY });
  }, [queryClient]);

  const setSession = useCallback(
    (user: User, token: string) => {
      setCsrfToken(token);
      queryClient.clear(); // never show a previous user's cached data
      queryClient.setQueryData(ME_KEY, user);
    },
    [queryClient],
  );

  const logout = useCallback(async () => {
    await api.POST("/api/v1/auth/logout").catch(() => undefined);
    setCsrfToken(null);
    queryClient.clear(); // drop every cached response belonging to this user
    queryClient.setQueryData(ME_KEY, null);
  }, [queryClient]);

  const value = useMemo(
    () => ({ user: me.data ?? null, loading: me.isPending, refresh, setSession, logout }),
    [me.data, me.isPending, refresh, setSession, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <AuthStateProvider>{children}</AuthStateProvider>
    </QueryClientProvider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
