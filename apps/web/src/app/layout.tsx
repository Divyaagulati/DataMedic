import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { AppHeader } from "@/components/app-header";
import { AuthProvider } from "@/components/auth-provider";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DataMedic — privacy-first data intelligence and ML reliability",
  description:
    "Privacy-first, domain-aware data intelligence and ML reliability platform: diagnose, experiment, transform, train, audit and export.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <AuthProvider>
          <AppHeader />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
          <footer className="border-t bg-card/60">
            <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-4 text-xs text-muted-foreground sm:flex-row sm:justify-between">
              <span>DataMedic MVP · privacy-first data intelligence and ML reliability</span>
              <span>Research and demonstration use only — not a medical device. Use synthetic or non-sensitive data.</span>
            </div>
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
