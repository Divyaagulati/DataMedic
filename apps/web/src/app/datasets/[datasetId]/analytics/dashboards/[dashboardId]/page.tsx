import { Suspense } from "react";

import { DashboardView } from "./dashboard-view";

export default function DashboardPage() {
  return (
    <Suspense>
      <DashboardView />
    </Suspense>
  );
}
