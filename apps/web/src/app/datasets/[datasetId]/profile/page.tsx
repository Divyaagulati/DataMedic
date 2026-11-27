import { Suspense } from "react";

import { ProfileDashboard } from "./profile-dashboard";

export default function DatasetProfilePage() {
  return (
    <Suspense>
      <ProfileDashboard />
    </Suspense>
  );
}
