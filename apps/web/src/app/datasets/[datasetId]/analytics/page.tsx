import { Suspense } from "react";

import { StudioView } from "./studio-view";

export default function AnalyticsStudioPage() {
  return (
    <Suspense>
      <StudioView />
    </Suspense>
  );
}
