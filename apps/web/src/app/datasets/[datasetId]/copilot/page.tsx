import { Suspense } from "react";

import { CopilotView } from "./copilot-view";

export default function DatasetCopilotPage() {
  return (
    <Suspense>
      <CopilotView />
    </Suspense>
  );
}
