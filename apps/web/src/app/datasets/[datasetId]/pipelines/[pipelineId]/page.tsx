import { Suspense } from "react";

import { PipelineView } from "./pipeline-view";

export default function PipelinePage() {
  return (
    <Suspense>
      <PipelineView />
    </Suspense>
  );
}
