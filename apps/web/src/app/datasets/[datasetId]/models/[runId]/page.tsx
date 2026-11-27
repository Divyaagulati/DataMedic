import { Suspense } from "react";

import { RunView } from "./run-view";

export default function TrainingRunPage() {
  return (
    <Suspense>
      <RunView />
    </Suspense>
  );
}
