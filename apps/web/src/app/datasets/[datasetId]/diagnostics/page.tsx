import { Suspense } from "react";

import { DiagnosticsView } from "./diagnostics-view";

export default function DatasetDiagnosticsPage() {
  return (
    <Suspense>
      <DiagnosticsView />
    </Suspense>
  );
}
