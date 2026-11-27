import { Suspense } from "react";

import { LabView } from "./lab-view";

export default function DatasetLabPage() {
  return (
    <Suspense>
      <LabView />
    </Suspense>
  );
}
