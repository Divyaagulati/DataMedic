import { Suspense } from "react";

import { BiomedicalView } from "./biomedical-view";

export default function DatasetBiomedicalPage() {
  return (
    <Suspense>
      <BiomedicalView />
    </Suspense>
  );
}
