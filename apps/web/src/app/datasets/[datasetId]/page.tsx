import { Suspense } from "react";

import { DatasetDetail } from "./dataset-detail";

export default function DatasetPage() {
  return (
    <Suspense>
      <DatasetDetail />
    </Suspense>
  );
}
