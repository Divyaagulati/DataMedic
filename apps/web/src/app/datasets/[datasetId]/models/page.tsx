import { Suspense } from "react";

import { ModelsView } from "./models-view";

export default function DatasetModelsPage() {
  return (
    <Suspense>
      <ModelsView />
    </Suspense>
  );
}
