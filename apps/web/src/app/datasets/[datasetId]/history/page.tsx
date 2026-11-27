import { Suspense } from "react";

import { HistoryView } from "./history-view";

export default function DatasetHistoryPage() {
  return (
    <Suspense>
      <HistoryView />
    </Suspense>
  );
}
