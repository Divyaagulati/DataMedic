import { Suspense } from "react";

import { PassportView } from "./passport-view";

export default function DatasetPassportPage() {
  return (
    <Suspense>
      <PassportView />
    </Suspense>
  );
}
