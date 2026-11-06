export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

const STATUS_LABELS: Record<string, string> = {
  upload_pending: "Waiting for upload",
  validating: "Validating",
  validated: "Validated",
  parsing: "Parsing",
  parse_failed: "Parse failed",
  ready: "Ready",
  rejected: "Rejected",
  deleting: "Deleting",
};

export function datasetStatusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status;
}

export const ACTIVE_DATASET_STATUSES = new Set(["validating", "parsing", "deleting"]);
