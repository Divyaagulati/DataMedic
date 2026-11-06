import { Badge } from "@/components/ui/badge";
import { datasetStatusLabel } from "@/lib/format";

const VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  validated: "default",
  ready: "default",
  succeeded: "default",
  rejected: "destructive",
  failed: "destructive",
  parse_failed: "destructive",
  cancelled: "outline",
};

export function StatusBadge({ status }: { status: string }) {
  return <Badge variant={VARIANT[status] ?? "secondary"}>{datasetStatusLabel(status)}</Badge>;
}
