/** RFC 9457 problem details returned by the API for every error. */
export interface Problem {
  type?: string;
  title: string;
  status: number;
  code: string;
  detail?: string;
  request_id?: string | null;
  errors?: { pointer: string; code: string; message?: string }[];
}

export function isProblem(value: unknown): value is Problem {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Problem).code === "string" &&
    typeof (value as Problem).status === "number"
  );
}

const FRIENDLY: Record<string, string> = {
  AUTH_REQUIRED: "Please sign in to continue.",
  AUTH_INVALID: "Those details were not accepted.",
  CSRF_FAILED: "Your session expired. Please refresh the page and try again.",
  FORBIDDEN: "You are not allowed to do that.",
  NOT_FOUND: "We could not find that item.",
  CONFLICT: "That action conflicts with the current state. Refresh and try again.",
  QUOTA_EXCEEDED: "You have reached a usage limit.",
  UPLOAD_TOO_LARGE: "The file is larger than the upload limit.",
  UNSUPPORTED_FORMAT: "That file type is not supported.",
  RATE_LIMITED: "Too many attempts. Please wait and try again.",
  JOB_LIMIT_EXCEEDED: "Too many jobs are running. Wait for some to finish.",
  VALIDATION_FAILED: "Some fields need attention.",
  INTERNAL: "Something went wrong on our side.",
};

/** A message safe to show to users: the server's detail if present, else a friendly default. */
export function problemMessage(error: unknown): string {
  if (isProblem(error)) {
    const fieldMessages = (error.errors ?? [])
      .map((e) => e.message)
      .filter((m): m is string => Boolean(m));
    if (fieldMessages.length > 0) return fieldMessages.join(" ");
    return error.detail ?? FRIENDLY[error.code] ?? error.title;
  }
  return "Could not reach the server. Check your connection and try again.";
}
