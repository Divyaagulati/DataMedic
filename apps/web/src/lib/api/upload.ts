import { API_BASE_URL, getCsrfToken, type Schemas } from "./client";
import { isProblem, type Problem } from "./problem";

export type UploadAccepted = Schemas["UploadAcceptedOut"];

/**
 * Streams a file to PUT /datasets/{id}/content. Uses XMLHttpRequest (not fetch) to report
 * upload progress. The browser sets Content-Length from the File automatically.
 */
export function uploadContent(
  datasetId: string,
  file: File,
  onProgress: (fraction: number) => void,
): Promise<UploadAccepted> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `${API_BASE_URL}/api/v1/datasets/${encodeURIComponent(datasetId)}/content`);
    xhr.withCredentials = true;
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    const csrf = getCsrfToken();
    if (csrf) xhr.setRequestHeader("X-CSRF-Token", csrf);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onerror = () => reject(new Error("network"));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status === 202) resolve(body as UploadAccepted);
      else reject(isProblem(body) ? (body as Problem) : new Error(`HTTP ${xhr.status}`));
    };
    xhr.send(file);
  });
}
