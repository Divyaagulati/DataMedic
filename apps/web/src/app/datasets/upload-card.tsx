"use client";

import { useState } from "react";

import { FormError } from "@/components/form-error";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";
import { problemMessage } from "@/lib/api/problem";
import { uploadContent } from "@/lib/api/upload";
import { formatBytes } from "@/lib/format";

export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;
export const ACCEPTED = ".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json,.jsonl";

export function UploadCard({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`The file is larger than the ${formatBytes(MAX_UPLOAD_BYTES)} limit.`);
      return;
    }
    setError(null);
    setProgress(0);
    try {
      const { data, error: problem } = await api.POST("/api/v1/datasets", {
        body: { name: name.trim() || file.name, filename: file.name, size_bytes: file.size },
      });
      if (!data) {
        setError(problemMessage(problem));
        setProgress(null);
        return;
      }
      await uploadContent(data.dataset.id, file, setProgress);
      setFile(null);
      setName("");
      onUploaded();
    } catch (e) {
      setError(problemMessage(e));
    } finally {
      setProgress(null);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload a dataset</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <FormError message={error} />
          <div className="space-y-2">
            <Label htmlFor="file">File (CSV, TSV, XLSX, XLS, Parquet, JSON — max 200 MB)</Label>
            <Input
              id="file"
              type="file"
              accept={ACCEPTED}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="name">Dataset name</Label>
            <Input
              id="name"
              value={name}
              maxLength={200}
              placeholder={file?.name ?? "e.g. ICU cohort 2024"}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-4">
            <Button type="submit" disabled={!file || progress !== null}>
              {progress === null ? "Upload" : `Uploading ${Math.round(progress * 100)}%`}
            </Button>
            {file ? (
              <span className="text-sm text-muted-foreground">{formatBytes(file.size)}</span>
            ) : null}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
