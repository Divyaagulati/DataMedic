"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { type FormEvent, useState } from "react";

import { DatasetNav } from "@/components/dataset-nav";
import { FormError } from "@/components/form-error";
import { VersionPicker } from "@/components/mvp/version-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { problemMessage } from "@/lib/api/problem";
import { AnalysisResult } from "@/components/analytics/analysis-result";
import { askCopilot, COPILOT_SUGGESTIONS, type CopilotAnswer } from "@/lib/mvp";
import { useDataset } from "@/lib/queries";

type Turn = { question: string; reply: CopilotAnswer | null; error?: string };

export function CopilotView() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const dataset = useDataset(datasetId);
  const [picked, setPicked] = useState<string>();
  const versionId = picked ?? dataset.data?.head_version?.id;
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);

  if (dataset.error) return <FormError message={problemMessage(dataset.error)} />;
  if (!dataset.data) return <p className="text-sm text-muted-foreground">Loading…</p>;
  const ds = dataset.data;

  async function ask(q: string) {
    const text = q.trim();
    if (!text || !versionId || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [...t, { question: text, reply: null }]);
    try {
      const reply = await askCopilot(versionId, text);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, reply } : x)));
    } catch (e) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: String((e as Error).message) } : x)));
    }
    setBusy(false);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void ask(question);
  }

  const last = turns.at(-1)?.reply;
  const suggestions = last?.suggestions?.length ? last.suggestions : COPILOT_SUGGESTIONS;

  return (
    <div className="space-y-6">
      <DatasetNav datasetId={datasetId} name={ds.name} status={ds.status} active="copilot" />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <VersionPicker datasetId={datasetId} value={versionId} onChange={setPicked} />
        <Badge variant="outline" title="Answers are composed by fixed rules from this version's stored analyses. No language model or external service is used.">
          Built-in rule-based assistant · no external AI
        </Badge>
      </div>
      <Card className="flex min-h-[28rem] flex-col">
        <CardHeader>
          <CardTitle>DataMedic Copilot</CardTitle>
          <CardDescription>
            Ask analytical questions (“average amount by region”, “histogram of age”, “monthly trend”) or about data
            quality, diagnostics, recommendations, models, reliability and next steps. Every number comes from the query
            engine; the interpreted query is shown. It never changes data, commits steps or trains models.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col gap-4">
          <div className="flex-1 space-y-4" aria-live="polite">
            {turns.length === 0 ? (
              <p className="text-sm text-muted-foreground">Choose a suggested question or type your own.</p>
            ) : null}
            {turns.map((t, i) => (
              <div key={i} className="space-y-2">
                <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2 text-sm text-primary-foreground">
                  {t.question}
                </div>
                <div className="max-w-[90%] rounded-2xl rounded-bl-sm border bg-muted/40 px-4 py-3 text-sm">
                  {t.error ? (
                    <span className="text-destructive">{t.error}</span>
                  ) : !t.reply ? (
                    <span className="text-muted-foreground">Thinking…</span>
                  ) : (
                    <div className="space-y-1">
                      {t.reply.answer.map((line, j) => (
                        <p key={j} className={line.startsWith("•") || /^\d+\./.test(line) ? "pl-2" : undefined}>
                          {line}
                        </p>
                      ))}
                      {t.reply.analysis && versionId ? (
                        <AnalysisResult analysis={t.reply.analysis} datasetId={datasetId} versionId={versionId} onAsk={(q) => void ask(q)} />
                      ) : null}
                      {t.reply.references.length ? (
                        <div className="flex flex-wrap gap-2 pt-2">
                          {t.reply.references.map((r) => (
                            <Link
                              key={r.href + r.label}
                              href={r.href}
                              className="rounded-full border bg-background px-2 py-0.5 text-xs text-primary hover:underline"
                            >
                              ↗ {r.label}
                            </Link>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <Button key={s} size="sm" variant="outline" disabled={busy || !versionId} onClick={() => void ask(s)}>
                {s}
              </Button>
            ))}
          </div>
          <form onSubmit={submit} className="flex gap-2">
            <Input
              aria-label="Question"
              value={question}
              maxLength={500}
              placeholder="Ask about this dataset version…"
              onChange={(e) => setQuestion(e.target.value)}
            />
            <Button type="submit" disabled={busy || !question.trim() || !versionId}>
              Ask
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
