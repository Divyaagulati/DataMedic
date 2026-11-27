"use client";

/**
 * Small, dependency-free chart primitives for the profiling dashboard.
 *
 * Rules (validated reference palette, see globals.css --viz-*): thin marks (<= 16px bars),
 * 4px rounded data-ends square at the baseline, 2px surface gaps, recessive axes, text in
 * text tokens (never the series color), a hover/focus tooltip on every mark, and a table
 * or list equivalent next to every chart.
 */

import { useState, type ReactNode } from "react";

import {
  binLabel,
  correlationColor,
  formatNumber,
  typeColor,
  TYPE_LABELS,
  TYPE_ORDER,
  type SemanticType,
} from "@/lib/profile";

type Tip = { x: number; y: number; content: ReactNode } | null;

export function useChartTooltip() {
  const [tip, setTip] = useState<Tip>(null);
  function bind(content: ReactNode, focusable = true) {
    return {
      onMouseMove: (e: React.MouseEvent) => setTip({ x: e.clientX, y: e.clientY, content }),
      onMouseLeave: () => setTip(null),
      ...(focusable
        ? {
            tabIndex: 0,
            onFocus: (e: React.FocusEvent<HTMLElement | SVGElement>) => {
              const r = e.currentTarget.getBoundingClientRect();
              setTip({ x: r.left + r.width / 2, y: r.top, content });
            },
            onBlur: () => setTip(null),
          }
        : {}),
    };
  }
  const node = tip ? (
    <div
      role="tooltip"
      className="pointer-events-none fixed z-50 max-w-xs rounded-md border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md"
      style={{ left: tip.x + 12, top: tip.y + 12 }}
    >
      {tip.content}
    </div>
  ) : null;
  return { bind, node };
}

export function StatTile({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-lg border bg-card px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {detail ? <div className="mt-0.5 text-xs text-muted-foreground">{detail}</div> : null}
    </div>
  );
}

export type BarItem = { key: string; label: string; value: number; display: string; tooltip?: ReactNode };

/** Horizontal bars, one hue (magnitude). Value printed at the bar tip. */
export function BarList({
  items,
  max,
  ariaLabel,
  labelWidth = "10rem",
}: {
  items: BarItem[];
  max?: number;
  ariaLabel: string;
  labelWidth?: string;
}) {
  const { bind, node } = useChartTooltip();
  const top = max ?? Math.max(1, ...items.map((i) => i.value));
  return (
    <div role="list" aria-label={ariaLabel} className="space-y-1">
      {items.map((item) => (
        <div
          key={item.key}
          role="listitem"
          aria-label={`${item.label}: ${item.display}`}
          className="grid items-center gap-3 rounded outline-none focus-visible:ring-2 focus-visible:ring-ring"
          style={{ gridTemplateColumns: `${labelWidth} 1fr` }}
          {...bind(item.tooltip ?? `${item.label}: ${item.display}`)}
        >
          <span className="truncate text-xs" title={item.label}>
            {item.label}
          </span>
          <span className="flex items-center gap-2">
            <span
              className="h-3.5 rounded-r-[4px]"
              style={{
                width: `${Math.max(item.value > 0 ? 0.5 : 0, (item.value / top) * 85)}%`,
                background: "var(--viz-seq)",
              }}
            />
            <span className="text-xs tabular-nums text-muted-foreground">{item.display}</span>
          </span>
        </div>
      ))}
      {node}
    </div>
  );
}

/** Columns for a histogram (numeric or datetime bins), one hue, 2px gaps. */
export function Histogram({
  kind,
  edges,
  counts,
  ariaLabel,
}: {
  kind: "numeric" | "datetime";
  edges: (number | string | null)[];
  counts: number[];
  ariaLabel: string;
}) {
  const { bind, node } = useChartTooltip();
  const top = Math.max(1, ...counts);
  const total = counts.reduce((a, b) => a + b, 0);
  const first = edges[0] ?? null;
  const last = edges[edges.length - 1] ?? null;
  const fmt = (v: number | string | null) =>
    kind === "datetime" ? (typeof v === "string" ? v.slice(0, 10) : "—") : formatNumber(v as number | null);
  return (
    <figure aria-label={ariaLabel} className="space-y-1">
      <div className="flex h-32 items-end gap-[2px] border-b" style={{ borderColor: "var(--viz-axis)" }}>
        {counts.map((count, i) => {
          const range = binLabel(kind, edges[i] ?? null, edges[i + 1] ?? null);
          return (
            <div
              key={i}
              className="flex h-full flex-1 items-end outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={`${range}: ${count} rows`}
              {...bind(
                <>
                  <div className="font-medium">{range}</div>
                  <div>
                    {formatNumber(count)} rows ({total ? ((100 * count) / total).toFixed(1) : 0}%)
                  </div>
                </>,
                false,
              )}
            >
              <div
                className="w-full rounded-t-[4px]"
                style={{ height: `${(count / top) * 100}%`, background: "var(--viz-seq)", minHeight: count ? 1 : 0 }}
              />
            </div>
          );
        })}
      </div>
      <figcaption className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
        <span>{fmt(first)}</span>
        <span>
          {counts.length} equal-width bins · tallest {formatNumber(top)} rows
        </span>
        <span>{fmt(last)}</span>
      </figcaption>
      {node}
    </figure>
  );
}

/** Thin segmented bar (parts of a whole) with a legend that carries names and counts. */
export function TypeMixBar({ counts }: { counts: Record<string, number> }) {
  const { bind, node } = useChartTooltip();
  const parts = TYPE_ORDER.map((type) => ({ type, n: counts[type] ?? 0 })).filter((p) => p.n > 0);
  const total = parts.reduce((a, p) => a + p.n, 0) || 1;
  return (
    <div className="space-y-3">
      <div className="flex h-2 gap-[2px]" role="img" aria-label="Column types">
        {parts.map((p) => (
          <div
            key={p.type}
            className="h-full rounded-full"
            style={{ width: `${(100 * p.n) / total}%`, background: typeColor(p.type) }}
            {...bind(`${TYPE_LABELS[p.type]}: ${p.n} column${p.n === 1 ? "" : "s"}`, false)}
          />
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {parts.map((p) => (
          <li key={p.type} className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full" style={{ background: typeColor(p.type) }} />
            {TYPE_LABELS[p.type]} <span className="text-muted-foreground">{p.n}</span>
          </li>
        ))}
      </ul>
      {node}
    </div>
  );
}

export function TypeBadge({ type }: { type: SemanticType }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs">
      <span className="size-2 rounded-full" style={{ background: typeColor(type) }} />
      {TYPE_LABELS[type]}
    </span>
  );
}

/** Correlation matrix as a diverging heatmap (-1 blue, 0 grey, +1 red). */
export function CorrelationHeatmap({
  columns,
  matrix,
  method,
}: {
  columns: string[];
  matrix: (number | null)[][];
  method: string;
}) {
  const { bind, node } = useChartTooltip();
  const cell = columns.length > 25 ? 12 : columns.length > 12 ? 18 : 28;
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-[2px]"
          style={{ gridTemplateColumns: `minmax(6rem, 10rem) repeat(${columns.length}, ${cell}px)` }}
          role="img"
          aria-label={`${method} correlation matrix of ${columns.length} columns; see the table of strongest pairs`}
        >
          {/* Column labels, read bottom-to-top, in the same order as the rows. */}
          <span />
          {columns.map((colName) => (
            <span
              key={`h-${colName}`}
              data-testid="heatmap-col-label"
              className="max-h-28 self-end truncate text-[11px] leading-none [writing-mode:vertical-rl] rotate-180"
              style={{ width: cell }}
              title={colName}
            >
              {colName}
            </span>
          ))}
          {columns.map((rowName, i) => (
            <Row key={rowName}>
              <span className="truncate pr-2 text-right text-[11px] leading-none" title={rowName} style={{ lineHeight: `${cell}px` }}>
                {rowName}
              </span>
              {columns.map((colName, j) => {
                const r = matrix[i]?.[j] ?? null;
                return (
                  <span
                    key={colName}
                    className="rounded-[2px]"
                    style={{ width: cell, height: cell, background: correlationColor(r) }}
                    {...bind(
                      <>
                        <div className="font-medium">
                          {rowName} × {colName}
                        </div>
                        <div>
                          {method} r = {r === null ? "not enough paired values" : r.toFixed(3)}
                        </div>
                      </>,
                      false,
                    )}
                  />
                );
              })}
            </Row>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
        <span>−1</span>
        <span
          className="h-2 w-40 rounded-full"
          style={{
            background:
              "linear-gradient(to right, var(--viz-div-neg), var(--viz-div-mid), var(--viz-div-pos))",
          }}
        />
        <span>+1</span>
        <span className="ml-2">blue: move in opposite directions · red: move together</span>
      </div>
      {node}
    </div>
  );
}

function Row({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
