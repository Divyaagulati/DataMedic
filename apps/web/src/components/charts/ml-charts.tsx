"use client";

/**
 * Chart primitives for model evaluation, following the same rules as ``charts.tsx``:
 * validated --viz-* tokens, thin 2px lines, recessive axes and reference lines, a tooltip on
 * every mark, a legend for two or more series, and a table equivalent next to every chart.
 * All inputs are server-side aggregates (curves of at most 200 points, binned counts).
 */

import type { ReactNode } from "react";

import { useChartTooltip } from "@/components/charts/charts";
import { linePath } from "@/lib/training";

const SIZE = 220;
const CAT = ["--viz-cat-1", "--viz-cat-2", "--viz-cat-3", "--viz-cat-4", "--viz-cat-5", "--viz-cat-6", "--viz-cat-7"];

export type Series = { key: string; label: string; points: number[][] };

/** Curves in the unit square (ROC, precision-recall, reliability) with a reference line. */
export function UnitCurve({
  series,
  reference,
  xLabel,
  yLabel,
  ariaLabel,
}: {
  series: Series[];
  reference: "diagonal" | { y: number } | null;
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
}) {
  const { bind, node } = useChartTooltip();
  const shown = series.slice(0, CAT.length);
  return (
    <figure className="space-y-2">
      <svg
        viewBox={`-28 -6 ${SIZE + 36} ${SIZE + 34}`}
        className="w-full max-w-[18rem]"
        role="img"
        aria-label={ariaLabel}
      >
        <rect x={0} y={0} width={SIZE} height={SIZE} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={t}>
            <line x1={t * SIZE} x2={t * SIZE} y1={0} y2={SIZE} stroke="var(--viz-grid)" />
            <line x1={0} x2={SIZE} y1={(1 - t) * SIZE} y2={(1 - t) * SIZE} stroke="var(--viz-grid)" />
          </g>
        ))}
        {reference === "diagonal" ? (
          <line x1={0} y1={SIZE} x2={SIZE} y2={0} stroke="var(--viz-axis)" strokeDasharray="4 4" />
        ) : reference ? (
          <line
            x1={0}
            x2={SIZE}
            y1={(1 - reference.y) * SIZE}
            y2={(1 - reference.y) * SIZE}
            stroke="var(--viz-axis)"
            strokeDasharray="4 4"
          />
        ) : null}
        {shown.map((s, i) => (
          <g key={s.key}>
            <path d={linePath(s.points, SIZE)} fill="none" stroke={`var(${CAT[i]})`} strokeWidth={2} />
            {s.points.length <= 12
              ? s.points.map(([x, y], j) => (
                  <circle
                    key={j}
                    cx={x * SIZE}
                    cy={(1 - y) * SIZE}
                    r={4}
                    fill={`var(${CAT[i]})`}
                    stroke="var(--viz-surface)"
                    strokeWidth={2}
                    {...bind(
                      <>
                        <div className="font-medium">{s.label}</div>
                        <div>
                          {xLabel} {x.toFixed(3)} · {yLabel} {y.toFixed(3)}
                        </div>
                      </>,
                    )}
                  />
                ))
              : null}
          </g>
        ))}
        <text x={SIZE / 2} y={SIZE + 24} textAnchor="middle" className="fill-muted-foreground text-[10px]">
          {xLabel}
        </text>
        <text
          x={-18}
          y={SIZE / 2}
          textAnchor="middle"
          transform={`rotate(-90 -18 ${SIZE / 2})`}
          className="fill-muted-foreground text-[10px]"
        >
          {yLabel}
        </text>
        {[0, 1].map((t) => (
          <text key={t} x={t * SIZE} y={SIZE + 12} textAnchor="middle" className="fill-muted-foreground text-[9px]">
            {t}
          </text>
        ))}
      </svg>
      {shown.length > 1 ? (
        <figcaption className="flex flex-wrap gap-3 text-xs">
          {shown.map((s, i) => (
            <span key={s.key} className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4" style={{ background: `var(${CAT[i]})` }} />
              {s.label}
            </span>
          ))}
        </figcaption>
      ) : null}
      {node}
    </figure>
  );
}

/** Count grid (confusion matrix, binned scatter): one sequential hue, 2px gaps, values in
 *  tooltips; ``showValues`` prints counts in small grids. Rows are drawn top to bottom. */
export function HeatGrid({
  rows,
  rowLabels,
  colLabels,
  ariaLabel,
  cellTip,
  showValues = false,
  cell = 28,
}: {
  rows: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  ariaLabel: string;
  cellTip: (r: number, c: number, value: number) => ReactNode;
  showValues?: boolean;
  cell?: number;
}) {
  const { bind, node } = useChartTooltip();
  const max = Math.max(1, ...rows.flat());
  const nCols = rows[0]?.length ?? 0;
  return (
    <div className="overflow-x-auto">
      <div
        role="img"
        aria-label={ariaLabel}
        className="inline-grid gap-[2px]"
        style={{ gridTemplateColumns: `${rowLabels ? "minmax(3rem, 8rem) " : ""}repeat(${nCols}, ${cell}px)` }}
      >
        {colLabels ? (
          <>
            {rowLabels ? <span /> : null}
            {colLabels.map((l, j) => (
              <span key={j} className="truncate text-center text-[10px] text-muted-foreground" title={l}>
                {l}
              </span>
            ))}
          </>
        ) : null}
        {rows.map((row, i) => (
          <div key={i} className="contents">
            {rowLabels ? (
              <span className="truncate pr-2 text-right text-[10px] text-muted-foreground" title={rowLabels[i]} style={{ lineHeight: `${cell}px` }}>
                {rowLabels[i]}
              </span>
            ) : null}
            {row.map((v, j) => (
              <span
                key={j}
                className="flex items-center justify-center rounded-[2px] text-[10px] tabular-nums"
                style={{
                  width: cell,
                  height: cell,
                  background:
                    v === 0
                      ? "var(--viz-surface)"
                      : `color-mix(in oklab, var(--viz-seq) ${Math.round(15 + (85 * v) / max)}%, var(--viz-surface))`,
                  color: v / max > 0.6 ? "white" : undefined,
                }}
                {...bind(cellTip(i, j, v), false)}
              >
                {showValues ? v : null}
              </span>
            ))}
          </div>
        ))}
      </div>
      {node}
    </div>
  );
}

/** Per-fold values of one metric for several models: one row per model, a dot per fold. */
export function FoldDots({
  rows,
  ariaLabel,
}: {
  rows: { key: string; label: string; values: (number | null)[] }[];
  ariaLabel: string;
}) {
  const { bind, node } = useChartTooltip();
  const present = rows.flatMap((r) => r.values.filter((v): v is number => v !== null));
  if (!present.length) return null;
  let lo = Math.min(...present);
  let hi = Math.max(...present);
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  const pos = (v: number) => `${4 + (92 * (v - lo)) / (hi - lo)}%`;
  return (
    <div role="list" aria-label={ariaLabel} className="space-y-1">
      {rows.map((r) => (
        <div key={r.key} role="listitem" className="grid items-center gap-3" style={{ gridTemplateColumns: "9rem 1fr" }}>
          <span className="truncate text-xs">{r.label}</span>
          <span className="relative h-5 rounded bg-muted/40">
            {r.values.map((v, i) =>
              v === null ? null : (
                <span
                  key={i}
                  className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-[var(--viz-surface)]"
                  style={{ left: pos(v), background: "var(--viz-seq)" }}
                  {...bind(`${r.label}, fold ${i + 1}: ${v.toFixed(4)}`)}
                />
              ),
            )}
          </span>
        </div>
      ))}
      <div className="grid gap-3 text-[10px] text-muted-foreground" style={{ gridTemplateColumns: "9rem 1fr" }}>
        <span />
        <span className="flex justify-between">
          <span>{lo.toFixed(3)}</span>
          <span>{hi.toFixed(3)}</span>
        </span>
      </div>
      {node}
    </div>
  );
}
