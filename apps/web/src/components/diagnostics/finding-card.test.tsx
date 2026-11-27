import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { FindingView } from "@/lib/diagnostics";

import { FindingCard, SeverityBadge } from "./finding-card";

afterEach(cleanup);

const base: FindingView = {
  id: "f1",
  check_id: "universal.duplicates.exact_rows",
  check_version: "1.0.0",
  domain: "universal",
  category: "duplicates",
  status: "detected",
  severity: "medium",
  title: "12 rows duplicate an earlier row",
  explanation: "12 rows are exact copies.",
  evidence: { duplicate_rows: 12, share_pct: 2.5, pairs: [{ a: "x", b: "y", phi: 0.91 }] },
  columns: ["x", "y"],
  affected_rows: 12,
  affected_pct: 2.5,
  recommendation: "Confirm whether these are true repeats.",
  limitations: ["Exact matching only."],
  not_assessed_reason: null,
  error_code: null,
  sampled: false,
  sample_info: null,
};

describe("FindingCard", () => {
  it("is collapsed by default and expands to show details", () => {
    render(
      <ul>
        <FindingCard finding={base} />
      </ul>,
    );
    const toggle = screen.getByRole("button", { name: /12 rows duplicate an earlier row/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("Suggested action")).toBeNull();
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Suggested action")).toBeTruthy();
    expect(screen.getByText("Confirm whether these are true repeats.")).toBeTruthy();
    expect(screen.getByText("Exact matching only.")).toBeTruthy();
    expect(screen.getByText("Duplicate rows")).toBeTruthy(); // evidence key made readable
    expect(screen.getByText("0.91")).toBeTruthy(); // nested evidence rendered
    expect(screen.getByText(/12 rows \(2.5%\)/)).toBeTruthy();
  });

  it("shows the reason for a not-assessed result and no severity badge", () => {
    render(
      <ul>
        <FindingCard
          finding={{
            ...base,
            status: "not_assessed",
            severity: "info",
            title: "Class imbalance",
            explanation: "Not assessed: no column has the role 'Target'.",
            evidence: {},
            recommendation: null,
            affected_rows: null,
            affected_pct: null,
            not_assessed_reason: "no column has the role 'Target'.",
          }}
        />
      </ul>,
    );
    expect(screen.queryByText("Medium")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Class imbalance/ }));
    expect(screen.getByText(/no column has the role 'Target'/)).toBeTruthy();
    expect(screen.queryByText("Evidence")).toBeNull();
  });
});

describe("SeverityBadge", () => {
  it("always carries a text label, never colour alone", () => {
    render(<SeverityBadge severity="critical" />);
    expect(screen.getByText("Critical")).toBeTruthy();
  });
});

describe("evidence rendering", () => {
  it("prettifies the check's own keys but shows data keys verbatim", () => {
    render(
      <ul>
        <FindingCard
          finding={{ ...base, evidence: { minority_share_pct: 3, classes: { no_show: 30, attended_ok: 970 } } }}
        />
      </ul>,
    );
    fireEvent.click(screen.getByRole("button", { name: /12 rows duplicate/ }));
    expect(screen.getByText("Minority share %")).toBeTruthy();
    expect(screen.getByText("no_show")).toBeTruthy(); // a class name, not "No show"
    expect(screen.getByText("attended_ok")).toBeTruthy();
  });

  it("never rounds a small affected share down to 0%", () => {
    render(
      <ul>
        <FindingCard finding={{ ...base, affected_rows: 1, affected_pct: 0.0167 }} />
      </ul>,
    );
    expect(screen.getByText(/1 rows \(<0.1%\)/)).toBeTruthy();
  });
});
