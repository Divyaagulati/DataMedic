import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { BarList, CorrelationHeatmap, Histogram, TypeMixBar } from "./charts";

afterEach(cleanup);

describe("BarList", () => {
  it("labels every bar and shows a tooltip on keyboard focus", () => {
    render(
      <BarList
        ariaLabel="Class distribution"
        items={[
          { key: "no", label: "no", value: 7, display: "7 · 70%" },
          { key: "yes", label: "yes", value: 3, display: "3 · 30%" },
        ]}
      />,
    );
    const items = screen.getAllByRole("listitem");
    expect(items.map((i) => i.getAttribute("aria-label"))).toEqual(["no: 7 · 70%", "yes: 3 · 30%"]);
    fireEvent.focus(items[1]);
    expect(screen.getByRole("tooltip").textContent).toBe("yes: 3 · 30%");
    fireEvent.blur(items[1]);
    expect(screen.queryByRole("tooltip")).toBeNull();
  });
});

describe("Histogram", () => {
  it("draws one column per bin with readable ranges", () => {
    render(<Histogram kind="numeric" edges={[0, 5, 10]} counts={[4, 1]} ariaLabel="Distribution of age" />);
    expect(screen.getByLabelText("0 – 5: 4 rows")).toBeTruthy();
    expect(screen.getByLabelText("5 – 10: 1 rows")).toBeTruthy();
  });
});

describe("TypeMixBar", () => {
  it("lists every present type with its count in the legend", () => {
    render(<TypeMixBar counts={{ numeric: 3, categorical: 1, boolean: 0 }} />);
    const legend = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(legend).toEqual(["Numeric 3", "Categorical 1"]);
  });
});

describe("CorrelationHeatmap", () => {
  it("renders an n×n grid with diverging colours", () => {
    const { container } = render(
      <CorrelationHeatmap columns={["a", "b"]} matrix={[[1, -0.5], [-0.5, 1]]} method="Pearson" />,
    );
    const cells = container.querySelectorAll("span.rounded-\\[2px\\]");
    expect(cells).toHaveLength(4);
    expect((cells[1] as HTMLElement).style.background).toContain("--viz-div-neg");
    // Both axes are labelled: one column label per column, in row order.
    const labels = screen.getAllByTestId("heatmap-col-label").map((el) => el.textContent);
    expect(labels).toEqual(["a", "b"]);
  });
});
