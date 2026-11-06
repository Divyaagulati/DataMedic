import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { FormError } from "./form-error";

describe("FormError", () => {
  afterEach(cleanup);

  it("renders nothing without a message", () => {
    const { container } = render(<FormError message={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders an accessible alert, escaping content", () => {
    render(<FormError message={"<img src=x onerror=alert(1)>"} />);
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("<img src=x onerror=alert(1)>");
    expect(alert.querySelector("img")).toBeNull();
  });
});
