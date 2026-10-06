/** The canvas is resized when its box changes, and only then.
 *
 * Every assignment to a canvas's `width` or `height` reallocates its backing
 * store, so a sizing rule that writes on every call costs a reallocation per
 * frame. The case that matters is a fractional pixel ratio, where the product
 * of box and ratio is never a whole number. */

import { describe, expect, test } from "vitest";

import { sizeCanvas } from "./canvas-size";
import type { SizableCanvas } from "./canvas-size";

/** A stand-in canvas that counts writes to its backing store. */
function fakeCanvas(box: { width: number; height: number }) {
  let width = 300;
  let height = 150;
  let writes = 0;
  const canvas: SizableCanvas & { readonly writes: number } = {
    get width() {
      return width;
    },
    set width(value: number) {
      width = value;
      writes += 1;
    },
    get height() {
      return height;
    },
    set height(value: number) {
      height = value;
      writes += 1;
    },
    getBoundingClientRect: () => box,
    get writes() {
      return writes;
    },
  };
  return canvas;
}

describe("sizeCanvas", () => {
  test("sizes the backing store to the box at the pixel ratio", () => {
    const canvas = fakeCanvas({ width: 920, height: 518 });
    expect(sizeCanvas(canvas, 2)).toEqual({ width: 920, height: 518 });
    expect([canvas.width, canvas.height]).toEqual([1840, 1036]);
  });

  test("leaves the store alone on the next frame at a fractional ratio", () => {
    // 919 x 1.25 = 1148.75: the unrounded product never equals the whole
    // number the store holds, which is exactly how a per-frame reallocation
    // hides on a scaled display.
    const canvas = fakeCanvas({ width: 919, height: 517 });
    sizeCanvas(canvas, 1.25);
    const afterFirst = canvas.writes;
    expect(afterFirst).toBeGreaterThan(0);
    for (let frame = 0; frame < 5; frame += 1) {
      sizeCanvas(canvas, 1.25);
    }
    expect(canvas.writes).toBe(afterFirst);
    expect([canvas.width, canvas.height]).toEqual([1149, 646]);
  });

  test("and resizes again when the box or the ratio really changes", () => {
    const box = { width: 919, height: 517 };
    const canvas = fakeCanvas(box);
    sizeCanvas(canvas, 1.25);
    const settled = canvas.writes;
    box.width = 640;
    sizeCanvas(canvas, 1.25);
    expect(canvas.writes).toBeGreaterThan(settled);
    expect(canvas.width).toBe(800);
    const afterBox = canvas.writes;
    sizeCanvas(canvas, 2);
    expect(canvas.writes).toBeGreaterThan(afterBox);
    expect(canvas.width).toBe(1280);
  });

  test("never hands back an empty box", () => {
    const canvas = fakeCanvas({ width: 0, height: 0.2 });
    expect(sizeCanvas(canvas, 1)).toEqual({ width: 1, height: 1 });
  });
});
