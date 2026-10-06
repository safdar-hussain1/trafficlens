/** Sizing a canvas's backing store to the box it is drawn in.
 *
 * Kept apart from `app.ts`, which owns the DOM and cannot be constructed in the
 * test suite, because the rule is easy to get subtly wrong and costly when it
 * is: assigning `width` or `height` reallocates and clears the backing store.
 * Comparing the store against the unrounded product of box and pixel ratio --
 * 919 CSS pixels at a ratio of 1.25 is 1148.75, which never equals the whole
 * number the store holds -- reallocated it on every frame on any display scaled
 * by a fraction, which is most Windows laptops. */

/** The parts of a canvas this needs, so a test can hand in a stand-in. */
export interface SizableCanvas {
  width: number;
  height: number;
  getBoundingClientRect(): { readonly width: number; readonly height: number };
}

/** Match the backing store to the canvas's CSS box at `dpr`, touching it only
 * when the rounded size actually changes. Returns the box in CSS pixels. */
export function sizeCanvas(canvas: SizableCanvas, dpr: number): { width: number; height: number } {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const backingWidth = Math.round(width * dpr);
  const backingHeight = Math.round(height * dpr);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  return { width, height };
}
