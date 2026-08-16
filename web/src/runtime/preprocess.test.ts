// letterbox is the one function in this repository that has to agree with
// Python to the BIT: Task 21 compares the two engines' detections, and a
// one-grey-level difference in the tensor moves every box in the frame. So
// these tests are not "does it look right" -- they compare against tensors
// `scripts/make_runtime_fixtures.py` produced by calling the real
// `trafficlens.detect.base.letterbox`, byte for byte.

import { describe, expect, it } from "vitest";

import { readFloat32, readBytes, readManifest } from "./fixture-loader";
import { letterboxGeometry, letterboxRgb, resizeBitExact } from "./preprocess";

const manifest = readManifest();

describe("letterboxGeometry", () => {
  it("reproduces Python's scale and pad for the shipped 1280x720 frame", () => {
    const want = manifest.letterbox.shipped;
    const got = letterboxGeometry(want.width, want.height, want.size);
    expect(got.scale).toBe(want.scale);
    expect(got.padX).toBe(want.padX);
    expect(got.padY).toBe(want.padY);
  });

  // Notes §1's measured case. `Math.round(358.5)` is 359; Python's `round` is
  // half-to-even and gives 358. Asserted directly rather than only through a
  // tensor, so the failure names the hazard instead of reporting "1179648
  // bytes differ".
  it("rounds a halfway resized dimension to even, as Python's round does", () => {
    expect(letterboxGeometry(1280, 717, 640).resizedHeight).toBe(358);
  });

  it("puts the odd pixel of an odd pad at the bottom, never the top", () => {
    const geometry = letterboxGeometry(100, 55, 64);
    expect(geometry.resizedHeight).toBe(35);
    expect(geometry.padY).toBe(14); // (64 - 35) // 2, floored
    expect(geometry.padY + geometry.resizedHeight).toBe(49); // 15 grey rows below
  });
});

describe("resizeBitExact", () => {
  // The clamping paths at the two ends of each axis are separate branches from
  // the interpolated middle, and a fixture that only downscales a large image
  // never reaches them.
  it("returns the source unchanged when nothing is resampled", () => {
    const src = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(Array.from(resizeBitExact(src, 3, 3, 1, 3, 3))).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9,
    ]);
  });

  // The coefficient quantisation inside the resize is cvRound -- half to EVEN.
  // Neither letterbox case above happens to contain a weight that lands on a
  // half-step, so without this case swapping that one rounding for Math.round
  // changes nothing any test can see (measured: it survived the mutation).
  // A 3 -> 256 upscale puts weights on 2.5, 8.5, 14.5 ... where the two rules
  // disagree.
  it("quantises half-step interpolation weights to even, as cvRound does", () => {
    const spec = manifest.resize.halfstep;
    const src = readBytes("resize_halfstep_src.bin");
    const want = readBytes("resize_halfstep_want.bin");
    const got = resizeBitExact(
      src,
      spec.srcWidth,
      spec.srcHeight,
      spec.channels,
      spec.dstWidth,
      spec.dstHeight,
    );
    expect(got).toEqual(want);
  });

  it("holds the edge value where the sample falls outside the source", () => {
    // Upscaling 2 -> 4 puts destination 0 left of the source centre and
    // destination 3 right of it; both clamp rather than extrapolate.
    const out = resizeBitExact(Uint8Array.from([0, 200]), 2, 1, 1, 4, 1);
    expect(out[0]).toBe(0);
    expect(out[3]).toBe(200);
  });
});

describe("letterboxRgb", () => {
  for (const [name, expected] of Object.entries(manifest.letterbox.cases)) {
    it(`produces Python's exact tensor bytes for the ${name} case`, () => {
      const source = readBytes(`letterbox_src_${name}.bin`);
      expect(source.length).toBe(expected.width * expected.height * 3);

      const got = letterboxRgb(source, expected.width, expected.height, expected.size);

      expect(got.scale).toBe(expected.scale);
      expect(got.padX).toBe(expected.padX);
      expect(got.padY).toBe(expected.padY);

      const want = readFloat32(`letterbox_want_${name}.bin`);
      expect(got.tensor.length).toBe(want.length);
      // Compare as arrays so a failure reports the first differing index
      // rather than just "not equal"; `toEqual` on typed arrays is exact.
      expect(got.tensor).toEqual(want);
    });
  }

  // Guards the fixtures themselves: if a later edit changes the shapes so that
  // neither case has a halfway rounding or an odd pad any more, the byte
  // comparisons above would still pass while covering strictly less. This
  // fails instead.
  it("keeps both letterbox hazards covered by the committed cases", () => {
    const cases = Object.values(manifest.letterbox.cases);
    expect(cases.some((c) => c.hasHalfwayRounding)).toBe(true);
    expect(cases.some((c) => c.padYIsOdd || c.padXIsOdd)).toBe(true);
  });

  it("fills the padding with the letterbox grey, scaled to [0, 1]", () => {
    const { size, width, height, padY } = manifest.letterbox.cases[
      "oddpad"
    ] as (typeof manifest.letterbox.cases)[string];
    const source = readBytes("letterbox_src_oddpad.bin");
    const { tensor } = letterboxRgb(source, width, height, size);
    // Row 0 is above the image on every channel.
    for (let channel = 0; channel < 3; channel += 1) {
      const row = channel * size * size;
      // fround because the tensor is a Float32Array: 114/255 is a float64
      // literal and reading it back narrows it.
      expect(tensor[row]).toBe(Math.fround(114 / 255));
      expect(tensor[row + size - 1]).toBe(Math.fround(114 / 255));
    }
    expect(padY).toBeGreaterThan(0);
  });
});
