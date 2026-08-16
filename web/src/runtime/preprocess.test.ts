// letterbox is the one function in this repository that has to agree with
// Python to the BIT: Task 21 compares the two engines' detections, and a
// one-grey-level difference in the tensor moves every box in the frame. So
// these tests are not "does it look right" -- they compare against tensors
// `scripts/make_runtime_fixtures.py` produced by calling the real
// `trafficlens.detect.base.letterbox`, byte for byte.

import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import { readFloat32, readBytes, readManifest } from "./fixture-loader";
import {
  type PixelReader,
  createCanvasPixelReader,
  frameSize,
  letterbox,
  letterboxGeometry,
  letterboxRgb,
  resizeBitExact,
} from "./preprocess";

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

// --- the browser boundary -----------------------------------------------------
//
// `letterbox()` reads pixels out of a <video> or canvas, and until these tests
// existed nothing covered it. Three mutations, each of which breaks the product
// in a way no user could diagnose, survived the whole 190-test suite:
//
//   scratch canvas sized to `size` (so drawImage resamples)   190 passed
//   videoWidth/videoHeight -> width/height                    190 passed
//   RGBA -> RGB reading channel 2 first (i.e. BGR)            190 passed
//
// All three are hazards the module's own comments name, which is exactly the
// shape of defect this session keeps producing: a documented rule with nothing
// enforcing it.

describe("frameSize", () => {
  it("takes a video's DECODED size, not its layout size", () => {
    // A styled <video>: CSS box 320x180, decoded frame 1280x720. Letterboxing
    // the CSS box would scale every returned box by the wrong factor.
    const video = { videoWidth: 1280, videoHeight: 720, width: 320, height: 180 };
    expect(frameSize(video as never)).toEqual({ width: 1280, height: 720 });
  });

  // The control on the same axis -- whether the source carries a decoded size:
  // a canvas has only width/height, and those ARE its pixels.
  it("takes a canvas's own width and height", () => {
    expect(frameSize({ width: 640, height: 480 } as never)).toEqual({
      width: 640,
      height: 480,
    });
  });
});

describe("createCanvasPixelReader", () => {
  interface Call {
    readonly args: readonly number[];
  }

  function spyCanvas() {
    const drawCalls: Call[] = [];
    const getCalls: Call[] = [];
    const sizes: Array<{ width: number; height: number }> = [];
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage: (_source: unknown, ...args: number[]) => {
          drawCalls.push({ args });
        },
        getImageData: (sx: number, sy: number, sw: number, sh: number) => {
          getCalls.push({ args: [sx, sy, sw, sh] });
          // R=10, G=20, B=30, A=255 everywhere: distinct per channel, so a
          // swapped read is visible rather than symmetrical.
          const data = new Uint8ClampedArray(sw * sh * 4);
          for (let i = 0; i < data.length; i += 4) {
            data[i] = 10;
            data[i + 1] = 20;
            data[i + 2] = 30;
            data[i + 3] = 255;
          }
          return { data, width: sw, height: sh };
        },
      }),
    };
    const factory = (width: number, height: number) => {
      sizes.push({ width, height });
      return canvas;
    };
    return { canvas, factory, drawCalls, getCalls, sizes };
  }

  it("sizes its surface to the SOURCE and never asks drawImage to rescale", () => {
    const spy = spyCanvas();
    createCanvasPixelReader(spy.factory)({} as never, 1280, 720);
    expect(spy.sizes).toEqual([{ width: 1280, height: 720 }]);
    expect(spy.canvas.width).toBe(1280);
    expect(spy.canvas.height).toBe(720);
    // Destination point only. A third and fourth argument would be a
    // destination SIZE, which hands the resampling to the browser's own
    // implementation-defined filter instead of resizeBitExact.
    expect(spy.drawCalls).toEqual([{ args: [0, 0] }]);
    expect(spy.getCalls).toEqual([{ args: [0, 0, 1280, 720] }]);
  });

  it("reuses one surface across frames of the same size", () => {
    const spy = spyCanvas();
    const read = createCanvasPixelReader(spy.factory);
    read({} as never, 64, 48);
    read({} as never, 64, 48);
    expect(spy.sizes.length).toBe(1);
    // The control: a different frame size must NOT reuse a mis-sized surface.
    read({} as never, 32, 24);
    expect(spy.sizes).toEqual([
      { width: 64, height: 48 },
      { width: 32, height: 24 },
    ]);
  });
});

describe("letterbox", () => {
  function pixels(width: number, height: number, rgba: readonly number[]): PixelReader {
    return (_source, w, h) => {
      const data = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < data.length; i += 4) {
        data[i] = rgba[0] as number;
        data[i + 1] = rgba[1] as number;
        data[i + 2] = rgba[2] as number;
        data[i + 3] = rgba[3] as number;
      }
      expect(w).toBe(width);
      expect(h).toBe(height);
      return { data, width: w, height: h };
    };
  }

  it("preserves channel order from RGBA through to the CHW tensor", () => {
    const size = 8;
    // A flat frame, so every pixel of the resized image carries the same
    // triple and the only thing under test is which channel lands in which
    // plane. R, G and B are distinct, so a BGR read is a different tensor.
    const { tensor } = letterbox(
      { width: 8, height: 8 } as never,
      size,
      pixels(8, 8, [10, 20, 30, 255]),
    );
    const plane = size * size;
    const centre = (size / 2) * size + size / 2;
    expect(tensor[centre]).toBe(Math.fround(10 / 255));
    expect(tensor[plane + centre]).toBe(Math.fround(20 / 255));
    expect(tensor[2 * plane + centre]).toBe(Math.fround(30 / 255));
  });

  it("reads pixels at the video's decoded size, not its layout size", () => {
    // pixels() asserts the dimensions it is asked for, so a reader asked for
    // 320x180 fails inside the fake rather than producing a wrong tensor.
    const video = { videoWidth: 64, videoHeight: 32, width: 320, height: 180 };
    const result = letterbox(video as never, 16, pixels(64, 32, [1, 2, 3, 255]));
    expect(result.scale).toBe(letterboxGeometry(64, 32, 16).scale);
  });

  it("refuses a frame that has not decoded yet", () => {
    expect(() =>
      letterbox({ videoWidth: 0, videoHeight: 0 } as never, 16, pixels(0, 0, [0, 0, 0, 0])),
    ).toThrow(/no decoded pixels/);
  });

  it("refuses a pixel buffer that is not RGBA at the frame's size", () => {
    const short: PixelReader = (_s, w, h) => ({
      data: new Uint8ClampedArray(w * h * 3),
      width: w,
      height: h,
    });
    expect(() => letterbox({ width: 8, height: 8 } as never, 8, short)).toThrow(/RGBA/);
  });
});

// --- the shape the product actually runs ---------------------------------------

describe("letterbox at the shipped shape", () => {
  // 1280x720 into 480 is what the demo runs on every frame, and neither committed
  // tensor fixture is that shape -- they are 128x69 and 100x55, chosen for the
  // rounding hazards. Its tensor is 2.7 MB, too big to commit; a sha256 is 32
  // bytes and pins it exactly.
  const shipped = manifest.letterbox.shipped;

  /** The generator's `synthetic_frame`, in RGB as a canvas would hand it over.
   * Python builds it in BGR and reverses; the channels are written here in the
   * order that reversal produces. */
  function syntheticFrameRgb(width: number, height: number): Uint8Array {
    const rgb = new Uint8Array(width * height * 3);
    for (let y = 0, i = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1, i += 3) {
        rgb[i] = (x * 7 + y * 13) % 256; // red: the sawtooth
        rgb[i + 1] = Math.floor((y * 255) / Math.max(height - 1, 1)); // green ramp
        rgb[i + 2] = Math.floor((x * 255) / Math.max(width - 1, 1)); // blue ramp
      }
    }
    return rgb;
  }

  it("reproduces the generator's source frame before letterboxing it", () => {
    // Checked separately so a drifting recipe fails HERE, naming the input,
    // rather than failing the tensor digest below where a broken letterbox and
    // a different frame are indistinguishable.
    const frame = syntheticFrameRgb(shipped.width, shipped.height);
    expect(createHash("sha256").update(frame).digest("hex")).toBe(shipped.sourceSha256);
  });

  it("produces Python's exact tensor for a 1280x720 frame into the 480px graph", () => {
    const frame = syntheticFrameRgb(shipped.width, shipped.height);
    const got = letterboxRgb(frame, shipped.width, shipped.height, shipped.size);
    expect(got.scale).toBe(shipped.scale);
    expect(got.padX).toBe(shipped.padX);
    expect(got.padY).toBe(shipped.padY);
    expect(got.tensor.length).toBe(3 * shipped.size * shipped.size);
    expect(createHash("sha256").update(new Uint8Array(got.tensor.buffer)).digest("hex")).toBe(
      shipped.tensorSha256,
    );
  });
});
