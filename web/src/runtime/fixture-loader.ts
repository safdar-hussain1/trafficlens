/** Test support: read the committed, PYTHON-GENERATED fixtures from disk.
 *
 * Nothing in the shipped app imports this -- it exists so the runtime tests
 * can be checked against bytes `scripts/make_runtime_fixtures.py` wrote, which
 * is the only direction that proves anything. A fixture the TypeScript side
 * produced would agree with the TypeScript side by construction. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const FIXTURES = new URL("./fixtures/", import.meta.url);

export function fixturePath(name: string): string {
  return fileURLToPath(new URL(name, FIXTURES));
}

export function readBytes(name: string): Uint8Array {
  return new Uint8Array(readFileSync(fixturePath(name)));
}

/** Read a fixture written by numpy's `float32` `tobytes()` -- little-endian,
 * densely packed, no header. */
export function readFloat32(name: string): Float32Array {
  const bytes = readBytes(name);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`${name}: ${bytes.byteLength} bytes is not a whole number of float32`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = view.getFloat32(i * 4, true);
  }
  return out;
}

export interface LetterboxCase {
  readonly width: number;
  readonly height: number;
  readonly size: number;
  readonly scale: number;
  readonly padX: number;
  readonly padY: number;
  readonly resizedWidth: number;
  readonly resizedHeight: number;
  readonly padXIsOdd: boolean;
  readonly padYIsOdd: boolean;
  readonly hasHalfwayRounding: boolean;
}

export interface ExpectedDetection {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly score: number;
  readonly classId: number;
  readonly className: string;
}

export interface DecodeCase {
  readonly columns: number;
  readonly scale: number;
  readonly padX: number;
  readonly padY: number;
  readonly expected: readonly ExpectedDetection[];
  readonly constructedCases?: Readonly<Record<string, number | readonly number[]>>;
}

export interface ResizeCase {
  readonly srcWidth: number;
  readonly srcHeight: number;
  readonly dstWidth: number;
  readonly dstHeight: number;
  readonly channels: number;
}

export interface Manifest {
  readonly letterbox: {
    readonly cases: Readonly<Record<string, LetterboxCase>>;
    readonly shipped: {
      readonly width: number;
      readonly height: number;
      readonly size: number;
      readonly scale: number;
      readonly padX: number;
      readonly padY: number;
      readonly sourceSha256: string;
      readonly tensorSha256: string;
    };
  };
  readonly resize: { readonly halfstep: ResizeCase };
  readonly decode: {
    readonly conf: number;
    readonly iou: number;
    readonly keepClassIds: readonly number[];
    readonly nClasses: number;
    readonly boundary: DecodeCase;
    readonly iouexact: DecodeCase;
    readonly real: DecodeCase;
  };
}

export function readManifest(): Manifest {
  return JSON.parse(readFileSync(fixturePath("manifest.json"), "utf8")) as Manifest;
}
