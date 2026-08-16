// The onnxruntime-web assets under web/public/ are copies of what npm shipped,
// served straight to the visitor. Nothing may edit them -- not a lint, not a
// guard, not a well-meant patch -- because the bytes the browser executes have
// to be the bytes the package published.
//
// The expected set is DERIVED by listing node_modules and applying the
// vendoring rule, never hand-typed. A hand-typed allowlist asserts against a
// set chosen by whoever wrote the test: drop a file from public/ and from the
// list in the same edit and the test still passes, which is the "passes while
// protecting nothing" defect this test exists to prevent. Deriving it means
// the package decides what must be present, and the floor below means the
// derivation cannot quietly match nothing.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const DIST = fileURLToPath(
  new URL("../../node_modules/onnxruntime-web/dist/", import.meta.url),
);
const PUBLIC = fileURLToPath(new URL("../../public/", import.meta.url));

/** The vendoring rule, stated once and applied to both directories: the WebGPU
 * entry point, plus the wasm loader it imports and that loader's binary.
 *
 * ASYNCIFY, not jsep. onnxruntime-web 1.27.0 ships `ort.webgpu.mjs` with the
 * jsep and jspi branches compiled out -- the built source reads
 * `false ? "...jsep.mjs" : false ? "...jspi.mjs" : true ? "...asyncify.mjs"`
 * -- so asyncify is the only pair it ever fetches. Vendoring the jsep pair
 * instead was tried first and produced exactly one symptom in the browser:
 * `Failed to fetch dynamically imported module:
 * .../ort-wasm-simd-threaded.asyncify.mjs`. The same build serves both
 * execution providers, which is why no separate plain-wasm pair is vendored. */
const VENDORED = /^(ort\.webgpu\.mjs|ort-wasm-simd-threaded\.asyncify\.(mjs|wasm))$/;

/** Every file the rule selects out of the installed package. */
function expectedFromPackage(): string[] {
  return readdirSync(DIST)
    .filter((name) => VENDORED.test(name))
    .sort();
}

describe("vendored onnxruntime assets", () => {
  it("derives a non-empty expected set from the installed package", () => {
    const expected = expectedFromPackage();
    // The floor. If onnxruntime-web is not installed, or renames these files,
    // the derivation collapses to [] and every byte comparison below becomes
    // vacuously true -- this is what stops that being silent.
    expect(expected.length).toBeGreaterThanOrEqual(3);
    expect(expected).toContain("ort.webgpu.mjs");
  });

  it("ships every derived file byte-identically to what npm published", () => {
    for (const name of expectedFromPackage()) {
      const published = readFileSync(`${DIST}${name}`);
      const vendored = readFileSync(`${PUBLIC}${name}`);
      expect(
        vendored.length,
        `${name} differs in length from the published copy`,
      ).toBe(published.length);
      expect(vendored.equals(published), `${name} is not byte-identical`).toBe(true);
    }
  });

  // The other direction, and the reason this is a pair rather than a single
  // assertion: the test above proves everything derived is present and
  // unmodified, but on its own it would tolerate public/ accumulating a dozen
  // more runtime files nobody meant to publish.
  it("ships no onnxruntime file the rule did not select", () => {
    const shipped = readdirSync(PUBLIC)
      .filter((name) => name.startsWith("ort"))
      .sort();
    expect(shipped).toEqual(expectedFromPackage());
  });
});
