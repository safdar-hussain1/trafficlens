/** The one invariant `app.ts` cannot state in a comment: the frame clock and
 * the pipeline restart together.
 *
 * `SessionPipeline.step()` retires a track on `frameIndex - lastSeen > maxAge`.
 * That is a subtraction, not a comparison with "now", so a pipeline kept across
 * a clock reset holds `lastSeen` values in the future, the difference goes
 * negative, and reaping stops -- `web/src/engine/pipeline.test.ts` pins that
 * mechanism in the engine where it lives.
 *
 * What was missing is the caller's half. `clearSession()` carried a comment
 * asserting "Both callers construct a new pipeline". They did not:
 * `selectSource()` calls `clearSession()` early and then RETURNS on its error
 * path -- a denied camera, an unloadable clip -- without ever reaching
 * `buildPipeline()`. Pressing Start afterwards ran a stale pipeline against a
 * clock that had just gone back to zero.
 *
 * `app.ts` owns the DOM, the video element and the onnxruntime session, so it
 * cannot be constructed in this suite. What can be checked is the property that
 * replaced the comment: the reset and the construction are one statement pair
 * in one method, and no other line in the file resets the clock. A structural
 * test is the weaker instrument, so it carries its own must-still-exist half:
 * the early return this was ever about has to still be there, or the guard is
 * passing over a case that no longer exists.
 */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "vitest";

const SOURCE = readFileSync(new URL("./app.ts", import.meta.url), "utf8");

/** The body of `private <name>(`, brace-matched from its opening brace. */
function methodBody(name: string): string {
  const declared = new RegExp(`private (?:async )?${name}\\(`).exec(SOURCE);
  expect(declared, `app.ts has no method named ${name}`).not.toBeNull();
  const signature = declared?.index ?? -1;
  const open = SOURCE.indexOf("{", signature);
  let depth = 0;
  for (let index = open; index < SOURCE.length; index += 1) {
    const character = SOURCE[index];
    if (character === "{") {
      depth += 1;
    } else if (character === "}") {
      depth -= 1;
      if (depth === 0) {
        return SOURCE.slice(open, index + 1);
      }
    }
  }
  throw new Error(`unbalanced braces reading ${name} out of app.ts`);
}

/** Statements, with comments removed: a sentence in a comment is what this
 * test exists because of, and must not be able to satisfy it. */
function statements(body: string): string {
  return body.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
}

describe("the frame clock and the pipeline", () => {
  test("are reset in the same method, and nowhere else", () => {
    const clearSession = statements(methodBody("clearSession"));
    expect(clearSession).toContain("this.frameIndex = 0;");
    expect(clearSession).toContain("this.pipeline = this.buildPipeline();");

    // No other line in the file may reset the clock. This is the assertion
    // that a future `frameIndex = 0` in the detect loop -- which is how this
    // failed the first time -- has to fail.
    const resets = statements(SOURCE).match(/this\.frameIndex\s*=\s*0/g) ?? [];
    expect(resets).toHaveLength(1);
  });

  test("and the path that used to skip the pipeline still exists", () => {
    // The must-still-exist half. If `selectSource` stopped returning early on
    // an unusable source, the test above would be guarding a case that had
    // gone away, and would pass by covering nothing.
    const selectSource = statements(methodBody("selectSource"));
    const cleared = selectSource.indexOf("this.clearSession();");
    const built = selectSource.indexOf("this.pipeline = this.buildPipeline();");
    const earlyReturn = selectSource.indexOf("return;");
    expect(cleared).toBeGreaterThan(-1);
    expect(built).toBeGreaterThan(-1);
    expect(earlyReturn).toBeGreaterThan(cleared);
    expect(earlyReturn).toBeLessThan(built);
  });

  test("the comment that replaced the false one does not assert the old claim", () => {
    // The sentence "Both callers construct a new pipeline" shipped while being
    // false, and nothing noticed, because nothing reads comments. This does.
    expect(SOURCE).not.toContain("Both callers construct a new pipeline");
  });
});
