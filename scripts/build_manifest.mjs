// Write `docs/BUILD_MANIFEST.json`: what `docs/` contains, and what it was
// built from.
//
// `docs/` is the published site and it is a build OUTPUT that lives in the
// repository. Nothing previously asserted that, so it could drift from the
// sources for any length of time -- a stale bundle serving next to a green test
// suite, with no check able to tell. This manifest closes that: it records a
// digest of every file the build reads and every file the build writes, and
// `tests/test_docs_build_manifest.py` recomputes both sides from the working
// tree. Edit a source without running the build and that test fails.
//
// Run automatically by the last step of `npm run build`, never by
// hand: a manifest written by hand would describe a build nobody performed.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const docs = join(root, "docs");
const web = join(root, "web");

const MANIFEST = join(docs, "BUILD_MANIFEST.json");

/** Test files are excluded from the input set on purpose: vitest specs are not
 * in the bundler's module graph, so editing one cannot change `docs/`, and
 * including them would make every test edit look like a stale build. */
const EXCLUDED_INPUT = /\.test\.ts$/;

function walk(directory, skip = () => false) {
  const out = [];
  for (const entry of readdirSync(directory)) {
    const full = join(directory, entry);
    if (skip(full)) {
      continue;
    }
    if (statSync(full).isDirectory()) {
      out.push(...walk(full, skip));
    } else {
      out.push(full);
    }
  }
  return out;
}

function digest(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function digestsOf(paths) {
  const out = {};
  for (const path of paths.sort()) {
    out[relative(root, path).split("\\").join("/")] = digest(path);
  }
  return out;
}

const inputs = [
  join(web, "index.html"),
  join(web, "vite.config.ts"),
  join(web, "tsconfig.json"),
  join(web, "package.json"),
  ...walk(join(web, "src"), (path) => EXCLUDED_INPUT.test(path)),
  ...walk(join(web, "public")),
];

const outputs = walk(docs, (path) => path === MANIFEST);

const manifest = {
  // No timestamp anywhere: the manifest has to be a pure function of the tree,
  // or every build would show as a change and the guard would be noise.
  note: "Written by scripts/build_manifest.mjs during `npm run build`. Do not edit.",
  inputs: digestsOf(inputs),
  outputs: digestsOf(outputs),
};

writeFileSync(MANIFEST, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(
  `build manifest: ${Object.keys(manifest.inputs).length} inputs, ` +
    `${Object.keys(manifest.outputs).length} outputs`,
);
