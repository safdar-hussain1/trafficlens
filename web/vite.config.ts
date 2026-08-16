import { defineConfig } from "vite";

// docs/ is the GitHub Pages site root, so the production build lands there and
// every asset URL is relative -- the site is served from a project subpath, not
// a domain root, and an absolute "/assets/..." would 404 there.
export default defineConfig({
  base: "./",
  build: {
    outDir: "../docs",
    emptyOutDir: true,
    // Sourcemaps embed the absolute path of every source file on the machine
    // that built them, which would put this repository's checkout location into
    // a tracked, publicly served artefact. The cost is no browser sourcemaps on
    // the published site; the engine's behaviour is pinned by tests, not by
    // stepping through bundled code.
    sourcemap: false,
    // public/ carries things the bundler must NOT touch: the exported model,
    // the onnxruntime-web entry point, and the wasm loader and binary it
    // pulls in. Vite copies publicDir verbatim into outDir, which is exactly
    // what is wanted -- `web/src/runtime/vendored.test.ts` asserts those files
    // are byte-identical to what npm published, and that assertion is only
    // worth something if the bytes it checks are the bytes the browser
    // executes. So the runtime is imported at RUN time from its published URL
    // (see session.ts) rather than through the module graph; bundling it would
    // have Rollup emit a transformed copy into assets/ and quietly leave the
    // vendored file unused.
    //
    // assetsInlineLimit 0 is the same principle in miniature: nothing that
    // does reach the build may become a data: URL, because an inlined copy of
    // a file is no longer that file.
    assetsInlineLimit: 0,
  },
});
