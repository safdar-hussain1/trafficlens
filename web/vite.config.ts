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
  },
});
