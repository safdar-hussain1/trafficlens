/** Ambient declarations for the two non-TypeScript things the bundle imports.
 *
 * Written out rather than pulled in with `"types": ["vite/client"]`: this
 * project's tsconfig keeps `types` empty on purpose, so that what the compiler
 * believes about the world is what is written here rather than whatever a
 * toolchain package happens to declare globally. */

/** `import "./ui/styles.css"` -- a side-effect import the bundler turns into a
 * stylesheet link. It has no value. */
declare module "*.css";

/** `await import("./fixtures/parity.json")` -- the bundler emits the parsed
 * JSON as the default export. Typed as `unknown` so a caller has to say what it
 * expects rather than being handed a shape nobody checked. */
declare module "*.json" {
  const value: unknown;
  export default value;
}
