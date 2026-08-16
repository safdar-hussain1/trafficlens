import { DETECT_DEFAULT_INPUT_SIZE, TRACK_MAX_AGE } from "./generated/constants";

// The control room is built later; this entry exists so the toolchain is proven
// end to end -- TypeScript typechecks, the generated constants are reachable
// from the build graph, and the bundle carries the same numbers the Python
// engine reads rather than a browser-side copy of them.
const mount = document.querySelector<HTMLElement>("#app");

if (mount) {
  const heading = document.createElement("h1");
  heading.textContent = "TrafficLens";

  const status = document.createElement("p");
  status.textContent =
    `Browser engine: detector input ${DETECT_DEFAULT_INPUT_SIZE}px, ` +
    `tracks survive ${TRACK_MAX_AGE} frames without a detection.`;

  mount.replaceChildren(heading, status);
}
