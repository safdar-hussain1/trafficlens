/** The entry point of the measurements page.
 *
 * Everything on it is static: the figures are baked from `reports/` into
 * `generated/reports.ts` and written into the authored sections by
 * `ui/results.ts`. No detector, no GPU and no download -- the page is the
 * evidence, and it reads the same on a machine that could never run the demo. */

import "./ui/styles.css";
import { installThemeToggle } from "./ui/controls";
import { mountResults } from "./ui/results";

// Who built this and where the source lives, once per load.
console.info("TrafficLens — built by Safdar Hussain · https://github.com/safdar-hussain1/trafficlens");

const toggle = document.getElementById("theme-toggle");
if (toggle instanceof HTMLButtonElement) {
  installThemeToggle(toggle);
}

// `mountResults` throws by design on a missing slot or an unaddressable
// figure, and that throw is what fails `results.test.ts` -- here it would also
// stop the page, so it is reported where a reader of the console will see it.
try {
  mountResults();
} catch (error) {
  console.error("the measured-results sections did not mount", error);
}
