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

markCurrentSection();

/** Mark the contents entry of the section being read, so a long page always
 * says where the reader is in it. Watched with an IntersectionObserver rather
 * than a scroll handler: no work happens while nothing crosses a boundary. */
function markCurrentSection(): void {
  const links = new Map<string, HTMLAnchorElement>();
  for (const link of document.querySelectorAll<HTMLAnchorElement>('.toc a[href^="#"]')) {
    links.set(link.hash.slice(1), link);
  }
  const sections = [...links.keys()]
    .map((id) => document.getElementById(id))
    .filter((section): section is HTMLElement => section !== null);
  if (sections.length === 0 || typeof IntersectionObserver === "undefined") {
    return;
  }
  const inView = new Set<string>();
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          inView.add(entry.target.id);
        } else {
          inView.delete(entry.target.id);
        }
      }
      // The first section, in reading order, that overlaps the band below the
      // sticky header is the one being read.
      const current = sections.find((section) => inView.has(section.id));
      if (current === undefined) {
        // Between two sections the last mark stands; above the first one --
        // back at the page's own heading -- nothing is being read yet.
        const first = sections[0] as HTMLElement;
        if (first.getBoundingClientRect().top > innerHeight * 0.45) {
          for (const link of links.values()) {
            link.removeAttribute("aria-current");
          }
        }
        return;
      }
      for (const [id, link] of links) {
        if (id === current.id) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      }
    },
    { rootMargin: "-90px 0px -55% 0px" },
  );
  for (const section of sections) {
    observer.observe(section);
  }
}
