/** The "How it works" pictures keep the promise their stylesheet makes: the
 * count fires when the car's centre reaches the line, and every trail dot
 * appears just after the vehicle has passed its position.
 *
 * The motion is CSS keyframes and the geometry is SVG markup, two files apart.
 * Change a drive distance or move a dot without re-deriving the percentages
 * and the picture would count a car before it reached the line, which is the
 * one thing the page exists to show it does not do. */

import { readFileSync } from "node:fs";

import { describe, expect, test } from "vitest";

const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");

/** The body of `@keyframes name { ... }`, brace-matched. */
function keyframes(name: string): string {
  const start = css.indexOf(`@keyframes ${name} {`);
  expect(start, `no @keyframes ${name}`).toBeGreaterThan(-1);
  let depth = 0;
  for (let index = css.indexOf("{", start); index < css.length; index += 1) {
    if (css[index] === "{") {
      depth += 1;
    } else if (css[index] === "}") {
      depth -= 1;
      if (depth === 0) {
        return css.slice(start, index + 1);
      }
    }
  }
  throw new Error(`unbalanced @keyframes ${name}`);
}

/** The translateX at the first and the last stop, and the percentage at which
 * the drive arrives. */
function drive(name: string): { from: number; to: number; arrivesAt: number } {
  const body = keyframes(name);
  const shifts = [...body.matchAll(/translateX\((-?[\d.]+)(?:px)?\)/g)].map((m) => Number(m[1]));
  const stops = [...body.matchAll(/([\d.]+)%/g)].map((m) => Number(m[1]));
  expect(shifts.length).toBe(2);
  return { from: shifts[0] as number, to: shifts[1] as number, arrivesAt: stops[1] as number };
}

/** The percentage at which a keyframe set first reaches its "after" stop. */
function switchesAt(name: string, after: string): number {
  const body = keyframes(name);
  const match = new RegExp(`([\\d.]+)%[,\\s]*(?:[\\d.]+%[,\\s]*)*\\{\\s*${after}`).exec(body);
  expect(match, `${name} never reaches ${after}`).not.toBeNull();
  return Number(match?.[1]);
}

/** Where an eased fade-in begins: the last stop still at opacity 0 before the
 * first stop above it. That is the moment the visitor first sees it move. */
function risesAt(name: string): number {
  const body = keyframes(name);
  const stops = [...body.matchAll(/((?:[\d.]+%[,\s]*)+)\{\s*opacity:\s*([\d.]+)/g)].map((m) => ({
    at: Math.max(...[...(m[1] ?? "").matchAll(/([\d.]+)%/g)].map((s) => Number(s[1]))),
    opacity: Number(m[2]),
  }));
  const firstVisible = stops.findIndex((stop) => stop.opacity > 0);
  expect(firstVisible, `${name} never becomes visible`).toBeGreaterThan(0);
  return (stops[firstVisible - 1] as { at: number }).at;
}

/** The x of every circle with this class prefix, in document order, inside the
 * illustration named. */
function circleXs(art: string, prefix: string): number[] {
  const start = html.indexOf(`step__art--${art}`);
  const end = html.indexOf("</svg>", start);
  const svg = html.slice(start, end);
  return [...svg.matchAll(new RegExp(`class="${prefix}--(\\d+)" cx="(-?[\\d.]+)"`, "g"))].map((m) =>
    Number(m[2]),
  );
}

describe("the Count picture", () => {
  const svgStart = html.indexOf("step__art--count");
  const svg = html.slice(svgStart, html.indexOf("</svg>", svgStart));

  test("counts the car at the moment its centre reaches the line", () => {
    const lineX = Number(/d="M(\d+) 30v120" stroke="#f7b500" stroke-width="4"/.exec(svg)?.[1]);
    const driveGroup = svg.slice(svg.indexOf("art-drive--count"));
    const body = /<rect x="(\d+)" y="\d+" width="(\d+)" height="20"/.exec(driveGroup);
    expect(body).not.toBeNull();
    const centre = Number(body?.[1]) + Number(body?.[2]) / 2;
    const motion = drive("art-drive-count");
    // Where along the drive the centre is at the line, as a share of the loop.
    const crossing = ((lineX - (centre + motion.from)) / (motion.to - motion.from)) * motion.arrivesAt;
    // The stepped switch (the counter going up, the tick on the chip) lands on
    // its stop; the eased ones (the "+1", the line's flash) start rising there.
    const fires: [string, number][] = [
      ["art-after", switchesAt("art-after", "opacity: 1")],
      ["art-plus", risesAt("art-plus")],
      ["art-glow", risesAt("art-glow")],
    ];
    for (const [name, at] of fires) {
      expect(
        Math.abs(at - crossing),
        `${name} fires at ${at}%, the car reaches the line at ${crossing.toFixed(1)}%`,
      ).toBeLessThanOrEqual(1.5);
    }
  });

  test("drops each trace dot only after the car's centre has passed it", () => {
    const xs = circleXs("count", "art-trace art-trace");
    expect(xs.length).toBeGreaterThan(3);
    const driveGroup = svg.slice(svg.indexOf("art-drive--count"));
    const body = /<rect x="(\d+)" y="\d+" width="(\d+)" height="20"/.exec(driveGroup);
    const centre = Number(body?.[1]) + Number(body?.[2]) / 2;
    const motion = drive("art-drive-count");
    xs.forEach((x, index) => {
      const passes = ((x - (centre + motion.from)) / (motion.to - motion.from)) * motion.arrivesAt;
      const appears = switchesAt(`art-trace-${index + 1}`, "transform: scale\\(1\\)");
      expect(appears).toBeGreaterThan(passes);
      expect(appears - passes).toBeLessThanOrEqual(6);
    });
  });
});

describe("the Track picture", () => {
  test("drops each trail dot only after the car's centre has passed it", () => {
    const start = html.indexOf("step__art--track");
    const svg = html.slice(start, html.indexOf("</svg>", start));
    // Car A: the first car in the drive group, and the first row of dots.
    const driveGroup = svg.slice(svg.indexOf("art-drive"));
    const body = /<rect x="(\d+)" y="\d+" width="(\d+)" height="20"/.exec(driveGroup);
    const centre = Number(body?.[1]) + Number(body?.[2]) / 2;
    const motion = drive("art-drive-track");
    const xs = circleXs("track", "art-dot art-dot").slice(0, 4);
    expect(xs).toHaveLength(4);
    xs.forEach((x, index) => {
      const passes = ((x - (centre + motion.from)) / (motion.to - motion.from)) * motion.arrivesAt;
      const appears = switchesAt(`art-dot-${index + 1}`, "transform: scale\\(1\\)");
      expect(appears).toBeGreaterThan(passes);
      expect(appears - passes).toBeLessThanOrEqual(6);
    });
  });
});
