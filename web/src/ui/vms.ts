/** The live total, drawn as an overhead variable-message sign.
 *
 * Motorway message signs are amber LED matrices behind a dark lens, and the
 * unlit LEDs stay faintly visible -- which is what makes one read as a sign
 * rather than as a number in a box. The digits here are the classic 5x7 matrix
 * shapes, so the count looks like the hardware a traffic counter would really
 * drive.
 *
 * The glyph maths is pure and tested; the DOM half only toggles a `data-on`
 * attribute per dot, so the stylesheet owns the colours, the glow and the
 * row-by-row refresh, and reduced motion is honoured in one place. */

export const GLYPH_COLUMNS = 5;
export const GLYPH_ROWS = 7;

/** Unlit columns between two characters, as on a real sign's module pitch. */
export const CHARACTER_GAP = 1;

/** Characters the sign always has room for. A count past 9999 widens it rather
 * than being cut: dropping a leading digit would show a different number. */
export const MIN_CHARACTERS = 4;

/** One string per row, top to bottom; "1" is a lit LED. The zero is open, not
 * slashed: at sign size a slashed zero reads as a letter. */
export const GLYPHS: Readonly<Record<string, readonly string[]>> = {
  "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
  "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
  "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
  "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
  "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
  "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
  "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
  "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
  "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
  " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
};

/** What the sign shows for a count: right-aligned and blank-padded, the way a
 * sign shows it -- leading blanks, never leading zeros.
 *
 * Anything that is not a whole, non-negative count shows a blank sign. The page
 * never prints a number it did not count, and a sign full of zeros would be
 * one. */
export function signText(value: number, minCharacters: number = MIN_CHARACTERS): string {
  const isCount = Number.isFinite(value) && Number.isInteger(value) && value >= 0;
  const digits = isCount ? String(value) : "";
  return digits.padStart(Math.max(minCharacters, digits.length), " ");
}

export interface Matrix {
  readonly columns: number;
  readonly rows: number;
  /** Row-major, `columns * rows` long. */
  readonly lit: readonly boolean[];
}

/** The whole sign face for `text`, including the unlit gap columns. */
export function matrixFor(text: string): Matrix {
  const characters = [...text];
  const columns =
    characters.length * GLYPH_COLUMNS + Math.max(0, characters.length - 1) * CHARACTER_GAP;
  const lit = new Array<boolean>(columns * GLYPH_ROWS).fill(false);
  characters.forEach((character, index) => {
    const glyph = GLYPHS[character];
    if (glyph === undefined) {
      throw new Error(`the sign has no glyph for ${JSON.stringify(character)}`);
    }
    const left = index * (GLYPH_COLUMNS + CHARACTER_GAP);
    glyph.forEach((row, y) => {
      for (let x = 0; x < GLYPH_COLUMNS; x += 1) {
        if (row[x] === "1") {
          lit[y * columns + left + x] = true;
        }
      }
    });
  });
  return { columns, rows: GLYPH_ROWS, lit };
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** Centre-to-centre spacing of the LEDs, in viewBox units. */
const PITCH = 10;
const RADIUS = 3.7;

/** How long the sign stays brighter after the count goes up, in ms. */
const BUMP_MS = 260;

/** A sign mounted in `host`. Recreated only when its width changes; otherwise a
 * new count flips exactly the dots that differ. */
export class VmsSign {
  private readonly host: HTMLElement;
  private dots: SVGCircleElement[] = [];
  private columns = 0;
  private shown: string | null = null;
  private value: number | null = null;
  private bumpTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(host: HTMLElement) {
    this.host = host;
  }

  show(value: number): void {
    const text = signText(value);
    if (text === this.shown) {
      return;
    }
    const matrix = matrixFor(text);
    if (matrix.columns !== this.columns) {
      this.build(matrix.columns, matrix.rows);
    }
    matrix.lit.forEach((on, index) => {
      const dot = this.dots[index];
      if (dot === undefined) {
        return;
      }
      if (on) {
        dot.setAttribute("data-on", "");
      } else {
        dot.removeAttribute("data-on");
      }
    });
    const rose = this.value !== null && value > this.value;
    this.shown = text;
    this.value = value;
    if (rose) {
      this.bump();
    }
  }

  private build(columns: number, rows: number): void {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${columns * PITCH} ${rows * PITCH}`);
    svg.setAttribute("focusable", "false");
    const dots: SVGCircleElement[] = [];
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < columns; x += 1) {
        const dot = document.createElementNS(SVG_NS, "circle");
        dot.setAttribute("cx", String(x * PITCH + PITCH / 2));
        dot.setAttribute("cy", String(y * PITCH + PITCH / 2));
        dot.setAttribute("r", String(RADIUS));
        dot.setAttribute("class", "vms__dot");
        // The refresh sweeps down the sign one row at a time, as a real one
        // does; the stylesheet turns the row into a delay.
        dot.setAttribute("style", `--row:${y}`);
        dots.push(dot);
      }
    }
    svg.append(...dots);
    this.host.replaceChildren(svg);
    this.dots = dots;
    this.columns = columns;
  }

  private bump(): void {
    this.host.classList.add("is-bump");
    if (this.bumpTimer !== null) {
      clearTimeout(this.bumpTimer);
    }
    this.bumpTimer = setTimeout(() => {
      this.host.classList.remove("is-bump");
      this.bumpTimer = null;
    }, BUMP_MS);
  }
}
