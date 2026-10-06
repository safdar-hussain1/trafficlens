/** The sign the live total is shown on.
 *
 * A counter that draws the wrong dots shows a different number, so the glyph
 * table and the layout are pinned rather than judged by eye: every digit is a
 * distinct 5x7 shape, characters sit one dark column apart, and a count wider
 * than the sign widens it instead of losing its leading digit. */

import { describe, expect, test } from "vitest";

import {
  CHARACTER_GAP,
  GLYPHS,
  GLYPH_COLUMNS,
  GLYPH_ROWS,
  MIN_CHARACTERS,
  matrixFor,
  signText,
} from "./vms";

const DIGITS = "0123456789";

describe("the glyphs", () => {
  test("every digit is seven rows of five LEDs", () => {
    for (const digit of DIGITS) {
      const glyph = GLYPHS[digit];
      expect(glyph, digit).toBeDefined();
      expect(glyph).toHaveLength(GLYPH_ROWS);
      for (const row of glyph ?? []) {
        expect(row).toMatch(new RegExp(`^[01]{${GLYPH_COLUMNS}}$`));
      }
    }
  });

  test("no two digits share a shape", () => {
    const shapes = new Set([...DIGITS].map((digit) => (GLYPHS[digit] ?? []).join("")));
    expect(shapes.size).toBe(DIGITS.length);
  });

  test("the blank is dark, and every digit lights something", () => {
    expect((GLYPHS[" "] ?? []).join("")).not.toContain("1");
    for (const digit of DIGITS) {
      expect((GLYPHS[digit] ?? []).join("")).toContain("1");
    }
  });
});

describe("signText", () => {
  test("right-aligns a count with blanks, never leading zeros", () => {
    expect(signText(0)).toBe("   0");
    expect(signText(127)).toBe(" 127");
    expect(signText(9999)).toBe("9999");
    expect(signText(0)).toHaveLength(MIN_CHARACTERS);
  });

  test("a count wider than the sign widens it rather than losing a digit", () => {
    expect(signText(12345)).toBe("12345");
    expect(signText(1234567)).toBe("1234567");
  });

  test("anything that is not a count shows a blank sign, not a number", () => {
    for (const value of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(signText(value)).toBe(" ".repeat(MIN_CHARACTERS));
    }
  });
});

describe("matrixFor", () => {
  test("lays characters out one dark column apart", () => {
    const matrix = matrixFor("10");
    expect(matrix.columns).toBe(2 * GLYPH_COLUMNS + CHARACTER_GAP);
    expect(matrix.rows).toBe(GLYPH_ROWS);
    expect(matrix.lit).toHaveLength(matrix.columns * matrix.rows);
    for (let y = 0; y < GLYPH_ROWS; y += 1) {
      expect(matrix.lit[y * matrix.columns + GLYPH_COLUMNS]).toBe(false);
    }
  });

  test("the dots lit for a digit are exactly its glyph, in its own cell", () => {
    const matrix = matrixFor(" 8");
    const left = GLYPH_COLUMNS + CHARACTER_GAP;
    const read: string[] = [];
    for (let y = 0; y < GLYPH_ROWS; y += 1) {
      let row = "";
      for (let x = 0; x < GLYPH_COLUMNS; x += 1) {
        row += matrix.lit[y * matrix.columns + left + x] === true ? "1" : "0";
      }
      read.push(row);
    }
    expect(read).toEqual(GLYPHS["8"]);
    // And the blank cell beside it stayed dark.
    for (let y = 0; y < GLYPH_ROWS; y += 1) {
      for (let x = 0; x < GLYPH_COLUMNS; x += 1) {
        expect(matrix.lit[y * matrix.columns + x]).toBe(false);
      }
    }
  });

  test("a character with no glyph throws rather than drawing a blank", () => {
    // A blank where a digit should be would show a different number.
    expect(() => matrixFor("1-2")).toThrow(/no glyph/);
  });
});
