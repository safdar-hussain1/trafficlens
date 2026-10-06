/** How a vehicle type is named and coloured, in one place.
 *
 * The box on the video, the bar in the readout and the line in the feed are
 * three views of one vehicle type, and a visitor links them by colour before
 * they read a word. So the mapping lives here and nowhere else, and colour
 * follows the type, never its rank: a source that has no buses does not
 * repaint the trucks.
 *
 * Only three types get a hue of their own. Three is the most the chart palette
 * keeps distinguishable in every pairing -- for full colour vision and under
 * simulated protanopia and deuteranopia -- and boxes on a video can sit beside
 * any other box, so every pair has to hold. Car, truck and bus take those three
 * because they are what the motorway clip carries; every other type shares one
 * neutral and is always named in text beside it, so no type is told apart by
 * colour alone. */

/** Types with their own hue, in the chart palette's slot order. */
export const COLOURED_TYPES = ["car", "truck", "bus"] as const;

export type ColourToken = "--class-car" | "--class-truck" | "--class-bus" | "--class-other";

/** The CSS custom property a type is drawn in. */
export function colourToken(className: string): ColourToken {
  switch (className) {
    case "car":
      return "--class-car";
    case "truck":
      return "--class-truck";
    case "bus":
      return "--class-bus";
    default:
      return "--class-other";
  }
}

/** The name a visitor reads: the detector's class, capitalised. */
export function displayName(className: string): string {
  return className.length === 0
    ? className
    : (className[0] as string).toUpperCase() + className.slice(1);
}

/** The order the readout lists types in: the coloured ones first, in palette
 * order, then every other type in the order it was given. Stable, so a row
 * never jumps while the counts change. */
export function displayOrder<T>(types: readonly T[], nameOf: (type: T) => string): T[] {
  const rank = (type: T, index: number): number => {
    const coloured = (COLOURED_TYPES as readonly string[]).indexOf(nameOf(type));
    return coloured === -1 ? COLOURED_TYPES.length + index : coloured;
  };
  return types
    .map((type, index) => ({ type, rank: rank(type, index) }))
    .sort((a, b) => a.rank - b.rank)
    .map((item) => item.type);
}
