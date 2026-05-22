/**
 * Deterministic color palette for up to 80 classes.
 * Uses HSL with evenly spaced hues for maximum visual distinction.
 */

const PALETTE: string[] = [];

// Generate 80 distinct colors
for (let i = 0; i < 80; i++) {
  const hue = (i * 137.508) % 360; // Golden angle for maximum spread
  const sat = 70 + (i % 3) * 10;
  const light = 50 + (i % 2) * 10;
  PALETTE.push(`hsl(${Math.round(hue)}, ${sat}%, ${light}%)`);
}

export function getClassColor(classId: number): string {
  return PALETTE[classId % PALETTE.length];
}

export function getClassColorRGB(classId: number): [number, number, number] {
  const hue = ((classId * 137.508) % 360) / 360;
  const sat = (70 + (classId % 3) * 10) / 100;
  const light = (50 + (classId % 2) * 10) / 100;

  // HSL to RGB conversion
  const c = (1 - Math.abs(2 * light - 1)) * sat;
  const x = c * (1 - Math.abs(((hue * 6) % 2) - 1));
  const m = light - c / 2;

  let r = 0, g = 0, b = 0;
  const h = hue * 6;
  if (h < 1) { r = c; g = x; }
  else if (h < 2) { r = x; g = c; }
  else if (h < 3) { g = c; b = x; }
  else if (h < 4) { g = x; b = c; }
  else if (h < 5) { r = x; b = c; }
  else { r = c; b = x; }

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

export { PALETTE };
