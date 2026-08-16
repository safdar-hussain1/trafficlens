/** The entry point, and the three things this bundle can be asked to do.
 *
 * The control room is the page. The other two modes exist so that claims made
 * on it can be checked from outside the browser: `?selftest=1` replays the
 * committed parity fixtures through the shipped engine and writes a verdict
 * into the tab title, and `?measure=1` times the real per-frame path and writes
 * the figures there too. Both are loaded lazily -- the fixture alone is 435 kB,
 * and a visitor who just wants to watch vehicles being counted should not pay
 * for the proof that the counting is right. */

import "./ui/styles.css";

const params = new URLSearchParams(location.search);

async function boot(): Promise<void> {
  if (params.get("selftest") === "1") {
    const { runSelftestPage } = await import("./selftest");
    await runSelftestPage();
    return;
  }
  if (params.get("measure") === "1") {
    const { runMeasurePage } = await import("./measure");
    await runMeasurePage(params);
    return;
  }
  const { mountControlRoom } = await import("./ui/app");
  const room = await mountControlRoom();
  // Reachable for the headless checks, which drive the same page a visitor
  // gets rather than a stripped-down harness that resembles it.
  (globalThis as { trafficlens?: unknown }).trafficlens = room;
}

void boot();
