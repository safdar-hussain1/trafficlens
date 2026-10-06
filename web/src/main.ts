/** The entry point of the demo page, and the three things it can be asked to do.
 *
 * The live demo is the page. The other two modes exist so that claims made on
 * it can be checked from outside the browser: `?selftest=1` replays the
 * committed parity fixtures through the shipped engine and writes a verdict
 * into the tab title, and `?measure=1` times the real per-frame path and writes
 * the figures there too. Both are loaded lazily -- the fixture alone is 435 kB,
 * and a visitor who just wants to watch vehicles being counted should not pay
 * for the proof that the counting is right. */

import "./ui/styles.css";

// Who built this and where the source lives, once per load, in every mode.
console.info("TrafficLens — built by Safdar Hussain · https://github.com/safdar-hussain1/trafficlens");

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
  // The accuracy tally first, and before anything is awaited on the hardware:
  // it is static, needs no GPU and no download, and it still means something
  // on a machine that cannot run the detector at all.
  //
  // Guarded, because `mountAccuracy` throws by design -- on a missing slot, and
  // on a method it cannot address in the bake -- and the tally and the demo
  // share no data path. A renamed slot is no reason a visitor cannot count
  // vehicles; the failure is reported rather than swallowed, and
  // `accuracy.test.ts` is what notices it.
  try {
    const { mountAccuracy } = await import("./ui/accuracy");
    mountAccuracy();
  } catch (error) {
    console.error("the accuracy tally did not mount; the live demo is unaffected", error);
  }

  const { mountControlRoom } = await import("./ui/app");
  const room = await mountControlRoom();
  // Reachable for the headless checks, which drive the same page a visitor
  // gets rather than a stripped-down harness that resembles it.
  (globalThis as { trafficlens?: unknown }).trafficlens = room;
}

void boot();
