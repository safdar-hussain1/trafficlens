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
  // The results sections first, and before anything is awaited on the hardware:
  // they are static, they need no GPU and no download, and they are the half of
  // the page that still means something on a machine that cannot run the
  // detector at all. The control room's own probe takes as long as it takes.
  //
  // Guarded, because `mountResults` throws by design -- on a missing slot, and on
  // a measurement it cannot address in the bake -- and the two halves of this page
  // share no data path at all. A slot that has been renamed in the static half is
  // no reason a visitor cannot count vehicles, and before this guard it was: the
  // rejected promise stopped `boot` before the control room mounted. The failure
  // is reported rather than swallowed, and `results.test.ts` asserts the slots and
  // the sections still agree -- the guard keeps the demo alive, the test is what
  // notices. Neither substitutes for the other.
  try {
    const { mountResults } = await import("./ui/results");
    mountResults();
  } catch (error) {
    console.error(
      "the measured-results sections did not mount; the control room is unaffected",
      error,
    );
  }

  const { mountControlRoom } = await import("./ui/app");
  const room = await mountControlRoom();
  // Reachable for the headless checks, which drive the same page a visitor
  // gets rather than a stripped-down harness that resembles it.
  (globalThis as { trafficlens?: unknown }).trafficlens = room;
}

void boot();
