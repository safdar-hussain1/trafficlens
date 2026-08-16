// The model is a 10.7 MB download on a page whose whole claim is that it runs
// locally. Two things therefore have to be true: the progress the visitor sees
// must be BYTES actually received rather than a spinner, and the second visit
// must not download it again.

import { describe, expect, it } from "vitest";

import { loadCachedBytes } from "./cache";

const BODY = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

function streamingResponse(body: Uint8Array, chunk: number, contentLength?: number) {
  let sent = 0;
  return {
    ok: true,
    status: 200,
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-length"
          ? contentLength === undefined
            ? null
            : String(contentLength)
          : null,
    },
    body: {
      getReader: () => ({
        read: async () => {
          if (sent >= body.length) return { done: true, value: undefined };
          const value = body.slice(sent, sent + chunk);
          sent += value.length;
          return { done: false, value };
        },
      }),
    },
    arrayBuffer: async () => body.buffer.slice(0) as ArrayBuffer,
  };
}

function fakeStorage() {
  const entries = new Map<string, Uint8Array>();
  const cache = {
    match: async (url: string) => {
      const hit = entries.get(url);
      if (hit === undefined) return undefined;
      return {
        ok: true,
        headers: { get: () => String(hit.length) },
        arrayBuffer: async () => hit.buffer.slice(0) as ArrayBuffer,
      };
    },
    put: async (url: string, response: { arrayBuffer(): Promise<ArrayBuffer> }) => {
      entries.set(url, new Uint8Array(await response.arrayBuffer()));
    },
  };
  return {
    entries,
    storage: { open: async () => cache },
  };
}

describe("loadCachedBytes", () => {
  it("reports progress in bytes actually received, ending at the total", async () => {
    const seen: Array<{ loaded: number; total: number }> = [];
    const bytes = await loadCachedBytes("/m.onnx", {
      fetchImpl: async () => streamingResponse(BODY, 3, BODY.length),
      storage: fakeStorage().storage,
      onProgress: (p) => seen.push({ loaded: p.loaded, total: p.total }),
    });

    expect(new Uint8Array(bytes)).toEqual(BODY);
    // 3 + 3 + 3 + 1: the reports follow the chunks, they are not interpolated.
    expect(seen.map((p) => p.loaded)).toEqual([3, 6, 9, 10]);
    expect(seen.every((p) => p.total === BODY.length)).toBe(true);
  });

  it("reports a zero total when the server sends no content-length", async () => {
    const seen: number[] = [];
    await loadCachedBytes("/m.onnx", {
      fetchImpl: async () => streamingResponse(BODY, 4, undefined),
      storage: fakeStorage().storage,
      onProgress: (p) => seen.push(p.total),
    });
    // Honest rather than guessed: a caller can tell "unknown" from "10 bytes"
    // and show an indeterminate bar instead of a wrong percentage.
    expect(seen.every((t) => t === 0)).toBe(true);
  });

  it("serves the second load from the cache without fetching again", async () => {
    const { storage, entries } = fakeStorage();
    let fetches = 0;
    const fetchImpl = async () => {
      fetches += 1;
      return streamingResponse(BODY, 10, BODY.length);
    };

    const first = await loadCachedBytes("/m.onnx", { fetchImpl, storage });
    expect(fetches).toBe(1);
    expect(entries.size).toBe(1);

    const reports: boolean[] = [];
    const second = await loadCachedBytes("/m.onnx", {
      fetchImpl,
      storage,
      onProgress: (p) => reports.push(p.fromCache),
    });
    expect(fetches).toBe(1);
    expect(new Uint8Array(second)).toEqual(new Uint8Array(first));
    // `every` on an empty array is true, so the length check is what stops
    // this passing when no progress is reported at all.
    expect(reports.length).toBeGreaterThan(0);
    expect(reports.every(Boolean)).toBe(true);
  });

  // The must-still-work half: Cache Storage is unavailable in a private window
  // in some browsers, and on any insecure origin. The model must still load.
  it("still loads when Cache Storage is unavailable", async () => {
    const bytes = await loadCachedBytes("/m.onnx", {
      fetchImpl: async () => streamingResponse(BODY, 5, BODY.length),
      storage: {
        open: async () => {
          throw new Error("SecurityError");
        },
      },
    });
    expect(new Uint8Array(bytes)).toEqual(BODY);
  });

  it("refuses a failed download rather than caching an error page", async () => {
    const { storage, entries } = fakeStorage();
    await expect(
      loadCachedBytes("/m.onnx", {
        storage,
        fetchImpl: async () => ({
          ok: false,
          status: 404,
          headers: { get: () => null },
          arrayBuffer: async () => new ArrayBuffer(0),
        }),
      }),
    ).rejects.toThrow(/404/);
    expect(entries.size).toBe(0);
  });
});
