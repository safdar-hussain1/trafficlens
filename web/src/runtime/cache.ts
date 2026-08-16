/** Downloading the model once, and showing the visitor the truth while it
 * happens.
 *
 * The model is 10.7 MB on a page whose whole point is that it runs locally.
 * Two things follow. The progress the visitor sees has to be bytes actually
 * received -- a spinner would be a claim that no one measured -- so the body
 * is streamed and each chunk reported as it lands. And a second visit must not
 * pay for it again, so the bytes go into Cache Storage, which unlike the HTTP
 * cache is explicit about whether a hit occurred. */

export interface LoadProgress {
  /** Bytes received so far. On a cache hit, the whole file at once. */
  readonly loaded: number;
  /** Total bytes, or 0 when the server sent no `content-length`. Zero means
   * UNKNOWN and is reported as such rather than guessed at, so a caller can
   * show an indeterminate bar instead of a wrong percentage. */
  readonly total: number;
  readonly fromCache: boolean;
}

/** Only what this module reads. Narrower than `Response` so the tests can
 * supply a fake without reimplementing the whole interface, and so a body that
 * cannot stream is representable. */
interface ResponseLike {
  readonly ok: boolean;
  readonly status: number;
  readonly headers: { get(name: string): string | null };
  readonly body?:
    | { getReader(): { read(): Promise<{ done: boolean; value?: Uint8Array | undefined }> } }
    | undefined;
  arrayBuffer(): Promise<ArrayBuffer>;
}

/** A Cache Storage hit: neither the status line nor the body stream is read
 * back out of the cache, only the bytes. */
interface CachedResponseLike {
  readonly ok: boolean;
  arrayBuffer(): Promise<ArrayBuffer>;
}

interface CacheLike {
  match(url: string): Promise<CachedResponseLike | undefined>;
  put(url: string, response: unknown): Promise<void>;
  /** Both optional: eviction is a courtesy to the visitor's disk, not a
   * correctness requirement, and a storage that cannot enumerate itself must
   * still be usable for the download that matters. */
  keys?(): Promise<readonly { readonly url: string }[]>;
  delete?(url: string): Promise<boolean>;
}

interface CacheStorageLike {
  open(name: string): Promise<CacheLike>;
}

export interface LoadOptions {
  readonly onProgress?: ((progress: LoadProgress) => void) | undefined;
  readonly storage?: CacheStorageLike | undefined;
  readonly fetchImpl?: ((url: string) => Promise<ResponseLike>) | undefined;
  readonly cacheName?: string | undefined;
  /** A token identifying the CONTENT at `url`, not the path to it. Supply the
   * model's digest here; see `cacheKeyFor`. */
  readonly version?: string | undefined;
}

export const MODEL_CACHE_NAME = "trafficlens-models-v1";

/** The key an entry is stored under.
 *
 * Cache Storage matches on the request and, unlike the HTTP cache, consults no
 * freshness metadata at all: an entry stored under a URL is served forever. So
 * a model replaced at the same path would strand every returning visitor on the
 * old graph while the page went on reporting the new graph's accuracy. Keying
 * on the content version turns a replacement into a miss, which is the only
 * behaviour that survives a redeploy. Callers with no version keep the bare URL
 * -- unchanged behaviour for anything whose bytes cannot change. */
export function cacheKeyFor(url: string, version?: string | undefined): string {
  return version === undefined || version === "" ? url : `${url}?v=${version}`;
}

/** Drop every other entry for the same url, whatever version it carried: the
 * old bytes can never be wanted again once the page has asked for new ones, and
 * a stale copy of a model this size is a real cost to the visitor's disk. */
async function evictOtherVersions(cache: CacheLike, url: string, keep: string): Promise<void> {
  if (cache.keys === undefined || cache.delete === undefined) {
    return;
  }
  try {
    for (const request of await cache.keys()) {
      const stored = request.url;
      if (stored === keep) {
        continue;
      }
      if (stored === url || stored.startsWith(`${url}?v=`)) {
        await cache.delete(stored);
      }
    }
  } catch {
    // Eviction is best-effort throughout; the visitor has the bytes either way.
  }
}

function defaultStorage(): CacheStorageLike | undefined {
  return typeof caches === "undefined" ? undefined : (caches as unknown as CacheStorageLike);
}

/** Fetch `url`, streaming progress, and serve it from Cache Storage next time.
 *
 * Cache Storage is best-effort throughout: it is unavailable on an insecure
 * origin and in some private-browsing modes, and a quota refusal on `put` is
 * routine for a file this size. None of those may stop the model loading, so
 * every cache interaction is guarded and the download stands on its own. */
export async function loadCachedBytes(
  url: string,
  options: LoadOptions = {},
): Promise<ArrayBuffer> {
  const { onProgress } = options;
  const storage = options.storage ?? defaultStorage();
  const fetchImpl =
    options.fetchImpl ?? ((target: string) => fetch(target) as unknown as Promise<ResponseLike>);
  const cacheName = options.cacheName ?? MODEL_CACHE_NAME;
  const cacheKey = cacheKeyFor(url, options.version);

  let cache: CacheLike | undefined;
  if (storage !== undefined) {
    try {
      cache = await storage.open(cacheName);
    } catch {
      cache = undefined;
    }
  }

  if (cache !== undefined) {
    try {
      const hit = await cache.match(cacheKey);
      if (hit !== undefined && hit.ok) {
        const bytes = await hit.arrayBuffer();
        onProgress?.({
          loaded: bytes.byteLength,
          total: bytes.byteLength,
          fromCache: true,
        });
        return bytes;
      }
    } catch {
      // A corrupt or unreadable entry is not worth reporting; fall through and
      // download, which is what the visitor needs to happen either way.
    }
  }

  const response = await fetchImpl(url);
  if (!response.ok) {
    // Thrown BEFORE any cache write: a 404 body is an error page, and caching
    // it would make the failure permanent for that visitor.
    throw new Error(`${url}: HTTP ${response.status}`);
  }

  const header = response.headers.get("content-length");
  const total = header === null ? 0 : Number(header);
  const reader = response.body?.getReader();

  let bytes: Uint8Array;
  if (reader === undefined) {
    // No streaming body (an older browser, or a mocked response): the download
    // still has to work, it just cannot be reported incrementally.
    bytes = new Uint8Array(await response.arrayBuffer());
    onProgress?.({ loaded: bytes.byteLength, total, fromCache: false });
  } else {
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value === undefined) {
        continue;
      }
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.({ loaded, total, fromCache: false });
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }

  if (cache !== undefined) {
    try {
      // A fresh Response rather than the fetched one: its body has been
      // consumed by the reader above and cannot be put a second time.
      await cache.put(cacheKey, new Response(bytes.buffer as ArrayBuffer));
      await evictOtherVersions(cache, url, cacheKey);
    } catch {
      // Quota refusals are expected at this size. The visitor has the bytes.
    }
  }
  return bytes.buffer as ArrayBuffer;
}
