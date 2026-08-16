/** Which graph the page runs, and which bytes those are.
 *
 * The digest is here for a specific bug rather than for decoration. Cache
 * Storage is keyed by request and bypasses HTTP freshness entirely, so a model
 * replaced at the same path is invisible to every returning visitor: they keep
 * running the old graph forever, and the page keeps claiming the accuracy of
 * the new one. Keying the cache on the CONTENT rather than the path makes a
 * replacement a miss, which is the behaviour a redeploy needs.
 *
 * `tests/test_web_model_asset.py` recomputes both values from the committed
 * file, so this cannot drift into a lie about bytes that are right there. */

/** Relative to the page, so the site works from a project subpath on Pages. */
export const MODEL_URL = "models/yolo11n-480.onnx";

/** sha256 of `web/public/models/yolo11n-480.onnx`. */
export const MODEL_SHA256 =
  "d7b9cad1308554bbd644931a73a2a4762f0a6c3e67038fb1c17b339e882f698a";

export const MODEL_BYTES = 10667823;

/** The square input the graph was exported at. NOT
 * `DETECT_DEFAULT_INPUT_SIZE`, which is 640: that is what the PYTHON engine
 * runs, and feeding 640 to this graph is an immediate shape error. */
export const MODEL_INPUT_SIZE = 480;

/** Enough of the digest to key a cache and still be readable in devtools. */
export const MODEL_CONTENT_VERSION = MODEL_SHA256.slice(0, 16);
