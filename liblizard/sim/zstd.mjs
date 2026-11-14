// zstd for the default client's pack (sim/pack.mjs; core/tx/pack.h natively) through liblizard/zstd/shim.c, which pins
// the parameters so every client writes the same frames; inside the codec's module (liblizard/build.sh), the instance
// sim/ob.mjs's default build shares. Handed to pack and unpack as `Z`, so a module that never packs loads none of it.
import { baseModule } from "./ob.mjs";

let M;
export async function init() {
  if (!M) M = await baseModule();
  return M;
}

// The frame of `bytes`, or null where the shim refuses (it does not, for a frame the pack makes).
export function compress(bytes) {
  const cap = M._lizard_zstd_bound(bytes.length), p = M._malloc(Math.max(1, bytes.length)), o = M._malloc(cap);
  M.HEAPU8.set(bytes, p);
  const n = M._lizard_zstd_compress(p, bytes.length, o, cap);
  const out = n < 0 ? null : M.HEAPU8.slice(o, o + n);
  M._free(p); M._free(o);
  return out;
}

// The `len` bytes a frame holds, or null where it is not a frame of exactly len bytes.
export function decompress(bytes, len) {
  const p = M._malloc(Math.max(1, bytes.length)), o = M._malloc(Math.max(1, len));
  M.HEAPU8.set(bytes, p);
  const rc = M._lizard_zstd_decompress(p, bytes.length, o, len);
  const out = rc ? null : M.HEAPU8.slice(o, o + len);
  M._free(p); M._free(o);
  return out;
}

// What a frame says it holds (its header is enough), or -1; the bytes the frame at the start of `bytes` takes, or -1.
function frameCall(fn, bytes) {
  const p = M._malloc(Math.max(1, bytes.length));
  M.HEAPU8.set(bytes, p);
  const r = fn(p, bytes.length);
  M._free(p);
  return r;
}
export const frameContent = (bytes) => frameCall(M._lizard_zstd_frame_content, bytes);
export const frameBytes = (bytes) => frameCall(M._lizard_zstd_frame_bytes, bytes);
export const bound = (n) => M._lizard_zstd_bound(n);

// What the pack takes: `Z` for sim/pack.mjs.
export const Z = { compress, decompress, frameContent, frameBytes, bound };
