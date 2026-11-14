// The default client's compression (header version 4, 2026-11-14: the transfer carries a file as it is and compresses
// nothing; how a client compresses is the client's): the web's twin of liblizard/core/tx/pack.h, byte for byte. A file
// goes as a zstd stream of independent frames, each of PACK_FRAME bytes of the file (liblizard/zstd/shim.h, the level
// and parameters pinned, so every client packs a file to the same bytes and the same root), named <name>.zst and typed
// `application/zstd; original="<type>"` (the original's `"` and `\` escaped by a `\`), where it pays: a sample of the
// file (three slices of 1 MiB, its start, middle and end; the whole file under 3 MiB) compressed to 90% or less, and
// the whole stream to 97% or less; else the file goes as it is. Any zstd reads the stream; a LIZARD receiver unpacks
// a file of that type and files it under its own name and type. `Z` is sim/zstd.mjs's, its init() already awaited.

export const PACK_FRAME = 2 ** 22, PACK_FRAME_MAX = 2 ** 26;
const PREFIX = 'application/zstd; original="', SAMPLE = 2 ** 20, HEAD = 18, SAMPLE_MAX = 0.9, WHOLE_MAX = 0.97;
const NAME_MAX = 255, TYPE_MAX = 158;

// The media type a packed file carries for a file of `type`; "" where it would pass the header's 158 bytes.
export function packType(type) {
  let t = PREFIX;
  for (const c of type) { if (c === '"' || c === "\\") t += "\\"; t += c; }
  t += '"';
  return t.length <= TYPE_MAX ? t : "";
}
// A packed file's original type, or null where `type` is not a packed file's.
export function packedType(type) {
  if (typeof type !== "string" || type.length <= PREFIX.length || !type.startsWith(PREFIX) || !type.endsWith('"')) return null;
  let o = "";
  for (let i = PREFIX.length; i + 1 < type.length; i++) {
    let c = type[i];
    if (c === '"') return null;
    if (c === "\\") { if (i + 2 >= type.length) return null; c = type[++i]; }
    o += c;
  }
  return o;
}
// A packed file's name back (<name>.zst less its ".zst"; another name as it is).
export const unpackedName = (name) => (name.length > 4 && name.endsWith(".zst") ? name.slice(0, -4) : name);
// UTF-8 cut to n bytes at a character's start
function cutName(name, n) {
  const b = new TextEncoder().encode(name);
  n = Math.min(n, b.length);
  while (n < b.length && n > 0 && (b[n] & 0xc0) === 0x80) n--;
  return new TextDecoder().decode(b.subarray(0, n));
}

// The file's bytes, its name and media type as the caller would send it: { packed, bytes, name, type }, packed where
// that pays, else the file as it is.
export function pack(bytes, name, type, Z) {
  const as = { packed: false, bytes, name, type };
  const L = bytes.length, pt = packType(/^[\x20-\x7e]*$/.test(type ?? "") ? type ?? "" : "");
  if (!L || !pt) return as;
  let s = bytes;
  if (L > 3 * SAMPLE) {
    s = new Uint8Array(3 * SAMPLE);
    [0, Math.floor(L / 2) - SAMPLE / 2, L - SAMPLE].forEach((at, k) => s.set(bytes.subarray(at, at + SAMPLE), k * SAMPLE));
  }
  const frame = (b) => { const f = Z.compress(b); if (!f) throw new Error("zstd refused a frame"); return f; };
  if (frame(s).length > SAMPLE_MAX * s.length) return as;
  const frames = [];
  let size = 0;
  for (let at = 0; at < L; at += PACK_FRAME) {
    const f = frame(bytes.subarray(at, Math.min(L, at + PACK_FRAME)));
    frames.push(f);
    size += f.length;
    if (size > WHOLE_MAX * L) return as;   // past the bar already
  }
  const out = new Uint8Array(size);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return { packed: true, bytes: out, name: name ? cutName(name, NAME_MAX - 4) + ".zst" : name, type: pt };
}

// A packed stream of `length` bytes unpacked: read(at, n) gives its bytes (n or fewer at its end), write(bytes) takes
// each frame's in order (either may be async). Throws where a frame does not say its bytes, holds more than
// PACK_FRAME_MAX, or is damaged or cut short.
export async function unpackWith(length, read, write, Z) {
  for (let at = 0; at < length;) {
    const c = Z.frameContent(await read(at, Math.min(HEAD, length - at)));
    if (c < 0 || c > PACK_FRAME_MAX) throw new Error("a frame that does not say its bytes, or holds more than 64 MiB");
    const b = await read(at, Math.min(Z.bound(c), length - at)), fb = Z.frameBytes(b);
    const plain = fb < 0 ? null : Z.decompress(b.subarray(0, fb), c);
    if (!plain) throw new Error("a frame damaged or cut short");
    await write(plain);
    at += fb;
  }
}
// The same for a stream in memory: the file's bytes.
export async function unpack(bytes, Z) {
  const parts = [];
  let n = 0;
  await unpackWith(bytes.length, (at, k) => bytes.subarray(at, at + k), (p) => { parts.push(p); n += p.length; }, Z);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
