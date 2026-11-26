// ai: The received files (2026-10-01): one folder a file in the
// ai: origin private file system, lizard-files/<id>/ holding the file as `data` and its meta.json ({ name, size,
// ai: root, at }), the id the first 16 hex digits of its BLAKE3 root (-2 on for the same file received again: a second
// ai: entry, as a download clicked twice, since 2026-11-20; it replaced the entry before). The
// ai: receiver's engine files a finished transfer there (engine-worker.mjs, bindings/wasm/engine.cpp lizw_file_away); the pages list, open and delete
// ai: here. A change is announced on a BroadcastChannel, so an open Home sees a file the receiver just filed.
// ai: Without OPFS (a private window, ?store=memory) nothing is kept: the receiver offers its file until the page closes.
export const DIR = "lizard-files";
const CHANNEL = "lizard-files";
const bc = typeof BroadcastChannel === "function" ? new BroadcastChannel(CHANNEL) : null;

async function folder(create = false) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create });
}

// ai: Whether this browser keeps files at all (OPFS reachable from a page).
export async function keeps() {
  try { await navigator.storage.getDirectory(); return true; } catch { return false; }
}

// ai: Every kept file, newest first: { id, name, size, root, at }. A folder whose meta.json does not read (a
// ai: filing cut short) is left out.
export async function list() {
  let d;
  try { d = await folder(); } catch { return []; }
  const out = [];
  for await (const [id, h] of d.entries()) {
    if (h.kind !== "directory") continue;
    try {
      const meta = JSON.parse(await (await (await h.getFileHandle("meta.json")).getFile()).text());
      out.push({ ...meta, id });
    } catch {}
  }
  return out.sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
}

// ai: A received file's type for the browser, from its name's extension (the transfer carries no media type since
// ai: 2026-11-22): the kinds a share sheet takes (Chromium's Web Share list), "" for any other.
const TYPES = { avif: "image/avif", bmp: "image/bmp", css: "text/css", csv: "text/csv", flac: "audio/flac", gif: "image/gif",
  htm: "text/html", html: "text/html", ico: "image/x-icon", jfif: "image/jpeg", jpeg: "image/jpeg", jpg: "image/jpeg",
  m4a: "audio/x-m4a", m4v: "video/mp4", mp3: "audio/mpeg", mp4: "video/mp4", mpeg: "video/mpeg", mpg: "video/mpeg",
  oga: "audio/ogg", ogg: "audio/ogg", ogv: "video/ogg", opus: "audio/ogg", pdf: "application/pdf", png: "image/png",
  svg: "image/svg+xml", text: "text/plain", tif: "image/tiff", tiff: "image/tiff", txt: "text/plain", wav: "audio/wav",
  weba: "audio/webm", webm: "video/webm", webp: "image/webp" };
export const typeOf = (name) => { const d = name.lastIndexOf("."); return d > 0 ? TYPES[name.slice(d + 1).toLowerCase()] ?? "" : ""; };

// ai: The file of an entry, under its own name (the bytes stay on disk: a File over the OPFS one).
export async function open(entry) {
  const d = await (await folder()).getDirectoryHandle(entry.id);
  const f = await (await d.getFileHandle("data")).getFile();
  const name = entry.name || "received.bin";
  return new File([f], name, { type: typeOf(name) });
}

export async function remove(id) {
  await (await folder()).removeEntry(id, { recursive: true });
  changed();
}
export async function removeAll() {
  await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }).catch(() => {});
  changed();
}

export function changed() { bc?.postMessage("changed"); }
// ai: cb on a change from another page (or the fountain worker), and whenever this page comes back into view.
export function watch(cb) {
  bc?.addEventListener("message", () => cb());
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") cb(); });
}

// ai: Transfers begun and not finished (2026-10-22): the receiver's folders under lizard-xfer (core/rx/store.h's file store on OPFS,
// ai: OpfsStore), one a transfer by its root's first 16 hex digits and its TOI, each with meta.json ({ name, length, source blocks,
// ai: verified, root, fraction, bytesIn }); a receiver resumes one when its file is in the light again. Newest
// ai: first: { ...meta, id, at }, at when its meta.json was last written.
export const XFER = "lizard-xfer";
const XFER_CHANNEL = "lizard-xfer";
export async function unfinished() {
  let d;
  try { d = await (await navigator.storage.getDirectory()).getDirectoryHandle(XFER); } catch { return []; }
  const out = [];
  for await (const [id, h] of d.entries()) {
    if (h.kind !== "directory") continue;
    try {
      await h.getFileHandle("header");
      const f = await (await h.getFileHandle("meta.json")).getFile();
      out.push({ ...JSON.parse(await f.text()), id, at: f.lastModified });
    } catch {}
  }
  return out.sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
}
// ai: An unfinished transfer let go: its folder removed, or, where an open receiver holds its files (OPFS refuses the
// ai: removal then), that receiver's engine asked to forget it (engine-worker.mjs listens on the channel).
export async function forgetUnfinished(id) {
  try { await (await (await navigator.storage.getDirectory()).getDirectoryHandle(XFER)).removeEntry(id, { recursive: true }); }
  catch (e) { if (e?.name === "NotFoundError") return; if (typeof BroadcastChannel !== "function") throw e; const bc = new BroadcastChannel(XFER_CHANNEL); bc.postMessage({ forget: id }); bc.close(); }
  changed();
}
