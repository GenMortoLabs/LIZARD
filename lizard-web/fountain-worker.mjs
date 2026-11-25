// The file's fountain, one per transfer, fed the new blocks of every decode worker. Off the page's thread because a
// solve takes long enough to drop camera frames.
//
// A LIZARD file is a chunked transfer (sim/xfer.mjs): the header comes in the light, each source block is its own
// RaptorQ source block, its chaining value made as it completes, and the file held to its BLAKE3 root at the last. Its blocks wait on disk,
// in the origin private file system (OPFS), where the browser has one: interleaved, every source block is in flight until the
// ai: end of the first cycle, and the file would otherwise be held in memory twice over. The header comes from the light
// ai: alone, and the page's config names only the store (?store=memory).
// ai: Every block names its file by its TOI (src/xfer.h, 2026-11-19; 0 none, a block under it ignored), and the receiver holds
// ai: several transfers at once, a TOI each, from its session header (the one control block): a data block of a
// ai: TOI it holds goes to it; any other waits in the receiver (sim/xfer.mjs HOLD_BLOCKS, 0.5 MB) until a header of its
// ai: TOI is read (it waited here until 2026-11-19). Every file it finishes is filed (offer).
import { init as initRq, Decoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { XferReceiver, MemoryStore, PAYLOAD, ID_BYTES, COMP_ZSTD, sameFile, idGet, idPut } from "../liblizard/sim/xfer.mjs";
import { unpack, unpackWith } from "../liblizard/sim/pack.mjs";

let rx = null, store = null, chain = Promise.resolve(), lastPost = 0, posted = new Set();
const REC = ID_BYTES + PAYLOAD, DIR = "lizard-xfer";

// Blocks and the file on disk, through sync access handles (a dedicated worker's only).
// ai: A folder a transfer under lizard-xfer, its id the root's first 16 hex digits (2026-10-22: kept across page loads,
// ai: liblizard/core/rx/store.h's file store in OPFS): header, c<k> (a source block's stored blocks, removed once the
// ai: source block is verified), out (the file, written at each source block's offset), verified (a source block's
// ai: index and its 32-byte chaining value each, 36 bytes; a folder of 4-byte records from before 2026-11-21 resumes
// ai: with none verified) and meta.json, which Home lists. A header of another root parks the transfer in hand (its
// ai: handles closed, its folder left); its own header read again resumes it, in this page or a later one. Opening the
// ai: store keeps the folders that hold a header and removes everything else (a finished file nobody filed, the layout
// ai: before). A finished transfer leaves its file alone in its folder, which fileAway moves out and removes.
const u32 = (b, o) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
const put32 = (b, o, v) => { b[o] = v & 255; b[o + 1] = (v >>> 8) & 255; b[o + 2] = (v >>> 16) & 255; b[o + 3] = v >>> 24; };
async function readAll(dir, name) {
  try { return new Uint8Array(await (await (await dir.getFileHandle(name)).getFile()).arrayBuffer()); } catch { return null; }
}
async function hasFile(dir, name) { try { await dir.getFileHandle(name); return true; } catch { return false; } }
class OpfsStore {
  static async open() {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    // Proved here, not assumed: a browser can have the directory and refuse the handle (a private window, an old engine).
    const probe = await (await dir.getFileHandle("probe", { create: true })).createSyncAccessHandle();
    probe.close();
    await dir.removeEntry("probe");
    for await (const [name, h] of dir.entries())
      if (h.kind !== "directory" || !(await hasFile(h, "header"))) await dir.removeEntry(name, { recursive: true }).catch(() => {});
    return new OpfsStore(dir);
  }
  // ai: another transfer's store: the same lizard-xfer folder, not cleaned again (sim/xfer.mjs XferReceiver: a store a transfer)
  make() { return new OpfsStore(this.dir); }
  constructor(dir) { this.kind = "opfs"; this.keeps = true; this.dir = dir; this.cur = null; this.id = ""; this.h = new Map(); this.out = null; this.ver = null; this.rec = new Uint8Array(REC); this.id4 = new Uint8Array(4); }
  // ai: a source block's records (c<k>) and its ids (i<k>, 4 bytes a record, 2026-11-15: the native store's, so a source block's
  // ai: distinct symbols are counted from 142 KB, not its 16.8 MB of records), two handles a source block
  async handle(c) {
    let e = this.h.get(c);
    if (!e) {
      e = { h: await (await this.cur.getFileHandle(`c${c}`, { create: true })).createSyncAccessHandle(),
            g: await (await this.cur.getFileHandle(`i${c}`, { create: true })).createSyncAccessHandle(), size: 0 };
      e.h.truncate(0); e.g.truncate(0); this.h.set(c, e);
    }
    return e;
  }
  async sync(name) { const h = await (await this.cur.getFileHandle(name, { create: true })).createSyncAccessHandle(); return { h, size: h.getSize() }; }
  closeAll() {
    for (const e of this.h.values()) { e.h.close(); e.g.close(); }
    this.h.clear();
    for (const k of ["out", "ver"]) { if (this[k]) { this[k].h.flush(); this[k].h.close(); } this[k] = null; }
    this.cur = null; this.id = "";
  }
  async open(id, header, length) {
    this.closeAll();
    let d = await this.dir.getDirectoryHandle(id, { create: true });
    const had = await readAll(d, "header");
    // ai: this file's folder where its header is this file's (sameFile: another sender's name kept as the first)
    const same = had && sameFile(await initOb(), had, header) && (await hasFile(d, "out"));
    if (!same) {
      await this.dir.removeEntry(id, { recursive: true }).catch(() => {});
      d = await this.dir.getDirectoryHandle(id, { create: true });
      this.cur = d; this.id = id;
      this.out = await this.sync("out"); this.out.h.truncate(length);
      const hh = await this.sync("header"); hh.h.write(header, { at: 0 }); hh.h.flush(); hh.h.close();   // ai: last, so a folder with one holds its output
      return { found: false };
    }
    this.cur = d; this.id = id;
    this.out = await this.sync("out");
    const v = (await readAll(d, "verified")) ?? new Uint8Array(0);
    const verified = [], cvs = new Map(), sourceBlocks = [];
    if (v.length % 36 === 0) for (let o = 0; o + 36 <= v.length; o += 36) { const c = u32(v, o); verified.push(c); cvs.set(c, v.slice(o + 4, o + 36)); }
    for await (const [name, h] of d.entries()) {
      if (h.kind !== "file" || !/^c\d+$/.test(name)) continue;
      const c = +name.slice(1), sh = await h.createSyncAccessHandle(), size = sh.getSize() - (sh.getSize() % REC);
      sh.truncate(size);   // ai: a record cut short by a closed page goes, so the next lands on a record's boundary
      const g = await (await d.getFileHandle(`i${c}`, { create: true })).createSyncAccessHandle();
      this.h.set(c, { h: sh, g, size });   // ai: its ids file made again from the records by ids() where it disagrees
      sourceBlocks.push(c);
    }
    return { found: true, verified, cvs, sourceBlocks };
  }
  // ai: the transfer in hand let go whole, its folder too
  async clear() { const id = this.id; this.closeAll(); if (id) await this.dir.removeEntry(id, { recursive: true }).catch(() => {}); }
  async park() { this.closeAll(); }
  // ai: a folder by its id (the root's 16 hex digits and the TOI's 12, sim/xfer.mjs folderOf), or every folder of a root
  // ai: (its hex or its first 16 digits; one named by the 16 alone before 2026-11-20 among them)
  async forget(id) {
    const folder = id.length !== 16 && id.length !== 64, names = [];
    for await (const [name] of this.dir.entries()) if (folder ? name === id : name.startsWith(id.slice(0, 16))) names.push(name);
    for (const name of names) { if (name === this.id) await this.clear(); else await this.dir.removeEntry(name, { recursive: true }).catch(() => {}); }
  }
  async append(c, esi, bytes) {
    const e = await this.handle(c), r = this.rec, i = this.id4;
    idPut(r, 0, esi);
    r.set(bytes.subarray(0, PAYLOAD), ID_BYTES);
    put32(i, 0, esi);
    e.h.write(r, { at: e.size }); e.g.write(i, { at: (e.size / REC) * 4 }); e.size += REC;
  }
  // ai: the records' symbols (i<k>); where the file does not hold exactly one a record (a folder from before it, or a
  // ai: page closed between a record's two writes) made again from the records
  async ids(c) {
    const e = this.h.get(c);
    if (!e) return new Uint32Array(0);
    const n = e.size / REC;
    if (e.g.getSize() === 4 * n) { const b = new Uint8Array(4 * n); e.g.read(b, { at: 0 }); return new Uint32Array(b.buffer); }
    const recs = await this.read(c), v = new Uint32Array(n), b = new Uint8Array(4 * n);
    for (let k = 0; k < n; k++) { v[k] = idGet(recs, k * REC); put32(b, 4 * k, v[k]); }
    e.g.truncate(0); e.g.write(b, { at: 0 });
    return v;
  }
  async read(c) { const e = this.h.get(c), b = new Uint8Array(e?.size ?? 0); if (e) e.h.read(b, { at: 0 }); return b; }
  async drop(c) {
    const e = this.h.get(c); if (!e) return;
    e.h.close(); e.g.close(); this.h.delete(c);
    await this.cur.removeEntry(`c${c}`).catch(() => {}); await this.cur.removeEntry(`i${c}`).catch(() => {});
  }
  async writeOut(off, bytes) { this.out.h.write(bytes, { at: off }); }
  // ai: the output flushed before the source block is noted: a source block noted is a source block on the disk
  async noteVerified(c, cv) {
    this.out.h.flush();
    this.ver ??= await this.sync("verified");
    const r = new Uint8Array(36); put32(r, 0, c); if (cv) r.set(cv, 4);
    this.ver.h.write(r, { at: this.ver.size }); this.ver.size += 36; this.ver.h.flush();
  }
  // ai: a file whose root failed (sim/xfer.mjs complete): nothing verified any more, every source block collected again
  async forgetVerified() {
    this.ver ??= await this.sync("verified");
    this.ver.h.truncate(0); this.ver.size = 0; this.ver.h.flush();
  }
  async noteMeta(json) {
    if (!this.cur) return;
    const h = await (await this.cur.getFileHandle("meta.json", { create: true })).createSyncAccessHandle(), b = ENC.encode(json);
    h.truncate(0); h.write(b, { at: 0 }); h.flush(); h.close();
  }
  // ai: the file whole: what was kept to resume goes, the file left in the folder for fileAway to move
  async finish() {
    const id = this.id, d = this.cur;
    this.out.h.flush(); this.out.h.close(); this.out = null;
    this.closeAll();
    for (const f of ["header", "verified", "meta.json"]) await d.removeEntry(f).catch(() => {});
    return { opfs: [DIR, id, "out"] };
  }
  held() { return 0; }
}

async function chunked(c) {
  const [M] = await Promise.all([initOb(), initRq(), initZstd()]);
  if (c.store === "memory" && store?.kind !== "memory") { store = new MemoryStore(); store.why = "asked for (?store=memory)"; rx = null; }
  if (!store) try { store = await OpfsStore.open(); } catch (e) { store = new MemoryStore(); store.why = String(e?.message ?? e); }
  rx ??= new XferReceiver({ M, Decoder, store });
  rx.onSolve = () => { report(false); };   // ai: the page sees a solve begin (the transfer's last source blocks: Decoding)
  await rx.reset(true);   // ai: the transfers held parked, as before a new page (2026-10-22; let go before)
}

// What the page shows and reports: the receiver's progress, and what this worker holds, since holding little is the
// point of the store.
async function report(force) {
  const now = performance.now();
  if (!force && now - lastPost < 250) return;
  lastPost = now;
  const M = await initOb();
  postMessage({ ...rx.progress(), type: "xfer", store: store.kind, storeWhy: store.why, mem: { codecHeap: M.HEAPU8.length, storeBytes: store.held() } });
}

// Once a file, the moment its last source block verifies (every file the receiver holds: several on air, 2026-11-19). A
// finished receiver still reads control blocks: a header with other bytes is the next transfer.
// ai: On disk, the finished file is filed into the received files first (fileAway), and the page told where it is.
// ai: A file the sender's client packed (sim/pack.mjs: its description's compression 1, zstd frames, its name <name>.zst)
// ai: is unpacked as it is filed (fileAway), or here in memory, under its name less the .zst; one that does not unpack
// ai: is offered as it came, the .zst kept. `sent` the bytes the
// ai: light carried, the transfer's.
async function offer() {
  const fresh = rx.results().filter((r) => !posted.has(r));
  await report(rx.done || fresh.length > 0);
  for (const res of fresh) {
    posted.add(res);
    let r = { ...res, sent: res.length };
    if (r.bytes && r.compression === COMP_ZSTD)
      try { const bytes = await unpack(r.bytes, Z); r = { ...r, bytes, length: bytes.length, name: plainName(r.name) }; }
      catch (e) { console.warn("fountain: the packed file did not unpack:", e); }
    // ai: a filing that fails (a full disk) leaves the file where it was, offered from there as before 2026-10-01
    const lib = r.opfs ? await fileAway(r).catch((e) => { console.warn("fountain: the file was not kept:", e); return null; }) : null;
    if (lib) r = { ...r, opfs: null, lib, name: lib.name, length: lib.size };
    postMessage({ ...r, type: "file", ok: true, chunked: true });
  }
}

// ai: The received files (2026-10-01, lizard-web/library.mjs, which reads them): lizard-files/<id>/data and its meta.json, the
// ai: id the root's first 16 hex digits, so the same file received again replaces its entry. The finished file leaves
// ai: its transfer's folder under lizard-xfer (OpfsStore) by a move where the browser has one, else a copy through sync
// ai: access handles, and the folder goes after it: nothing of the transfer is left behind. A packed file (sim/pack.mjs)
// ai: is unpacked into data instead, under its name less the .zst (one that does not unpack is moved as it came).
const LIB = "lizard-files", ENC = new TextEncoder();
// ai: a packed file's name back: <name>.zst less its .zst (sim/pack.mjs names it so), another as it is
const plainName = (n) => (n.length > 4 && n.endsWith(".zst") ? n.slice(0, -4) : n);
async function fileAway(r) {
  const path = r.opfs, name = path.at(-1), root = await navigator.storage.getDirectory();
  let from = root;
  for (const d of path.slice(0, -1)) from = await from.getDirectoryHandle(d);
  // ai: its folder the root's first 16 hex digits, -2 on where one is taken: a file received again is a second entry
  // ai: (2026-11-20, a download clicked twice; it replaced the first before), as the apps' kept/ (session.cpp)
  const lib = await root.getDirectoryHandle(LIB, { create: true });
  let id = r.root.slice(0, 16);
  for (let n = 2; await lib.getDirectoryHandle(id).then(() => true, () => false); n++) id = `${r.root.slice(0, 16)}-${n}`;
  const d = await lib.getDirectoryHandle(id, { create: true }), src = await from.getFileHandle(name);
  let moved = false, size = r.length;
  let fileName = r.name || "received.bin";
  if (r.compression === COMP_ZSTD) {
    const a = await src.createSyncAccessHandle(), b = await (await d.getFileHandle("data", { create: true })).createSyncAccessHandle();
    let out = 0;
    try {
      b.truncate(0);
      await unpackWith(a.getSize(), (at, n) => { const buf = new Uint8Array(n); a.read(buf, { at }); return buf; }, (p) => { b.write(p, { at: out }); out += p.length; }, Z);
      b.flush();
      moved = true; size = out; fileName = plainName(fileName);
    } catch (e) { console.warn("fountain: the packed file did not unpack:", e); b.truncate(0); }
    a.close(); b.close();
    if (moved) await from.removeEntry(name);
  }
  if (!moved && typeof src.move === "function") try { await src.move(d, "data"); moved = true; } catch {}
  if (!moved) {
    const a = await src.createSyncAccessHandle(), b = await (await d.getFileHandle("data", { create: true })).createSyncAccessHandle(), buf = new Uint8Array(1 << 20);
    b.truncate(0);
    for (let at = 0, n; (n = a.read(buf, { at })) > 0; at += n) b.write(buf.subarray(0, n), { at });
    b.flush(); b.close(); a.close();
    await from.removeEntry(name);
  }
  // ai: the transfer's folder, empty now (OpfsStore.finish took the rest)
  if (path.length > 2) { let parent = root; for (const d of path.slice(0, -2)) parent = await parent.getDirectoryHandle(d); await parent.removeEntry(path.at(-2), { recursive: true }).catch(() => {}); }
  const meta = { name: fileName, size, root: r.root, at: Date.now() };
  const m = await (await d.getFileHandle("meta.json", { create: true })).createSyncAccessHandle(), bytes = ENC.encode(JSON.stringify(meta));
  m.truncate(0); m.write(bytes, { at: 0 }); m.flush(); m.close();
  if (typeof BroadcastChannel === "function") { const bc = new BroadcastChannel(LIB); bc.postMessage("changed"); bc.close(); }
  return { ...meta, id };
}

async function blocks({ ids, tois, bytes }) {
  if (!rx) return;
  const all = new Uint8Array(bytes);
  for (let k = 0; k < ids.length; k++) await rx.add(ids[k], tois[k], all.subarray(k * PAYLOAD, (k + 1) * PAYLOAD));
  return offer();
}

// ai: A config starts a new transfer: what was held goes. { type: "config", config: { store } }. A finished file was
// ai: filed in the received files (fileAway) and stays there; the pages delete those (lizard-web/library.mjs). The
// ai: "clear" message, the receiver's Receive again, went with that button on 2026-10-05.
async function handle(m) {
  // ai: the page's, once its camera stopped for a file in: every finished transfer's TOIs let go (sim/xfer.mjs
  // ai: releaseDone), so the same file at a later Start is received again (2026-11-20)
  if (m.type === "release") return rx?.releaseDone();
  if (m.type !== "config") return blocks(m);
  await chunked(m.config);
  await offer();
}
onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", message: String(e?.message ?? e) }));
};
// ai: Home's Delete of an unfinished transfer whose files this worker holds open (library.mjs forgetUnfinished):
// ai: forgotten in turn with the blocks, then the received files' channel told, so Home draws again.
if (typeof BroadcastChannel === "function") new BroadcastChannel("lizard-xfer").onmessage = ({ data }) => {
  if (typeof data?.forget !== "string") return;
  chain = chain.then(async () => { if (rx) await rx.forget(data.forget); const bc = new BroadcastChannel(LIB); bc.postMessage("changed"); bc.close(); }).catch(() => {});
};
