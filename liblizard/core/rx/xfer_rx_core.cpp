// ai: XferRxCore and XferJudge (xfer_rx_core.h): the web's transfer chain ported, with no thread of its own (split out of
// ai: xfer_rx.cpp 2026-10-03, every line of the chain as it was). Rx is sim/xfer.mjs XferReceiver and
// ai: lizard-web/fountain-worker.mjs take; the judge is sim/phy.mjs blockJudge and lizard-web/recv.mjs take's dedupe. Each function
// ai: names the JS it ports; where the port differs, an ai: comment says so and why.
#include "xfer_rx_core.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

extern "C" {
#include "shake.h"
#include "xfer.h"
// ai: liblizard/wirehair/shim.cpp, the web's own ABI to Wirehair (the profile pinned there), compiled natively
int lizard_wh_init(void);
int lizard_wh_decoder_create(uint32_t message_bytes, uint32_t block_bytes, uint32_t seed_attempt, void** codec_out);
int lizard_wh_decode(void* codec, uint32_t block_id, const void* data, uint32_t bytes);
int lizard_wh_recover(void* codec, void* out, uint32_t capacity);
void lizard_wh_free(void* codec);
}

namespace lizard {

static constexpr uint32_t PAYLOAD = XFER_PAYLOAD, REC = XFER_BLOCK;
// ai: lizard-web/fountain-worker.mjs LIVE_BLOCKS, HOLD_BLOCKS; lizard-web/recv.mjs SEEN_KEEP
static constexpr int64_t LIVE_BLOCKS = 4096, NEVER = INT64_MAX / 2;
static constexpr size_t HOLD_BLOCKS = 1024, SEEN_KEEP = 1 << 16;
// ai: meta.json written again every this many blocks stored (about 1 MB) or a second of storing, whichever comes first,
// ai: as well as at each chunk verified and at a park, so a receiver stopped mid-chunk leaves an app's list a line at
// ai: most that far behind
static constexpr uint32_t META_EVERY = 2048;
static constexpr double META_MS = 1000;
enum { COLLECTING, HOT, VERIFIED };

static std::string hexOf(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
  return s;
}
static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// ai: a JSON string: quotes, backslashes and control bytes escaped, UTF-8 passed through
static std::string jstr(const std::string& v) {
  std::string o = "\"";
  for (unsigned char ch : v) {
    if (ch == '"' || ch == '\\') { o += '\\'; o += char(ch); }
    else if (ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
    else o += char(ch);
  }
  return o + "\"";
}

struct Bits {
  // ai: a chunk's symbols in dense runs of bits, each from its own first symbol (sim/xfer.mjs Bits, 2026-11-14): a send
  // ai: counts a chunk's symbols on from a random start, so a send's are one run and a resumed transfer's sends a run
  // ai: each; a symbol within NEAR of a run joins it (a frame's slots come shuffled)
  static constexpr uint32_t NEAR = 4096;
  struct Run { uint32_t base; std::vector<uint8_t> a; };
  std::vector<Run> runs;
  uint32_t n = 0;
  bool has(uint32_t i) const {
    for (const auto& r : runs) { const uint32_t d = i - r.base; if (i >= r.base && d < 8 * r.a.size() && ((r.a[d >> 3] >> (d & 7)) & 1)) return true; }
    return false;
  }
  void add(uint32_t i) {
    if (has(i)) return;
    Run* r = nullptr;
    for (auto& q : runs) if (i + NEAR >= q.base && i < q.base + 8 * q.a.size() + NEAR) { r = &q; break; }
    if (!r) { runs.push_back({i & ~7u, std::vector<uint8_t>(64)}); r = &runs.back(); }
    if (i < r->base) { const uint32_t nb = i & ~7u; r->a.insert(r->a.begin(), (r->base - nb) / 8, uint8_t(0)); r->base = nb; }
    const uint32_t d = i - r->base;
    if ((d >> 3) >= r->a.size()) r->a.resize(std::max(2 * r->a.size(), size_t(d >> 3) + 1));
    r->a[d >> 3] |= uint8_t(1 << (d & 7)); n++;
  }
  void orWith(const Bits& o) {
    for (const auto& r : o.runs) for (size_t i = 0; i < r.a.size(); i++) if (r.a[i]) for (int b = 0; b < 8; b++) if ((r.a[i] >> b) & 1) add(r.base + uint32_t(8 * i + b));
  }
};

// ai: sim/fountain.mjs Decoder over the shim
struct Wh {
  void* h = nullptr;
  uint32_t bytes = 0, lastId = 0;
  ~Wh() { if (h) lizard_wh_free(h); }
  int create(uint32_t messageBytes, uint32_t seed) {
    bytes = messageBytes;
    lastId = (messageBytes + PAYLOAD - 1) / PAYLOAD - 1;
    return lizard_wh_decoder_create(messageBytes, PAYLOAD, seed, &h);
  }
  // ai: 1 once enough blocks are in, 0 for more, negative on Wirehair's refusal; the last source block travels padded
  // ai: and Wirehair wants its true length
  int add(uint32_t id, const uint8_t* p) { return lizard_wh_decode(h, id, p, id == lastId ? bytes - lastId * PAYLOAD : PAYLOAD); }
  std::vector<uint8_t> recover() {
    std::vector<uint8_t> o(bytes);
    const int n = lizard_wh_recover(h, o.data(), bytes);
    o.resize(n < 0 ? 0 : size_t(n));
    return o;
  }
};

struct XferRxCore::Rx {
  // ai: recs: the blocks the store holds for the chunk; mark: how many it held when this transfer's header was last
  // ai: read. The blocks after the mark are provisional: a sender that moved to another file sends that file's blocks
  // ai: under the same ids until its header is read, so a park cuts each chunk back to its mark.
  struct Ch { Bits have, ban; int state = COLLECTING; std::unique_ptr<Wh> dec; uint32_t recs = 0, mark = 0; };
  struct Hdr {
    std::array<uint8_t, XFER_PAYLOAD> bytes;
    xfer_header_t h;
    std::string name, type, root;
    uint64_t size = 0;
    uint32_t mcount = 0, mhave = 0;
    // ai: len and K: each chunk's bytes and blocks, the header's (version 4: a chunk goes as its own bytes, its seed
    // ai: attempt the header's, xfer_chunk_seed); a chunk is solved once the list checks (one chunk: at once)
    std::vector<uint32_t> K, len;
    std::vector<uint8_t> mgot, list;
    bool listOk = false;
    bool ready() const { return h.chunks == 1 || listOk; }
  };
  std::unique_ptr<Store> store;
  std::function<void(const std::string&)> log;
  std::atomic<uint64_t> gen{0};   // ai: headers of another root read (a judge's dedupe clears on a change)
  std::atomic<uint64_t> taken{0};   // ai: XferRxCore::taken: data blocks added to a chunk still collecting, ever
  std::string lastRoot;   // ai: lizard-web/recv.mjs xferRoot: the root the dedupe was last cleared for
  std::unique_ptr<Hdr> hdr;
  // ai: ordered where the JS Map is in insertion order: only the order of solves within one block's take differs
  std::map<uint32_t, Ch> ch;
  std::map<uint32_t, std::array<uint8_t, XFER_PAYLOAD>> raw;
  uint32_t verified = 0, rejected = 0, manifestRejected = 0, refused = 0, over = 0;
  double ms = 0;
  double firstMs = 0, doneMs = 0;   // ai: XferProgress.secs: this transfer's first data block taken, its finish (nowMs)
  bool done = false;
  // ai: the store could not keep this transfer (store.h: over the memory cap, a write refused): it ends here, never
  // ai: done, its blocks no longer taken; the store's error says why. A new header begins afresh.
  bool failed = false;
  void stop() { if (!failed) say("xfer: " + (hdr ? hdr->name : std::string("the transfer")) + " stopped: " + store->error); failed = true; }
  std::string path;
  // ai: a folder resumed is replayed into this transfer (its manifest blocks are not noted again)
  bool replaying = false;
  uint32_t sinceMeta = 0;   // ai: blocks stored since meta.json was last written, and when (nowMs)
  double metaMs = 0;
  // ai: the fountain worker's
  int64_t sinceHeader = NEVER;
  std::vector<std::pair<uint32_t, std::array<uint8_t, XFER_PAYLOAD>>> held;
  size_t heldAt = 0;
  uint64_t forwarded = 0, doneAt = 0;

  Rx(std::unique_ptr<Store> s, std::function<void(const std::string&)> l) : store(std::move(s)), log(std::move(l)) {
    store->log = log;
    static const int wh = lizard_wh_init();
    if (wh) say("xfer: wirehair init failed");
  }
  void say(const std::string& s) { if (log) log(s); }

  // ai: lizard-web/fountain-worker.mjs chunked (a config) and handle's clear: a new receiver, the held blocks let go, and on
  // ai: a clear the finished files too (refused kept, as the web's receiver keeps it)
  void clearAll() {
    reset(false);
    store->removeAll();
    lastRoot.clear();
    sinceHeader = NEVER; held.clear(); heldAt = 0; forwarded = 0; doneAt = 0;
  }

  // ai: lizard-web/fountain-worker.mjs take
  void block(const uint8_t* b) {
    forwarded++;
    const uint32_t id = xfer_id_get(b);
    const uint8_t* p = b + XFER_ID_BYTES;
    if (xfer_id_chunk(id) == XFER_CONTROL) {
      add(id, p);
      if (id == XFER_ID_HEADER && hdr) { sinceHeader = 0; if (!held.empty()) release(); }
    } else if (hdr && sinceHeader++ < LIVE_BLOCKS) {
      add(id, p);
    } else {
      std::pair<uint32_t, std::array<uint8_t, XFER_PAYLOAD>> r{id, {}};
      memcpy(r.second.data(), p, PAYLOAD);
      if (held.size() < HOLD_BLOCKS) held.push_back(r); else { held[heldAt] = r; heldAt = (heldAt + 1) % HOLD_BLOCKS; }
    }
    if (done && !doneAt) doneAt = forwarded;
  }
  // ai: the held blocks, oldest first, into the transfer whose header was just read
  void release() {
    auto ring = std::move(held);
    held.clear();
    const size_t at = ring.size() < HOLD_BLOCKS ? 0 : heldAt;
    heldAt = 0;
    for (size_t k = 0; k < ring.size(); k++) { auto& r = ring[(at + k) % ring.size()]; add(r.first, r.second.data()); }
  }

  // ai: XferReceiver.reset (refused is kept, as there): the transfer in hand parked for its header to come again
  // ai: (keep; a store that keeps no transfers lets it go), or let go whole
  void reset(bool keep) {
    if (keep && hdr && !done && !failed && store->keeps()) {
      for (auto& [c, s] : ch) if (s.state != VERIFIED && s.recs > s.mark) { store->truncate(c, s.mark); s.recs = s.mark; }
      saveMeta();
    }
    ch.clear(); raw.clear(); hdr.reset();
    verified = rejected = manifestRejected = over = 0;
    ms = 0; firstMs = doneMs = 0; done = false; failed = false; path.clear(); sinceMeta = 0;
    if (keep) store->park(); else store->clear();
  }
  // ai: a transfer let go by its root (an app's list's Delete): the one in hand, or a parked one
  void forget(const std::string& root) {
    const std::string id = root.substr(0, 16);
    if (hdr && hdr->root.compare(0, id.size(), id) == 0) { reset(false); return; }
    store->forget(id);
  }
  // ai: the transfer's line for an app's list of unfinished transfers (store.h noteMeta)
  void saveMeta() {
    sinceMeta = 0; metaMs = nowMs();
    if (!hdr || !store->keeps()) return;
    XferProgress p;
    progress(p);
    char f[32];
    snprintf(f, sizeof f, "%.4f", p.fraction);
    store->noteMeta("{\"name\":" + jstr(p.name) + ",\"type\":" + jstr(p.type) + ",\"length\":" + std::to_string(p.length) +
                    ",\"chunks\":" + std::to_string(p.chunks) + ",\"verified\":" + std::to_string(p.verified) + ",\"root\":\"" + p.root +
                    "\",\"fraction\":" + f + ",\"bytesIn\":" + std::to_string(p.bytesIn) + "}");
  }

  Ch& state(uint32_t c) { return ch[c]; }

  // ai: XferReceiver.add
  void add(uint32_t id, const uint8_t* p) {
    const uint32_t c = xfer_id_chunk(id), sym = xfer_id_symbol(id);
    if (c == XFER_CONTROL) {
      if (sym == 0) return header(p);
      if (sym <= XFER_MAX_MANIFEST) return manifestBlock(sym - 1, p);
      return;   // ai: reserved: this version ignores it
    }
    // ai: a store that keeps transfers keeps them by their header's folder: no data block before a header (block()
    // ai: holds those for the next header itself)
    if (done || failed || (hdr && c >= hdr->h.chunks) || (!hdr && store->keeps())) return;
    if (!firstMs) firstMs = nowMs();
    Ch& s = state(c);
    if (s.state == VERIFIED || s.have.has(sym) || s.ban.has(sym)) return;
    s.have.add(sym);
    taken.fetch_add(1, std::memory_order_relaxed);
    if (s.dec) {
      // ai: a store that keeps the transfer keeps this chunk's blocks too, so a resume rebuilds the decoder
      if (store->keeps()) { if (!store->append(c, sym, p)) return stop(); s.recs++; }
      const int rc = s.dec->add(sym, p);
      if (rc < 0) say("xfer: wirehair decode " + std::to_string(rc) + " on chunk " + std::to_string(c));
      if (rc == 1) recovered(c);
      return;
    }
    if (!store->append(c, sym, p)) return stop();
    s.recs++;
    if (++sinceMeta >= META_EVERY || nowMs() - metaMs >= META_MS) saveMeta();
    if (hdr && hdr->ready() && s.have.n >= hdr->K[c]) solve(c);   // ai: once the list checks (one chunk: at once)
  }

  // ai: XferReceiver.header, the light's only: a different header is a new transfer, and everything held is let go
  void header(const uint8_t* p) {
    if (hdr && !memcmp(p, hdr->bytes.data(), PAYLOAD)) { for (auto& [c, s] : ch) s.mark = s.recs; return; }
    auto h = std::make_unique<Hdr>();
    if (xfer_header_parse(p, &h->h)) { refused++; return; }   // ai: not a header this receiver reads
    if (hdr) reset(!failed);
    memcpy(h->bytes.data(), p, PAYLOAD);
    h->name.assign((const char*)h->h.name, h->h.name_len);
    h->type.assign((const char*)h->h.type, h->h.type_len);
    h->root = hexOf(h->h.root, XFER_CV);
    h->size = uint64_t(1) << h->h.chunk_log2;
    const uint32_t chunks = h->h.chunks;
    h->mcount = uint32_t(xfer_manifest_blocks(chunks));
    h->mgot.assign(h->mcount, 0);
    h->list.assign(std::max<size_t>(1, size_t(XFER_CV) * chunks), 0);
    h->K.resize(chunks); h->len.resize(chunks);
    for (uint32_t c = 0; c < chunks; c++) { h->len[c] = uint32_t(xfer_chunk_len(&h->h, c)); h->K[c] = xfer_blocks(h->len[c]); }
    // ai: lizard-web/recv.mjs: a header of another root clears the page's dedupe (the producer's, told by the generation)
    if (!lastRoot.empty() && lastRoot != h->root) gen.fetch_add(1);
    lastRoot = h->root;
    hdr = std::move(h);
    say("xfer: header " + hdr->name + ", " + std::to_string(hdr->h.length) + " B, " + std::to_string(chunks) + " chunks, root " + hdr->root);
    for (auto it = ch.begin(); it != ch.end();) {   // ai: blocks of another transfer
      if (it->first >= chunks) { store->drop(it->first); it = ch.erase(it); } else ++it;
    }
    Store::Before before;
    if (!store->open(hdr->root.substr(0, 16), p, hdr->h.length, before)) return stop();
    if (!chunks) {   // ai: an empty file: its root is BLAKE3 of nothing, which is all there is to check
      uint8_t e[XFER_CV];
      xfer_b3_hash(nullptr, 0, e);
      if (!memcmp(e, hdr->h.root, XFER_CV)) finish(); else refused++;
      return;
    }
    if (before.found && resume(before)) return;
    auto pending = std::move(raw);
    raw.clear();
    for (auto& [m, b] : pending) manifestBlock(m, b.data());
    saveMeta();
    if (hdr && hdr->ready()) solveReady();
  }
  // ai: A folder of this transfer from before (store.h): its manifest blocks replayed, its verified chunks taken as
  // ai: verified, and each collecting chunk's symbols marked had from its stored blocks; true when that finished it
  bool resume(const Store::Before& b) {
    replaying = true;
    for (auto& [m, pay] : b.manifest) if (pay.size() == PAYLOAD) manifestBlock(m, pay.data());
    replaying = false;
    const uint32_t chunks = hdr->h.chunks;
    for (uint32_t c : b.verified) if (c < chunks && ch[c].state != VERIFIED) { ch[c].state = VERIFIED; verified++; }
    uint64_t had = 0;
    for (uint32_t c : b.chunks) {
      if (c >= chunks || ch[c].state == VERIFIED) { store->drop(c); continue; }
      Ch& s = ch[c];
      const auto recs = store->read(c);
      for (size_t o = 0; o + REC <= recs.size(); o += REC) s.have.add(xfer_id_get(recs.data() + o));
      s.recs = s.mark = uint32_t(recs.size() / REC);
      had += s.have.n;
    }
    say("xfer: resumed " + hdr->name + ", " + std::to_string(verified) + " of " + std::to_string(chunks) + " chunks verified, " +
        std::to_string(had) + " blocks in hand");
    if (verified == chunks) { finish(); return true; }
    return false;
  }
  // ai: XferReceiver.solveReady: every chunk with its blocks in hand, now that the list checks
  void solveReady() {
    std::vector<uint32_t> ready;
    for (auto& [c, s] : ch) if (s.state == COLLECTING && s.have.n >= hdr->K[c]) ready.push_back(c);
    for (uint32_t c : ready) if (hdr && !failed && !done && ch[c].state == COLLECTING) solve(c);
  }

  // ai: XferReceiver.manifestBlock
  void manifestBlock(uint32_t m, const uint8_t* p) {
    if (!hdr) { if (raw.size() < 4096) memcpy(raw[m].data(), p, PAYLOAD); return; }   // ai: kept until there is a root to check it by
    Hdr& h = *hdr;
    if (failed || h.listOk || m >= h.mcount || h.mgot[m]) return;
    if (xfer_manifest_parse(p, &h.h, m, h.list.data())) return;   // ai: another transfer's (its tag), or not a manifest block
    h.mgot[m] = 1;
    if (!replaying) store->noteManifest(m, p);
    if (++h.mhave < h.mcount) return;
    if (xfer_manifest_check(h.list.data(), h.h.chunks, h.h.root)) {
      // ai: every block passed its parse and the list still does not make the root: one passed its CRC wrongly, which
      // ai: cannot be told, so all are collected again; the sender repeats them
      std::fill(h.mgot.begin(), h.mgot.end(), 0);
      h.mhave = 0; manifestRejected++;
      return;
    }
    h.listOk = true;
    solveReady();
  }

  // ai: XferReceiver.solve and solveOnce
  void solve(uint32_t c) {
    const double t = nowMs();
    solveOnce(c);
    ms += nowMs() - t;
  }
  void solveOnce(uint32_t c) {
    Hdr& h = *hdr;
    Ch& s = ch[c];
    auto recs = store->read(c);
    if (h.K[c] == 1) {
      std::vector<uint8_t> out;
      if (recs.size() >= REC) out.assign(recs.begin() + XFER_ID_BYTES, recs.begin() + XFER_ID_BYTES + h.len[c]);
      store->drop(c); s.recs = s.mark = 0;
      return check(c, out);
    }
    auto d = std::make_unique<Wh>();
    if (int rc = d->create(h.len[c], uint32_t(xfer_chunk_seed(&h.h, c))); rc < 0) {
      // ai: the JS throws here; a header that passed its CRC wrongly is the only way, and its chunk is collected afresh
      say("xfer: wirehair decoder " + std::to_string(rc) + " for chunk " + std::to_string(c));
      store->drop(c);
      return reject(c, s);
    }
    bool ok = false;
    for (size_t o = 0; o + REC <= recs.size() && !ok; o += REC) {
      const int rc = d->add(xfer_id_get(recs.data() + o), recs.data() + o + XFER_ID_BYTES);
      if (rc < 0) say("xfer: wirehair decode " + std::to_string(rc) + " on chunk " + std::to_string(c));
      ok = rc == 1;
    }
    // ai: the decoder holds them now; a store that keeps the transfer keeps them until the chunk is verified, so a
    // ai: resume rebuilds a chunk Wirehair wanted more blocks for
    if (ok || !store->keeps()) { store->drop(c); s.recs = s.mark = 0; }
    s.dec = std::move(d);
    if (!ok) { s.state = HOT; over++; return; }   // ai: Wirehair wanted a block more than K (a few percent of chunks): fed as they come
    recovered(c);
  }
  void recovered(uint32_t c) {
    Ch& s = ch[c];
    auto out = s.dec->recover();
    s.dec.reset();
    check(c, out);
  }
  void check(uint32_t c, const std::vector<uint8_t>& out) {
    Hdr& h = *hdr;
    Ch& s = ch[c];
    if (!verify(c, out)) return reject(c, s);
    if (!store->writeOut(uint64_t(c) * h.size, out.data(), out.size())) return stop();
    store->noteVerified(c);
    accept(c, s);
  }
  // ai: xfer_chunk_check: the chunk's chaining value against the list (already checked against the root), or, for a
  // ai: file of one chunk, its hash against the root
  bool verify(uint32_t c, const std::vector<uint8_t>& out) {
    Hdr& h = *hdr;
    return xfer_chunk_check(&h.h, c, out.data(), out.size(), h.h.chunks >= 2 ? h.list.data() : nullptr) == 0;
  }
  void accept(uint32_t c, Ch& s) {
    s.state = VERIFIED; s.have = Bits(); s.ban = Bits(); verified++;
    store->drop(c); s.recs = s.mark = 0;   // ai: a kept chunk's blocks (store.h keeps), nothing otherwise
    if (verified == hdr->h.chunks) finish(); else saveMeta();
  }
  // ai: a chunk that decoded to the wrong bytes: which block passed its CRC wrongly cannot be told, so every symbol that
  // ai: went into it is refused from now on and the chunk collected again from fresh ids
  void reject(uint32_t c, Ch& s) {
    s.ban.orWith(s.have); s.have = Bits(); s.state = COLLECTING; rejected++;
    s.dec.reset();
    store->drop(c); s.recs = s.mark = 0;
    say("xfer: chunk " + std::to_string(c) + " rejected");
  }
  void finish() {
    saveMeta();   // ai: the finished file's name, type and root beside it until it is taken out (store.h fileStore)
    path = store->finish(hdr->name);
    if (!store->error.empty()) { path.clear(); return stop(); }
    done = true;
    doneMs = nowMs();
    if (!firstMs) firstMs = doneMs;   // ai: an empty file: no data block, no time
    say("xfer: " + hdr->name + " verified, " + path);
  }

  // ai: XferReceiver.progress, fractionDone and the fountain worker's report
  void progress(XferProgress& p) const {
    p = XferProgress{};
    p.error = store->error;
    p.refused = refused; p.forwarded = forwarded; p.doneAt = doneAt; p.held = uint32_t(held.size());
    if (!hdr) return;
    const Hdr& h = *hdr;
    p.header = true; p.live = sinceHeader < LIVE_BLOCKS;
    p.name = h.name; p.type = h.type; p.root = h.root; p.length = h.h.length; p.chunks = h.h.chunks;
    p.verified = verified; p.manifest = h.mcount; p.manifestHave = h.mhave;
    p.rejected = rejected; p.manifestRejected = manifestRejected; p.over = over; p.solveMs = ms;
    p.done = done; p.failed = failed; p.path = path;
    p.secs = firstMs ? ((done ? doneMs : nowMs()) - firstMs) / 1000 : 0;
    p.per.resize(h.h.chunks);
    for (uint32_t c = 0; c < h.h.chunks; c++) {
      auto it = ch.find(c);
      const Ch* s = it == ch.end() ? nullptr : &it->second;
      p.per[c] = !s ? 0 : s->state == VERIFIED ? 255 : uint8_t(std::min<uint64_t>(99, 100ull * s->have.n / std::max<uint32_t>(1, h.K[c])));
      if (s) p.bytesIn += s->state == VERIFIED ? h.len[c] : std::min<uint64_t>(h.len[c], uint64_t(s->have.n) * PAYLOAD);
    }
    // ai: XferReceiver.bytesIn and fractionDone
    if (done) p.bytesIn = h.h.length;
    p.fraction = done ? 1 : h.h.length ? std::min(1.0, double(p.bytesIn) / double(h.h.length)) : 0;
  }
};

XferRxCore::XferRxCore(std::unique_ptr<Store> store, std::function<void(const std::string&)> log)
    : rx_(std::make_unique<Rx>(std::move(store), std::move(log))) {}
XferRxCore::~XferRxCore() = default;
void XferRxCore::block(const uint8_t* b) { rx_->block(b); }
void XferRxCore::clearAll() { rx_->clearAll(); }
void XferRxCore::forget(const std::string& root) { rx_->forget(root); }
void XferRxCore::progress(XferProgress& p) const { rx_->progress(p); }
const std::atomic<uint64_t>& XferRxCore::rootGen() const { return rx_->gen; }
const std::atomic<uint64_t>& XferRxCore::taken() const { return rx_->taken; }
const std::vector<uint8_t>* XferRxCore::data() const { return rx_->done ? rx_->store->data() : nullptr; }

void XferJudge::remember(uint32_t id) {
  seenIds_.insert(id);
  if (seenIds_.size() >= SEEN_KEEP) { seenOld_ = std::move(seenIds_); seenIds_ = {}; }
}

// ai: lizard-web/recv.mjs take's dedupe, and its reset on a new root (the page's xfer message; here the core's generation)
bool XferJudge::take(const uint8_t* b, const Pass& pass) {
  if (const uint64_t g = gen_.load(); g != genSeen_) { genSeen_ = g; seenIds_.clear(); seenOld_.clear(); }
  const uint32_t id = xfer_id_get(b);
  if (xfer_id_chunk(id) == XFER_CONTROL) { pass(b); return false; }
  if (seenIds_.count(id) || seenOld_.count(id)) return false;
  remember(id);
  pass(b);
  return true;
}

// ai: sim/phy.mjs blockJudge, then lizard-web/recv.mjs take: a test frame's new ids are counted and remembered as the page
// ai: does, and none of its blocks passed on
FrameVerdict XferJudge::frame(const uint8_t* blocks, size_t count, const Pass& pass) {
  FrameVerdict v;
  std::vector<uint8_t> same(count, 0);
  uint8_t want[XFER_PAYLOAD];
  int data = 0;
  for (size_t k = 0; k < count; k++) {
    const uint8_t* b = blocks + k * XFER_BLOCK;
    const uint32_t id = xfer_id_get(b);
    if (xfer_id_chunk(id) == XFER_CONTROL) continue;
    data++;
    stream_fill(id, want, XFER_PAYLOAD);
    same[k] = !memcmp(want, b + XFER_ID_BYTES, XFER_PAYLOAD);
    v.test |= same[k] != 0;
  }
  v.judged = v.test ? data : 0;
  // ai: blockJudge's got holds the frame's control blocks first, then its data blocks: the order the worker takes them in
  for (int pass2 = 0; pass2 < 2; pass2++)
    for (size_t k = 0; k < count; k++) {
      const uint8_t* b = blocks + k * XFER_BLOCK;
      const uint32_t id = xfer_id_get(b);
      const bool control = xfer_id_chunk(id) == XFER_CONTROL;
      if (control != (pass2 == 0)) continue;
      if (v.test && !control && !same[k]) { v.bad++; continue; }
      v.seen++;
      if (!v.test) { v.fresh += take(b, pass); continue; }
      if (!control && !seenIds_.count(id) && !seenOld_.count(id)) { remember(id); v.fresh++; }
    }
  return v;
}

void XferJudge::clear() {
  seenIds_.clear(); seenOld_.clear();
  genSeen_ = gen_.load();
}

}  // namespace lizard
