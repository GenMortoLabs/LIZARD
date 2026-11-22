// ai: XferRxCore and XferJudge (xfer_rx_core.h): the web's transfer chain ported, with no thread of its own (split out of
// ai: xfer_rx.cpp 2026-10-03, every line of the chain as it was). Rx is sim/xfer.mjs XferReceiver and
// ai: lizard-web/fountain-worker.mjs take; the judge is sim/phy.mjs blockJudge and lizard-web/recv.mjs take's dedupe. Each function
// ai: names the JS it ports; where the port differs, an ai: comment says so and why.
#include "xfer_rx_core.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <system_error>
#include <thread>

extern "C" {
#include "rq.h"
#include "rq_plan.h"
#include "xfer.h"
}

namespace lizard {

static constexpr uint32_t PAYLOAD = XFER_PAYLOAD, REC = XFER_BLOCK;
// ai: lizard-web/fountain-worker.mjs LIVE_BLOCKS, HOLD_BLOCKS; lizard-web/recv.mjs SEEN_KEEP
static constexpr int64_t LIVE_BLOCKS = 4096, NEVER = INT64_MAX / 2;   // ai: XferProgress.live
static constexpr size_t HOLD_BLOCKS = 1024, SEEN_KEEP = 1 << 16;
// ai: meta.json written again every this many blocks stored (about 1 MB) or a second of storing, whichever comes first,
// ai: as well as at each source block verified and at a park, so a receiver stopped mid-source block leaves an app's list a line at
// ai: most that far behind
static constexpr uint32_t META_EVERY = 2048;
static constexpr double META_MS = 1000;
// ai: SOLVING: its K symbols in, its solve with a solver or waiting for one (the pool's, below)
enum { COLLECTING, HOT, VERIFIED, SOLVING };

static std::string hexOf(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
  return s;
}
// ai: a transfer's folder in a store: the root's first 16 hex digits and its TOI's 12 (a file's TOI its content's, so
// ai: the same file from any sender is one folder)
static std::string keyOf(const std::string& rootHex, xfer_toi_t toi) {
  char t[16];
  snprintf(t, sizeof t, "%012llx", (unsigned long long)(toi & XFER_TOI_MASK));
  return rootHex.substr(0, 16) + t;
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

// ai: A source block's refused symbols (a rejected source block's: one of them passed its CRC wrongly), sorted: rare, so a sorted
// ai: list. Since 2026-11-15 a source block keeps no set of the symbols it holds: a send's ids are each pass of a source block's 2^24
// ai: in an order of its own (core/tx/xfer_tx.cpp esiOf), so they never repeat within a send; a camera's repeats are
// ai: the judge's (seenIds); the rare repeat across sends (K / 2^24 of a source block's) is dropped at its solve. The dense runs
// ai: of bits held until then (Bits, 2026-11-14) would have grown to 2 MiB a source block over shuffled ids.
// ai: v radix-sorted (24-bit keys, two stable passes of 12 bits), O(n)
void radix24(std::vector<uint32_t>& v) {
  std::vector<uint32_t> t(v.size());
  for (int shift = 0; shift < 24; shift += 12) {
    std::array<uint32_t, 4097> cnt{};
    for (uint32_t x : v) cnt[((x >> shift) & 0xfff) + 1]++;
    for (int b = 0; b < 4096; b++) cnt[b + 1] += cnt[b];
    for (uint32_t x : v) t[cnt[(x >> shift) & 0xfff]++] = x;
    v.swap(t);
  }
}

// ai: sim/fountain.mjs Decoder: a source block's RaptorQ decoder (src/rq.h), its symbols the ids' 18 bits; it takes every
// ai: symbol it is given and solves from K on
struct Dec {
  rq_dec_t* d = nullptr;
  uint32_t bytes = 0;
  ~Dec() { rq_dec_free(d); }
  bool create(uint32_t messageBytes) {
    bytes = messageBytes;
    d = rq_dec_new(messageBytes, PAYLOAD, 1u << XFER_ESI_BITS);
    return d != nullptr;
  }
  // ai: 1 once the source block is whole, 0 for more, -1 out of memory
  int add(uint32_t sym, const uint8_t* p) { return rq_dec_add(d, sym, p); }
  // ai: a symbol kept with no solve tried (0, -1 out of memory): a rank-short source block's records, its next add solving
  int hold(uint32_t sym, const uint8_t* p) { return rq_dec_hold(d, sym, p); }
  std::vector<uint8_t> recover() {
    std::vector<uint8_t> o(bytes);
    if (rq_dec_recover(d, o.data())) o.clear();
    return o;
  }
};

struct XferRxCore::Rx {
  // ai: recs: the blocks the store holds for the source block, every one this transfer's by its TOI (src/xfer.h; until
  // ai: 2026-11-19 the blocks after the header's last read were provisional and a park cut each source block back to them).
  // ai: asked: the blocks forwarded when its solve was asked for (doneAt); askedRecs: the records held then, all a job
  // ai: solves (an inline solve's), the records after them the blocks a solver's time let in
  // ai: dups: of the source block's records, those its last solve found repeating an earlier one's symbol (a repeat across
  // ai: sends: the source block keeps no set of its symbols); it solves again once recs - dups reaches K
  struct SbState { int state = COLLECTING; std::unique_ptr<Dec> dec; uint32_t recs = 0, askedRecs = 0, dups = 0; uint64_t asked = 0; };
  struct Hdr {
    std::vector<uint8_t> bytes;   // ai: its description (a session header's, src/xfer.h)
    xfer_header_t h;
    xfer_toi_t toi = 0;   // ai: its blocks' TOI (its content's, src/xfer.h)
    std::string name, root;
    uint64_t size = 0;
    // ai: len and K: each source block's bytes and source symbols, the header's (a source block goes as its own bytes); a
    // ai: source block is solved once its K symbols are in; cvs: each source block's chaining value as it decodes (two
    // ai: source blocks or more), combined and held to the root at the last (src/xfer.h xfer_root_check)
    std::vector<uint32_t> K, len;
    std::vector<uint8_t> cvs;
  };

  // ai: The solvers (2026-11-14): a source block whose K symbols are in is solved by a thread of a pool, a physical core each
  // ai: (XferRx's), so the source blocks the interleave leaves to the end of a transfer are solved side by side, where they
  // ai: went one after another on this thread. A job carries the source block's records, its length and what its check needs
  // ai: (the header, the list); the solver holds the records in a decoder, lets them go, solves, recovers and checks the
  // ai: source block; the result is taken back on this thread (settle), where the store and the source block's state are touched. At
  // ai: most one job a solver is out (its records read as it goes): the rest wait as source block numbers, so a burst holds
  // ai: about 36 MB a solver at 16 MiB source blocks, not every source block's records at once. No pool (solvers 0: the library's,
  // ai: and the wasm's, which has no threads): the job runs here, inline, as before. A job names its transfer by its
  // ai: serial (one a transfer opened), so a solve of a transfer let go is dropped when it comes back.
  // ai: at: the blocks forwarded when the source block's solve was asked for (doneAt, below)
  struct Job { uint32_t c = 0; uint64_t serial = 0, at = 0; uint32_t len = 0, nrecs = 0; std::vector<uint8_t> recs; xfer_header_t h{}; };
  // ai: cv: the source block's chaining value (two source blocks or more), xfer_sb_check's
  struct Done { uint32_t c = 0; uint64_t serial = 0, at = 0; uint32_t nrecs = 0, dups = 0; bool solved = false, ok = false, oom = false, few = false; std::vector<uint8_t> out; std::unique_ptr<Dec> dec; double ms = 0; uint8_t cv[XFER_CV] = {}; };

  // ai: One transfer this receiver holds (sim/xfer.mjs Transfer, 2026-11-19): a file, from
  // ai: its session header's description on, each source block checked against its root, its blocks in a store
  // ai: of its own (the root store's make()).
  struct Obj {
    Rx& rx;
    uint64_t serial;
    std::unique_ptr<Store> store;
    std::unique_ptr<Hdr> hdr;
    // ai: each source block's state by its number (2026-11-15: a flat table, sized to the header's source blocks, where an ordered map
    // ai: took a tree walk a block and a lookup a source block a snapshot); in the source blocks' order, as the map was (the JS Map is in
    // ai: insertion order: only the order of solves within one block's take differs). A source block never touched is COLLECTING
    // ai: with nothing held, which every reader takes as the map's absent entry.
    std::vector<SbState> sbs;
    uint32_t verified = 0, rejected = 0, over = 0;
    double ms = 0;
    double firstMs = 0, doneMs = 0;   // ai: XferProgress.secs: this transfer's first data block taken, its finish (nowMs)
    bool done = false;
    // ai: the store could not keep this transfer (store.h: over the memory cap, a write refused): it ends here, never
    // ai: done, its blocks no longer taken; the store's error says why. Its header again (allow) begins it afresh.
    bool failed = false;
    std::string path;
    uint32_t sinceMeta = 0;   // ai: blocks stored since meta.json was last written, and when (nowMs)
    double metaMs = 0;
    uint64_t doneAt = 0;   // ai: the blocks forwarded when the file finished
    uint64_t lastAsked = 0;   // ai: the latest block at which a source block verified became complete (doneAt)

    Obj(Rx& r, uint64_t s, std::unique_ptr<Store> st) : rx(r), serial(s), store(std::move(st)) { store->log = rx.log; }
    void say(const std::string& s) { rx.say(s); }
    void stop() { if (!failed) say("xfer: " + (hdr ? hdr->name : std::string("the transfer")) + " stopped: " + store->error); failed = true; }
    // ai: K symbols in hand, as far as the source block knows: its records less the repeats its last solve found
    bool full(uint32_t c, const SbState& s) const { return hdr && s.recs >= hdr->K[c] + s.dups; }
    SbState& state(uint32_t c) { if (c >= sbs.size()) sbs.resize(size_t(c) + 1); return sbs[c]; }

    // ai: XferReceiver.open for a new transfer from its description (desc, n bytes, a session header's):
    // ai: its folder resumed or begun; true where that ended it (resumed whole, an empty file, or the store refused it)
    bool open(const uint8_t* desc, size_t n, xfer_toi_t toi, const xfer_header_t& parsed) {
      auto h = std::make_unique<Hdr>();
      h->h = parsed;
      h->toi = toi;
      h->bytes.assign(desc, desc + n);
      h->name.assign((const char*)h->h.name, h->h.name_len);
      h->root = hexOf(h->h.root, XFER_CV);
      h->size = uint64_t(1) << h->h.sb_log2;
      const uint32_t sourceBlocks = h->h.source_blocks;
      h->cvs.assign(std::max<size_t>(1, size_t(XFER_CV) * sourceBlocks), 0);
      h->K.resize(sourceBlocks); h->len.resize(sourceBlocks);
      for (uint32_t c = 0; c < sourceBlocks; c++) { h->len[c] = uint32_t(xfer_sb_len(&h->h, c)); h->K[c] = xfer_symbols(h->len[c]); }
      hdr = std::move(h);
      say("xfer: header " + hdr->name + ", " + std::to_string(hdr->h.length) + " B, " + std::to_string(sourceBlocks) + " chunks, root " + hdr->root);
      sbs.resize(sourceBlocks);
      Store::Before before;
      if (!store->open(keyOf(hdr->root, toi), hdr->bytes.data(), hdr->bytes.size(), hdr->h.length, before)) { stop(); return true; }
      if (!sourceBlocks) {   // ai: an empty file: its root is BLAKE3 of nothing, which is all there is to check
        uint8_t e[XFER_CV];
        xfer_b3_hash(nullptr, 0, e);
        if (!memcmp(e, hdr->h.root, XFER_CV)) finish(); else rx.refused++;
        return true;
      }
      return before.found && resume(before);
    }
    // ai: XferReceiver.reset for this transfer: parked for its header to come again (keep; a store that keeps no
    // ai: transfers lets it go), or let go whole. Its solves out are dropped when they come back (its serial gone).
    void close(bool keep) {
      if (keep && hdr && !done && !failed && store->keeps()) saveMeta();
      if (keep) store->park(); else store->clear();
    }
    // ai: the transfer's line for an app's list of unfinished transfers (store.h noteMeta)
    void saveMeta() {
      sinceMeta = 0; metaMs = nowMs();
      if (!hdr || !store->keeps()) return;
      XferProgress p;
      progress(p, false);
      char f[32];
      snprintf(f, sizeof f, "%.4f", p.fraction);
      store->noteMeta("{\"name\":" + jstr(p.name) + ",\"compression\":" + std::to_string(p.compression) + ",\"length\":" + std::to_string(p.length) +
                      ",\"sourceBlocks\":" + std::to_string(p.sourceBlocks) + ",\"verified\":" + std::to_string(p.verified) + ",\"root\":\"" + p.root +
                      "\",\"fraction\":" + f + ",\"bytesIn\":" + std::to_string(p.bytesIn) + "}");
    }

    // ai: XferReceiver's data path: a block of this transfer's TOI, its symbol
    void add(xfer_id_t id, const uint8_t* p) { addSym(xfer_id_sbn(id), xfer_id_esi(id), p); }
    // ai: one symbol of source block c (PAYLOAD bytes at p)
    void addSym(uint32_t c, uint32_t sym, const uint8_t* p) {
      if (done || failed || !hdr || c >= hdr->h.source_blocks) return;
      if (!firstMs) firstMs = nowMs();
      SbState& s = state(c);
      if (s.state == VERIFIED) return;
      rx.taken.fetch_add(1, std::memory_order_relaxed);
      if (s.dec) {
        // ai: kept until the source block is verified, by any store: a resume makes the decoder again from them
        if (!store->append(c, sym, p)) return stop();
        s.recs++;
        const int rc = s.dec->add(sym, p);
        if (rc < 0) say("xfer: RaptorQ out of memory on chunk " + std::to_string(c));
        if (rc == 1) recovered(c);
        return;
      }
      if (!store->append(c, sym, p)) return stop();
      s.recs++;
      if (++sinceMeta >= META_EVERY || nowMs() - metaMs >= META_MS) saveMeta();
      if (s.state == COLLECTING && full(c, s)) solve(c);
    }

    // ai: A folder of this transfer from before (store.h): its verified source blocks taken as verified, their chaining
    // ai: values back, and each collecting source block's symbols marked had from its stored blocks; true when that
    // ai: finished it
    bool resume(const Store::Before& b) {
      const uint32_t sourceBlocks = hdr->h.source_blocks;
      for (const auto& [c, cv] : b.verified) if (c < sourceBlocks && sbs[c].state != VERIFIED) {
        sbs[c].state = VERIFIED; verified++;
        if (sourceBlocks >= 2 && cv.size() == XFER_CV) std::memcpy(hdr->cvs.data() + size_t(c) * XFER_CV, cv.data(), XFER_CV);
      }
      uint64_t had = 0;
      for (uint32_t c : b.sourceBlocks) {
        if (c >= sourceBlocks || sbs[c].state == VERIFIED) { store->drop(c); continue; }
        SbState& s = sbs[c];
        s.recs = uint32_t(store->read(c).size() / REC);
        had += s.recs;
      }
      say("xfer: resumed " + hdr->name + ", " + std::to_string(verified) + " of " + std::to_string(sourceBlocks) + " chunks verified, " +
          std::to_string(had) + " blocks in hand");
      if (verified == sourceBlocks) { complete(); return done || failed; }
      return false;
    }
    // ai: XferReceiver.solveReady: every source block with its blocks in hand (those held before the header)
    void solveReady() {
      std::vector<uint32_t> ready;
      for (uint32_t c = 0; c < sbs.size(); c++) if (sbs[c].state == COLLECTING && full(c, sbs[c])) ready.push_back(c);
      for (uint32_t c : ready) if (hdr && !failed && !done && sbs[c].state == COLLECTING) solve(c);
    }

    // ai: XferReceiver.solve: source block c's K symbols are in. With solvers, waiting for one (dispatch); without, solved here.
    void solve(uint32_t c) {
      SbState& s = sbs[c];
      // ai: its distinct symbols counted from the store's ids first (142 KB read, not the records' 16.8 MB; 2026-11-15):
      // ai: fewer than K (repeats across sends) and the source block collects on, no records read and no solver taken. The A/B
      // ai: that asked for it: a sender started again at half, 1 GiB, the tail 12% longer with each source block's first try
      // ai: reading its records for nothing (research/build/idorder-2026-11-15/ab_tail.cpp)
      std::vector<uint32_t> ids = store->ids(c);
      for (auto& v : ids) v &= XFER_ESI_MASK;
      radix24(ids);
      const uint32_t distinct = uint32_t(std::unique(ids.begin(), ids.end()) - ids.begin());
      if (distinct < hdr->K[c]) {
        s.dups = s.recs - std::min(s.recs, distinct);
        return;
      }
      s.state = SOLVING;
      s.asked = rx.forwarded;
      s.askedRecs = s.recs;
      if (rx.solvers) { rx.waiting.push_back({serial, c}); rx.waitN++; return rx.dispatch(); }
      commit(run(jobOf(c)));
    }
    Job jobOf(uint32_t c) {
      Job j;
      j.c = c; j.serial = serial; j.at = sbs[c].asked; j.len = hdr->len[c]; j.h = hdr->h;
      j.recs = store->read(c);
      j.recs.resize(std::min(j.recs.size(), size_t(sbs[c].askedRecs) * REC));
      j.nrecs = uint32_t(j.recs.size() / REC);
      return j;
    }
    // ai: reject for a job's result: the symbols the solve took dropped (as the inline reject's), the blocks stored
    // ai: since its ask kept as the source block's first ones, as they would be after an inline reject
    void rejectSolved(uint32_t c, SbState& s, uint32_t nrecs) {
      const auto recs = store->read(c);
      const size_t cut = std::min(recs.size(), size_t(nrecs) * REC);
      s.state = COLLECTING; rejected++;
      s.dec.reset();
      store->drop(c); s.recs = s.dups = 0;
      say("xfer: chunk " + std::to_string(c) + " rejected");
      for (size_t o = cut; o + REC <= recs.size(); o += REC) {
        const uint32_t sym = uint32_t(xfer_id_get(recs.data() + o));
        if (!store->append(c, sym, recs.data() + o + XFER_ID_BYTES)) return stop();
        s.recs++;
      }
      if (hdr && full(c, s)) solve(c);
    }
    // ai: a job's result, on the receiving end's thread
    void commit(Done d) {
      ms += d.ms;
      if (!hdr || done || d.c >= sbs.size() || sbs[d.c].state != SOLVING) return;
      SbState& s = sbs[d.c];
      if (d.few) {
        s.dups = d.dups;
        s.state = COLLECTING;
        if (full(d.c, s)) solve(d.c);   // ai: the repeats' worth came while it solved
        return;
      }
      if (d.oom) {
        // ai: the JS throws here: out of memory, the source block collected afresh
        say("xfer: no RaptorQ decoder for chunk " + std::to_string(d.c));
        return rejectSolved(d.c, s, d.nrecs);
      }
      if (d.solved) {
        if (!d.ok) return rejectSolved(d.c, s, d.nrecs);
        if (!store->writeOut(uint64_t(d.c) * hdr->size, d.out.data(), d.out.size())) return stop();
        if (hdr->h.source_blocks >= 2) std::memcpy(hdr->cvs.data() + size_t(d.c) * XFER_CV, d.cv, XFER_CV);
        store->noteVerified(d.c, hdr->h.source_blocks >= 2 ? d.cv : nullptr);
        // ai: the file done at the block that completed its last source block to complete, as inline solves have it, not when a
        // ai: solver came back (which the solvers' timing moves)
        lastAsked = std::max(lastAsked, d.at);
        if (verified + 1 == hdr->h.source_blocks && !doneAt) doneAt = lastAsked;
        return accept(d.c, s);
      }
      // ai: RaptorQ wanted a block more than K (about 1% of source blocks): the blocks stored since the job's feed it, the rest
      // ai: as they come (add), kept until the source block is verified
      s.dec = std::move(d.dec); s.state = HOT; over++;
      if (s.recs > d.nrecs) {
        const auto recs = store->read(d.c);
        for (size_t o = size_t(d.nrecs) * REC; o + REC <= recs.size(); o += REC)
          if (s.dec->add(uint32_t(xfer_id_get(recs.data() + o)), recs.data() + o + XFER_ID_BYTES) == 1) return recovered(d.c);
      }
    }
    void recovered(uint32_t c) {
      lastAsked = std::max(lastAsked, rx.forwarded);   // ai: a source block fed on (HOT) completes at this block
      SbState& s = sbs[c];
      auto out = s.dec->recover();
      s.dec.reset();
      check(c, out);
    }
    void check(uint32_t c, const std::vector<uint8_t>& out) {
      Hdr& h = *hdr;
      SbState& s = sbs[c];
      if (!verify(c, out)) return reject(c, s);
      if (!store->writeOut(uint64_t(c) * h.size, out.data(), out.size())) return stop();
      store->noteVerified(c, h.h.source_blocks >= 2 ? h.cvs.data() + size_t(c) * XFER_CV : nullptr);
      accept(c, s);
    }
    // ai: xfer_sb_check as the source block decodes: its length, and for a file of one source block its hash against the
    // ai: root; for two or more its chaining value into cvs, for the root at the last
    bool verify(uint32_t c, const std::vector<uint8_t>& out) {
      Hdr& h = *hdr;
      return xfer_sb_check(&h.h, c, out.data(), out.size(), h.cvs.data() + size_t(c) * XFER_CV) == 0;
    }
    void accept(uint32_t c, SbState& s) {
      s.state = VERIFIED; verified++;
      store->drop(c); s.recs = s.dups = 0;   // ai: a kept source block's blocks (store.h keeps), nothing otherwise
      if (verified == hdr->h.source_blocks) complete(); else saveMeta();
    }
    // ai: XferReceiver.complete: the last source block in. A file of one was checked against the root as it decoded; of two
    // ai: or more the chaining values are combined and held to the root. A file that fails (only a broken sender makes
    // ai: one: every block passed its CRC-64) is refused whole, every source block collected again from the blocks to come
    void complete() {
      Hdr& h = *hdr;
      if (h.h.source_blocks >= 2 && xfer_root_check(h.cvs.data(), h.h.source_blocks, h.h.root)) {
        rejected += h.h.source_blocks; verified = 0;
        for (SbState& s : sbs) { s.state = COLLECTING; s.dec.reset(); s.recs = s.dups = 0; }
        store->forgetVerified();
        say("xfer: " + h.name + "'s chunks do not make its root; collected again");
        return;
      }
      finish();
    }
    // ai: a source block that decoded to the wrong bytes (only a broken sender's: a wrong block past its CRC-64 is out of
    // ai: reach): which block was wrong cannot be told, so the symbols that went into it are dropped and the source block
    // ai: collected again. No list of refused symbols is kept (2026-11-21): a sender outside the model is not defended
    // ai: against beyond memory safety
    void reject(uint32_t c, SbState& s) {
      s.state = COLLECTING; rejected++;
      s.dec.reset();
      store->drop(c); s.recs = s.dups = 0;
      say("xfer: chunk " + std::to_string(c) + " rejected");
    }
    void finish() {
      saveMeta();   // ai: the finished file's name, compression and root beside it until it is taken out (store.h fileStore)
      path = store->finish(hdr->name);
      if (!store->error.empty()) { path.clear(); return stop(); }
      done = true;
      doneMs = nowMs();
      if (!firstMs) firstMs = doneMs;   // ai: an empty file: no data block, no time
      rx.finished.push_back(serial);
      say("xfer: " + hdr->name + " verified, " + path);
    }

    // ai: XferReceiver.progress, fractionDone and the fountain worker's report, this transfer's part
    // ai: perSb: per filled too (liz_rx_source_blocks' and the checks'; the snapshots for the app's polls take none, 2026-11-15)
    void progress(XferProgress& p, bool perSb) const {
      p.error = store->error;
      p.refusal = store->refusal; p.limit = store->limit; p.free = store->free;
      if (!hdr) return;
      const Hdr& h = *hdr;
      p.header = true;
      p.name = h.name; p.compression = h.h.compression; p.root = h.root; p.length = h.h.length; p.sourceBlocks = h.h.source_blocks;
      p.verified = verified;
      p.rejected = rejected; p.over = over; p.solveMs = ms;
      p.done = done; p.failed = failed; p.path = path; p.doneAt = doneAt;
      p.secs = firstMs ? ((done ? doneMs : nowMs()) - firstMs) / 1000 : 0;
      p.bytesIn = 0; p.solving = 0;
      if (perSb) p.per.resize(h.h.source_blocks);
      for (uint32_t c = 0; c < h.h.source_blocks; c++) {
        const SbState* s = c < sbs.size() ? &sbs[c] : nullptr;
        if (perSb) p.per[c] = !s ? 0 : s->state == VERIFIED ? 255 : uint8_t(std::min<uint64_t>(99, 100ull * (s->recs - std::min(s->recs, s->dups)) / std::max<uint32_t>(1, h.K[c])));
        if (s) p.bytesIn += s->state == VERIFIED ? h.len[c] : std::min<uint64_t>(h.len[c], uint64_t(s->recs - std::min(s->recs, s->dups)) * PAYLOAD);
        if (s && s->state == SOLVING) p.solving++;
      }
      p.inHand = !done && p.solving > 0 && p.verified + p.solving == h.h.source_blocks;
      // ai: XferReceiver.bytesIn and fractionDone
      if (done) p.bytesIn = h.h.length;
      p.fraction = done ? 1 : h.h.length ? std::min(1.0, double(p.bytesIn) / double(h.h.length)) : 0;
    }
  };

  std::unique_ptr<Store> root;   // ai: the store handed in: each transfer's made from it (make), its folder's own ops (removeAll, forget)
  std::function<void(const std::string&)> log;
  std::atomic<uint64_t> taken{0};   // ai: XferRxCore::taken: data blocks added to a source block still collecting, ever
  // ai: the transfers held, by TOI, and the order they opened in (progress' files); focus the file a single file's view
  // ai: shows (the last whose header opened one; XferProgress' own fields), bySerial a job's way back to its transfer
  std::map<uint64_t, std::unique_ptr<Obj>> objs;   // ai: by TOI
  std::vector<Obj*> order;
  // ai: the finished transfers whose TOIs were let go (releaseDone), kept for progress' lines and the file in focus
  std::vector<std::unique_ptr<Obj>> released;
  std::map<uint64_t, Obj*> bySerial;
  Obj* focus = nullptr;
  uint64_t serials = 0;
  uint32_t refused = 0, descsRefused = 0;
  // ai: every description held (a session header's), by the session's TOI
  std::map<uint64_t, std::vector<uint8_t>> descs;
  // ai: data blocks forwarded since the last that went to a transfer held (live), and the blocks of a TOI no transfer
  // ai: holds yet, held (their ids tagged) until a header of theirs is read
  int64_t sinceOwn = NEVER;
  struct Held { xfer_id_t id; xfer_toi_t toi; std::array<uint8_t, XFER_PAYLOAD> p; };
  std::vector<Held> held;
  size_t heldAt = 0;
  uint64_t forwarded = 0;
  // ai: the files finished since the last block (their serials): their doneAt the blocks forwarded then, where no solve's
  // ai: ask set it first
  std::vector<uint64_t> finished;
  void noteDone() {
    for (uint64_t s : finished) if (Obj* t = find(s); t && !t->doneAt) t->doneAt = forwarded;
    finished.clear();
  }

  unsigned solvers = 0;
  std::function<void()> wake;
  std::mutex jmu;
  std::condition_variable jcv;
  std::deque<Job> jobs;
  std::deque<Done> dones;
  bool jstop = false;
  std::vector<std::thread> pool;
  std::deque<std::pair<uint64_t, uint32_t>> waiting;   // ai: source blocks SOLVING with no solver yet (their transfer's serial), in the order they came
  std::atomic<uint32_t> waitN{0};        // ai: its size, for another thread's solving()
  std::atomic<uint32_t> out{0};          // ai: jobs handed to the solvers and not yet taken back

  Rx(std::unique_ptr<Store> s, std::function<void(const std::string&)> l, unsigned n, std::function<void()> w)
      : root(std::move(s)), log(std::move(l)), wake(std::move(w)) {
    root->log = log;
    for (unsigned i = 0; i < n; i++) try { pool.emplace_back([this] { solver(); }); } catch (const std::system_error&) { break; }
    solvers = unsigned(pool.size());
  }
  ~Rx() {
    { std::lock_guard<std::mutex> lk(jmu); jstop = true; }
    jcv.notify_all();
    for (auto& t : pool) t.join();
  }
  void solver() {
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(jmu);
        jcv.wait(lk, [&] { return jstop || !jobs.empty(); });
        if (jstop) return;
        j = std::move(jobs.front());
        jobs.pop_front();
      }
      Done d = run(std::move(j));
      { std::lock_guard<std::mutex> lk(jmu); dones.push_back(std::move(d)); }
      if (wake) wake();
    }
  }
  // ai: a job, on any thread: nothing of the Rx touched
  static Done run(Job j) {
    Done d;
    d.c = j.c; d.serial = j.serial; d.at = j.at; d.nrecs = j.nrecs;
    const double t = nowMs();
    // ai: the solve over the records where they lie (src/rq_plan.h): the plan from their symbols' ESIs, the symbol work
    // ai: reading each payload in its record, the source block recovered straight into its bytes; no copy into a decoder
    // ai: (2026-11-15, the S26: a 1 GiB burst's tail 2.28 and 2.35 s against 2.54 and 2.89 held in a decoder, the solve
    // ai: work 13 to 16% less)
    // ai: each symbol once (a repeat across sends, K / 2^24 of a source block's, stored as it came: the source block keeps no set of
    // ai: its symbols since 2026-11-15), the first record of an ESI kept, the order the records came in
    // ai: the records' indexes radix-sorted by their 24-bit ESIs (two stable passes of 12 bits, so an ESI's first
    // ai: record stays first), each ESI's first kept, then back in arrival order by one pass over the kept marks
    const size_t all = j.recs.size() / REC;
    std::vector<uint32_t> key(all), at(all), tmp(all);
    for (size_t k = 0; k < all; k++) key[k] = uint32_t(xfer_id_get(j.recs.data() + k * REC)) & XFER_ESI_MASK, at[k] = uint32_t(k);
    for (int shift = 0; shift < 24; shift += 12) {
      std::array<uint32_t, 4097> cnt{};
      for (uint32_t k : at) cnt[((key[k] >> shift) & 0xfff) + 1]++;
      for (int b = 0; b < 4096; b++) cnt[b + 1] += cnt[b];
      for (uint32_t k : at) tmp[cnt[(key[k] >> shift) & 0xfff]++] = k;
      at.swap(tmp);
    }
    std::vector<uint8_t> keep(all, 0);
    for (size_t i = 0; i < all; i++) keep[at[i]] = i == 0 || key[at[i]] != key[at[i - 1]];
    at.clear();
    for (size_t k = 0; k < all; k++) if (keep[k]) at.push_back(uint32_t(k));
    const auto esiAt = [&](uint32_t k) { return key[k]; };
    const size_t n = at.size();
    // ai: fewer than K symbols once repeats are dropped: no solve and no decoder (a decoder would hold the source block's
    // ai: symbols in memory, and near a resumed transfer's end every source block would at once); the source block collects that many
    // ai: more (commit)
    d.dups = uint32_t(all - n);
    if (n < xfer_symbols(j.len)) {
      d.few = true;
      d.ms = nowMs() - t;
      return d;
    }
    std::vector<uint32_t> esi(n);
    std::vector<const uint8_t*> sym(n);
    for (size_t k = 0; k < n; k++) esi[k] = esiAt(at[k]), sym[k] = j.recs.data() + size_t(at[k]) * REC + XFER_ID_BYTES;
    rq_plan_t* pl = nullptr;
    const int pr = rq_plan_for(j.len, PAYLOAD, uint32_t(n), esi.data(), &pl);
    if (pr == 1) {
      // ai: C not cleared first: rq_plan_exec writes every column of it (2026-11-15; a vector's 17 MB of zeros before)
      std::unique_ptr<uint8_t[]> C;
      try { C.reset(new uint8_t[size_t(pl->p.L) * PAYLOAD]); d.out.resize(j.len); } catch (const std::bad_alloc&) { d.oom = true; }
      if (!d.oom && (rq_plan_exec(pl, PAYLOAD, uint32_t(n), sym.data(), C.get()) || rq_plan_recover(pl, C.get(), PAYLOAD, j.len, d.out.data()))) d.oom = true;
      rq_plan_free(pl);
      if (!d.oom) {
        d.solved = true;
        d.ok = xfer_sb_check(&j.h, j.c, d.out.data(), d.out.size(), d.cv) == 0;
      }
    } else if (pr < 0) {
      d.oom = true;
    } else {
      // ai: a rank short (about 0.5% of source blocks at K): a decoder holding the records, back to take the blocks as they come
      auto dec = std::make_unique<Dec>();
      if (!dec->create(j.len)) d.oom = true;
      for (size_t k = 0; !d.oom && k < n; k++)
        if (dec->hold(esi[k], sym[k]) < 0) d.oom = true;
      if (!d.oom) d.dec = std::move(dec);
    }
    d.ms = nowMs() - t;
    return d;
  }
  Obj* find(uint64_t serial) const { auto it = bySerial.find(serial); return it == bySerial.end() ? nullptr : it->second; }
  // ai: the solves finished since, taken in (each by its transfer, one let go since dropped); then the source blocks waiting
  // ai: given the solvers freed
  void settle() {
    std::deque<Done> got;
    { std::lock_guard<std::mutex> lk(jmu); got.swap(dones); }
    for (auto& d : got) {
      out--;
      if (Obj* t = find(d.serial)) t->commit(std::move(d));
    }
    dispatch();
    noteDone();
  }
  void dispatch() {
    while (!waiting.empty() && out < solvers) {
      const auto [serial, c] = waiting.front();
      waiting.pop_front();
      waitN--;
      Obj* t = find(serial);
      if (!t || !t->hdr || c >= t->sbs.size() || t->sbs[c].state != SOLVING) continue;
      Job j = t->jobOf(c);
      out++;
      { std::lock_guard<std::mutex> lk(jmu); jobs.push_back(std::move(j)); }
      jcv.notify_one();
    }
  }
  uint32_t solving() const { return out + waitN; }
  void say(const std::string& s) { if (log) log(s); }

  // ai: lizard-web/fountain-worker.mjs chunked (a config) and handle's clear: a new receiver, the held blocks let go, and on
  // ai: a clear the finished files too (refused kept, as the web's receiver keeps it)
  void clearAll() {
    while (!objs.empty()) close(objs.begin()->first, false);
    order.clear(); released.clear(); focus = nullptr; descs.clear();
    root->removeAll();
    sinceOwn = NEVER; held.clear(); heldAt = 0; forwarded = 0;
  }
  // ai: A transfer let go: parked (keep) or its folder gone; its solves queued taken out, those out dropped when they
  // ai: come back (its serial), its source blocks waiting skipped (dispatch)
  void close(uint64_t key, bool keep) {
    auto it = objs.find(key);
    if (it == objs.end()) return;
    Obj* t = it->second.get();
    {
      std::lock_guard<std::mutex> lk(jmu);
      for (auto j = jobs.begin(); j != jobs.end();)
        if (j->serial == t->serial) { j = jobs.erase(j); out--; } else ++j;
    }
    t->close(keep);
    bySerial.erase(t->serial);
    order.erase(std::remove(order.begin(), order.end(), t), order.end());
    const bool wasFocus = focus == t;
    objs.erase(it);
    if (wasFocus) {
      focus = nullptr;
      if (!order.empty()) focus = order.back();
    }
  }
  // ai: XferRxCore::releaseDone: every finished transfer's TOI let go, the transfer kept
  // ai: for progress, so a header under them again is a new transfer: the file received again (2026-11-20)
  void releaseDone() {
    std::vector<uint64_t> gone;
    for (auto& [key, t] : objs) if (t->done) gone.push_back(key);
    for (uint64_t key : gone) {
      auto it = objs.find(key);
      bySerial.erase(it->second->serial);
      released.push_back(std::move(it->second));
      objs.erase(it);
    }
  }

  // ai: lizard-web/fountain-worker.mjs take: a block by its TOI (src/xfer.h). A data block goes to its object's transfer;
  // ai: any other is held until a header names its object; control blocks are read as they come.
  // ai: A block under TOI 0 (none) is ignored (SPEC 7.5).
  void block(const uint8_t* b) {
    const xfer_id_t id = xfer_id_get(b);
    const xfer_toi_t toi = xfer_toi_get(b);
    if (toi == XFER_TOI_NONE) return;
    forwarded++;
    const uint8_t* p = b + XFER_ID_BYTES;
    if (xfer_id_sbn(id) == XFER_CONTROL) { control(id, toi, p); return noteDone(); }
    if (route(id, toi, p)) { sinceOwn = 0; return noteDone(); }
    if (sinceOwn < NEVER) sinceOwn++;
    Held r{id, toi, {}};
    memcpy(r.p.data(), p, PAYLOAD);
    if (held.size() < HOLD_BLOCKS) held.push_back(r); else { held[heldAt] = r; heldAt = (heldAt + 1) % HOLD_BLOCKS; }
  }
  // ai: a data block to the transfer of its TOI; false where no transfer held names it
  bool route(xfer_id_t id, xfer_toi_t toi, const uint8_t* p) {
    auto it = objs.find(toi);
    if (it == objs.end()) return false;
    it->second->add(id, p);
    return true;
  }
  // ai: XferReceiver.add's control path: the session header; any other control id reserved, ignored
  void control(xfer_id_t id, xfer_toi_t toi, const uint8_t* p) {
    if (xfer_id_esi(id) == 0) session(p, toi);
  }
  // ai: the held blocks of a transfer's object, oldest first, into it; the rest kept
  void release(uint64_t key) {
    if (held.empty() || !objs.count(key)) return;
    auto ring = std::move(held);
    held.clear();
    const size_t at = ring.size() < HOLD_BLOCKS ? 0 : heldAt;
    heldAt = 0;
    std::vector<Held> mine;
    for (size_t k = 0; k < ring.size(); k++) {
      auto& r = ring[(at + k) % ring.size()];
      if (r.toi == key) mine.push_back(std::move(r)); else held.push_back(std::move(r));
    }
    if (!mine.empty()) sinceOwn = 0;
    for (auto& r : mine) route(r.id, r.toi, r.p.data());
  }

  // ai: XferReceiver.session: a session header (src/xfer.h) under the session's TOI: its description, the send's one
  // ai: file's map, read at once and its file opened (listed). One whose description does not read, names the session's
  // ai: own TOI as its file's, or carries a key to be understood that this build does not know, opens nothing.
  void session(const uint8_t* p, xfer_toi_t toi) {
    xfer_session_t s;
    if (xfer_session_parse(p, &s) || !toi) { refused++; return; }   // ai: not a session header this receiver reads
    std::vector<uint8_t> desc(p + s.at, p + s.at + s.len);
    xfer_toi_t t = 0;
    if (!fileOf(desc, toi, t)) return;
    descs[toi] = desc;
    listed(desc.data(), desc.size(), t);
  }
  // ai: A description's file, read under the header's TOI (src/xfer.h xfer_desc_read), its TOI into t; false
  // ai: (descsRefused) where the read refuses it or it carries a key to be understood that this build does not know
  bool fileOf(const std::vector<uint8_t>& desc, xfer_toi_t toi, xfer_toi_t& t) {
    xfer_header_t h;
    if (xfer_desc_read(desc.data(), desc.size(), toi, &h, &t) || h.critical) {
      descsRefused++;
      say("xfer: a session's description refused");
      return false;
    }
    return true;
  }
  // ai: XferReceiver.listed: a description's file (under its content's TOI): the same file held again, from this session
  // ai: or another's, opens nothing; a new one opens beside the others
  void listed(const uint8_t* desc, size_t n, xfer_toi_t toi) {
    xfer_header_t h;
    if (xfer_desc_read(desc, n, 0, &h, nullptr)) { refused++; return; }
    auto it = objs.find(toi);
    Obj* t = it == objs.end() ? nullptr : it->second.get();
    if (t && t->hdr && xfer_same_file(&h, &t->hdr->h)) return;
    openObj(desc, n, toi);
  }
  // ai: XferReceiver.open: an object opened from its description (desc, n bytes, an entry's form): refused under TOI 0
  // ai: (none), or carrying a critical attribute this build does not know; the one
  // ai: held under its key parked, its folder resumed or begun, its held blocks into it. The object, or null.
  Obj* openObj(const uint8_t* desc, size_t n, xfer_toi_t toi) {
    xfer_header_t h;
    if (xfer_desc_read(desc, n, 0, &h, nullptr) || !toi || h.critical) { refused++; return nullptr; }
    const uint64_t key = toi;
    if (auto it = objs.find(key); it != objs.end()) close(key, !it->second->failed);
    auto made = std::make_unique<Obj>(*this, ++serials, root->make());
    Obj& o = *made;
    o.store->limit = root->limit; o.store->room = root->room;
    objs[key] = std::move(made);
    order.push_back(&o);
    bySerial[o.serial] = &o;
    focus = &o;
    if (!o.open(desc, n, toi, h)) {
      // ai: its data blocks held before it
      release(key);
      o.saveMeta();
      if (o.hdr && !o.done && !o.failed) o.solveReady();
    }
    auto it = objs.find(key);
    return it != objs.end() && it->second.get() == &o ? &o : nullptr;
  }
  // ai: XferRxCore::allow: the bounds on every transfer; one the store refused is let go (it wrote nothing) and one
  // ai: whose store failed mid-way (a write refused, the disk full) parked as it stands, so its description opens it
  // ai: again (the folder resumed): a session header's description held, opened again at once
  void allow(uint64_t lim, double room) {
    root->limit = lim; root->room = room;
    std::vector<std::pair<uint64_t, bool>> again;
    for (auto& [key, t] : objs) {
      t->store->limit = lim; t->store->room = room;
      if (!t->failed) continue;
      say("xfer: " + (t->hdr ? t->hdr->name : std::string("the transfer")) + " taken afresh at its next header");
      again.push_back({key, t->store->refusal.empty()});
    }
    for (auto [key, keep] : again) close(key, keep);
    root->error.clear(); root->refusal.clear(); root->free = 0;
    std::vector<std::pair<xfer_toi_t, std::vector<uint8_t>>> files;   // ai: (a description's file's TOI, the description)
    for (const auto& [key, d] : descs) { xfer_toi_t t = 0; if (fileOf(d, key, t)) files.push_back({t, d}); }
    for (const auto& [toi, d] : files) listed(d.data(), d.size(), toi);
  }
  // ai: transfers let go by their root (an app's list's Delete: every one of that file, held or parked), or by a
  // ai: folder's id (keyOf: the root's 16 hex digits and the TOI's 12), that one
  void forget(const std::string& rootHex) {
    const bool folder = rootHex.size() != 16 && rootHex.size() != 64;
    std::vector<uint64_t> gone;
    for (auto& [key, t] : objs)
      if (t->hdr && t->hdr->root.compare(0, 16, rootHex, 0, 16) == 0 && (!folder || rootHex.compare(0, 28, keyOf(t->hdr->root, t->hdr->toi)) == 0)) gone.push_back(key);
    for (uint64_t key : gone) close(key, false);
    root->forget(rootHex);
  }

  // ai: XferReceiver.progress: the file in focus's progress as before several (its fields), every file's line (files,
  // ai: in the order their transfers opened), the descriptions', and done once every file held is
  void progress(XferProgress& p, bool perSb) const {
    p = XferProgress{};
    p.error = root->error;
    if (focus) focus->progress(p, perSb);
    p.refused = refused; p.forwarded = forwarded; p.held = uint32_t(held.size());
    p.live = sinceOwn < LIVE_BLOCKS && focus;
    p.done = false; p.doneAt = 0;
    bool all = true;
    for (const Obj* tp : order) {
      const Obj& t = *tp;
      if (!t.hdr) continue;
      XferProgress q;
      t.progress(q, false);
      XferFileProgress f;
      f.toi = t.hdr->toi; f.name = q.name; f.compression = q.compression; f.root = q.root; f.path = q.path; f.error = q.failed ? q.error : std::string();
      f.length = q.length; f.bytesIn = q.bytesIn; f.sourceBlocks = q.sourceBlocks; f.verified = q.verified; f.fraction = q.fraction;
      f.done = q.done; f.failed = q.failed; f.secs = q.secs;
      p.files.push_back(std::move(f));
      all = all && t.done;
      p.doneAt = std::max(p.doneAt, t.doneAt);
    }
    p.descsRefused = descsRefused;
    p.descs = uint32_t(descs.size());   // ai: each description held
    p.done = all && !p.files.empty();
    if (!p.done) p.doneAt = 0;
  }
};

XferRxCore::XferRxCore(std::unique_ptr<Store> store, std::function<void(const std::string&)> log, unsigned solvers,
                       std::function<void()> wake)
    : rx_(std::make_unique<Rx>(std::move(store), std::move(log), solvers, std::move(wake))) {}
void XferRxCore::settle() { rx_->settle(); }
uint32_t XferRxCore::solving() const { return rx_->solving(); }
XferRxCore::~XferRxCore() = default;
void XferRxCore::block(const uint8_t* b) { rx_->block(b); }
void XferRxCore::clearAll() { rx_->clearAll(); }
void XferRxCore::releaseDone() { rx_->releaseDone(); }
void XferRxCore::forget(const std::string& root) { rx_->forget(root); }
void XferRxCore::progress(XferProgress& p, bool perSb) const { rx_->progress(p, perSb); }
void XferRxCore::allow(uint64_t limit, double room) { rx_->allow(limit, room); }
const std::atomic<uint64_t>& XferRxCore::taken() const { return rx_->taken; }
const std::vector<uint8_t>* XferRxCore::data() const { return rx_->focus && rx_->focus->done ? rx_->focus->store->data() : nullptr; }

void XferJudge::remember(const Seen& id) {
  seenIds_.insert(id);
  if (seenIds_.size() >= SEEN_KEEP) { seenOld_ = std::move(seenIds_); seenIds_ = {}; }
}

// ai: lizard-web/recv.mjs take's dedupe: an id is its file's (its TOI), so the sets need no clearing at a new file
// ai: (cleared at each header of another root until 2026-11-19, when ids repeated across files)
bool XferJudge::take(const uint8_t* b, const Pass& pass) {
  const xfer_id_t id = xfer_id_get(b);
  if (xfer_id_sbn(id) == XFER_CONTROL) { pass(b); return false; }
  const Seen s{id, xfer_toi_get(b)};   // ai: an id under its object
  if (seenIds_.count(s) || seenOld_.count(s)) return false;
  remember(s);
  pass(b);
  return true;
}

// ai: sim/phy.mjs blockJudge, then lizard-web/recv.mjs take: a test frame's new ids are counted and remembered as the page
// ai: does, and none of its blocks passed on
FrameVerdict XferJudge::frame(const uint8_t* blocks, size_t count, const Pass& pass) {
  FrameVerdict v;
  std::vector<uint8_t> same(count, 0);
  int data = 0;
  for (size_t k = 0; k < count; k++) {
    const uint8_t* b = blocks + k * XFER_BLOCK;
    const xfer_id_t id = xfer_id_get(b);
    if (xfer_id_sbn(id) == XFER_CONTROL) continue;
    data++;
    // ai: compared as the stream is made (src/xfer.h stream_matches): a file's block stops at its first 64 bytes, one
    // ai: BLAKE3 compression (2026-11-15; the whole stream made and compared before)
    same[k] = uint8_t(stream_matches(id, b + XFER_ID_BYTES, XFER_PAYLOAD));
    v.test |= same[k] != 0;
  }
  v.judged = v.test ? data : 0;
  // ai: blockJudge's got holds the frame's control blocks first, then its data blocks: the order the worker takes them in
  for (int pass2 = 0; pass2 < 2; pass2++)
    for (size_t k = 0; k < count; k++) {
      const uint8_t* b = blocks + k * XFER_BLOCK;
      const xfer_id_t id = xfer_id_get(b);
      const bool control = xfer_id_sbn(id) == XFER_CONTROL;
      if (control != (pass2 == 0)) continue;
      if (v.test && !control && !same[k]) { v.bad++; continue; }
      v.seen++;
      if (!v.test) { v.fresh += take(b, pass); continue; }
      const Seen s{id, xfer_toi_get(b)};
      if (!control && !seenIds_.count(s) && !seenOld_.count(s)) { remember(s); v.fresh++; }
    }
  return v;
}

void XferJudge::clear() { seenIds_.clear(); seenOld_.clear(); }

}  // namespace lizard
