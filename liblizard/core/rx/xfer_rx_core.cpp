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
static constexpr int64_t LIVE_BLOCKS = 4096, NEVER = INT64_MAX / 2;
static constexpr size_t HOLD_BLOCKS = 1024, SEEN_KEEP = 1 << 16;
// ai: meta.json written again every this many blocks stored (about 1 MB) or a second of storing, whichever comes first,
// ai: as well as at each chunk verified and at a park, so a receiver stopped mid-chunk leaves an app's list a line at
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

// ai: A chunk's refused symbols (a rejected chunk's: one of them passed its CRC wrongly), sorted: rare, so a sorted
// ai: list. Since 2026-11-15 a chunk keeps no set of the symbols it holds: a send's ids are each pass of a chunk's 2^24
// ai: in an order of its own (core/tx/xfer_tx.cpp symbolOf), so they never repeat within a send; a camera's repeats are
// ai: the judge's (seenIds); the rare repeat across sends (K / 2^24 of a chunk's) is dropped at its solve. The dense runs
// ai: of bits held until then (Bits, 2026-11-14) would have grown to 2 MiB a chunk over shuffled ids.
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
struct Ban {
  std::vector<uint32_t> s;
  bool has(uint32_t i) const { return std::binary_search(s.begin(), s.end(), i); }
  // ai: a rejected chunk's records' symbols, all at once: appended, radix-sorted, repeats dropped (an insert a symbol
  // ai: kept the list sorted at O(n) each, O(n^2) for a chunk's 35,621)
  void addAll(const std::vector<uint8_t>& recs, size_t bytes) {
    for (size_t o = 0; o + REC <= bytes; o += REC) s.push_back(uint32_t(xfer_id_get(recs.data() + o)) & XFER_SYMBOL_MASK);
    radix24(s);
    s.erase(std::unique(s.begin(), s.end()), s.end());
  }
};

// ai: sim/fountain.mjs Decoder: a chunk's RaptorQ decoder (src/rq.h), its symbols the ids' 18 bits; it takes every
// ai: symbol it is given and solves from K on
struct Dec {
  rq_dec_t* d = nullptr;
  uint32_t bytes = 0;
  ~Dec() { rq_dec_free(d); }
  bool create(uint32_t messageBytes) {
    bytes = messageBytes;
    d = rq_dec_new(messageBytes, PAYLOAD, 1u << XFER_SYMBOL_BITS);
    return d != nullptr;
  }
  // ai: 1 once the chunk is whole, 0 for more, -1 out of memory
  int add(uint32_t sym, const uint8_t* p) { return rq_dec_add(d, sym, p); }
  // ai: a symbol kept with no solve tried (0, -1 out of memory): a rank-short chunk's records, its next add solving
  int hold(uint32_t sym, const uint8_t* p) { return rq_dec_hold(d, sym, p); }
  std::vector<uint8_t> recover() {
    std::vector<uint8_t> o(bytes);
    if (rq_dec_recover(d, o.data())) o.clear();
    return o;
  }
};

struct XferRxCore::Rx {
  // ai: recs: the blocks the store holds for the chunk; mark: how many it held when this transfer's header was last
  // ai: read. The blocks after the mark are provisional: a sender that moved to another file sends that file's blocks
  // ai: under the same ids until its header is read, so a park cuts each chunk back to its mark.
  // ai: asked: the blocks forwarded when its solve was asked for (doneAt); askedRecs: the records held then, all a job
  // ai: solves (an inline solve's), the records after them the blocks a solver's time let in
  // ai: dups: of the chunk's records, those its last solve found repeating an earlier one's symbol (a repeat across
  // ai: sends: the chunk keeps no set of its symbols); it solves again once recs - dups reaches K
  struct Ch { Ban ban; int state = COLLECTING; std::unique_ptr<Dec> dec; uint32_t recs = 0, mark = 0, askedRecs = 0, dups = 0; uint64_t asked = 0; };
  struct Hdr {
    std::array<uint8_t, XFER_PAYLOAD> bytes;
    xfer_header_t h;
    std::string name, type, root;
    uint64_t size = 0;
    uint32_t mcount = 0, mhave = 0;
    // ai: len and K: each chunk's bytes and source symbols, the header's (a chunk goes as its own bytes); a
    // ai: chunk is solved once the list checks (one chunk: at once)
    std::vector<uint32_t> K, len;
    std::vector<uint8_t> mgot, list;
    // ai: the list once it checks, shared with the solvers (a header of another root lets the Hdr go under them)
    std::shared_ptr<const std::vector<uint8_t>> listDone;
    bool listOk = false;
    bool ready() const { return h.chunks == 1 || listOk; }
  };
  std::unique_ptr<Store> store;
  std::function<void(const std::string&)> log;
  std::atomic<uint64_t> gen{0};   // ai: headers of another root read (a judge's dedupe clears on a change)
  std::atomic<uint64_t> taken{0};   // ai: XferRxCore::taken: data blocks added to a chunk still collecting, ever
  std::string lastRoot;   // ai: lizard-web/recv.mjs xferRoot: the root the dedupe was last cleared for
  std::unique_ptr<Hdr> hdr;
  // ai: each chunk's state by its number (2026-11-15: a flat table, sized to the header's chunks, where an ordered map
  // ai: took a tree walk a block and a lookup a chunk a snapshot); in the chunks' order, as the map was (the JS Map is in
  // ai: insertion order: only the order of solves within one block's take differs). A chunk never touched is COLLECTING
  // ai: with nothing held, which every reader takes as the map's absent entry.
  std::vector<Ch> ch;
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
  std::vector<std::pair<xfer_id_t, std::array<uint8_t, XFER_PAYLOAD>>> held;
  size_t heldAt = 0;
  uint64_t forwarded = 0, doneAt = 0;

  // ai: The solvers (2026-11-14): a chunk whose K symbols are in is solved by a thread of a pool, a physical core each
  // ai: (XferRx's), so the chunks the interleave leaves to the end of a transfer are solved side by side, where they
  // ai: went one after another on this thread. A job carries the chunk's records, its length and what its check needs
  // ai: (the header, the list); the solver holds the records in a decoder, lets them go, solves, recovers and checks the
  // ai: chunk; the result is taken back on this thread (settle), where the store and the chunk's state are touched. At
  // ai: most one job a solver is out (its records read as it goes): the rest wait as chunk numbers, so a burst holds
  // ai: about 36 MB a solver at 16 MiB chunks, not every chunk's records at once. No pool (solvers 0: the library's,
  // ai: and the wasm's, which has no threads): the job runs here, inline, as before. `epoch` moves at every reset, so a
  // ai: solve of a transfer let go is dropped when it comes back.
  // ai: at: the blocks forwarded when the chunk's solve was asked for (doneAt, below)
  struct Job { uint32_t c = 0; uint64_t epoch = 0, at = 0; uint32_t len = 0, nrecs = 0; std::vector<uint8_t> recs; xfer_header_t h{}; std::shared_ptr<const std::vector<uint8_t>> list; };
  struct Done { uint32_t c = 0; uint64_t epoch = 0, at = 0; uint32_t nrecs = 0, dups = 0; bool solved = false, ok = false, oom = false, few = false; std::vector<uint8_t> out; std::unique_ptr<Dec> dec; double ms = 0; };
  unsigned solvers = 0;
  std::function<void()> wake;
  std::mutex jmu;
  std::condition_variable jcv;
  std::deque<Job> jobs;
  std::deque<Done> dones;
  bool jstop = false;
  std::vector<std::thread> pool;
  std::deque<uint32_t> waiting;          // ai: chunks SOLVING with no solver yet, in the order they came
  std::atomic<uint32_t> waitN{0};        // ai: its size, for another thread's solving()
  std::atomic<uint32_t> out{0};          // ai: jobs handed to the solvers and not yet taken back
  uint64_t epoch = 0;

  Rx(std::unique_ptr<Store> s, std::function<void(const std::string&)> l, unsigned n, std::function<void()> w)
      : store(std::move(s)), log(std::move(l)), wake(std::move(w)) {
    store->log = log;
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
    d.c = j.c; d.epoch = j.epoch; d.at = j.at; d.nrecs = j.nrecs;
    const double t = nowMs();
    // ai: the solve over the records where they lie (src/rq_plan.h): the plan from their symbols' ESIs, the symbol work
    // ai: reading each payload in its record, the chunk recovered straight into its bytes; no copy into a decoder
    // ai: (2026-11-15, the S26: a 1 GiB burst's tail 2.28 and 2.35 s against 2.54 and 2.89 held in a decoder, the solve
    // ai: work 13 to 16% less)
    // ai: each symbol once (a repeat across sends, K / 2^24 of a chunk's, stored as it came: the chunk keeps no set of
    // ai: its symbols since 2026-11-15), the first record of an ESI kept, the order the records came in
    // ai: the records' indexes radix-sorted by their 24-bit ESIs (two stable passes of 12 bits, so an ESI's first
    // ai: record stays first), each ESI's first kept, then back in arrival order by one pass over the kept marks
    const size_t all = j.recs.size() / REC;
    std::vector<uint32_t> key(all), at(all), tmp(all);
    for (size_t k = 0; k < all; k++) key[k] = uint32_t(xfer_id_get(j.recs.data() + k * REC)) & XFER_SYMBOL_MASK, at[k] = uint32_t(k);
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
    // ai: fewer than K symbols once repeats are dropped: no solve and no decoder (a decoder would hold the chunk's
    // ai: symbols in memory, and near a resumed transfer's end every chunk would at once); the chunk collects that many
    // ai: more (commit)
    d.dups = uint32_t(all - n);
    if (n < xfer_blocks(j.len)) {
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
        d.ok = xfer_chunk_check(&j.h, j.c, d.out.data(), d.out.size(), j.h.chunks >= 2 && j.list ? j.list->data() : nullptr) == 0;
      }
    } else if (pr < 0) {
      d.oom = true;
    } else {
      // ai: a rank short (about 0.5% of chunks at K): a decoder holding the records, back to take the blocks as they come
      auto dec = std::make_unique<Dec>();
      if (!dec->create(j.len)) d.oom = true;
      for (size_t k = 0; !d.oom && k < n; k++)
        if (dec->hold(esi[k], sym[k]) < 0) d.oom = true;
      if (!d.oom) d.dec = std::move(dec);
    }
    d.ms = nowMs() - t;
    return d;
  }
  // ai: the solves finished since, taken in; then the chunks waiting given the solvers freed
  void settle() {
    std::deque<Done> got;
    { std::lock_guard<std::mutex> lk(jmu); got.swap(dones); }
    for (auto& d : got) { out--; commit(std::move(d)); }
    dispatch();
  }
  void dispatch() {
    while (!waiting.empty() && out < solvers) {
      const uint32_t c = waiting.front();
      waiting.pop_front();
      waitN--;
      if (!hdr || c >= ch.size() || ch[c].state != SOLVING) continue;
      Job j = jobOf(c);
      out++;
      { std::lock_guard<std::mutex> lk(jmu); jobs.push_back(std::move(j)); }
      jcv.notify_one();
    }
  }
  Job jobOf(uint32_t c) {
    Job j;
    j.c = c; j.epoch = epoch; j.at = ch[c].asked; j.len = hdr->len[c]; j.h = hdr->h; j.list = hdr->listDone;
    j.recs = store->read(c);
    j.recs.resize(std::min(j.recs.size(), size_t(ch[c].askedRecs) * REC));
    j.nrecs = uint32_t(j.recs.size() / REC);
    return j;
  }
  uint32_t solving() const { return out + waitN; }
  uint64_t lastAsked = 0;   // ai: the latest block at which a chunk verified since the reset became complete (doneAt)
  // ai: reject for a job's result: the symbols the solve took refused (as the inline reject's), the blocks stored
  // ai: since its ask kept as the chunk's first fresh ones, as they would be after an inline reject
  void rejectSolved(uint32_t c, Ch& s, uint32_t nrecs) {
    const auto recs = store->read(c);
    const size_t cut = std::min(recs.size(), size_t(nrecs) * REC);
    s.ban.addAll(recs, cut);
    s.state = COLLECTING; rejected++;
    s.dec.reset();
    store->drop(c); s.recs = s.mark = s.dups = 0;
    say("xfer: chunk " + std::to_string(c) + " rejected");
    for (size_t o = cut; o + REC <= recs.size(); o += REC) {
      const uint32_t sym = uint32_t(xfer_id_get(recs.data() + o));
      if (s.ban.has(sym)) continue;
      if (!store->append(c, sym, recs.data() + o + XFER_ID_BYTES)) return stop();
      s.recs++;
    }
    if (hdr && hdr->ready() && full(c, s)) solve(c);
  }
  // ai: a job's result, on this thread
  void commit(Done d) {
    ms += d.ms;
    if (d.epoch != epoch || !hdr || d.c >= ch.size() || ch[d.c].state != SOLVING) return;   // ai: a transfer let go since
    Ch& s = ch[d.c];
    if (d.few) {
      s.dups = d.dups;
      s.state = COLLECTING;
      if (full(d.c, s)) solve(d.c);   // ai: the repeats' worth came while it solved
      return;
    }
    if (d.oom) {
      // ai: the JS throws here: out of memory, the chunk collected afresh
      say("xfer: no RaptorQ decoder for chunk " + std::to_string(d.c));
      return rejectSolved(d.c, s, d.nrecs);
    }
    if (d.solved) {
      if (!d.ok) return rejectSolved(d.c, s, d.nrecs);
      if (!store->writeOut(uint64_t(d.c) * hdr->size, d.out.data(), d.out.size())) return stop();
      store->noteVerified(d.c);
      // ai: the file done at the block that completed its last chunk to complete, as inline solves have it, not when a
      // ai: solver came back (which the solvers' timing moves)
      lastAsked = std::max(lastAsked, d.at);
      if (verified + 1 == hdr->h.chunks && !doneAt) doneAt = lastAsked;
      return accept(d.c, s);
    }
    // ai: RaptorQ wanted a block more than K (about 1% of chunks): the blocks stored since the job's feed it, the rest
    // ai: as they come (add), kept until the chunk is verified
    s.dec = std::move(d.dec); s.state = HOT; over++;
    if (s.recs > d.nrecs) {
      const auto recs = store->read(d.c);
      for (size_t o = size_t(d.nrecs) * REC; o + REC <= recs.size(); o += REC)
        if (s.dec->add(uint32_t(xfer_id_get(recs.data() + o)), recs.data() + o + XFER_ID_BYTES) == 1) return recovered(d.c);
    }
  }
  void say(const std::string& s) { if (log) log(s); }
  // ai: K symbols in hand, as far as the chunk knows: its records less the repeats its last solve found
  bool full(uint32_t c, const Ch& s) const { return hdr && s.recs >= hdr->K[c] + s.dups; }

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
    const xfer_id_t id = xfer_id_get(b);
    const uint8_t* p = b + XFER_ID_BYTES;
    if (xfer_id_chunk(id) == XFER_CONTROL) {
      add(id, p);
      if (id == XFER_ID_HEADER && hdr) { sinceHeader = 0; if (!held.empty()) release(); }
    } else if (hdr && sinceHeader++ < LIVE_BLOCKS) {
      add(id, p);
    } else {
      std::pair<xfer_id_t, std::array<uint8_t, XFER_PAYLOAD>> r{id, {}};
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
      for (uint32_t c = 0; c < ch.size(); c++) { Ch& s = ch[c]; if (s.state != VERIFIED && s.recs > s.mark) { store->truncate(c, s.mark); s.recs = s.mark; s.dups = 0; } }
      saveMeta();
    }
    ch.clear(); raw.clear(); hdr.reset();
    // ai: the solves of this transfer dropped: those queued at once, those out when they come back (epoch)
    epoch++;
    lastAsked = 0;
    waiting.clear();
    waitN = 0;
    {
      std::lock_guard<std::mutex> lk(jmu);
      out -= uint32_t(jobs.size());
      jobs.clear();
    }
    verified = rejected = manifestRejected = over = 0;
    ms = 0; firstMs = doneMs = 0; done = false; failed = false; path.clear(); sinceMeta = 0;
    if (keep) store->park(); else store->clear();
  }
  // ai: XferRxCore::allow: a transfer the store refused is let go (it wrote nothing), and one whose store failed
  // ai: mid-way (a write refused, the disk full) parked as it stands, so its header opens it again (the folder resumed)
  void allow(uint64_t lim, double room) {
    store->limit = lim;
    store->room = room;
    if (!failed) return;
    say("xfer: " + (hdr ? hdr->name : std::string("the transfer")) + " taken afresh at its next header");
    if (!store->refusal.empty()) reset(false);
    else {
      failed = false;
      reset(true);
    }
    store->error.clear();
    store->refusal.clear();
    store->free = 0;
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
    progress(p, false);
    char f[32];
    snprintf(f, sizeof f, "%.4f", p.fraction);
    store->noteMeta("{\"name\":" + jstr(p.name) + ",\"type\":" + jstr(p.type) + ",\"length\":" + std::to_string(p.length) +
                    ",\"chunks\":" + std::to_string(p.chunks) + ",\"verified\":" + std::to_string(p.verified) + ",\"root\":\"" + p.root +
                    "\",\"fraction\":" + f + ",\"bytesIn\":" + std::to_string(p.bytesIn) + "}");
  }

  Ch& state(uint32_t c) { if (c >= ch.size()) ch.resize(size_t(c) + 1); return ch[c]; }

  // ai: XferReceiver.add
  void add(xfer_id_t id, const uint8_t* p) {
    const uint32_t c = xfer_id_chunk(id), sym = xfer_id_symbol(id);
    if (c == XFER_CONTROL) {
      if (sym == 0) return header(p);
      if (sym <= XFER_MAX_MANIFEST) return manifestBlock(sym - 1, p);
      return;   // ai: reserved: ignored
    }
    // ai: a store that keeps transfers keeps them by their header's folder: no data block before a header (block()
    // ai: holds those for the next header itself)
    if (done || failed || (hdr && c >= hdr->h.chunks) || (!hdr && store->keeps())) return;
    if (!firstMs) firstMs = nowMs();
    Ch& s = state(c);
    if (s.state == VERIFIED || s.ban.has(sym)) return;
    taken.fetch_add(1, std::memory_order_relaxed);
    if (s.dec) {
      // ai: kept until the chunk is verified, by any store: a resume makes the decoder again from them
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
    if (s.state == COLLECTING && hdr && hdr->ready() && full(c, s)) solve(c);   // ai: once the list checks (one chunk: at once)
  }

  // ai: XferReceiver.header, the light's only: a different header is a new transfer, and everything held is let go
  void header(const uint8_t* p) {
    if (hdr && !memcmp(p, hdr->bytes.data(), PAYLOAD)) { for (Ch& s : ch) s.mark = s.recs; return; }
    auto h = std::make_unique<Hdr>();
    if (xfer_header_parse(p, &h->h)) { refused++; return; }   // ai: not a header this receiver reads
    // ai: the same file under another name or type (another sender's header, xfer_same_file): the transfer in hand, as
    // ai: its own header again; the name and type it read first kept
    if (hdr && xfer_same_file(&h->h, &hdr->h)) { for (Ch& s : ch) s.mark = s.recs; return; }
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
    for (uint32_t c = chunks; c < ch.size(); c++) store->drop(c);   // ai: blocks of another transfer
    ch.resize(chunks);
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
      s.recs = s.mark = uint32_t(store->read(c).size() / REC);
      had += s.recs;
    }
    say("xfer: resumed " + hdr->name + ", " + std::to_string(verified) + " of " + std::to_string(chunks) + " chunks verified, " +
        std::to_string(had) + " blocks in hand");
    if (verified == chunks) { finish(); return true; }
    return false;
  }
  // ai: XferReceiver.solveReady: every chunk with its blocks in hand, now that the list checks
  void solveReady() {
    std::vector<uint32_t> ready;
    for (uint32_t c = 0; c < ch.size(); c++) if (ch[c].state == COLLECTING && full(c, ch[c])) ready.push_back(c);
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
    h.listDone = std::make_shared<const std::vector<uint8_t>>(h.list);
    solveReady();
  }

  // ai: XferReceiver.solve: chunk c's K symbols are in. With solvers, waiting for one (dispatch); without, solved here.
  void solve(uint32_t c) {
    Ch& s = ch[c];
    // ai: its distinct symbols counted from the store's ids first (142 KB read, not the records' 16.8 MB; 2026-11-15):
    // ai: fewer than K (repeats across sends) and the chunk collects on, no records read and no solver taken. The A/B
    // ai: that asked for it: a sender started again at half, 1 GiB, the tail 12% longer with each chunk's first try
    // ai: reading its records for nothing (research/build/idorder-2026-11-15/ab_tail.cpp)
    std::vector<uint32_t> ids = store->ids(c);
    for (auto& v : ids) v &= XFER_SYMBOL_MASK;
    radix24(ids);
    const uint32_t distinct = uint32_t(std::unique(ids.begin(), ids.end()) - ids.begin());
    if (distinct < hdr->K[c]) {
      s.dups = s.recs - std::min(s.recs, distinct);
      return;
    }
    s.state = SOLVING;
    s.asked = forwarded;
    s.askedRecs = s.recs;
    if (solvers) { waiting.push_back(c); waitN++; return dispatch(); }
    commit(run(jobOf(c)));
  }
  void recovered(uint32_t c) {
    lastAsked = std::max(lastAsked, forwarded);   // ai: a chunk fed on (HOT) completes at this block
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
    s.state = VERIFIED; s.ban = Ban(); verified++;
    store->drop(c); s.recs = s.mark = s.dups = 0;   // ai: a kept chunk's blocks (store.h keeps), nothing otherwise
    if (verified == hdr->h.chunks) finish(); else saveMeta();
  }
  // ai: a chunk that decoded to the wrong bytes: which block passed its CRC wrongly cannot be told, so every symbol that
  // ai: went into it is refused from now on and the chunk collected again from fresh ids
  void reject(uint32_t c, Ch& s) {
    const auto recs = store->read(c);   // ai: the symbols that went into it (a fed-on chunk's, all it holds)
    s.ban.addAll(recs, recs.size());
    s.state = COLLECTING; rejected++;
    s.dec.reset();
    store->drop(c); s.recs = s.mark = s.dups = 0;
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
  // ai: perChunk: per filled too (liz_rx_chunks' and the checks'; the snapshots for the app's polls take none, 2026-11-15)
  void progress(XferProgress& p, bool perChunk) const {
    p = XferProgress{};
    p.error = store->error;
    p.refusal = store->refusal; p.limit = store->limit; p.free = store->free;
    p.refused = refused; p.forwarded = forwarded; p.doneAt = doneAt; p.held = uint32_t(held.size());
    if (!hdr) return;
    const Hdr& h = *hdr;
    p.header = true; p.live = sinceHeader < LIVE_BLOCKS;
    p.name = h.name; p.type = h.type; p.root = h.root; p.length = h.h.length; p.chunks = h.h.chunks;
    p.verified = verified; p.manifest = h.mcount; p.manifestHave = h.mhave;
    p.rejected = rejected; p.manifestRejected = manifestRejected; p.over = over; p.solveMs = ms;
    p.done = done; p.failed = failed; p.path = path;
    p.secs = firstMs ? ((done ? doneMs : nowMs()) - firstMs) / 1000 : 0;
    if (perChunk) p.per.resize(h.h.chunks);
    for (uint32_t c = 0; c < h.h.chunks; c++) {
      const Ch* s = c < ch.size() ? &ch[c] : nullptr;
      if (perChunk) p.per[c] = !s ? 0 : s->state == VERIFIED ? 255 : uint8_t(std::min<uint64_t>(99, 100ull * (s->recs - std::min(s->recs, s->dups)) / std::max<uint32_t>(1, h.K[c])));
      if (s) p.bytesIn += s->state == VERIFIED ? h.len[c] : std::min<uint64_t>(h.len[c], uint64_t(s->recs - std::min(s->recs, s->dups)) * PAYLOAD);
      if (s && s->state == SOLVING) p.solving++;
    }
    p.inHand = !done && h.ready() && p.solving > 0 && p.verified + p.solving == h.h.chunks;
    // ai: XferReceiver.bytesIn and fractionDone
    if (done) p.bytesIn = h.h.length;
    p.fraction = done ? 1 : h.h.length ? std::min(1.0, double(p.bytesIn) / double(h.h.length)) : 0;
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
void XferRxCore::forget(const std::string& root) { rx_->forget(root); }
void XferRxCore::progress(XferProgress& p, bool perChunk) const { rx_->progress(p, perChunk); }
void XferRxCore::allow(uint64_t limit, double room) { rx_->allow(limit, room); }
const std::atomic<uint64_t>& XferRxCore::rootGen() const { return rx_->gen; }
const std::atomic<uint64_t>& XferRxCore::taken() const { return rx_->taken; }
const std::vector<uint8_t>* XferRxCore::data() const { return rx_->done ? rx_->store->data() : nullptr; }

void XferJudge::remember(xfer_id_t id) {
  seenIds_.insert(id);
  if (seenIds_.size() >= SEEN_KEEP) { seenOld_ = std::move(seenIds_); seenIds_ = {}; }
}

// ai: lizard-web/recv.mjs take's dedupe, and its reset on a new root (the page's xfer message; here the core's generation)
bool XferJudge::take(const uint8_t* b, const Pass& pass) {
  if (const uint64_t g = gen_.load(); g != genSeen_) { genSeen_ = g; seenIds_.clear(); seenOld_.clear(); }
  const xfer_id_t id = xfer_id_get(b);
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
  int data = 0;
  for (size_t k = 0; k < count; k++) {
    const uint8_t* b = blocks + k * XFER_BLOCK;
    const xfer_id_t id = xfer_id_get(b);
    if (xfer_id_chunk(id) == XFER_CONTROL) continue;
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
