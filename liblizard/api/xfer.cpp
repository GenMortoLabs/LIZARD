// ai: Layer 3, the transfer (lizard.h): the sending end (core/tx/xfer_tx.h: the file's source blocks, header and
// ai: schedule, or the test stream as the senders make it) and the receiving end with no thread (core/rx/
// ai: xfer_rx_core.h: the judge, the dedupe, the fountain and the BLAKE3 checks, in a memory or folder store).
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "api.hpp"
#include "xfer_rx_core.h"
#include "refresh.h"
#include "xfer_tx.h"

extern "C" {
#include "xfer.h"
}

struct liz_tx {
  std::vector<uint8_t> owned;
  std::unique_ptr<lizard::XferTx> x;   // ai: null for the test stream
  uint32_t next = 0;    // ai: the test stream's next bare id
  uint64_t toi = 1;     // ai: and its TOI (48 bits)
  bool made = false;    // ai: its first frame made (its TOI then fixed)
  uint64_t length = 0;
  std::vector<xfer_id_t> ids;
};

struct liz_rx {
  std::unique_ptr<lizard::XferRxCore> core;
  std::unique_ptr<lizard::XferJudge> judge;
  lizard::XferProgress p;
  std::string meta;
};

using namespace liz;

static liz_tx* fileTx(std::vector<uint8_t> owned, const uint8_t* data, size_t length, const char* name, int flags, int sbLog2 = -1) {
  static const uint8_t empty = 0;
  auto t = std::make_unique<liz_tx>();
  t->owned = std::move(owned);
  if (!t->owned.empty()) data = t->owned.data();
  try {
    t->x = std::make_unique<lizard::XferTx>(length ? data : &empty, length, name ? name : "", sbLog2, "", std::function<bool()>{}, false,
                                            uint8_t(flags & LIZ_TX_ZSTD ? XFER_COMP_ZSTD : XFER_COMP_NONE));
  } catch (const std::runtime_error& e) {
    throw Error(LIZ_E_TOOBIG, e.what());   // ai: XferTx refuses a file only for its size (source blocks)
  }
  t->length = length;
  return t.release();
}

extern "C" {

LIZ_API liz_tx* liz_tx_new(const uint8_t* data, size_t length, const char* name, int flags) {
  return guardNew<liz_tx>([&] {
    need(data || !length, "the file's bytes");
    std::vector<uint8_t> owned;
    if (flags & LIZ_TX_COPY) owned.assign(data, data + length);
    return fileTx(std::move(owned), data, length, name, flags);
  });
}

LIZ_API liz_tx* liz_tx_new_sized(const uint8_t* data, size_t length, const char* name, int flags, int sbLog2) {
  return guardNew<liz_tx>([&] {
    need(data || !length, "the file's bytes");
    need(sbLog2 >= XFER_LOG2_MIN && sbLog2 <= XFER_LOG2_MAX, "sb_log2 within the header's (10 to 24)");
    std::vector<uint8_t> owned;
    if (flags & LIZ_TX_COPY) owned.assign(data, data + length);
    return fileTx(std::move(owned), data, length, name, flags, sbLog2);
  });
}

LIZ_API liz_tx* liz_tx_new_read(liz_read_fn read, void* user, uint64_t length, const char* name, int flags, int sbLog2, const char* spillDir) {
  return guardNew<liz_tx>([&] {
    need(read, "a reader");
    need(sbLog2 == -1 || (sbLog2 >= XFER_LOG2_MIN && sbLog2 <= XFER_LOG2_MAX), "sb_log2 -1 or within the header's (10 to 24)");
    xfer_header_t h{};
    const int k = sbLog2 >= 0 ? sbLog2 : xfer_sb_log2_for(length);
    if (k < 0 || xfer_header_init(&h, length, k)) throw Error(LIZ_E_TOOBIG, "the file is too large for the transfer's source blocks");
    auto t = std::make_unique<liz_tx>();
    try {
      t->x = std::make_unique<lizard::XferTx>([read, user](uint64_t off, size_t n, uint8_t* into) { return read(user, off, n, into) == 1; }, length,
                                              name ? name : "", sbLog2, spillDir ? spillDir : "", std::function<bool()>{}, false,
                                              uint8_t(flags & LIZ_TX_ZSTD ? XFER_COMP_ZSTD : XFER_COMP_NONE));
    } catch (const std::runtime_error& e) {
      throw Error(LIZ_E_IO, e.what());   // ai: the size held above: a read or the spill refused
    }
    t->length = length;
    return t.release();
  });
}

LIZ_API liz_tx* liz_tx_new_path(const char* path, const char* name, int flags) {
  return guardNew<liz_tx>([&] {
    need(path, "a path");
    const std::filesystem::path p = std::filesystem::u8path(path);
    std::ifstream f(p, std::ios::binary);
    if (!f) throw Error(LIZ_E_IO, std::string("cannot read ") + path);
    std::vector<uint8_t> owned((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) throw Error(LIZ_E_IO, std::string("cannot read ") + path);
    const std::string base = p.filename().u8string();
    const size_t length = owned.size();
    return fileTx(std::move(owned), nullptr, length, name ? name : base.c_str(), flags & ~LIZ_TX_COPY);
  });
}

LIZ_API liz_tx* liz_tx_new_test(uint32_t first) {
  return guardNew<liz_tx>([&] {
    auto t = std::make_unique<liz_tx>();
    // ai: the senders' rule: a random first id well below the control source block's, odd
    // ai: and a random TOI, nonzero, as any send's (the format has no test stream: TOI 0 is none)
    std::random_device rd;
    if (!first) first = (rd() & 0x3fffffffu) | 1u;
    t->next = first;
    const xfer_toi_t toi = xfer_toi_t(rd() & 0xffffffu) << 24 | (rd() & 0xffffffu);
    t->toi = toi ? toi : 1;
    return t.release();
  });
}

LIZ_API void liz_tx_free(liz_tx* t) { delete t; }

LIZ_API int liz_tx_seed(liz_tx* t, uint32_t seed) {
  return guard([&] {
    need(t, "a sender");
    if (!t->x) {   // ai: the test stream: its TOI from the seed, as a send's session header's
      if (t->made) throw Error(LIZ_E_STATE, "the test stream's TOI is fixed once its first frame is made");
      t->toi = lizard::sessionToi(seed);
      return LIZ_OK;
    }
    if (!t->x->reseed(seed)) throw Error(LIZ_E_STATE, "the schedule is drawn once its first frame is made");
    return LIZ_OK;
  });
}

LIZ_API int liz_refresh_places(int subch, int blocks, const int every[4], int* place) {
  return guard([&] {
    need(every, "the four delays");
    need(place, "the places");
    lizard_tier_t tier[LIZARD_TIERS];
    const int tiers = lizard_tiers_for(subch, tier);
    int laid = 0;
    for (int t = 0; t < tiers; t++) laid += tier[t].blocks;
    if (tiers <= 0 || laid != blocks) throw Error(LIZ_E_ARG, "LIZARD-" + std::to_string(subch) + " lays " + std::to_string(laid) + " blocks, not " + std::to_string(blocks));
    lizard::refresh::places(tier, tiers, blocks, every, place);
    return LIZ_OK;
  });
}

LIZ_API int liz_refresh_held(const int* place, int blocks, int total, uint64_t seq, int before, uint8_t* held) {
  return guard([&] {
    need(place, "the places");
    need(held, "held");
    if (blocks < 1 || total < 0 || total % blocks) throw Error(LIZ_E_ARG, std::to_string(total) + " blocks are no frame of " + std::to_string(blocks) + "-block symbols");
    return lizard::refresh::held(place, blocks, total, seq, before != 0, held);
  });
}

LIZ_API int liz_tx_next(liz_tx* t, int n, uint8_t* blocks) {
  return guard([&] {
    need(t && blocks && n >= 1, "a sender, at least a block and a place for them");
    t->ids.resize(size_t(n));
    if (t->x) {
      const std::string f = t->x->fault();   // ai: a worker that failed to make a source block: the file cannot all go
      if (!f.empty()) throw Error(LIZ_E_TOOBIG, f);
      t->x->frameIds(t->ids.data(), n);
    }
    else {
      for (int i = 0; i < n; i++) t->ids[i] = t->next++;
      t->made = true;
    }
    int data = 0;
    for (int i = 0; i < n; i++) {
      uint8_t* o = blocks + size_t(i) * LIZ_BLOCK;
      xfer_id_put(o, t->ids[i]);
      xfer_toi_put(o, t->x ? t->x->toiFor(t->ids[i]) : t->toi);   // ai: every id under its TOI: the file's, its session header's or the test stream's
      if (t->x) t->x->block(t->ids[i], o + LIZ_ID_BYTES);
      else stream_fill(t->ids[i], o + LIZ_ID_BYTES, LIZ_PAYLOAD);
      data += t->x ? t->x->isFileData(t->ids[i]) : 1;
    }
    return data;
  });
}

LIZ_API int liz_tx_info_get(const liz_tx* t, liz_tx_info* out) {
  return guard([&] {
    need(t && out, "a sender and a place for its info");
    liz_tx_info i{};
    i.test = !t->x;
    if (t->x) {
      // ai: the carousel's cycle once drawn (at the first frame after every source block is made), the one it will
      // ai: have from the moment they are (lizard.h: 0 until every source block is made)
      i.length = t->length; i.source_blocks = t->x->sourceBlocks(); i.cycle = t->x->cycle() ? t->x->cycle() : t->x->cycleWhenMade();
      std::memcpy(i.root, t->x->root(), sizeof i.root);
      i.seed = t->x->seed(); i.sb_log2 = t->x->sbLog2();
    }
    *out = i;
    return LIZ_OK;
  });
}

LIZ_API liz_rx* liz_rx_new(const liz_store* store) {
  return guardNew<liz_rx>([&] {
    auto r = std::make_unique<liz_rx>();
    std::unique_ptr<lizard::Store> s;
    // ai: a folder's own subfolder, as the app's receiver keeps it: nothing outside it is removed (lizard.h liz_store)
    if (store && store->dir) s = lizard::fileStore(std::string(store->dir) + "/lizard-xfer");
    else s = lizard::memoryStore(store ? store->max_bytes : 0);
    if (!s->error.empty()) throw Error(LIZ_E_IO, s->error);
    r->core = std::make_unique<lizard::XferRxCore>(std::move(s));
    r->judge = std::make_unique<lizard::XferJudge>();
    return r.release();
  });
}

LIZ_API void liz_rx_free(liz_rx* r) { delete r; }

LIZ_API int liz_rx_frame(liz_rx* r, const uint8_t* verified, int count, liz_verdict* out) {
  return guard([&] {
    need(r && count >= 0 && (verified || !count), "a receiver and the frame's blocks");
    const lizard::FrameVerdict v = r->judge->frame(verified, size_t(count), [r](const uint8_t* b) { r->core->block(b); });
    if (out) *out = liz_verdict{v.seen, v.bad, v.judged, v.fresh, v.test};
    return LIZ_OK;
  });
}

LIZ_API int liz_rx_progress(liz_rx* r, liz_progress* out) {
  return guard([&] {
    need(r && out, "a receiver and a place for its progress");
    r->core->progress(r->p);
    const auto& p = r->p;
    liz_progress o{};
    o.header = p.header; o.done = p.done; o.length = p.length; o.bytes_in = p.bytesIn; o.fraction = p.fraction;
    o.source_blocks = p.sourceBlocks; o.verified = p.verified; o.rejected = p.rejected; o.solve_ms = p.solveMs;
    o.forwarded = p.forwarded; o.done_at = p.doneAt; o.refused = p.refused; o.over = p.over;
    *out = o;
    return LIZ_OK;
  });
}

LIZ_API int liz_rx_source_blocks(liz_rx* r, uint8_t* per, int cap) {
  return guard([&] {
    need(r && (per || cap <= 0), "a receiver and a place for its source blocks");
    r->core->progress(r->p, true);
    const int n = int(r->p.per.size());
    if (cap > 0) std::memcpy(per, r->p.per.data(), size_t(std::min(n, cap)));
    return n;
  });
}

LIZ_API const char* liz_rx_meta(liz_rx* r, int which) {
  const char* s = "";
  guard([&] {
    need(r, "a receiver");
    r->core->progress(r->p);
    const auto& p = r->p;
    switch (which) {
      case LIZ_META_NAME: r->meta = p.name; break;
      case LIZ_META_ROOT: r->meta = p.root; break;
      case LIZ_META_PATH: r->meta = p.path; break;
      case LIZ_META_ERROR: r->meta = p.error; break;
      default: need(false, "which is a LIZ_META_* value");
    }
    s = r->meta.c_str();
    return LIZ_OK;
  });
  return s;
}

LIZ_API int liz_rx_data(liz_rx* r, const uint8_t** data, size_t* length) {
  return guard([&] {
    need(r && data && length, "a receiver and places for the file");
    const std::vector<uint8_t>* d = r->core->data();
    if (!d) throw Error(LIZ_E_STATE, "no file yet, or the store is a directory's");
    *data = d->data();
    *length = d->size();
    return LIZ_OK;
  });
}

LIZ_API void liz_rx_clear(liz_rx* r) {
  guard([&] {
    need(r, "a receiver");
    r->judge->clear();
    r->core->clearAll();
    return LIZ_OK;
  });
}

}  // extern "C"
