// ai: The transfer's sending end (2026-10-01): liblizard/sim/xfer.mjs XferSender ported name for name, over
// ai: liblizard/src/xfer.c (the chunks' chaining values, their root, the header and the manifest) and Wirehair
// ai: through the web's shim (liblizard/wirehair/shim.cpp), as the receiving end (rx/xfer_rx.h) takes them. A file
// ai: goes in 2^chunk_log2 chunks (4 MiB up to a file of 64 GiB, xfer_chunk_log2_for), a fountain over each chunk's
// ai: own bytes (header version 4, 2026-11-14: the transfer compresses nothing, a client packs a file before it, pack.h);
// ai: the header (with the two Wirehair seed attempts, every chunk's but the last and the last's) every other control
// ai: slot and the manifest (each chunk's chaining value) between, the control cycle within an eighth of a lap; data
// ai: blocks by the schedule, shuffled in a frame.
// ai: The schedule's seed (2026-11-14): each chunk's first symbol, each round's order of the chunks and each frame's
// ai: shuffle are drawn from it, a random one a sender unless set before the first frame (reseed), so no send repeats
// ai: another's ids or their places in the frame, and no period of the schedule meets a periodic loss (an exact round
// ai: of every chunk put each chunk in frames of one parity where the chunks were twice a frame's data blocks). The
// ai: same ids and bytes as the JS sender for the same file and seed (tools/tx_check.cpp holds it to one). The chunks
// ai: are made a worker a core (2026-11-06): a chunk's Wirehair encoder and chaining value depend on no other chunk, so
// ai: the bytes are the serial sender's.
// ai: Streaming (2026-11-06): the constructor hashes the chunks' own bytes (their chaining values and the root, all the
// ai: header needs with the two seeds) and returns; the workers make the chunks (Wirehair) behind it while frames go. Until the
// ai: last is made, a frame carries one control block (the header, or a manifest block once its 12 chunks are made)
// ai: and the data of the chunks made so far in turn, fresh symbols each; then the schedule as before, every chunk's
// ai: symbols going on from where they were. A receiver keeps a chunk's blocks from the header on and solves once the
// ai: manifest is whole, so it needs nothing new. A file of one chunk is made before the constructor returns, and so
// ai: is every file under emscripten (no threads).
// ai: The spill (2026-11-14; a folder given): no chunk keeps its encoder (about 1.75 times its bytes); its symbols go to
// ai: one unlinked file in segments of a quarter of its blocks (at least 64), three a chunk (the one being sent, the
// ai: next, and the one before, which a frame's ids may still name), about 0.75 times the file. A chunk entering a
// ai: segment asks for the one after; one thread makes it (the chunk's encoder again from the mapped file, the
// ai: segment's symbols, the encoder let go), a segment's airtime ahead. Memory: the encoders being made, a worker's
// ai: each. The ids and bytes are the same either way.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "xfer.h"
}

namespace lizard {

// ai: the prepare's workers: one a physical core (the hardware threads where the system does not say; 1 under
// ai: emscripten); the client's pack (pack.h) takes as many
unsigned physicalCores();

class XferTx {
 public:
  // ai: data: the file's bytes, alive as long as this (the app maps the file); name and type as the header carries them
  // ai: (a name cut to 255 bytes at a character, a type kept only where it is printable ASCII of at most 158).
  // ai: Throws std::runtime_error where the file cannot go (too many chunks, a chunk Wirehair seeded apart). Returns
  // ai: with the root and the header made (the chaining values hashed a thread a physical core); the chunks are made
  // ai: after it by a worker a physical core, each taking the next chunk, until every one is (fault() says a failure);
  // ai: the destructor stops and joins them. chunkLog2 -1: xfer_chunk_log2_for's for the length (4 MiB to 64 GiB).
  // ai: spillDir: a folder for the spill (above; "" none: every encoder kept), its file made there and unlinked at once.
  XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2 = -1,
         const std::string& spillDir = "");
  ~XferTx();
  // ai: The schedule drawn from seed in place of the random one; before the first frame only (false after: the ids a
  // ai: schedule in hand gave are not given again).
  bool reseed(uint32_t seed);
  uint32_t seed() const { return seed_s_; }
  XferTx(const XferTx&) = delete;
  XferTx& operator=(const XferTx&) = delete;

  // ai: The ids of the next frame of n blocks (advances the schedule: a frame asked for twice is two frames).
  void frameIds(uint32_t* out, int n);
  // ai: What block `id` carries: 472 bytes into out.
  void block(uint32_t id, uint8_t* out) const;

  uint32_t chunks() const { return chunks_; }
  uint32_t lap() const { return lap_; }           // ai: data blocks a lap of the schedule (0 until every chunk is made)
  const uint8_t* root() const { return root_; }   // ai: 32 bytes, b3sum's answer for the file
  std::string rootHex() const;
  bool made() const { return made_.load(std::memory_order_acquire) == chunks_; }   // ai: every chunk made
  // ai: Waits until every chunk is made (tx_check ids, whose frames from then are the serial sender's); throws a
  // ai: worker's failure.
  void waitMade();
  std::string fault() const;   // ai: a worker's failure, "" none: the transfer cannot finish

 private:
  uint32_t next();   // ai: the schedule's next data id (sim/xfer.mjs Schedule.next)
  void seedWith(uint32_t seed);   // ai: each chunk's first symbol and the rounds' generator from seed
  void newRound();    // ai: the next round's order of the chunks
  void make(uint32_t c);   // ai: chunk c made (a worker's)
  void work();        // ai: a worker: the next chunk until none is left, or a failure stops them
  void schedule();    // ai: the schedule, once every chunk is made (frameIds')
  void warm(uint32_t* out, int n);   // ai: a frame's ids before that
  void took(uint32_t c);             // ai: a symbol of chunk c named (the spill's count, and its next segment asked for)
  void writeSegment(uint32_t c, void* enc, uint64_t n, uint32_t base);   // ai: segment n's symbols from base into its slot
  void spiller();                    // ai: the spill's thread: segments asked for, in order

  size_t size_;
  uint32_t chunks_ = 0;
  const uint8_t* data_ = nullptr;
  std::atomic<uint32_t> lap_{0};
  std::vector<uint8_t> cvs_;     // ai: every chunk's chaining value (two chunks or more)
  std::vector<size_t> len_;      // ai: a chunk's own bytes
  std::vector<uint32_t> K_;      // ai: Wirehair source blocks a chunk (1: sent unfountained)
  uint8_t seedFull_ = 0, seedLast_ = 0;   // ai: the header's Wirehair seed attempts: every chunk's but the last, the last's
  struct WhFree { void operator()(void* e) const; };
  std::vector<std::unique_ptr<void, WhFree>> enc_;   // ai: a chunk's encoder, null where K is 1
  std::vector<std::vector<uint8_t>> one_;            // ai: the bytes of a chunk of one block, else empty
  uint8_t root_[32]{}, header_[XFER_PAYLOAD]{};
  std::vector<std::vector<uint8_t>> manifest_;
  // ai: the schedule (Schedule): symbols a chunk sent, the round's place, each chunk's weight a round and accumulator,
  // ai: the round's order of the chunks; the seed, and the rounds' generator from it (xorshift32)
  std::vector<uint32_t> sym_, w_, acc_, perm_;
  uint32_t i_ = 0, max_ = 0, seed_s_ = 0, rs_ = 1;
  // ai: the control cycle: how many data blocks between control slots, since the last, which control is next
  uint32_t every_ = 1, since_ = UINT32_MAX, ctl_ = 0, frame_ = 0;
  // ai: streaming: a chunk's slots written before its ready_ is set, a manifest block's before its mready_ (the
  // ai: workers'); left_, chunks of a manifest group not yet made; next_, the next chunk a worker takes
  std::unique_ptr<std::atomic<uint8_t>[]> ready_, mready_;
  std::unique_ptr<std::atomic<uint32_t>[]> left_;
  std::atomic<uint32_t> made_{0}, next_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
  mutable std::mutex faultMu_;
  std::string fault_;
  bool scheduled_ = false;   // ai: frameIds' own: the schedule built
  uint32_t turn_ = 0;        // ai: the warm frames' turn over the chunks made
  // ai: the spill: its file (-1 none), each chunk's segment length and slots' offset, its first symbol (base_), the
  // ai: symbols of it named (taken_, frameIds'), the base its segments hold (segBase_) and the newest segment written
  // ai: (have_, -1 none); the segments asked for, the thread that makes them
  intptr_t spill_ = -1;
  std::vector<uint32_t> seg_, base_, segBase_;
  std::vector<uint64_t> off_, taken_;
  std::unique_ptr<std::atomic<int64_t>[]> have_;
  mutable std::mutex spillMu_;
  mutable std::condition_variable spillCv_;
  std::deque<std::pair<uint32_t, uint64_t>> asks_;
  std::thread spillThread_;
};

}  // namespace lizard
