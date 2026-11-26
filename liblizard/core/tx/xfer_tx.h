// ai: The transfer's sending end (2026-10-01; the web's JS sender, sim/xfer.mjs XferSender, ported name for name, its twin until 2026-11-25), over
// ai: liblizard/src/xfer.c (the source blocks' chaining values, their root, the header) and RaptorQ (liblizard/src/rq.h), as
// ai: the receiving end (rx/xfer_rx.h) takes them. A file goes in 2^sb_log2 source blocks (16 MiB, xfer_sb_log2_for; a
// ai: file to 65,535 source blocks, about 1 TiB), a RaptorQ source block over each source block's own bytes (the transfer compresses
// ai: nothing, a client packs a file before it, pack.h); the session header on a control slot about every eighth of a
// ai: cycle (no manifest since 2026-11-21); data blocks by the carousel, shuffled in a frame.
// ai: The TOIs (src/xfer.h): the file's its content's (xfer_toi_of), whatever the seed, so every sender's blocks of the
// ai: same file are one object's; the session header's the session's, drawn from the seed (sessionToi).
// ai: frameIds gives bare ids, toiFor each one's TOI.
// ai: The carousel's seed (2026-11-14): each source block's symbols' order (each pass of its 2^24 ids its own, 2026-11-15), each round's order of the source blocks and each frame's
// ai: shuffle are drawn from it, a random one a sender unless set before the first frame (reseed), so no send repeats
// ai: another's ids or their places in the frame, and no period of the carousel meets a periodic loss (an exact round
// ai: of every source block put each source block in frames of one parity where the source blocks were twice a frame's data blocks). The
// ai: same ids and bytes as the JS sender for the same file and seed (tools/tx_check.cpp holds it to one). The source blocks
// ai: are made a worker a core (2026-11-06): a source block's encoder and chaining value depend on no other source block, so
// ai: the bytes are the serial sender's.
// ai: Streaming (2026-11-06): the constructor hashes the source blocks' own bytes (their chaining values and the root, all the
// ai: header needs) and returns; the workers make the source blocks (their encoders) behind
// ai: it while frames go. Until the last is made, a frame carries the header alone, no data (since 2026-11-16: the
// ai: source blocks made so far in turn before, which left those made first ahead for the whole transfer); then the
// ai: carousel, every source block starting it level. A receiver solves each source block as its symbols come, so it
// ai: needs nothing new. A file of one source block is made before the constructor returns, and so
// ai: is every file in the plain wasm module (no threads; the threaded one, the web's, makes them as natively).
// ai: Deferred (2026-11-16, the sender's blocks on the GPU): with `defer` the constructor makes no source block, opens no spill
// ai: and starts no worker; the sender's first configure starts them (makeOnCpu), with a sink where the GPU streams the
// ai: blocks (core/tx/gpu_stream.h): each source block's encoder as it is made, then let go (no spill, nothing held
// ai: twice), block() giving zeros for its data ids (the GPU makes them).
// ai: The spill (2026-11-14; a folder given): no source block keeps its encoder (its L intermediate symbols, about 1.02 times its
// ai: bytes); its symbols go to
// ai: one unlinked file in segments of a quarter of its blocks (at least 64), three a source block (the one being sent, the
// ai: next, and the one before, which a frame's ids may still name), about 0.75 times the file. A source block entering a
// ai: segment asks for the one after; one thread makes it (the source block's encoder again from the mapped file, the
// ai: segment's symbols, the encoder let go), a segment's airtime ahead. Memory: the encoders being made, a worker's
// ai: each. The ids and bytes are the same either way.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "rq.h"
#include "xfer.h"
}

namespace lizard {

// ai: the prepare's workers: one a physical core (the hardware threads where the system does not say, and in the
// ai: threaded wasm module; 1 in the plain one); the client's pack (pack.h) takes as many
unsigned physicalCores();

struct XferDesc;

// ai: A session's TOI from a send's seed (xfer_tx.cpp: two 24-bit mixes, 0 taken as 1), one past avoid where it is
// ai: that; the test stream's too where it is seeded (lizard.h liz_tx_seed)
xfer_toi_t sessionToi(uint32_t seed, xfer_toi_t avoid = 0);

class XferTx {
 public:
  // ai: data: the file's bytes, alive as long as this (the app maps the file); name as the header carries it (one past
  // ai: 255 bytes cut in its stem, its extension kept: src/xfer.h xfer_name_cut).
  // ai: Throws std::runtime_error where the file cannot go (too many source blocks, an encoder out of memory). Returns
  // ai: with the root and the header made (the chaining values hashed a thread a physical core); the source blocks are made
  // ai: after it by a worker a physical core, each taking the next source block, until every one is (fault() says a failure);
  // ai: the destructor stops and joins them. sbLog2 -1: xfer_sb_log2_for's for the length (16 MiB).
  // ai: spillDir: a folder for the spill (above; "" none: every encoder kept), its file made there and unlinked at once.
  // ai: stopped, where given, is asked before each source block's hash: true stops the constructor with std::runtime_error
  // ai: ("cancelled") (a make the app let go, 2026-11-16; the source blocks after it stop at the destructor as before).
  // ai: defer: no source block made (above, Deferred) until makeOnCpu.
  // ai: compression: the bytes' compression, an attribute of the description (XFER_COMP_*: 1 the default client's zstd
  // ai: stream, pack.h). Every send is one file (several go as one tar): its description, the file's one CBOR map,
  // ai: in its session header (a 255-byte name always fits).
  XferTx(const uint8_t* data, size_t length, const std::string& name, int sbLog2 = -1, const std::string& spillDir = "",
         const std::function<bool()>& stopped = {}, bool defer = false, uint8_t compression = 0);
  // ai: The file's bytes read as the transfer asks for them (2026-11-25, a file past what memory holds: the web's picked
  // ai: file through the threaded wasm module): read(off, n, into) puts the n bytes of the file at off into `into`, true
  // ai: when whole; called from the transfer's threads, several at once. Each reads one source block at a time into a
  // ai: buffer of its own (the hash's, a make's, the spill's), so with a spill the memory is a source block a thread, not
  // ai: the file. A read that fails is the hash's failure (the constructor throws) or a make's (fault). The rest as above.
  using Reader = std::function<bool(uint64_t off, size_t n, uint8_t* into)>;
  XferTx(Reader read, uint64_t length, const std::string& name, int sbLog2 = -1, const std::string& spillDir = "",
         const std::function<bool()>& stopped = {}, bool defer = false, uint8_t compression = 0);
  ~XferTx();
  // ai: The carousel and the TOIs drawn from seed in place of the random one; before the first frame only (false
  // ai: after: the ids a carousel in hand gave are not given again).
  bool reseed(uint32_t seed);
  uint32_t seed() const { return seed_s_; }
  XferTx(const XferTx&) = delete;
  XferTx& operator=(const XferTx&) = delete;

  // ai: The bare ids of the next frame of n blocks, each under toiFor's TOI (advances the carousel: a frame asked for
  // ai: twice is two frames).
  void frameIds(xfer_id_t* out, int n);
  // ai: What tagged block `id` carries: 460 bytes into out.
  void block(xfer_id_t id, uint8_t* out) const;

  uint32_t sourceBlocks() const { return sourceBlocks_; }
  uint32_t cycle() const { return cycle_; }           // ai: data blocks a cycle of the carousel (0 until every source block is made)
  // ai: the cycle the carousel will have, once every source block is made and before the carousel is drawn (at the first frame
  // ai: after): what an export waits on before it paints, so its first frame carries data (0 until then)
  uint32_t cycleWhenMade() const;
  const uint8_t* root() const { return root_; }   // ai: 32 bytes, b3sum's answer for the file
  uint32_t sbLog2() const { return hdr_.sb_log2; }   // ai: the source block size's power of two (the header's k)
  // ai: the file's TOI (src/xfer.h xfer_toi_of, its content's), and the TOI an id goes under: the session header the
  // ai: session's, every other the file's
  xfer_toi_t toi() const { return toi_; }
  xfer_toi_t toiFor(xfer_id_t id) const { return xfer_id_sbn(id) == XFER_CONTROL ? headerToi() : toi_; }
  // ai: a block of the file's own data (a source block's): what a cycle counts
  bool isFileData(xfer_id_t id) const { return xfer_id_sbn(id) != XFER_CONTROL; }
  std::string rootHex() const;
  bool made() const { return made_.load(std::memory_order_acquire) == sourceBlocks_; }   // ai: every source block made
  // ai: Waits until every source block is made (tx_check ids, whose frames from then are the serial sender's); throws a
  // ai: worker's failure.
  void waitMade();
  std::string fault() const;   // ai: a worker's failure, "" none: the transfer cannot finish
  // ai: Deferred: the source blocks made by the CPU's workers from here (a thread of their own opening the spill first, where
  // ai: there is no sink); once (a call after the first does nothing). sink: each source block's encoder as it is made, on its
  // ai: worker, let go after (false: stopped, the transfer's fault).
  using Sink = std::function<bool(uint32_t c, const rq_enc_t* enc)>;
  void makeOnCpu(Sink sink = {});
  uint32_t madeCount() const { return made_.load(std::memory_order_acquire); }
  size_t sbSize() const { return size_; }        // ai: 2^sb_log2: source block c starts at c sbSize() in the file
  uint64_t length() const { return length_; }        // ai: the file's bytes
  size_t sbBytes(uint32_t c) const { return len_[c]; }
  uint32_t controls() const { return 1; }   // ai: the header, the one control block
  // ai: the header's 460 bytes; the source symbols over every source block (sum of K)
  const uint8_t* header() const { return header_; }
  uint32_t sourceSymbols() const;
  // ai: the TOI its session headers go under: the session's (its description's)
  xfer_toi_t headerToi() const;

 private:
  xfer_id_t next();   // ai: the carousel's next data id
  void seedWith(uint32_t seed);   // ai: each source block's first symbol and the rounds' generator from seed
  void drawTois();                // ai: the file's TOI from its content (xfer_toi_of); once the root is known
  void newRound();    // ai: the next round's order of the source blocks
  void make(uint32_t c);   // ai: source block c made (a worker's)
  void writeHeader();      // ai: header_ from hdr_
  void makeAll();          // ai: makeOnCpu's thread: the spill opened, then the workers
  void work();        // ai: a worker: the next source block until none is left, or a failure stops them
  std::vector<uint32_t> weights() const;   // ai: each source block's visits a cycle (carousel's)
  void buildCarousel();    // ai: the carousel, once every source block is made (frameIds')
  void warm(xfer_id_t* out, int n);   // ai: a frame's ids before that
  void took(uint32_t c);             // ai: a symbol of source block c named (the spill's count, and its next segment asked for)
  void writeSegment(uint32_t c, const rq_enc_t* enc, uint64_t n, uint32_t base);   // ai: segment n's symbols from base into its slot
  void spiller();                    // ai: the spill's thread: segments asked for, in order

  XferTx(const uint8_t* data, Reader read, uint64_t length, const std::string& name, int sbLog2, const std::string& spillDir,
         const std::function<bool()>& stopped, bool defer, uint8_t compression);
  // ai: source block c's bytes: in place in the file (data_), else read into buf (read_)
  const uint8_t* bytesOf(uint32_t c, std::vector<uint8_t>& buf) const;
  size_t size_;
  uint64_t length_ = 0;
  uint32_t sourceBlocks_ = 0;
  const uint8_t* data_ = nullptr;   // ai: the file in memory, or null where read_ reads it
  Reader read_;
  std::atomic<uint32_t> cycle_{0};
  std::vector<uint8_t> cvs_;     // ai: every source block's chaining value (two source blocks or more)
  std::vector<size_t> len_;      // ai: a source block's own bytes
  std::vector<uint32_t> K_;      // ai: RaptorQ source symbols a source block
  struct EncFree { void operator()(rq_enc_t* e) const; };
  std::vector<std::unique_ptr<rq_enc_t, EncFree>> enc_;   // ai: a source block's encoder (none where the spill holds its symbols)
  std::unique_ptr<XferDesc> desc_;   // ai: a file alone: its description (the session header, its TOI)
  uint8_t root_[32]{}, header_[XFER_PAYLOAD]{};
  xfer_header_t hdr_{};
  bool headed_ = false;   // ai: hdr_ whole (the root and name in)
  // ai: the file's TOI (its content's)
  xfer_toi_t toi_ = 0;
  // ai: the carousel (Carousel): each source block's key (its symbols' order, esiOf) and the symbols it sent, the round's
  // ai: place, each source block's weight a round and accumulator, the round's order of the source blocks; the seed, and the rounds'
  // ai: generator from it (xorshift32)
  std::vector<uint32_t> key_, w_, acc_, perm_;
  std::vector<uint64_t> cnt_;
  uint32_t i_ = 0, max_ = 0, seed_s_ = 0, rs_ = 1;
  // ai: the control cycle: how many data blocks between control slots, since the last, which control is next
  uint32_t every_ = 1, since_ = UINT32_MAX, frame_ = 0;   // ai: since_: frames since the last control block (UINT32_MAX: none yet)
  // ai: streaming: a source block's slots written before its ready_ is set (the workers'); next_, the next source block a worker takes
  std::unique_ptr<std::atomic<uint8_t>[]> ready_;
  std::atomic<uint32_t> made_{0}, next_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
  mutable std::mutex faultMu_;
  std::string fault_;
  bool carouselBuilt_ = false;   // ai: frameIds' own: the carousel built
  // ai: the source blocks being made (makeOnCpu called, or the constructor's), makeOnCpu's thread, its sink
  std::atomic<bool> making_{false};
  std::thread maker_;
  Sink sink_;
  std::string spillDir_;     // ai: deferred: the spill's folder and length, opened by makeAll
  uint64_t spillBytes_ = 0;
  // ai: the spill: its file (-1 none), each source block's segment length and slots' offset, its key (base_), the
  // ai: symbols of it named (taken_, frameIds'), the key its segments hold (segBase_) and the newest segment written
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

// ai: A session's description (src/xfer.h, SPEC 7.7 and 7.11): the send's one file's CBOR map
// ai: (src/xfer.h xfer_desc_write; no session-wide keys yet), in the session header, read from one block. Under the
// ai: session's TOI (the sender's); the session header written once, here.
struct XferDesc {
  XferDesc(std::vector<uint8_t> desc, xfer_toi_t toi);
  xfer_toi_t toi = 0;
  std::vector<uint8_t> bytes;        // ai: the description
  uint8_t header[XFER_PAYLOAD]{};    // ai: the session header
  // ai: what an id under its TOI carries: the session header (other ids zeros)
  void block(xfer_id_t id, uint8_t* out) const;
};

}  // namespace lizard
