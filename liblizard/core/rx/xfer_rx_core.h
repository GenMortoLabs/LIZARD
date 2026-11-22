// ai: The transfer's receiving end with no thread of its own (2026-10-03, for the library, which runs it on the caller's
// ai: thread and in a wasm build with no threads at all; xfer_rx.h is the app's threaded form over the same two parts):
// ai:   XferRxCore  lizard-web/fountain-worker.mjs take (the live-header rule, the held ring) and sim/xfer.mjs XferReceiver
// ai:               (source block states, bans, a new transfer on a header with other bytes): CRC-verified blocks in, a file out,
// ai:               each source block checked against the file's BLAKE3 root, kept in a Store (store.h: files or memory)
// ai:   XferJudge   sim/phy.mjs blockJudge (a frame is the test stream's when a verified data block carries its id's
// ai:               BLAKE3, and its other data blocks are then bad) and lizard-web/recv.mjs take (the dedupe: two rotating id
// ai:               sets of 65,536, control ids always passed on, a test frame's blocks never; an id is its file's, by its
// ai:               TOI, so nothing clears them but a Clear)
// ai: The layouts are src/xfer.h's, with RaptorQ (src/rq.h), BLAKE3 (liblizard/vendor/blake3)
// ai: and zstd (liblizard/vendor/zstd, liblizard/zstd/shim.c; 2026-10-05: a source block sent as a frame is decompressed on recovery, before its hash).
// ai: A solve (RaptorQ and BLAKE3 over a 16 MiB source block) is tens of ms: run inside the block() that completes the source block,
// ai: or with solvers (2026-11-14, XferRx's: a physical core each) on a pool's thread, its result taken in by settle().
// ai: Since 2026-10-22 a store that keeps transfers (store.h keeps, a file store) holds each in a folder of its own, and
// ai: a transfer's header read again, by a later receiver on the same folder, resumes it (its verified source blocks with
// ai: their chaining values, and the blocks of the rest, from the folder). Since 2026-11-19 several transfers at once, a TOI each (src/xfer.h),
// ai: each from a session header's description (SPEC 7.11), each in a store the handed-in one makes (store.h make).
// ai: A file's TOI is its content's (src/xfer.h xfer_toi_of): the same file from any sender, in any session, is one
// ai: transfer and one folder (the root's 16 hex digits and the TOI's 12); no manifest (each source block's chaining
// ai: value made as it decodes, the root checked at the last).
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "store.h"

extern "C" {
#include "xfer.h"
}

namespace lizard {

struct Block { std::array<uint8_t, XFER_BLOCK> bytes; };
static_assert(sizeof(Block) == XFER_BLOCK, "a vector of Blocks is the packed blocks");

// ai: What a frame's verified blocks came to: seen, those not judged bad (control blocks and repeats in); bad, a test
// ai: frame's data blocks that are not the stream's; test, the frame was the test stream's; judged, the data blocks
// ai: held to the stream (all of them on a test frame, else 0); fresh, data ids new to the dedupe (a test frame's
// ai: counted too, for goodput, but none of its blocks passed on).
struct FrameVerdict { int seen = 0, bad = 0, judged = 0, fresh = 0; bool test = false; };

// ai: One file a receiver holds (2026-11-19): its line in XferProgress.files, XferProgress' own fields for it
struct XferFileProgress {
  uint64_t toi = 0;   // ai: 48 bits (src/xfer.h)
  std::string name, root, path, error;
  int compression = 0;   // ai: the description's (src/xfer.h XFER_COMP_*): 1 the bytes a zstd stream (pack.h)
  uint64_t length = 0, bytesIn = 0;
  uint32_t sourceBlocks = 0, verified = 0;
  double fraction = 0, secs = 0;
  bool done = false, failed = false;
};

// ai: The file in focus (the last file whose header opened a transfer: a single file's view, as before several) in its
// ai: own fields; every file held in files; done once every file held is.
struct XferProgress {
  bool header = false;        // ai: a header read (this transfer's)
  bool live = false;          // ai: and a data block of a transfer held within the last LIVE_BLOCKS data blocks
  std::string name, root;   // ai: the header's name; root, BLAKE3 of the file as hex (b3sum's)
  int compression = 0;       // ai: the header's (src/xfer.h XFER_COMP_*)
  uint64_t length = 0;        // ai: the file's bytes
  uint32_t sourceBlocks = 0, verified = 0;
  std::vector<uint8_t> per;   // ai: each source block's share in hand, 0 to 99, 255 verified
  double fraction = 0;        // ai: sim/xfer.mjs fractionDone: bytesIn over length, 0 to 1
  uint64_t bytesIn = 0;       // ai: the file's bytes in: a verified source block's all, another's blocks in hand to its length
  bool done = false;
  // ai: solving: source blocks whose K symbols are in, their solve waiting for a solver or running (with solvers; a source block
  // ai: solved inside block() is never seen so); inHand: the list checked and every source block verified or solving, the file
  // ai: not yet done: the light's part over, the solves finishing (the apps' Decoding; a solve a symbol short sends its
  // ai: source block back to collecting, about one in 200)
  uint32_t solving = 0;
  bool inHand = false;
  std::string path;           // ai: the verified file, once done (a file store's; "" for memory)
  std::string error;          // ai: the store's failure in this transfer (store.h), "" none
  bool failed = false;        // ai: the store could not keep the file: the transfer ended there, never done
  // ai: the store's refusal of the file (store.h, 2026-11-15): "limit", over the largest file the receiver takes;
  // ai: "room", too little free space for it; "" none. limit: the largest file then (0 none); free: the bytes free then
  std::string refusal;
  uint64_t limit = 0, free = 0;
  // ai: the transfer's own clock, s: from its first data block taken (the blocks held for its header count from that
  // ai: header) to its last source block verified, or to this snapshot while it runs; 0 before a data block (2026-10-10; an
  // ai: app's own poll clock carried one file's start into the next file's time). Time with no blocks coming (a camera
  // ai: stopped mid-transfer) counts.
  double secs = 0;
  // ai: counts, the web's names: rejected source blocks, headers refused, source blocks RaptorQ wanted a block
  // ai: more than K for, solve ms; forwarded blocks handed to the worker (control ones in), doneAt how many had been
  // ai: when the file finished (0 before), held the blocks waiting for a header, queued those not yet taken, lost those
  // ai: the queue had no room for (the last two the threaded form's)
  uint32_t rejected = 0, refused = 0, over = 0;
  double solveMs = 0;
  uint64_t forwarded = 0, doneAt = 0, lost = 0;
  uint32_t held = 0, queued = 0;
  // ai: every file held, in the order their transfers opened; the sessions' descriptions held, and those refused
  // ai: (src/xfer.h xfer_desc_read)
  std::vector<XferFileProgress> files;
  uint32_t descs = 0, descsRefused = 0;
};

class XferRxCore {
 public:
  // ai: solvers: threads that solve source blocks side by side (0: each on the caller's thread, inside block()); wake: called
  // ai: from a solver's thread when a solve is done, so the caller's thread calls settle().
  XferRxCore(std::unique_ptr<Store> store, std::function<void(const std::string&)> log = nullptr, unsigned solvers = 0,
             std::function<void()> wake = nullptr);
  ~XferRxCore();
  XferRxCore(const XferRxCore&) = delete;
  XferRxCore& operator=(const XferRxCore&) = delete;

  // ai: One block a judge passed on (the fountain worker's take).
  void block(const uint8_t* blk);
  // ai: With solvers: the solves done since, taken in (a source block verified, refused or fed on), and the source blocks waiting
  // ai: handed to the solvers freed. On the thread that calls block().
  void settle();
  // ai: Source blocks with the solvers or waiting for one (0 without solvers).
  uint32_t solving() const;
  // ai: Every transfer held and every file received let go (the page's Clear); transfers a store parked stay.
  void clearAll();
  // ai: Every finished transfer's TOIs let go, the transfer kept in progress (its line, the file in focus): the same
  // ai: file in the light again is received again, as a download clicked twice (2026-11-20; the receive session's,
  // ai: when it stops the camera for a file in).
  void releaseDone();
  // ai: Transfers let go by their root (hex, or its first 16 digits: every one of that file), or one by its folder's id
  // ai: (22 hex digits): held, or parked by the store (store.h keeps; 2026-10-22, an app's Delete of an unfinished transfer).
  void forget(const std::string& root);
  // ai: perSb: XferProgress.per made too (an O(source blocks) list: the C API's liz_rx_source_blocks and the checks ask for it)
  void progress(XferProgress& p, bool perSb = false) const;
  // ai: The receiver's bounds (store.h limit and room), and a transfer the store refused for them, or failed mid-way
  // ai: (parked as it stands), taken afresh at its next header (a Start after it, or the bounds changed: refused again
  // ai: if it is still past them, failing again if the store still cannot write).
  void allow(uint64_t limit, double room);
  // ai: Data blocks added to a source block still collecting since this was made, never reset (2026-10-10): what a transfer
  // ai: actually took, a block held for a header counted once its header lets it in, none of a source block already verified
  // ai: or of a symbol refused. Any thread may read it; a receiver's rate is its change over a window.
  const std::atomic<uint64_t>& taken() const;
  // ai: The file in focus's bytes, once done, where the store keeps them in memory, else null.
  const std::vector<uint8_t>* data() const;

  struct Rx;

 private:
  std::unique_ptr<Rx> rx_;
};

class XferJudge {
 public:
  using Pass = std::function<void(const uint8_t*)>;
  // ai: A frame's verified blocks, packed (472 B each): judged, deduplicated, and each one the transfer should take
  // ai: handed to pass, control blocks first.
  FrameVerdict frame(const uint8_t* blocks, size_t count, const Pass& pass);
  // ai: One block past the judge: true when its id is new data. A control block is always passed on (its id is every
  // ai: transfer's; only its bytes say which) and is never new.
  bool take(const uint8_t* blk, const Pass& pass);
  // ai: The ids forgotten (the page's Clear).
  void clear();

 private:
  // ai: an id under its object: one bare id of two objects is two blocks
  struct Seen { xfer_id_t id; xfer_toi_t toi; bool operator==(const Seen& o) const { return id == o.id && toi == o.toi; } };
  struct SeenHash { size_t operator()(const Seen& s) const { return std::hash<uint64_t>()(s.id ^ (s.toi * 0x9e3779b97f4a7c15ull)); } };
  std::unordered_set<Seen, SeenHash> seenIds_, seenOld_;
  void remember(const Seen& s);
};

}  // namespace lizard
