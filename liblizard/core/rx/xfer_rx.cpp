// ai: XferRx (xfer_rx.h): the receiving end's threaded form, the app's: the judge on the producer's thread, a queue,
// ai: and the core (xfer_rx_core.h) on a worker thread with files under the store folder, so take never waits on a
// ai: solve. The chain itself is xfer_rx_core.cpp's.
#include "xfer_rx.h"

#include "xfer_tx.h"   // ai: physicalCores

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace lizard {

// ai: the queue's bound, 31 MB: a 16 MiB source block's solve and hash are tens of ms, so this is seconds of backlog at any
// ai: format's rate; past it a block is lost (counted), which the fountain covers as any other loss
static constexpr size_t QUEUE_MAX = 1 << 16;

XferRx::XferRx(const std::string& storeDir, std::function<void(const std::string&)> log) : log_(std::move(log)) {
  // ai: a solver a physical core, at most SOLVERS_MAX: a transfer's last source blocks, which the interleave completes together,
  // ai: solve side by side. Past 4 the solves share DRAM's bandwidth (a 16 MiB solve streams ~36 MB, past a CCD's L3):
  // ai: on the 9950X a 1 GiB file done 1.59 s after its last block with 4, 1.06 s with 16 at 600 MB more and 1.7x the
  // ai: work (2026-11-14), so 4. LIZ_SOLVERS=<n> the lab's count, uncapped (0: each solve on the worker)
  constexpr unsigned SOLVERS_MAX = 4;
  const char* ns = getenv("LIZ_SOLVERS");
  const unsigned solvers = ns && *ns ? unsigned(atoi(ns)) : std::min(physicalCores(), SOLVERS_MAX);
  rx_ = std::make_unique<XferRxCore>(fileStore(storeDir + "/lizard-xfer"), log_, solvers, [this] {
    { std::lock_guard<std::mutex> lk(mu_); settle_ = true; }
    cv_.notify_one();
  });
  judge_ = std::make_unique<XferJudge>();
  rx_->progress(snap_, sourceBlocks_);
  worker_ = std::thread([this] { run(); });
}

XferRx::~XferRx() {
  { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
  cv_.notify_all();
  idle_.notify_all();
  worker_.join();
}

void XferRx::push(uint8_t kind, const uint8_t* b) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (kind == 0 && queue_.size() >= QUEUE_MAX) { lost_++; return; }
    queue_.push_back(Item{kind, {}});
    if (b) memcpy(queue_.back().b.data(), b, XFER_BLOCK);
  }
  cv_.notify_one();
}

void XferRx::run() {
  std::deque<Item> batch;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      busy_ = false;
      idle_.notify_all();
      cv_.wait(lk, [&] { return stop_ || !queue_.empty() || settle_; });
      if (stop_) return;
      batch.swap(queue_);
      settle_ = false;
      busy_ = true;
    }
    rx_->settle();
    for (size_t k = 0; k < batch.size(); k++) {
      if (batch[k].kind != 0) {
        if (batch[k].kind == 1) rx_->clearAll();
        else if (batch[k].kind == 4) rx_->releaseDone();
        else if (batch[k].kind == 3) {
          uint64_t lim;
          double room;
          memcpy(&lim, batch[k].b.data(), 8);
          memcpy(&room, batch[k].b.data() + 8, 8);
          rx_->allow(lim, room);
        } else rx_->forget(std::string(reinterpret_cast<const char*>(batch[k].b.data())));
        { std::lock_guard<std::mutex> lk(pmu_); rx_->progress(snap_, sourceBlocks_); }
        { std::lock_guard<std::mutex> lk(mu_); clearsDone_++; }
        idle_.notify_all();
        continue;
      }
      rx_->block(batch[k].b.data());
      // ai: a snapshot every 256 blocks, so progress moves through a long backlog, at most every SNAP_MS (2026-11-15: a
      // ai: snapshot is O(source blocks), and a 65,535-source block file's at every 256 blocks was a worker's second each second)
      if ((k & 255) == 255) snapshot(false);
    }
    batch.clear();
    rx_->settle();
    snapshot(true);
  }
}

// ai: the worker's snapshot for progress(): idle (the queue empty, the worker about to wait) always, so a reader after
// ai: drain() sees every block; otherwise at most every SNAP_MS (the apps poll about every 100 ms)
static constexpr double SNAP_MS = 50;
void XferRx::snapshot(bool atEnd) {
  const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (t - snapAt_ < SNAP_MS) {
    if (!atEnd) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (!queue_.empty()) return;
  }
  snapAt_ = t;
  std::lock_guard<std::mutex> lk(pmu_);
  rx_->progress(snap_, sourceBlocks_);
}

bool XferRx::take(const uint8_t* b) {
  std::lock_guard<std::mutex> lk(prod_);
  return judge_->take(b, [this](const uint8_t* p) { push(0, p); });
}

FrameVerdict XferRx::frame(const std::vector<Block>& verified) {
  std::lock_guard<std::mutex> lk(prod_);
  return judge_->frame(verified.empty() ? nullptr : verified.front().bytes.data(), verified.size(), [this](const uint8_t* p) { push(0, p); });
}

XferProgress XferRx::progress() {
  XferProgress p;
  { std::lock_guard<std::mutex> lk(pmu_); p = snap_; }
  std::lock_guard<std::mutex> lk(mu_);
  p.queued = uint32_t(queue_.size());
  p.lost = lost_;
  return p;
}

// ai: the producers' lock held through the worker's clear, so no frame lands between the judge's and the core's
void XferRx::clear() {
  std::lock_guard<std::mutex> lk(prod_);
  judge_->clear();
  ask(1, nullptr);
}

void XferRx::forget(const std::string& root) {
  std::lock_guard<std::mutex> lk(prod_);
  judge_->clear();
  std::array<uint8_t, XFER_BLOCK> b{};
  memcpy(b.data(), root.data(), std::min(root.size(), b.size() - 1));
  ask(2, b.data());
}

void XferRx::releaseDone() {
  std::lock_guard<std::mutex> lk(prod_);
  judge_->clear();
  ask(4, nullptr);
}

void XferRx::allow(uint64_t limit, double room) {
  std::lock_guard<std::mutex> lk(prod_);
  std::array<uint8_t, XFER_BLOCK> b{};
  memcpy(b.data(), &limit, 8);
  memcpy(b.data() + 8, &room, 8);
  ask(3, b.data());
}

// ai: a clear, a forget or an allow queued behind what was queued before it, and waited for
void XferRx::ask(uint8_t kind, const uint8_t* b) {
  std::unique_lock<std::mutex> lk(mu_);
  queue_.push_back(Item{kind, {}});
  if (b) memcpy(queue_.back().b.data(), b, XFER_BLOCK);
  const uint64_t ticket = ++clearsAsked_;
  cv_.notify_one();
  idle_.wait(lk, [&] { return clearsDone_ >= ticket || stop_; });
}

void XferRx::drain() {
  std::unique_lock<std::mutex> lk(mu_);
  idle_.wait(lk, [&] { return stop_ || (queue_.empty() && !busy_ && !settle_ && !rx_->solving()); });
}

}  // namespace lizard
