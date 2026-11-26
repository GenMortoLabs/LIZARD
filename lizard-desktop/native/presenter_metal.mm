// ai: The desktop sender's presenter on macOS (presenter.h; 2026-11-25, Metal's own in place of the Vulkan presenter
// ai: through MoltenVK): the CAMetalLayer over the code area's canvas (macos.mm) presented on the sender's Metal device
// ai: (the painter's frames and the presents on one GPU), one thread of its own as presenter.cpp's. The same schedule and
// ai: tally (presenter_common.h): a picture taken from the sender's ring (Sender::takeSlot, no copy) when one is due,
// ai: held round(refresh / fps) refreshes where that is whole within 1%, else a grid; each present one refresh, the
// ai: layer's display sync its FIFO (three drawables, the most Core Animation gives: the presents queue two refreshes
// ai: ahead, against eight images on Vulkan); a present's time on screen the drawable's presented time (its handler, the
// ai: twin of present_wait's), a present never shown (presentedTime 0) unknown. Each refresh the drawable is cleared to
// ai: the code's paper (a render pass's clear) and the picture put in its place by Scaling's choice:
// ai:   - Bilinear (the default): the ring frame expanded into the slot's BGRA8 texture (expand.slang), then stretched
// ai:     to its place on the drawable by the texture unit's bilinear filter (scale_linear.slang, Metal's alone: Metal's
// ai:     blit encoder copies 1:1, where Vulkan's blit filters);
// ai:   - Lanczos: the grey frame copied into the slot's buffer, stretched by Lanczos-3 across (scale_h.slang) and down
// ai:     (scale_v.slang) into a BGRA8 texture the place's size, copied 1:1 into the drawable;
// ai:   - None: the expanded texture copied 1:1, its middle where the place is smaller.
// ai: The kernels gen's MSL (out/msl/send_*.metal and their maps, wg::readModule), compiled at the start. No present
// ai: timing grid (Vulkan's VK_EXT_present_timing has no Metal twin used here: `timing` false, as on RADV).
#include "presenter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "macos.h"
#include "presenter_common.h"
#include "priority.h"
#include "sender.h"
#include "wg.h"

namespace lizard {

using namespace pc;

namespace {

// ai: a kernel of the sender's tree (wg::readModule's MSL and map): its pipeline, workgroup size and argument slots
struct Kernel {
  id<MTLComputePipelineState> ps = nil;
  MTLSize local = MTLSizeMake(1, 1, 1);
  wg::Module m;
  uint32_t at(uint32_t binding, char kind) const {
    if (binding < m.bind.size())
      for (const auto& s : m.bind[binding]) if (s.kind == kind) return s.index;
    throw std::runtime_error("the kernel " + m.name + " binds nothing at " + std::to_string(binding));
  }
  uint32_t push() const {
    if (!m.hasPush) throw std::runtime_error("the kernel " + m.name + " has no push constants");
    return m.push.index;
  }
};

Kernel kernel(id<MTLDevice> dev, const std::string& assets, const std::string& name) {
  Kernel k;
  k.m = wg::readModule(assets, name);
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
  else o.fastMathEnabled = NO;
  NSError* err = nil;
  NSString* src = [[NSString alloc] initWithBytes:k.m.msl.data() length:k.m.msl.size() encoding:NSUTF8StringEncoding];
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
  if (!lib) throw std::runtime_error("the presenter's kernel " + name + " did not compile: " + std::string(err ? [err.localizedDescription UTF8String] : "no reason given"));
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:k.m.entry.c_str()]];
  if (!fn) throw std::runtime_error("the presenter's kernel " + name + " has no " + k.m.entry);
  k.ps = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!k.ps) throw std::runtime_error("the presenter's kernel " + name + ": " + std::string(err ? [err.localizedDescription UTF8String] : "no pipeline"));
  k.local = MTLSizeMake(k.m.local[0], k.m.local[1], k.m.local[2]);
  return k;
}

// ai: the host's steady clock (ms, nowMs) of a Core Animation host time (s, CACurrentMediaTime's: the drawables'
// ai: presented times), by the two read together
double hostToSteadyMs(double caSeconds) {
  const double off = nowMs() - CACurrentMediaTime() * 1000.0;
  return caSeconds * 1000.0 + off;
}

}  // namespace

// ai: A picture's home (presenter.cpp's Slot): Lanczos's grey buffer (four pixels a word, rows rw words apart), or
// ai: Bilinear's and None's BGRA8 texture, which the expand kernel writes; three, so the one filled is never in flight.
struct MtlSlot {
  id<MTLBuffer> buf = nil;
  id<MTLTexture> img = nil;
  uint32_t rw = 0;
  bool grey = false, none = false;
  int w = 0, h = 0;
  int64_t usedBy = -2;
};

struct Presenter::Impl {
  PresentTarget target;
  Sender* sender = nullptr;
  std::mutex* senderLock = nullptr;
  PresentOptions opt;
  std::atomic<int> fps{60};
  std::atomic<double> size{1.0};
  std::atomic<double> edge{0.0};
  std::atomic<bool> paused{false}, quit{false};
  std::atomic<bool> vsync{false}, vsyncDirty{false};
  std::atomic<int> scaling{0};
  std::mutex wakeMu;
  std::condition_variable wake;
  std::thread thread;

  wg::Device* W = nullptr;
  std::string name;
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  CAMetalLayer* layer = nil;
  CGSize drawable = CGSizeMake(0, 0);
  Kernel expand, scaleH, scaleV, linear;
  id<MTLSamplerState> lin = nil;
  // ai: frames in flight: N, each with its command buffer (waited before its objects are used again), the ring buffer
  // ai: it read, and the Lanczos stretch's own objects
  static constexpr int N = 3;
  dispatch_semaphore_t inflight = nil;
  id<MTLCommandBuffer> cbs[N] = {nil, nil, nil};
  std::shared_ptr<wg::Buffer> ringHeld[N];
  struct Stretch {
    id<MTLBuffer> mid = nil, tab = nil;
    id<MTLTexture> out = nil;
    size_t tabCap = 0, midCap = 0;
    int key[4] = {0, 0, 0, 0}, tx = 0, ty = 0, yo = 0;
  } stretches[N];
  std::vector<MtlSlot> slots = std::vector<MtlSlot>(3);
  static constexpr int MARK_BITS = 12, MARK_PX = 8, MARK_W = 2 * MARK_BITS * MARK_PX;
  id<MTLBuffer> markBuf = nil;
  Tally win;
  FILE* csv = nullptr;

  // ai: the presents waiting for their presented handler (by id) and those whose handler came, in a box the handlers
  // ai: hold (a drawable's handler may come after the presenter is gone)
  struct ShownBox {
    std::mutex mu;
    std::map<uint64_t, Shown> sent;
    std::vector<Shown> done;
    std::atomic<uint64_t> shownId{0};
  };
  std::shared_ptr<ShownBox> box = std::make_shared<ShownBox>();

  std::atomic<bool> snapAsk{false};
  std::mutex snapMu;
  std::condition_variable snapCv;
  bool snapDone = false;
  std::vector<unsigned char> snapPx;
  int snapW = 0, snapH = 0;

  double t0 = 0, lastWin = 0, due = 0, modeHz = 0, measured = 0, early = 0;
  std::vector<double> firstAcq;
  int cur = -1;
  uint64_t picture = 0, presents = 0, pid = 0, winPresents = 0;
  int64_t frame = 0;
  uint32_t epoch = 0;
  bool wasPaused = false;
  struct Held {
    bool on = false, decided = false;
    uint64_t where = 0;
    double origHz = 0, hz = 0;
    int fps = 0;
  } held;

  std::mutex statsMu;
  std::string statsJson, error;

  void say(const std::string& m) { if (opt.log) opt.log(m); }
  double refreshHz() const { return measured > 0 ? measured : modeHz > 0 ? modeHz : early > 0 ? early : 60.0; }
  double period() const {
    double p = refreshHz() / std::max(1, fps.load());
    const double r = std::round(p);
    if (r >= 1 && std::fabs(p - r) < 0.01 * r) p = r;
    return p;
  }

  void setup();
  void run();
  void teardown();
  void presentOnce(bool live);
  void queryMode() { modeHz = mac::layerRefreshHz(target.metalLayer); }
  void holdMode();
  void releaseMode();
  void rateMoved();
  void drain();
  void window(double secs, bool last = false);
  void takeSnapshot();
  // ai: frame f's command buffer done (every frame up to frame - N is: each present waits the one N before it)
  void waitFrame(int64_t f) {
    if (f < 0 || f <= frame - N) return;
    id<MTLCommandBuffer> c = cbs[f % N];
    if (c) [c waitUntilCompleted];
  }
};

// ai: The sender's device (wg's Metal device, made by Sender::prepareGpu): the layer set up for it, a queue of this
// ai: presenter's own, the four kernels.
void Presenter::Impl::setup() {
  W = sender->device();
  if (!W) throw std::runtime_error("the sender has no Metal device to present from" + (sender->prepareWhy().empty() ? std::string() : ": " + sender->prepareWhy()));
  if (!target.metalLayer) throw std::runtime_error("no layer to present to");
  name = W->name;
  dev = (__bridge id<MTLDevice>)W->mtl;
  queue = [dev newCommandQueue];
  if (!queue) throw std::runtime_error(name + ": no command queue for the presenter");
  layer = (__bridge CAMetalLayer*)target.metalLayer;
  // ai: on this thread, which has no run loop: an explicit transaction, then a flush that commits it (macos.mm)
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  layer.device = dev;
  layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  layer.framebufferOnly = NO;   // ai: written by compute and blits, not a render pass's alone
  layer.maximumDrawableCount = 3;
  layer.displaySyncEnabled = YES;
  CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  layer.colorspace = srgb;   // ai: an 8-bit UNORM grey as painted, as the Vulkan chain's SRGB_NONLINEAR
  CGColorSpaceRelease(srgb);
  [CATransaction commit];
  [CATransaction flush];
  const std::string assets = sender->assets();
  expand = kernel(dev, assets, "send_expand");
  scaleH = kernel(dev, assets, "send_scale_h");
  scaleV = kernel(dev, assets, "send_scale_v");
  linear = kernel(dev, assets, "send_scale_linear");
  MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
  sd.minFilter = MTLSamplerMinMagFilterLinear;
  sd.magFilter = MTLSamplerMinMagFilterLinear;
  sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
  sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
  sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
  sd.normalizedCoordinates = YES;
  lin = [dev newSamplerStateWithDescriptor:sd];
  inflight = dispatch_semaphore_create(N);
  if (opt.mark) markBuf = [dev newBufferWithLength:(NSUInteger)MARK_W * MARK_PX * 4 * N options:MTLResourceStorageModeShared];
  if (!opt.csv.empty()) {
    csv = std::fopen(opt.csv.c_str(), "w");
    if (!csv) throw std::runtime_error(opt.csv + " could not be written");
    std::fprintf(csv, "id,picture,fresh,acquired_ms,submitted_ms,shown_ms,queued,target_ms\n");
  }
  queryMode();
  t0 = lastWin = nowMs();
  if (csv) std::fprintf(csv, "# t0_steady_ms=%s\n", fmt("%.3f", t0).c_str());
  window(0);
  say("presenter: " + name + " (Metal), a queue of its own, times from the drawables' presented handlers" +
      (modeHz > 0 ? ", the mode's refresh " + fmt("%.3f", modeHz) + " Hz" : ""));
}

// ai: VSync's hold (presenter.h setVsync) on the main display, as presenter.cpp's macOS branch was
void Presenter::Impl::holdMode() {
  if (!vsync) { releaseMode(); held.decided = false; return; }
  const int f = std::max(1, fps.load());
  auto fpsText = [&] { return std::to_string(f) + " pictures a second"; };
  const uint32_t display = mac::mainDisplay();
  if (held.on && held.where != display) { releaseMode(); held.decided = false; }
  if (held.decided && held.where == display && held.fps == f) return;
  double now = 0;
  const std::vector<double> hz = mac::displayRates(display, &now);
  // ai: a built-in panel's mode says 0: its refresh is the one measured
  const double origHz = held.on ? held.origHz : now > 0 ? now : measured;
  if (origHz <= 0) return;
  held.decided = true; held.where = display; held.fps = f;
  if (evenAt(origHz, f)) { releaseMode(); return; }
  const int b = pickRate(hz, f);
  if (b < 0) { say("presenter: VSync: no mode of this display shows " + fpsText() + " evenly"); releaseMode(); return; }
  if (now > 0 && std::fabs(hz[static_cast<size_t>(b)] - now) < 0.01) return;
  if (!mac::setDisplayRate(display, hz[static_cast<size_t>(b)])) {
    say("presenter: VSync: the display refused " + fmt("%.2f", hz[static_cast<size_t>(b)]) + " Hz");
    return;
  }
  if (!held.on) held.origHz = origHz;
  held.on = true;
  held.hz = hz[static_cast<size_t>(b)];
  rateMoved();
  say("presenter: VSync: the monitor moved from " + fmt("%.2f", held.origHz) + " to " + fmt("%.2f", held.hz) + " Hz for " + fpsText());
}

void Presenter::Impl::releaseMode() {
  if (!held.on) return;
  held.on = false;
  mac::restoreDisplays();
  rateMoved();
  say("presenter: VSync: the monitor back at " + fmt("%.2f", held.origHz) + " Hz");
}

void Presenter::Impl::rateMoved() {
  queryMode();
  measured = 0;
  early = 0;
  firstAcq.clear();
  win.gaps.clear();
  epoch++;
  due = static_cast<double>(presents);
}

void Presenter::Impl::drain() {
  const double refreshMs = 1000.0 / refreshHz();
  std::vector<Shown> all;
  {
    std::lock_guard<std::mutex> l(box->mu);
    all.swap(box->done);
  }
  std::sort(all.begin(), all.end(), [](const Shown& a, const Shown& b) { return a.id < b.id; });
  for (const Shown& s : all) {
    win.add(s, refreshMs);
    if (csv)
      std::fprintf(csv, "%llu,%llu,%d,%s,%s,%s,%d,%s\n", static_cast<unsigned long long>(s.id), static_cast<unsigned long long>(s.picture),
                   s.fresh ? 1 : 0, fmt("%.3f", s.acquiredMs).c_str(), fmt("%.3f", s.submittedMs).c_str(), fmt("%.3f", s.shownMs).c_str(), s.queued,
                   fmt("%.3f", s.targetMs).c_str());
  }
}

void Presenter::Impl::window(double secs, bool last) {
  drain();
  const double m = win.measuredHz(modeHz);
  if (m > 0) measured = m;
  queryMode();
  if (!last) holdMode();
  const MtlSlot* s = cur >= 0 ? &slots[cur] : nullptr;
  const double per = secs > 0 ? secs : 1;
  std::string st = "{\"shownFps\":" + fmt("%.1f", secs > 0 ? win.pictures / per : 0) +
                   ",\"presentsPerSec\":" + fmt("%.1f", secs > 0 ? winPresents / per : 0) + ",\"hz\":" + fmt("%.3f", refreshHz()) +
                   ",\"modeHz\":" + fmt("%.3f", modeHz) + ",\"held\":" + win.heldJson() + ",\"missed\":" + std::to_string(win.missed) +
                   ",\"behind\":" + std::to_string(win.behind) + ",\"queued\":" + fmt("%.1f", win.queuedN ? win.queuedSum / static_cast<double>(win.queuedN) : 0.0) +
                   ",\"queuedMin\":" + std::to_string(std::max(0, win.queuedMin)) + ",\"images\":" + std::to_string(N) +
                   ",\"paused\":" + (paused ? "true" : "false") + ",\"surface\":\"" +
                   std::to_string(static_cast<int>(drawable.width)) + "x" + std::to_string(static_cast<int>(drawable.height)) + "\",\"frame\":\"" +
                   std::to_string(s ? s->w : 0) + "x" + std::to_string(s ? s->h : 0) + "\",\"presentWait\":true" +
                   ",\"vsync\":" + (vsync ? "true" : "false") + ",\"modeFrom\":" + fmt("%.3f", held.on ? held.origHz : 0.0) +
                   ",\"timing\":false,\"late\":0,\"slips\":0,\"errMs\":0.000,\"device\":\"" + esc(name) + "\",\"error\":\"";
  {
    std::lock_guard<std::mutex> l(statsMu);
    statsJson = st + esc(error) + "\",\"secs\":" + fmt("%.3f", secs) + "}";
  }
  win.reset();
  winPresents = 0;
}

void Presenter::Impl::presentOnce(bool live) {
  @autoreleasepool {
    // ai: the layer's size in pixels; none (minimised, not laid out yet): nothing to present to
    uint32_t pw = 0, ph = 0;
    if (!mac::layerPixels(target.metalLayer, &pw, &ph) || !pw || !ph) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      return;
    }
    if (pw != static_cast<uint32_t>(drawable.width) || ph != static_cast<uint32_t>(drawable.height)) {
      drawable = CGSizeMake(pw, ph);
      [CATransaction begin];
      [CATransaction setDisableActions:YES];
      layer.drawableSize = drawable;
      [CATransaction commit];
      [CATransaction flush];
      epoch++;
      say("presenter: the drawables at " + std::to_string(pw) + " x " + std::to_string(ph) + ", BGRA8 UNORM, " + std::to_string(N) + " deep, display sync");
    }
    // ai: a frame in flight's objects free (at most 100 ms, as the Vulkan presenter's acquire, so a stop or a stats
    // ai: window is never held up), then a drawable (Core Animation's wait, at most a second)
    if (dispatch_semaphore_wait(inflight, dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC))) return;
    const int fl = static_cast<int>(frame % N);
    if (cbs[fl]) [cbs[fl] waitUntilCompleted];
    ringHeld[fl].reset();
    id<CAMetalDrawable> d = [layer nextDrawable];
    if (!d) { dispatch_semaphore_signal(inflight); return; }
    const double acq = nowMs() - t0;
    if (measured <= 0 && modeHz <= 0 && frame >= N && firstAcq.size() < 16) {
      firstAcq.push_back(acq);
      if (firstAcq.size() >= 6) {
        std::vector<double> g;
        for (size_t k = 1; k < firstAcq.size(); k++) g.push_back(firstAcq[k] - firstAcq[k - 1]);
        std::nth_element(g.begin(), g.begin() + static_cast<long>(g.size() / 2), g.end());
        if (g[g.size() / 2] > 0.5) early = 1000.0 / g[g.size() / 2];
      }
    }

    // ai: a new picture when one is due and painted (presenter.cpp's rule without the timing grid)
    bool fresh = false;
    int ringSlot = -1;
    std::shared_ptr<wg::Buffer> ring;
    uint32_t ringFS = 0, ringRW = 0;
    const double i = static_cast<double>(presents);
    if (live && i + 0.5 >= due) {
      const int p = (cur + 1) % static_cast<int>(slots.size());
      waitFrame(slots[p].usedBy);
      if (lockFor(*senderLock, 2.0)) {
        std::lock_guard<std::mutex> hold(*senderLock, std::adopt_lock);
        const int fw = sender->width(), fh = sender->side();
        uint64_t seq = 0;
        if (fw > 0 && fh > 0 && sender->takeSlot(&seq, &ringSlot)) {
          ring = sender->ring();
          ringFS = sender->ringFrameWords();
          ringRW = sender->ringRowWords();
          MtlSlot& sp = slots[p];
          const int how = scaling.load();
          sp.grey = how == static_cast<int>(Scaling::Lanczos);
          sp.none = how == static_cast<int>(Scaling::Unscaled);
          if (sp.grey) {
            if (!sp.buf || sp.buf.length < static_cast<NSUInteger>(ringFS) * 4) sp.buf = [dev newBufferWithLength:static_cast<NSUInteger>(ringFS) * 4 options:MTLResourceStorageModePrivate];
            sp.w = fw; sp.h = fh; sp.rw = ringRW;
          } else if (!sp.img || sp.w != fw || sp.h != fh) {
            MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:fw height:fh mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            td.storageMode = MTLStorageModePrivate;
            sp.img = [dev newTextureWithDescriptor:td];
            sp.w = fw; sp.h = fh;
          }
          if ((sp.grey && !sp.buf) || (!sp.grey && !sp.img)) throw std::runtime_error("no memory for a picture of " + std::to_string(fw) + " x " + std::to_string(fh));
          if (cur >= 0) picture++;
          cur = p;
          fresh = true;
          const double per = period();
          due = std::max(due + per, i + per);
        }
      }
      if (!fresh && cur >= 0) win.behind++;
    }

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) throw std::runtime_error(name + ": no command buffer for a present");
    id<MTLBuffer> ringBuf = ring ? (__bridge id<MTLBuffer>)ring->mtl : nil;
    if (fresh && ring && slots[cur].grey) {
      // ai: the grey picture (painted on the painter's queue and complete before the sender marked it ready) into the
      // ai: slot's buffer
      ringHeld[fl] = ring;
      id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
      [be copyFromBuffer:ringBuf sourceOffset:static_cast<NSUInteger>(ringSlot) * ringFS * 4 toBuffer:slots[cur].buf destinationOffset:0 size:static_cast<NSUInteger>(ringFS) * 4];
      [be endEncoding];
    } else if (fresh && ring) {
      ringHeld[fl] = ring;
      MtlSlot& s = slots[cur];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:expand.ps];
      const uint32_t pc6[6] = {static_cast<uint32_t>(ringSlot) * ringFS, ringRW, static_cast<uint32_t>(s.w), static_cast<uint32_t>(s.h), sender->ink(), sender->paper()};
      [ce setBytes:pc6 length:sizeof pc6 atIndex:expand.push()];
      [ce setBuffer:ringBuf offset:0 atIndex:expand.at(0, 'b')];
      [ce setTexture:s.img atIndex:expand.at(1, 't')];
      [ce dispatchThreadgroups:MTLSizeMake((static_cast<uint32_t>(s.w) + 15) / 16, (static_cast<uint32_t>(s.h) + 15) / 16, 1) threadsPerThreadgroup:expand.local];
      [ce endEncoding];
    }
    // ai: round the picture the code's paper (Sender::paper, white by default)
    const uint32_t paper = sender->paper();
    {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = d.texture;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.colorAttachments[0].clearColor = MTLClearColorMake(((paper >> 16) & 0xff) / 255.0, ((paper >> 8) & 0xff) / 255.0, (paper & 0xff) / 255.0, 1.0);
      id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
      [re endEncoding];
    }
    const int W2 = static_cast<int>(drawable.width), H2 = static_cast<int>(drawable.height);
    if (cur >= 0) {
      // ai: where the picture goes (presenter.cpp's): centred, the largest that fits with the edge's light round it,
      // ai: times the size share; None at its own pixels, its middle where the place is smaller
      MtlSlot& s = slots[cur];
      const double e = 2 * edge.load() * s.h;
      const double fit = std::min(static_cast<double>(W2) / (s.w + e), static_cast<double>(H2) / (s.h + e));
      const double share = std::clamp(size.load(), 0.25, 1.0);
      const double scale = s.none ? 1.0 : fit * share;
      const int dw = std::min(W2, static_cast<int>(std::lround(s.w * scale)));
      const int dh = std::min(H2, static_cast<int>(std::lround(s.h * scale)));
      const int ox = (W2 - dw) / 2, oy = (H2 - dh) / 2;
      if (dw > 0 && dh > 0) {
        if (s.none) {
          id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
          [be copyFromTexture:s.img sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake((s.w - dw) / 2, (s.h - dh) / 2, 0) sourceSize:MTLSizeMake(dw, dh, 1)
                    toTexture:d.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(ox, oy, 0)];
          [be endEncoding];
        } else if (s.grey) {
          // ai: Lanczos-3 across then down into this frame's own texture, copied 1:1; the table, buffer and texture made
          // ai: again where the sizes change (this frame's command buffer waited above, so none is in use)
          Stretch& st = stretches[fl];
          const int key[4] = {s.w, s.h, dw, dh};
          if (!std::equal(key, key + 4, st.key)) {
            const int tx = stretchTaps(s.w, dw), ty = stretchTaps(s.h, dh);
            const size_t need = static_cast<size_t>(dw) * static_cast<size_t>(tx + 1) + static_cast<size_t>(dh) * static_cast<size_t>(ty + 1);
            if (need > st.tabCap) { st.tab = [dev newBufferWithLength:need * sizeof(float) options:MTLResourceStorageModeShared]; st.tabCap = need; }
            float* tab = static_cast<float*>(st.tab.contents);
            stretchAxis(tab, s.w, dw, tx);
            stretchAxis(tab + static_cast<size_t>(dw) * static_cast<size_t>(tx + 1), s.h, dh, ty);
            st.tx = tx; st.ty = ty; st.yo = dw * (tx + 1);
            const size_t mid = static_cast<size_t>(dw) * static_cast<size_t>(s.h) * 4;
            if (mid > st.midCap) { st.mid = [dev newBufferWithLength:mid options:MTLResourceStorageModePrivate]; st.midCap = mid; }
            MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:dw height:dh mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            td.storageMode = MTLStorageModePrivate;
            st.out = [dev newTextureWithDescriptor:td];
            if (!st.tab || !st.mid || !st.out) throw std::runtime_error("no memory for the stretch to " + std::to_string(dw) + " x " + std::to_string(dh));
            std::copy(key, key + 4, st.key);
          }
          const uint32_t sp[10] = {static_cast<uint32_t>(s.w), static_cast<uint32_t>(s.h), static_cast<uint32_t>(dw), static_cast<uint32_t>(dh),
                                   static_cast<uint32_t>(st.tx), static_cast<uint32_t>(st.ty), static_cast<uint32_t>(st.yo), s.rw, sender->ink(), sender->paper()};
          id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
          [ce setComputePipelineState:scaleH.ps];
          [ce setBytes:sp length:sizeof sp atIndex:scaleH.push()];
          [ce setBuffer:s.buf offset:0 atIndex:scaleH.at(0, 'b')];
          [ce setBuffer:st.mid offset:0 atIndex:scaleH.at(1, 'b')];
          [ce setBuffer:st.tab offset:0 atIndex:scaleH.at(2, 'b')];
          [ce dispatchThreadgroups:MTLSizeMake((static_cast<uint32_t>(dw) + 127) / 128, (static_cast<uint32_t>(s.h) + 15) / 16, 1) threadsPerThreadgroup:scaleH.local];   // ai: scale_h.slang's ROWS
          [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [ce setComputePipelineState:scaleV.ps];
          [ce setBytes:sp length:sizeof sp atIndex:scaleV.push()];
          [ce setBuffer:st.mid offset:0 atIndex:scaleV.at(1, 'b')];
          [ce setBuffer:st.tab offset:0 atIndex:scaleV.at(2, 'b')];
          [ce setTexture:st.out atIndex:scaleV.at(3, 't')];
          [ce dispatchThreadgroups:MTLSizeMake((static_cast<uint32_t>(dw) + 63) / 64, (static_cast<uint32_t>(dh) + 3) / 4, 1) threadsPerThreadgroup:scaleV.local];
          [ce endEncoding];
          id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
          [be copyFromTexture:st.out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(dw, dh, 1)
                    toTexture:d.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(ox, oy, 0)];
          [be endEncoding];
        } else {
          // ai: Bilinear: the expanded picture to its place by the texture unit's bilinear filter
          const int32_t lp[4] = {ox, oy, dw, dh};
          id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
          [ce setComputePipelineState:linear.ps];
          [ce setBytes:lp length:sizeof lp atIndex:linear.push()];
          [ce setTexture:s.img atIndex:linear.at(0, 't')];
          [ce setSamplerState:lin atIndex:linear.at(1, 's')];
          [ce setTexture:d.texture atIndex:linear.at(2, 't')];
          [ce dispatchThreadgroups:MTLSizeMake((static_cast<uint32_t>(dw) + 15) / 16, (static_cast<uint32_t>(dh) + 15) / 16, 1) threadsPerThreadgroup:linear.local];
          [ce endEncoding];
        }
      }
      s.usedBy = frame;
    }
    if (markBuf && W2 >= MARK_W && H2 >= MARK_PX) {
      const size_t slot = static_cast<size_t>(MARK_W) * MARK_PX * 4;
      uint8_t* m = static_cast<uint8_t*>(markBuf.contents) + static_cast<size_t>(fl) * slot;
      const uint32_t v = static_cast<uint32_t>(picture) & ((1u << MARK_BITS) - 1);
      for (int b = 0; b < 2 * MARK_BITS; b++) {
        const uint32_t bit = b < MARK_BITS ? (v >> (MARK_BITS - 1 - b)) & 1u : ((v >> (2 * MARK_BITS - 1 - b)) & 1u) ^ 1u;
        const uint8_t g = bit ? 0 : 255;
        for (int yy = 0; yy < MARK_PX; yy++)
          for (int xx = 0; xx < MARK_PX; xx++) {
            uint8_t* q = m + (static_cast<size_t>(yy) * MARK_W + static_cast<size_t>(b) * MARK_PX + xx) * 4;
            q[0] = q[1] = q[2] = g;
            q[3] = 255;
          }
      }
      id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
      [be copyFromBuffer:markBuf sourceOffset:static_cast<NSUInteger>(fl) * slot sourceBytesPerRow:MARK_W * 4 sourceBytesPerImage:slot sourceSize:MTLSizeMake(MARK_W, MARK_PX, 1)
               toTexture:d.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
      [be endEncoding];
    }

    const uint64_t id = ++pid;
    if (live && cur >= 0) {
      Shown s;
      s.id = id;
      s.picture = picture;
      s.fresh = fresh;
      s.acquiredMs = acq;
      s.epoch = epoch;
      s.queued = static_cast<int>(id - box->shownId.load());
      std::lock_guard<std::mutex> l(box->mu);
      box->sent[id] = s;
    }
    const double origin = t0;
    std::shared_ptr<ShownBox> shared = box;
    [d addPresentedHandler:^(id<MTLDrawable> dr) {
      const CFTimeInterval at = dr.presentedTime;
      std::lock_guard<std::mutex> l(shared->mu);
      auto it = shared->sent.find(id);
      if (it == shared->sent.end()) return;
      Shown s = it->second;
      shared->sent.erase(it);
      s.shownMs = at > 0 ? hostToSteadyMs(at) - origin : -1;
      if (at > 0) shared->shownId = std::max(shared->shownId.load(), id);
      shared->done.push_back(s);
    }];
    [cb presentDrawable:d];
    dispatch_semaphore_t sem = inflight;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) { (void)done; dispatch_semaphore_signal(sem); }];
    [cb commit];
    cbs[fl] = cb;
    {
      std::lock_guard<std::mutex> l(box->mu);
      auto it = box->sent.find(id);
      if (it != box->sent.end()) it->second.submittedMs = nowMs() - t0;
    }
    if (live) presents++;
    winPresents++;
    frame++;
  }
}

void Presenter::Impl::run() {
  try {
    while (!quit) {
      @autoreleasepool {
        if (vsyncDirty.exchange(false)) holdMode();
        const double now = nowMs();
        if (now - lastWin >= 1000) {
          window((now - lastWin) / 1000);
          lastWin = now;
        }
        if (snapAsk) takeSnapshot();
        const bool p = paused;
        if (p) {
          // ai: paused: the last picture presented again a few times a second, none taken
          wasPaused = true;
          std::unique_lock<std::mutex> l(wakeMu);
          wake.wait_for(l, std::chrono::milliseconds(100), [&] { return quit.load() || !paused.load() || snapAsk.load(); });
          if (quit) break;
          if (snapAsk) { l.unlock(); takeSnapshot(); }
          if (!paused) continue;
        } else if (wasPaused) {
          due = static_cast<double>(presents);
          epoch++;
          wasPaused = false;
        }
        presentOnce(!p);
      }
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> l(statsMu);
    error = e.what();
  }
  if (!error.empty()) say("presenter: stopped: " + error);
}

// ai: The slot on screen read back (Presenter::snapshot): its BGRA texture's pixels as RGBA, or the grey words along the
// ai: line from the code's ink to its paper, as the expand kernel draws them
void Presenter::Impl::takeSnapshot() {
  std::vector<unsigned char> px;
  int w = 0, h = 0;
  @autoreleasepool {
    try {
      if (cur >= 0 && (slots[cur].grey ? slots[cur].buf != nil : slots[cur].img != nil) && slots[cur].w > 0 && slots[cur].h > 0) {
        const MtlSlot& s = slots[cur];
        const NSUInteger bytes = s.grey ? static_cast<NSUInteger>(s.rw) * s.h * 4 : static_cast<NSUInteger>(s.w) * s.h * 4;
        id<MTLBuffer> back = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!back) throw std::runtime_error("no memory for the snapshot");
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        if (s.grey) [be copyFromBuffer:s.buf sourceOffset:0 toBuffer:back destinationOffset:0 size:bytes];
        else
          [be copyFromTexture:s.img sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(s.w, s.h, 1)
                     toBuffer:back destinationOffset:0 destinationBytesPerRow:static_cast<NSUInteger>(s.w) * 4 destinationBytesPerImage:bytes];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) throw std::runtime_error("the snapshot's copy failed");
        px.resize(static_cast<size_t>(s.w) * s.h * 4);
        if (s.grey) {
          const auto* words = static_cast<const uint32_t*>(back.contents);
          const uint32_t ink = sender->ink(), pap = sender->paper();
          unsigned char lut[256][3];
          for (int g = 0; g < 256; g++)
            for (int c = 0; c < 3; c++) {
              const int a = static_cast<int>((ink >> (16 - 8 * c)) & 0xffu), b = static_cast<int>((pap >> (16 - 8 * c)) & 0xffu);
              lut[g][c] = static_cast<unsigned char>(std::lround(a + (b - a) * g / 255.0));
            }
          for (int y = 0; y < s.h; y++)
            for (int x = 0; x < s.w; x++) {
              const unsigned g = (words[static_cast<size_t>(y) * s.rw + (x >> 2)] >> (8 * (x & 3))) & 0xffu;
              unsigned char* q = &px[(static_cast<size_t>(y) * s.w + x) * 4];
              q[0] = lut[g][0]; q[1] = lut[g][1]; q[2] = lut[g][2];
              q[3] = 255;
            }
        } else {
          const auto* bgra = static_cast<const unsigned char*>(back.contents);
          for (size_t k = 0; k < px.size(); k += 4) { px[k] = bgra[k + 2]; px[k + 1] = bgra[k + 1]; px[k + 2] = bgra[k]; px[k + 3] = bgra[k + 3]; }
        }
        w = s.w;
        h = s.h;
      }
    } catch (const std::exception& e) {
      say(std::string("presenter: snapshot: ") + e.what());
      px.clear();
    }
  }
  {
    std::lock_guard<std::mutex> l(snapMu);
    snapPx = std::move(px);
    snapW = w;
    snapH = h;
    snapDone = true;
    snapAsk = false;
  }
  snapCv.notify_all();
}

void Presenter::Impl::teardown() {
  @autoreleasepool {
    for (int k = 0; k < N; k++) if (cbs[k]) [cbs[k] waitUntilCompleted];
    // ai: the presented handlers of the last presents a moment to come (the window's last second counts them)
    if (t0 > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      window((nowMs() - lastWin) / 1000, true);
    }
    for (int k = 0; k < N; k++) { cbs[k] = nil; ringHeld[k].reset(); stretches[k] = Stretch{}; }
    for (auto& s : slots) s = MtlSlot{};
    markBuf = nil;
    expand = scaleH = scaleV = linear = Kernel{};
    queue = nil;
    layer = nil;
    dev = nil;
  }
  releaseMode();
  if (csv) std::fclose(csv);
  csv = nullptr;
}

const wg::DeviceExtras& Presenter::deviceExtras() {
  static const wg::DeviceExtras none;   // ai: a Metal device needs nothing asked for its drawables
  return none;
}

std::unique_ptr<Presenter> Presenter::start(const PresentTarget& target, Sender* sender, std::mutex* senderLock,
                                            const PresentOptions& opt, std::string* err) {
  if (!sender || !senderLock) {
    if (err) *err = "no sender to present";
    return nullptr;
  }
  auto p = std::make_unique<Impl>();
  p->target = target;
  p->sender = sender;
  p->senderLock = senderLock;
  p->opt = opt;
  p->fps = std::clamp(opt.fps, 1, 240);
  p->size = std::clamp(opt.size, 0.25, 1.0);
  p->edge = std::clamp(opt.edge, 0.0, 1.0);
  p->vsync = opt.vsync;
  p->scaling = static_cast<int>(opt.scaling);
  p->vsyncDirty = opt.vsync;
  std::promise<std::string> ready;
  std::future<std::string> started = ready.get_future();
  Impl* raw = p.get();
  raw->thread = std::thread([raw, pr = std::move(ready)]() mutable {
    lizard::sys::raiseThread();
    try {
      @autoreleasepool { raw->setup(); }
    } catch (const std::exception& e) {
      raw->teardown();
      pr.set_value(e.what());
      return;
    }
    pr.set_value("");
    raw->run();
    raw->teardown();
  });
  const std::string why = started.get();
  if (!why.empty()) {
    raw->thread.join();
    if (err) *err = why;
    return nullptr;
  }
  return std::unique_ptr<Presenter>(new Presenter(std::move(p)));
}

Presenter::Presenter(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

void Presenter::setFps(int fps) {
  p_->fps = std::clamp(fps, 1, 240);
  p_->vsyncDirty = true;
}
void Presenter::setVsync(bool on) {
  p_->vsync = on;
  p_->vsyncDirty = true;
}
void Presenter::setScaling(Scaling s) { p_->scaling = static_cast<int>(s); }
void Presenter::setSize(double size, double edge) {
  p_->size = std::clamp(size, 0.25, 1.0);
  p_->edge = std::clamp(edge, 0.0, 1.0);
}
void Presenter::setPaused(bool on) {
  {
    std::lock_guard<std::mutex> l(p_->wakeMu);
    p_->paused = on;
  }
  p_->wake.notify_all();
}

bool Presenter::snapshot(std::vector<unsigned char>* rgba, int* w, int* h) {
  Impl& m = *p_;
  std::unique_lock<std::mutex> l(m.snapMu);
  m.snapDone = false;
  {
    std::lock_guard<std::mutex> wl(m.wakeMu);
    m.snapAsk = true;
  }
  m.wake.notify_all();
  if (!m.snapCv.wait_for(l, std::chrono::seconds(1), [&] { return m.snapDone; }) || m.snapPx.empty()) return false;
  *rgba = std::move(m.snapPx);
  *w = m.snapW;
  *h = m.snapH;
  return true;
}

std::string Presenter::stats() {
  std::lock_guard<std::mutex> l(p_->statsMu);
  return p_->statsJson;
}

void Presenter::stop() {
  {
    std::lock_guard<std::mutex> l(p_->wakeMu);
    p_->quit = true;
  }
  p_->wake.notify_all();
  if (p_->thread.joinable()) p_->thread.join();
}

Presenter::~Presenter() { stop(); }

}  // namespace lizard
