// ai: wg over Metal (wg.h; 2026-11-25, in place of Vulkan through MoltenVK on Apple's systems): the same WebGPU rules
// ai: as wg.cpp's over Vulkan, each kept the Metal way:
// ai:   - every buffer and texture reads as zeros until written: a buffer the host maps is zeroed by the host, a
// ai:     private one filled at the head of the next submit, a texture's layers copied from a zeroed buffer there;
// ai:   - queue writes land before the next submit's commands: the head is a command buffer of its own (the inits, the
// ai:     uploads' copies, then the writes, a blit encoder each), committed before the batch's, and one queue runs its
// ai:     command buffers in the order they are committed;
// ai:   - each dispatch sees what the commands before it wrote: a concurrent compute encoder with a memory barrier
// ai:     before every dispatch but its first and an alongside one; the copies and fills in blit encoders of their own,
// ai:     one a command (Metal orders encoders by their tracked resources, which every object here is);
// ai:   - no timestamps (Features::timestamps false: resolveQuerySet writes zeros, as Vulkan's does without them).
// ai: A kernel is gen's MSL (out/msl/<name>.metal, slangc's from the one Slang source), compiled by the system's Metal
// ai: compiler at its first pipeline (math in its safe mode: no fast math; MSL has no `precise`, so the few `precise`
// ai: sums may contract there: float noise), each module's library kept for the device's life; its binding map says
// ai: where each Vulkan binding's object goes and the workgroup's size, which Metal takes at the dispatch.
// ai: Every call that makes Metal objects runs in an autorelease pool: the decoder's and the painter's threads have none
// ai: of their own, and a long run would hold every command buffer and encoder it ever made.
#include "wg.h"
#include "json.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace wg {

namespace {

double steadyMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
std::string str(NSString* s) { return s ? std::string([s UTF8String]) : std::string(); }
// ai: a retained object's handle let go
void drop(void*& p) {
  if (p) CFRelease(p);
  p = nullptr;
}
template <class T> void* keep(T o) { return o ? (__bridge_retained void*)o : nullptr; }

id<MTLDevice> D(const Device* d) { return (__bridge id<MTLDevice>)d->mtl; }
id<MTLBuffer> B(const Buffer& b) { return (__bridge id<MTLBuffer>)b.mtl; }
id<MTLTexture> T(const Texture& t) { return (__bridge id<MTLTexture>)t.mtl; }

// ai: a command buffer's fault as wg's errors: out of memory, else the device lost to its user (a GPU fault, a timeout,
// ai: the device gone: nothing it wrote can be trusted)
[[noreturn]] void fault(Device* dev, id<MTLCommandBuffer> cb) {
  NSError* e = cb.error;
  const std::string m = "a command buffer on " + dev->name + " failed: " + (e ? str(e.localizedDescription) : std::string("no reason given"));
  if (e && e.code == MTLCommandBufferErrorOutOfMemory) { OutOfMemory x(m); x.code = (int)e.code; throw x; }
  dev->lost = true;
  Lost x(m);
  x.code = e ? (int)e.code : 0;
  throw x;
}

// ai: the open encoder of kind k (1 compute, 2 blit), ending another; a blit encoder is new at every call, so each copy
// ai: is ordered after everything before it
id<MTLComputeCommandEncoder> compute(Encoder& e, bool* fresh) {
  if (e.open == 1) { *fresh = false; return (__bridge id<MTLComputeCommandEncoder>)e.enc; }
  if (e.open) { [(__bridge id<MTLCommandEncoder>)e.enc endEncoding]; drop(e.enc); }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)e.cb;
  id<MTLComputeCommandEncoder> c = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  if (!c) throw Error("no compute encoder on " + e.dev->name);
  e.enc = keep(c);
  e.open = 1;
  e.bound = false;
  *fresh = true;
  return c;
}
id<MTLBlitCommandEncoder> blit(Encoder& e) {
  if (e.open) { [(__bridge id<MTLCommandEncoder>)e.enc endEncoding]; drop(e.enc); }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)e.cb;
  id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
  if (!b) throw Error("no blit encoder on " + e.dev->name);
  e.enc = keep(b);
  e.open = 2;
  return b;
}
void endOpen(Encoder& e) {
  if (e.open) { [(__bridge id<MTLCommandEncoder>)e.enc endEncoding]; drop(e.enc); }
  e.open = 0;
}

// ai: the compute encoder for a dispatch: open, a barrier behind what came before in it (unless alongside), the
// ai: pipeline and its group's objects set where they changed or the encoder is new
id<MTLComputeCommandEncoder> forDispatch(Encoder& e, bool alongside) {
  if (!e.pipe) throw Error("a dispatch before a pipeline");
  if (!e.group) throw Error("a dispatch before a bind group");
  bool fresh = false;
  id<MTLComputeCommandEncoder> c = compute(e, &fresh);
  if (!fresh && !alongside) [c memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
  e.commands++;
  if (e.bound) return c;
  const Pipeline& p = *e.pipe;
  const BindGroup& g = *e.group;
  [c setComputePipelineState:(__bridge id<MTLComputePipelineState>)p.mtl];
  const auto& entries = g.bgl->entries;
  for (size_t i = 0; i < entries.size() && i < p.bind.size(); i++)
    for (const Module::Slot& s : p.bind[i]) {
      if (s.kind == 's') { [c setSamplerState:(__bridge id<MTLSamplerState>)e.dev->linear atIndex:s.index]; continue; }
      void* r = g.res[i];
      if (!r) throw Error("bind group " + g.id + ": binding " + std::to_string(i) + " has no object for pipeline " + p.id);
      if (s.kind == 't') [c setTexture:(__bridge id<MTLTexture>)r atIndex:s.index];
      else [c setBuffer:(__bridge id<MTLBuffer>)r offset:0 atIndex:s.index];
    }
  e.bound = true;
  return c;
}

MTLSize localOf(const Pipeline& p) { return MTLSizeMake(p.local[0], p.local[1], p.local[2]); }

}  // namespace

std::string Device::autoId(char kind) { return std::string("n") + kind + std::to_string(nextId++); }
bool Device::has(const char* ext) const { return std::find(enabled.begin(), enabled.end(), ext) != enabled.end(); }

std::unique_ptr<Device> Device::create(const std::string& want, bool validate, std::function<void(const std::string&)> log, const DeviceExtras&) {
  @autoreleasepool {
    auto d = std::make_unique<Device>();
    d->log = log ? log : [](const std::string&) {};
    if (validate) d->log("validation asked: Metal's is the environment's (MTL_DEBUG_LAYER=1)");
    // ai: want: a substring of the device's name (LIZ_VK_DEVICE, the hosts' one name for it), else the system's default
    id<MTLDevice> pick = nil;
    std::string names;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    for (id<MTLDevice> x in all) {
      names += std::string(names.empty() ? "" : ", ") + str(x.name);
      if (!want.empty() && !pick && lower(str(x.name)).find(lower(want)) != std::string::npos) pick = x;
    }
    if (!want.empty() && !pick) throw Error("no Metal device named like \"" + want + "\" among " + (names.empty() ? std::string("none") : names));
    if (!pick) pick = MTLCreateSystemDefaultDevice();
    if (!pick) throw Error("no Metal device");
    d->mtl = keep(pick);
    d->name = str(pick.name);
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    d->driver = "Metal, macOS " + std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion) + "." + std::to_string(v.patchVersion);
    d->vendorId = 0x106B;   // ai: Apple's PCI vendor id
    d->deviceId = (uint32_t)(pick.registryID & 0xffffffffu);
    d->kind = d->name.find("Paravirtual") != std::string::npos ? Kind::Virtual : pick.hasUnifiedMemory ? Kind::Integrated : Kind::Discrete;
    d->hangMs = HANG_MS;
    if (const char* h = getenv("LIZ_HANG_MS")) d->hangMs = (uint32_t)std::max(0, atoi(h));
    id<MTLCommandQueue> q = [pick newCommandQueue], q2 = [pick newCommandQueue];
    if (!q || !q2) throw Error(d->name + ": no command queue");
    d->queue = keep(q);
    d->queue2 = keep(q2);
    d->twoQueues = true;
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.normalizedCoordinates = YES;
    d->linear = keep([pick newSamplerStateWithDescriptor:sd]);
    d->libs = keep([NSMutableDictionary new]);

    auto& F = d->features;
    F.f16 = F.storage16 = F.int8 = F.int16 = true;
    F.dot = false;   // ai: no packed dot in MSL: the setup takes the variants without it (core/dec/setup.cpp variantsFor)
    // ai: SIMD-group shuffles: Apple's GPUs from the A13's family (Apple6) and Mac2's
    F.subgroupShuffle = [pick supportsFamily:MTLGPUFamilyApple6] || [pick supportsFamily:MTLGPUFamilyMac2];
    F.subgroupSize = 32;
    auto& L = d->limits;
    L.maxBufferSize = pick.maxBufferLength;
    L.maxStorageBufferRange = std::min<uint64_t>(pick.maxBufferLength, 0xffffffffull);
    L.maxComputeSharedMemorySize = (uint32_t)pick.maxThreadgroupMemoryLength;
    L.maxPerStageStorageBuffers = 31;   // ai: Metal's buffer argument table
    L.maxImageArrayLayers = 2048;
    L.maxComputeWorkGroupInvocations = (uint32_t)pick.maxThreadsPerThreadgroup.width;
    L.maxUniformBufferRange = 65536;
    L.minStorageAlign = 16;
    L.minUniformAlign = 256;
    L.deviceLocalBytes = pick.recommendedMaxWorkingSetSize;

    char line[512];
    snprintf(line, sizeof line, "device %s (%s): f16 %d, int8 %d, int16 %d, dot %d, subgroup shuffle %d (size %u), queues 2, timestamps 0; shared memory %u B, storage range %llu MB, buffer %llu MB, working set %llu MB",
             d->name.c_str(), d->driver.c_str(), F.f16, F.int8, F.int16, F.dot, F.subgroupShuffle, F.subgroupSize, L.maxComputeSharedMemorySize,
             (unsigned long long)(L.maxStorageBufferRange >> 20), (unsigned long long)(L.maxBufferSize >> 20), (unsigned long long)(L.deviceLocalBytes >> 20));
    d->log(line);
    return d;
  }
}

Device::~Device() {
  if (mtl && !abandoned) {
    try { waitIdle(); } catch (const std::exception&) {}
    if (queue2) { @autoreleasepool { id<MTLCommandBuffer> cb = [(__bridge id<MTLCommandQueue>)queue2 commandBuffer]; [cb commit]; [cb waitUntilCompleted]; } }
  }
  writes.clear();
  inits.clear();
  staging.clear();
  drop(last2);
  drop(libs);
  drop(linear);
  drop(queue2);
  drop(queue);
  drop(mtl);
}

// ai: Bounded as a Ticket is (an empty command buffer behind everything on the queue); nothing on an abandoned device.
void Device::waitIdle() {
  if (abandoned) return;
  Encoder e = encoder();
  submit(e)->wait();
}

void Device::loadCache(const std::string& path) { cachePath = path; }
void Device::saveCache() {}

std::shared_ptr<Buffer> Device::createBuffer(uint64_t size, uint32_t usage, const std::string& id, bool zero) {
  @autoreleasepool {
    auto b = std::make_shared<Buffer>();
    b->dev = this;
    b->size = size;
    b->usage = usage;
    b->id = id.empty() ? autoId('b') : id;
    const bool host = usage & (MAP_READ | MAP_WRITE);
    // ai: whole u32s (Metal's fills and copies on macOS take multiples of 4), 16 at least, as Vulkan's
    const uint64_t len = (std::max<uint64_t>(size, 16) + 3) & ~3ull;
    id<MTLBuffer> m = [D(this) newBufferWithLength:len options:(host ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate)];
    if (!m) throw OutOfMemory("no Metal buffer of " + std::to_string(size) + " B on " + name);
    b->mtl = keep(m);
    allocated += size;
    if (host) {
      b->mapped = (uint8_t*)m.contents;
      memset(b->mapped, 0, (size_t)len);
      b->coherent = true;
    } else if (zero) {
      void* raw = b->mtl;
      std::lock_guard<std::mutex> l(mu);
      inits.push_back({b.get(), [raw, len](Cmd c) {
        [(__bridge id<MTLBlitCommandEncoder>)c fillBuffer:(__bridge id<MTLBuffer>)raw range:NSMakeRange(0, len) value:0];
      }});
    }
    return b;
  }
}

Buffer::~Buffer() {
  if (!dev) return;
  {
    std::lock_guard<std::mutex> l(dev->mu);
    auto& w = dev->writes;
    w.erase(std::remove_if(w.begin(), w.end(), [this](const Device::Write& x) { return x.dst == this; }), w.end());
    auto& in = dev->inits;
    in.erase(std::remove_if(in.begin(), in.end(), [this](const Device::Init& x) { return x.owner == this; }), in.end());
  }
  if (mtl) dev->allocated -= size;
  if (dev->abandoned) return;
  drop(mtl);
}

// ai: A texture: r8unorm, a 2D array of `layers`, read by the kernels and copied into; zeroed at the next submit from a
// ai: zeroed buffer of one layer, copied into each.
std::shared_ptr<Texture> Device::createTexture(uint32_t w, uint32_t h, uint32_t layers, const std::string& id, bool storage) {
  @autoreleasepool {
    auto t = std::make_shared<Texture>();
    t->dev = this;
    t->w = w; t->h = h; t->layers = layers;
    t->storage = storage;
    t->id = id.empty() ? autoId('t') : id;
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = MTLPixelFormatR8Unorm;
    td.width = w;
    td.height = h;
    td.arrayLength = layers;
    td.mipmapLevelCount = 1;
    td.storageMode = MTLStorageModePrivate;
    td.usage = storage ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
    id<MTLTexture> m = [D(this) newTextureWithDescriptor:td];
    if (!m) throw OutOfMemory("no Metal texture of " + std::to_string(w) + " x " + std::to_string(h) + " x " + std::to_string(layers) + " on " + name);
    t->mtl = keep(m);
    allocated += (uint64_t)w * h * layers;
    id<MTLBuffer> z = [D(this) newBufferWithLength:std::max<uint64_t>(16, (uint64_t)w * h) options:MTLResourceStorageModeShared];
    if (!z) throw OutOfMemory("no Metal buffer to clear a texture on " + name);
    memset(z.contents, 0, (size_t)z.length);
    void* raw = t->mtl;
    std::lock_guard<std::mutex> l(mu);
    inits.push_back({t.get(), [raw, z, w, h, layers](Cmd c) {
      id<MTLBlitCommandEncoder> be = (__bridge id<MTLBlitCommandEncoder>)c;
      for (uint32_t k = 0; k < layers; k++)
        [be copyFromBuffer:z sourceOffset:0 sourceBytesPerRow:w sourceBytesPerImage:(NSUInteger)w * h sourceSize:MTLSizeMake(w, h, 1)
                 toTexture:(__bridge id<MTLTexture>)raw destinationSlice:k destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    }});
    return t;
  }
}

Texture::~Texture() {
  if (!dev) return;
  {
    std::lock_guard<std::mutex> l(dev->mu);
    auto& in = dev->inits;
    in.erase(std::remove_if(in.begin(), in.end(), [this](const Device::Init& x) { return x.owner == this; }), in.end());
  }
  if (mtl) dev->allocated -= (uint64_t)w * h * layers;
  if (dev->abandoned) return;
  drop(mtl);
}

std::shared_ptr<BindGroupLayout> Device::createBindGroupLayout(const std::vector<Bind>& entries, const std::string& id) {
  auto L = std::make_shared<BindGroupLayout>();
  L->dev = this;
  L->entries = entries;
  L->id = id.empty() ? autoId('L') : id;
  return L;
}

BindGroupLayout::~BindGroupLayout() {}

// ai: gen's MSL (<folder>/name.metal) and its map (<folder>/name.json: entry, local, bind, push; gen.mjs slangMsl)
Module readModule(const Read& read, const std::string& name, const std::string& folder) {
  const std::string dir = folder.empty() ? std::string("msl") : folder;
  std::vector<uint8_t> src, map;
  try { src = read(dir + "/" + name + ".metal"); map = read(dir + "/" + name + ".json"); } catch (const std::exception&) {}
  if (src.empty() || map.empty()) throw Error(dir + "/" + name + ".metal: missing, or its map is (liblizard/gen/gen.mjs)");
  Module m;
  m.name = name;
  m.msl.assign(src.begin(), src.end());
  try {
    const auto j = nlohmann::json::parse(map.begin(), map.end());
    m.entry = j.at("entry").get<std::string>();
    for (int k = 0; k < 3; k++) m.local[k] = j.at("local").at(k).get<uint32_t>();
    for (const auto& el : j.at("bind").items()) {
      const size_t i = (size_t)std::stoul(el.key());
      if (m.bind.size() <= i) m.bind.resize(i + 1);
      for (const auto& s : el.value()) m.bind[i].push_back({s.at(0).get<std::string>().at(0), s.at(1).get<uint32_t>()});
    }
    if (j.contains("push") && !j.at("push").is_null()) { m.hasPush = true; m.push = {j["push"].at(0).get<std::string>().at(0), j["push"].at(1).get<uint32_t>()}; }
  } catch (const std::exception& e) {
    throw Error(dir + "/" + name + ".json: " + e.what());
  }
  return m;
}

Module readModule(const std::string& dir, const std::string& name) {
  return readModule([&](const std::string& rel) {
    std::ifstream f(std::filesystem::u8path(dir + "/" + rel), std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }, name);
}

std::shared_ptr<Pipeline> Device::createPipeline(const Module& m, std::shared_ptr<BindGroupLayout> bgl, const std::string& id, const std::string& label, const std::vector<uint32_t>& spec) {
  @autoreleasepool {
    if (m.msl.empty()) throw Error("module " + m.name + " has no MSL (liblizard/gen/gen.mjs)");
    if (!spec.empty()) throw Error("module " + m.name + ": specialisation constants are Vulkan's alone");
    auto p = std::make_shared<Pipeline>();
    p->dev = this;
    p->bgl = bgl;
    p->id = id.empty() ? autoId('p') : id;
    p->label = label;
    id<MTLLibrary> lib = nil;
    {
      std::lock_guard<std::mutex> l(libMu);
      NSMutableDictionary* cache = (__bridge NSMutableDictionary*)libs;
      NSString* key = [NSString stringWithUTF8String:m.name.c_str()];
      lib = cache[key];
      if (!lib) {
        MTLCompileOptions* o = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
        else o.fastMathEnabled = NO;
        NSError* err = nil;
        NSString* src = [[NSString alloc] initWithBytes:m.msl.data() length:m.msl.size() encoding:NSUTF8StringEncoding];
        lib = [D(this) newLibraryWithSource:src options:o error:&err];
        if (!lib) throw Error("module " + m.name + (label.empty() ? "" : " (" + label + ")") + " did not compile: " + str(err.localizedDescription));
        cache[key] = lib;
      }
    }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:m.entry.c_str()]];
    if (!fn) throw Error("module " + m.name + " has no kernel " + m.entry);
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [D(this) newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) throw Error("pipeline " + p->id + (label.empty() ? "" : " (" + label + ")") + ": " + str(err.localizedDescription));
    const uint64_t lanes = (uint64_t)m.local[0] * m.local[1] * m.local[2];
    if (lanes > ps.maxTotalThreadsPerThreadgroup)
      throw Error("pipeline " + p->id + (label.empty() ? "" : " (" + label + ")") + " needs " + std::to_string(lanes) + " lanes a workgroup; " + name + " runs it with " +
                  std::to_string(ps.maxTotalThreadsPerThreadgroup));
    p->mtl = keep(ps);
    std::copy(m.local, m.local + 3, p->local);
    p->bind = m.bind;
    return p;
  }
}

Pipeline::~Pipeline() {
  if (!dev || dev->abandoned) return;
  drop(mtl);
}

std::shared_ptr<BindGroup> Device::createBindGroup(std::shared_ptr<BindGroupLayout> bgl, const std::vector<Resource>& entries, const std::string& id) {
  uint32_t nSmp = 0;
  for (auto b : bgl->entries) if (b == Bind::Sampler) nSmp++;
  if (entries.size() + nSmp != bgl->entries.size()) throw Error("bind group of " + std::to_string(entries.size()) + " entries on layout " + bgl->id + " of " + std::to_string(bgl->entries.size() - nSmp) + " resources");
  auto g = std::make_shared<BindGroup>();
  g->dev = this;
  g->bgl = bgl;
  g->id = id.empty() ? autoId('g') : id;
  g->res.assign(bgl->entries.size(), nullptr);
  for (uint32_t i = 0, k = 0; i < bgl->entries.size(); i++) {
    if (bgl->entries[i] == Bind::Sampler) continue;
    if (bgl->entries[i] == Bind::Tex) {
      if (!entries[k].tex) throw Error("bind group on " + bgl->id + ": binding " + std::to_string(i) + " wants a texture");
      g->res[i] = entries[k].tex->mtl;
    } else {
      if (!entries[k].buf) throw Error("bind group on " + bgl->id + ": binding " + std::to_string(i) + " wants a buffer");
      g->res[i] = entries[k].buf->mtl;
    }
    k++;
  }
  return g;
}

BindGroup::~BindGroup() {}

std::shared_ptr<QuerySet> Device::createQuerySet(uint32_t count, const std::string& id) {
  auto q = std::make_shared<QuerySet>();
  q->dev = this;
  q->count = count;
  q->id = id.empty() ? autoId('q') : id;
  return q;
}

QuerySet::~QuerySet() {}

void Device::writeBuffer(Buffer& b, uint64_t off, const void* data, uint64_t size) {
  if (off + size > b.size) throw Error("writeBuffer past " + b.id + "'s " + std::to_string(b.size) + " bytes");
  if ((off & 3) || (size & 3)) throw Error("writeBuffer to " + b.id + " at " + std::to_string(off) + ", " + std::to_string(size) + " B: WebGPU's writes are whole u32s");
  if (!size) return;
  Write w{&b, off, std::vector<uint8_t>((const uint8_t*)data, (const uint8_t*)data + size)};
  std::lock_guard<std::mutex> l(mu);
  writes.push_back(std::move(w));
}

void Device::upload(Buffer& b, uint64_t off, const void* data, uint64_t size) {
  if (off + size > b.size) throw Error("upload past " + b.id + "'s " + std::to_string(b.size) + " bytes");
  if (!size) return;
  auto s = createBuffer(size, MAP_WRITE, autoId('s'));
  memcpy(s->mapped, data, (size_t)size);
  void* src = s->mtl;
  void* dst = b.mtl;
  std::lock_guard<std::mutex> l(mu);
  inits.push_back({&b, [src, dst, off, size](Cmd c) {
    [(__bridge id<MTLBlitCommandEncoder>)c copyFromBuffer:(__bridge id<MTLBuffer>)src sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)dst destinationOffset:off size:size];
  }, 1});
  staging.push_back(s);
}

Encoder Device::encoder() {
  @autoreleasepool {
    Encoder e;
    e.dev = this;
    id<MTLCommandBuffer> cb = [(__bridge id<MTLCommandQueue>)queue commandBuffer];
    if (!cb) throw Error("no command buffer on " + name);
    e.cb = keep(cb);
    return e;
  }
}

Encoder::Encoder(Encoder&& o) noexcept { *this = std::move(o); }
Encoder& Encoder::operator=(Encoder&& o) noexcept {
  if (this == &o) return *this;
  if (open && enc) { [(__bridge id<MTLCommandEncoder>)enc endEncoding]; }
  drop(enc);
  drop(cb);
  dev = o.dev; cb = o.cb; enc = o.enc; open = o.open; group = o.group; bound = o.bound;
  commands = o.commands; inPass = o.inPass; passQs = o.passQs; passEnd = o.passEnd; reset = std::move(o.reset); pipe = o.pipe;
  o.cb = nullptr; o.enc = nullptr; o.open = 0;
  return *this;
}
// ai: an encoder never submitted: its open encoder ended and its command buffer let go, never committed
Encoder::~Encoder() {
  if (open && enc) { @autoreleasepool { [(__bridge id<MTLCommandEncoder>)enc endEncoding]; } }
  drop(enc);
  drop(cb);
}

void Encoder::hazard() { commands++; }

void Encoder::beginComputePass(QuerySet* qs, int begin, int end) {
  (void)begin;
  if (inPass) throw Error("a pass begun inside a pass");
  inPass = true;
  passQs = qs;
  passEnd = end;
}

void Encoder::endPass() {
  if (!inPass) throw Error("end with no pass");
  inPass = false;
  passQs = nullptr;
}

void Encoder::setPipeline(const Pipeline& p) {
  if (pipe != &p) bound = false;
  pipe = &p;
}

void Encoder::setBindGroup(const BindGroup& g) {
  if (!pipe) throw Error("a bind group before a pipeline");
  if (group != &g) bound = false;
  group = &g;
}

void Encoder::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!x || !y || !z) return;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> c = forDispatch(*this, false);
    [c dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:localOf(*pipe)];
  }
}

void Encoder::dispatchAlongside(uint32_t x, uint32_t y, uint32_t z) {
  if (!x || !y || !z) return;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> c = forDispatch(*this, true);
    [c dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:localOf(*pipe)];
  }
}

void Encoder::dispatchIndirectAlongside(const Buffer& b, uint64_t offset) {
  @autoreleasepool {
    id<MTLComputeCommandEncoder> c = forDispatch(*this, true);
    [c dispatchThreadgroupsWithIndirectBuffer:B(b) indirectBufferOffset:offset threadsPerThreadgroup:localOf(*pipe)];
  }
}

void Encoder::dispatchIndirect(const Buffer& b, uint64_t offset) {
  @autoreleasepool {
    id<MTLComputeCommandEncoder> c = forDispatch(*this, false);
    [c dispatchThreadgroupsWithIndirectBuffer:B(b) indirectBufferOffset:offset threadsPerThreadgroup:localOf(*pipe)];
  }
}

void Encoder::copyBufferToBuffer(const Buffer& src, uint64_t so, const Buffer& dst, uint64_t dof, uint64_t size) {
  if (!size) return;
  @autoreleasepool {
    commands++;
    [blit(*this) copyFromBuffer:B(src) sourceOffset:so toBuffer:B(dst) destinationOffset:dof size:size];
  }
}

void Encoder::copyBufferToTexture(const Buffer& src, uint64_t offset, uint32_t stride, const Texture& tex, uint32_t layer, uint32_t w, uint32_t h) {
  @autoreleasepool {
    commands++;
    const NSUInteger row = stride ? stride : w;
    [blit(*this) copyFromBuffer:B(src) sourceOffset:offset sourceBytesPerRow:row sourceBytesPerImage:row * h sourceSize:MTLSizeMake(w, h, 1)
                      toTexture:T(tex) destinationSlice:layer destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
  }
}

void Encoder::copyTextureToTexture(const Texture& src, uint32_t srcLayer, const Texture& dst, uint32_t dstLayer, uint32_t w, uint32_t h) {
  @autoreleasepool {
    commands++;
    [blit(*this) copyFromTexture:T(src) sourceSlice:srcLayer sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                       toTexture:T(dst) destinationSlice:dstLayer destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
  }
}

void Encoder::copyTextureToBuffer(const Texture& src, uint32_t layer, uint32_t w, uint32_t h, const Buffer& dst, uint64_t offset) {
  @autoreleasepool {
    commands++;
    [blit(*this) copyFromTexture:T(src) sourceSlice:layer sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                        toBuffer:B(dst) destinationOffset:offset destinationBytesPerRow:w destinationBytesPerImage:(NSUInteger)w * h];
  }
}

void Encoder::clearBuffer(const Buffer& b, uint64_t offset, uint64_t size) {
  if (size == ~0ull) size = b.size - offset;
  if (!size) return;
  @autoreleasepool {
    commands++;
    [blit(*this) fillBuffer:B(b) range:NSMakeRange(offset, size) value:0];
  }
}

// ai: no timestamps on Metal (Features::timestamps): zeros, as Vulkan's resolve writes where a device has none
void Encoder::resolveQuerySet(const QuerySet& qs, uint32_t first, uint32_t count, const Buffer& dst, uint64_t offset) {
  (void)qs; (void)first;
  @autoreleasepool {
    commands++;
    [blit(*this) fillBuffer:B(dst) range:NSMakeRange(offset, 8ull * count) value:0];
  }
}

// ai: The head (wg.cpp's): the new objects' fills and clears, the uploads' copies, then the queue's writes (each from a
// ai: buffer of its bytes the command buffer holds), a blit encoder each phase.
static void recordHead(Device& d, id<MTLCommandBuffer> cb) {
  for (int phase = 0; phase < 2; phase++) {
    bool any = false;
    for (auto& f : d.inits) any = any || f.phase == phase;
    if (!any) continue;
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    for (auto& f : d.inits) if (f.phase == phase) f.f((__bridge void*)be);
    [be endEncoding];
  }
  if (!d.writes.empty()) {
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    for (auto& w : d.writes) {
      id<MTLBuffer> s = [D(&d) newBufferWithBytes:w.data.data() length:w.data.size() options:MTLResourceStorageModeShared];
      if (!s) throw OutOfMemory("no Metal buffer for a write of " + std::to_string(w.data.size()) + " B on " + d.name);
      [be copyFromBuffer:s sourceOffset:0 toBuffer:B(*w.dst) destinationOffset:w.off size:w.data.size()];
    }
    [be endEncoding];
  }
  d.inits.clear();
  d.writes.clear();
}

std::shared_ptr<Ticket> Device::submit(Encoder& enc) {
  if (enc.inPass) throw Error("submit with a pass open");
  if (!enc.cb) throw Error("submit of an encoder with no command buffer");
  @autoreleasepool {
    endOpen(enc);
    auto t = std::make_shared<Ticket>();
    t->dev = this;
    std::lock_guard<std::mutex> lock(mu);
    if (!inits.empty() || !writes.empty()) {
      id<MTLCommandBuffer> head = [(__bridge id<MTLCommandQueue>)queue commandBuffer];
      if (!head) throw Error("no command buffer on " + name);
      recordHead(*this, head);
      [head commit];
    }
    id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)enc.cb;
    enc.cb = nullptr;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) { (void)done; dispatch_semaphore_signal(sem); }];
    [cb commit];
    t->t0 = steadyMs();
    t->cb = keep(cb);
    t->sem = keep(sem);
    t->keep = std::move(staging);
    staging.clear();
    return t;
  }
}

void Device::flush() {
  { std::lock_guard<std::mutex> l(mu); if (inits.empty() && writes.empty()) return; }
  Encoder e = encoder();
  submit(e)->wait();
}

void Device::copyOnQueue2(const Buffer& src, uint64_t srcOff, const Buffer& dst, uint64_t dstOff, uint64_t size) {
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  id<MTLCommandBuffer> cb = nil;
  @autoreleasepool {
    std::lock_guard<std::mutex> p(pool2Mu);
    cb = [(__bridge id<MTLCommandQueue>)queue2 commandBuffer];
    if (!cb) throw Error("no command buffer on " + name + "'s second queue");
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:B(src) sourceOffset:srcOff toBuffer:B(dst) destinationOffset:dstOff size:size];
    [be endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) { (void)done; dispatch_semaphore_signal(sem); }];
    [cb commit];
    drop(last2);
    last2 = keep(cb);
  }
  const long late = dispatch_semaphore_wait(sem, hangMs ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)hangMs * 1000000) : DISPATCH_TIME_FOREVER);
  if (late) throw Error("a copy on the second queue did not finish in " + std::to_string(hangMs) + " ms");
  if (cb.status == MTLCommandBufferStatusError) fault(this, cb);
}

void Device::waitQueue2() {
  id<MTLCommandBuffer> cb = nil;
  {
    std::lock_guard<std::mutex> p(pool2Mu);
    cb = (__bridge id<MTLCommandBuffer>)last2;
  }
  if (cb) [cb waitUntilCompleted];
}

void Device::flushMapped(Buffer&) {}

bool Ticket::done() {
  if (!cb || finished) return true;
  if (dev->lost) throw Lost("the device is lost (" + dev->name + ")");
  id<MTLCommandBuffer> c = (__bridge id<MTLCommandBuffer>)cb;
  const MTLCommandBufferStatus s = c.status;
  if (s == MTLCommandBufferStatusCompleted) { finished = true; return true; }
  if (s == MTLCommandBufferStatusError) fault(dev, c);
  if (dev->hangMs && steadyMs() - t0 > dev->hangMs) hung();
  return false;
}

void Ticket::wait() {
  if (!cb || finished) return;
  if (dev->lost) throw Lost("the device is lost (" + dev->name + ")");
  const double left = dev->hangMs ? std::max(0.0, dev->hangMs - (steadyMs() - t0)) : 0;
  dispatch_semaphore_t s = (__bridge dispatch_semaphore_t)sem;
  if (dispatch_semaphore_wait(s, dev->hangMs ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(left * 1e6)) : DISPATCH_TIME_FOREVER)) hung();
  dispatch_semaphore_signal(s);   // ai: the signal back, for the destructor's wait
  finished = true;
  id<MTLCommandBuffer> c = (__bridge id<MTLCommandBuffer>)cb;
  if (c.status == MTLCommandBufferStatusError) fault(dev, c);
}

// ai: wg.cpp's rule: a submit past the bound loses the device to its user; Metal gets SETTLE_MS more to finish it, and
// ai: doing neither the device is abandoned
void Ticket::hung() {
  const uint32_t ms = dev->hangMs;
  dev->lost = true;
  dispatch_semaphore_t s = (__bridge dispatch_semaphore_t)sem;
  const bool settled = !dispatch_semaphore_wait(s, dispatch_time(DISPATCH_TIME_NOW, (int64_t)SETTLE_MS * 1000000));
  if (settled) { dispatch_semaphore_signal(s); finished = true; }
  else dev->abandoned = true;
  const std::string m = "a submit not done in " + std::to_string(ms) + " ms on " + dev->name + ": taken as hung, " +
                        (settled ? "done " + std::to_string((int)(steadyMs() - t0)) + " ms after it went" : "and abandoned (not done " + std::to_string(SETTLE_MS) + " ms later)");
  if (dev->log) dev->log("device: " + m);
  Lost e(m);
  throw e;
}

Ticket::~Ticket() {
  if (!dev || dev->abandoned) return;   // ai: the command buffer left (Device::abandoned)
  if (sem && !finished) {
    const int64_t ns = dev->hangMs ? (int64_t)(dev->hangMs + SETTLE_MS) * 1000000 : -1;
    if (dispatch_semaphore_wait((__bridge dispatch_semaphore_t)sem, ns < 0 ? DISPATCH_TIME_FOREVER : dispatch_time(DISPATCH_TIME_NOW, ns))) {
      dev->lost = true;
      dev->abandoned = true;
      return;
    }
  }
  drop(cb);
  drop(sem);
}

const uint8_t* Device::read(Buffer& b, uint64_t off, uint64_t size) {
  (void)size;
  if (!b.mapped) throw Error("read of " + b.id + ", which is not MAP_READ");
  return b.mapped + off;
}

}  // namespace wg
