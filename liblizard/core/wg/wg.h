// ai: wg: the part of WebGPU the decoder's host uses, over Vulkan, or over Metal on Apple's systems (2026-11-25,
// ai: wg_metal.mm, in place of Vulkan through MoltenVK: one backend a build, LIZ_VK or LIZ_METAL). The host keeps
// ai: WebGPU's rules (it was first written for WebGPU) here rather than in every caller:
// ai:   - every buffer and texture reads as zeros until written (a fill at creation);
// ai:   - queue writes land before the commands of the next submit and after everything submitted before them;
// ai:   - each dispatch sees everything the commands before it wrote (a memory barrier between commands; WebGPU on
// ai:     Vulkan runs a pass's dispatches one after the other too, so this is the web's order, not a new one);
// ai:   - textures are 2D arrays of r8unorm, sampled with textureLoad, kept in the GENERAL layout for their life;
// ai:   - timestamps come back in nanoseconds.
// ai: A kernel reaches either as gen writes it (liblizard/gen/gen.mjs, one Slang source): SPIR-V, or MSL with its
// ai: binding map (readModule).
// ai: Threads: one thread at a time records and submits (the decoder's: encoder(), an Encoder's commands, submit(),
// ai: flush()), and the command pool is touched by that thread alone. Any thread may create and destroy buffers and
// ai: textures, write a buffer and drop a Ticket: the pending writes and fills and the queue are under `mu`, and a
// ai: dropped Ticket's command buffers wait in `retired` for the recording thread to free (Vulkan wants a pool's
// ai: allocations, frees and recording in one thread at a time). The camera ingest has its own pool (pool2) and, where
// ai: the family has two, the second queue (queue2, core/ingest/camera.cpp); the sender's device has two users of
// ai: queue2, the desktop's presenter and the stream's chunk copies (core/tx/gpu_stream.cpp). Every submit, present and
// ai: wait on queue2 takes queue2Mu(): `mu2` where it is a queue of its own, `mu` where it is the queue itself.
// ai: (Until 2026-11-17 a user of a queue of its own took no lock, as if it were the queue's one thread: the
// ai: presenter's presents and the copies of a 3.1 GB file's chunks raced on the 4090's, a SIGSEGV in NVIDIA's submit.)
// ai: (Until 2026-09-30 nothing was locked: the receiver's camera thread freed command buffers through a slot's
// ai: ticket and, on the host-luma path, recorded and submitted its uploads, against the decoder thread on the
// ai: same pool and queue; the 4090 lost the device once in a run of the receiver on it, Xid 69.)
#pragma once
#if defined(__APPLE__)
#define LIZ_METAL 1
#else
#define LIZ_VK 1
#endif
#if LIZ_VK
#if defined(__ANDROID__) && !defined(VK_USE_PLATFORM_ANDROID_KHR)
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif
// ai: Vulkan loaded at run time (2026-10-03, for the library: third_party/volk): no link to libvulkan, so a machine with
// ai: no Vulkan loads the code and decodes on the CPU, and Windows needs no import library. Every vk* name is volk's
// ai: pointer, set by Device::create (volkLoadInstance: device calls dispatch through the loader, so two devices share a
// ai: process, the decoder's and the painter's).
#include "volk.h"
#endif
// ai: Metal (wg_metal.mm): the objects' handles are its Objective-C objects, held here as retained opaque pointers
// ai: (void*), so plain C++ includes this header; only the .mm files touch them.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace wg {

// ai: WebGPU's usage bits (GPUBufferUsage), as the setup and the host's code state them.
enum : uint32_t { MAP_READ = 1, MAP_WRITE = 2, COPY_SRC = 4, COPY_DST = 8, UNIFORM = 64, STORAGE = 128, INDIRECT = 256, QUERY_RESOLVE = 512 };
// ai: A bind group layout entry: what the shader binds there. Sampler: the device's one filtering sampler, immutable in
// ai: the layout (Device::linearSampler), so a bind group takes no resource for it.
enum class Bind : uint8_t { Tex, RO, RW, Uniform, Sampler };

// ai: code: the backend's own (Vulkan's VkResult, Metal's MTLCommandBufferError), 0 none
struct Error : std::runtime_error { using std::runtime_error::runtime_error; int code = 0; };
// ai: An allocation the device refused: the host halves its batch on this, as the web does on an out-of-memory scope.
struct OutOfMemory : Error { using Error::Error; };
// ai: The device lost (VK_ERROR_DEVICE_LOST, a Metal command buffer's error) or taken as hung (a submit not done within Device::hangMs; 2026-10-31).
// ai: Nothing on it can be trusted after it: its user drops every object on it and the device, and makes another.
struct Lost : Error { using Error::Error; };

// ai: A hang's bounds (2026-10-31): a submit not done HANG_MS after it went is taken as hung (Ticket::wait, done);
// ai: the driver then has SETTLE_MS more to finish it or lose the device (its own watchdog), past which the device is
// ai: abandoned (Device::abandoned). A batch over the decoder's 400 ms cap is halved long before HANG_MS.
constexpr uint32_t HANG_MS = 4000, SETTLE_MS = 10000;

struct Limits {
  uint64_t maxBufferSize = 0, maxStorageBufferRange = 0, deviceLocalBytes = 0;
  uint32_t maxComputeSharedMemorySize = 0, maxPerStageStorageBuffers = 0, maxImageArrayLayers = 0, maxComputeWorkGroupInvocations = 0;
  uint32_t maxUniformBufferRange = 0, minStorageAlign = 0, minUniformAlign = 0;
};

struct Features {
  bool f16 = false, storage16 = false, int8 = false, dot = false, subgroupShuffle = false, timestamps = false;
  bool zeroInit = false;   // ai: shaderZeroInitializeWorkgroupMemory: the SPIR-V zeroes workgroup memory by initializer
  bool int16 = false;      // ai: shaderInt16: 16-bit integers in workgroup memory and registers (a pipeline's "int16" need)
  // ai: the camera's zero-copy ingest (core/ingest/): YCbCr sampling, an r8 storage image, timeline semaphores, and
  // ai: on Android the AHardwareBuffer import; all four or the ingest copies luma instead
  bool ycbcr = false, storageR8 = false, timeline = false, ahb = false, foreign = false;
  uint32_t subgroupSize = 0;
};

// ai: What a device is (2026-10-25: Cpu a software rasterizer on the host's cores, lavapipe or SwiftShader, which the
// ai: receiver's auto passes over for the C, core/rx/receiver.cpp): Vulkan's physical device type, or Metal's device
// ai: (Integrated where its memory is the host's, Virtual a hypervisor's paravirtual device).
enum class Kind : uint8_t { Other, Integrated, Discrete, Virtual, Cpu };

// ai: A kernel as a device takes it, gen's output (readModule): SPIR-V words (Vulkan), or MSL (Metal) with its binding
// ai: map: the entry's name, the workgroup's size (Metal takes it at the dispatch), and for each Vulkan binding the Metal
// ai: argument slots it takes ('b' a buffer, 't' a texture, 's' a sampler; liblizard/gen/gen.mjs slangMsl).
struct Module {
  std::string name;
  std::vector<uint32_t> spirv;
  std::string msl, entry;
  uint32_t local[3] = {1, 1, 1};
  struct Slot { char kind; uint32_t index; };
  std::vector<std::vector<Slot>> bind;   // ai: by Vulkan binding; empty where the kernel declares none
  bool hasPush = false;
  Slot push{'b', 0};
};
// ai: gen's module `name`, read through `read` (a tree-relative path's bytes: the decoder's ReadFile, core/dec/setup.h):
// ai: <folder>/name.spv on Vulkan, <folder>/name.metal and its map <folder>/name.json on Metal, the folder "spv" or
// ai: "msl" unless named (LIZ_SPV, the lab's). Throws where it is missing or does not read. The second form reads the
// ai: tree under `dir` (the assets: liblizard/out, or the app's copy), by UTF-8 paths.
using Read = std::function<std::vector<uint8_t>(const std::string&)>;
Module readModule(const Read& read, const std::string& name, const std::string& folder = "");
Module readModule(const std::string& dir, const std::string& name);

struct Device;

struct Buffer {
  Device* dev = nullptr;
#if LIZ_VK
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
#else
  void* mtl = nullptr;   // ai: id<MTLBuffer>
#endif
  uint64_t size = 0;
  uint32_t usage = 0;
  uint8_t* mapped = nullptr;   // ai: MAP_READ buffers, host-visible and mapped for life
  bool coherent = true;
  std::string id;
  ~Buffer();
};

struct Texture {
  Device* dev = nullptr;
#if LIZ_VK
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;   // ai: the whole array, 2D_ARRAY, as texture_2d_array reads it
#else
  void* mtl = nullptr;   // ai: id<MTLTexture>, MTLTextureType2DArray
#endif
  uint32_t w = 0, h = 0, layers = 0;
  bool storage = false;
  std::string id;
  ~Texture();
};

struct BindGroupLayout {
  Device* dev = nullptr;
#if LIZ_VK
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
#endif
  std::vector<Bind> entries;
  std::string id;
  ~BindGroupLayout();
};

struct Pipeline {
  Device* dev = nullptr;
#if LIZ_VK
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
#else
  void* mtl = nullptr;   // ai: id<MTLComputePipelineState>
  uint32_t local[3] = {1, 1, 1};
  std::vector<std::vector<Module::Slot>> bind;
#endif
  std::shared_ptr<BindGroupLayout> bgl;
  std::string id, label;
  ~Pipeline();
};

// ai: A resource bound at a binding: a buffer (whole, as every group here binds them) or a texture's array view.
struct Resource {
  Buffer* buf = nullptr;
  Texture* tex = nullptr;
  Resource(Buffer* b) : buf(b) {}
  Resource(Texture* t) : tex(t) {}
  Resource(const std::shared_ptr<Buffer>& b) : buf(b.get()) {}
  Resource(const std::shared_ptr<Texture>& t) : tex(t.get()) {}
};

struct BindGroup {
  Device* dev = nullptr;
#if LIZ_VK
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
#else
  // ai: each binding's object in the layout's order (null at a Sampler binding: the device's), bound at each dispatch
  // ai: where the pipeline's map puts it; the buffers and textures, which own their objects, outlive the group by the
  // ai: host's rule as Vulkan's descriptors do
  std::vector<void*> res;
#endif
  std::shared_ptr<BindGroupLayout> bgl;
  std::string id;
  ~BindGroup();
};

struct QuerySet {
  Device* dev = nullptr;
#if LIZ_VK
  VkQueryPool pool = VK_NULL_HANDLE;
#endif
  uint32_t count = 0;
  std::string id;
  ~QuerySet();
};

// ai: A command buffer being recorded: WebGPU's command encoder and compute pass in one. A pass is only a timestamp
// ai: pair's scope here (Vulkan has no compute pass); every command is behind a memory barrier. On Metal the commands go
// ai: into a concurrent compute encoder (a memory barrier before every command but an alongside one) or a blit encoder
// ai: (the copies and fills), one after the other in the command buffer as they come.
struct Encoder {
  Device* dev = nullptr;
#if LIZ_VK
  VkCommandBuffer cb = VK_NULL_HANDLE;
#else
  void* cb = nullptr;    // ai: id<MTLCommandBuffer>, released by submit (or with an encoder never submitted)
  void* enc = nullptr;   // ai: the open id<MTLComputeCommandEncoder> or id<MTLBlitCommandEncoder>
  int open = 0;          // ai: 0 none, 1 compute, 2 blit
  const struct BindGroup* group = nullptr;
  bool bound = false;    // ai: the pipeline's and group's objects set on the open compute encoder
  Encoder() = default;
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  Encoder(Encoder&& o) noexcept;
  Encoder& operator=(Encoder&& o) noexcept;
  ~Encoder();
#endif
  int commands = 0;
  bool inPass = false;
  QuerySet* passQs = nullptr;
  int passEnd = -1;
  std::vector<QuerySet*> reset;     // ai: query sets reset at the head of this command buffer
  const Pipeline* pipe = nullptr;

  void beginComputePass(QuerySet* qs = nullptr, int begin = -1, int end = -1);
  void endPass();
  void setPipeline(const Pipeline& p);
  void setBindGroup(const BindGroup& g);
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  // ai: a dispatch with no barrier before it, so it may run alongside the commands since the last barrier: only for a
  // ai: dispatch that reads nothing they write and writes nothing they read or write, but through atomics
  void dispatchAlongside(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  void dispatchIndirectAlongside(const Buffer& b, uint64_t offset);
  void dispatchIndirect(const Buffer& b, uint64_t offset);
  void copyBufferToBuffer(const Buffer& src, uint64_t so, const Buffer& dst, uint64_t dof, uint64_t size);
  // ai: rows of w bytes, `stride` apart in src from `offset`, into layer `layer` of tex at (0, 0).
  void copyBufferToTexture(const Buffer& src, uint64_t offset, uint32_t stride, const Texture& tex, uint32_t layer, uint32_t w, uint32_t h);
  // ai: layer to layer, w x h from (0, 0) (the ring's slot into a lane's frame, as the web's copyTextureToTexture).
  void copyTextureToTexture(const Texture& src, uint32_t srcLayer, const Texture& dst, uint32_t dstLayer, uint32_t w, uint32_t h);
  // ai: a layer's w x h from (0, 0) into a buffer, rows of w bytes (an r8 texture: LIZ_DUMP's look at the ring)
  void copyTextureToBuffer(const Texture& src, uint32_t layer, uint32_t w, uint32_t h, const Buffer& dst, uint64_t offset);
  void clearBuffer(const Buffer& b, uint64_t offset = 0, uint64_t size = ~0ull);
  void resolveQuerySet(const QuerySet& qs, uint32_t first, uint32_t count, const Buffer& dst, uint64_t offset);
  void hazard();
};

// ai: A submitted command buffer's fence: done() polls it, wait() blocks on it. On Metal the command buffer itself and a
// ai: semaphore its completion handler signals (a bounded wait).
struct Ticket {
  Device* dev = nullptr;
#if LIZ_VK
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer cbs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
#else
  void* cb = nullptr;    // ai: id<MTLCommandBuffer>
  void* sem = nullptr;   // ai: dispatch_semaphore_t
  bool finished = false;
#endif
  std::vector<std::shared_ptr<Buffer>> keep;   // ai: staging the head copied from
  double t0 = 0;   // ai: when it was submitted (steady ms), for the hang's bound
  // ai: done() and wait() throw Lost on a lost device or past the hang's bound (Device::hangMs)
  bool done();
  void wait();
  ~Ticket();

 private:
  [[noreturn]] void hung();
};

// ai: A host's extras for Device::create: instance and device extensions (a window surface, the swapchain, present_id
// ai: and present_wait, whose features are enabled with them where the device has both), and a queue family that draws
// ai: (the first family that both computes and draws is taken anyway; this refuses a device with none).
// ai: Vulkan's alone: a Metal device has no extensions to ask for (its presenter's drawables need nothing of it).
struct DeviceExtras {
  std::vector<std::string> instanceExts, deviceExts;
  bool graphics = false;
  // ai: the queues at the global priority LOW where the device has VK_KHR_global_priority or VK_EXT_global_priority
  // ai: (2026-10-27, for a receiver's decoder: the screen, the preview and the compositor go first, the decode in its
  // ai: slack); the driver's default where it has neither. `lowPriority` on the Device says which took.
  bool lowPriority = false;
};

struct Device {
  // ai: The hang's bound in ms (HANG_MS; 0 none: a software device's, whose batches take seconds; LIZ_HANG_MS sets it).
  // ai: lost: the device lost or a submit past the bound (every Ticket then throws Lost). abandoned: a hung submit the
  // ai: driver neither finished nor lost in SETTLE_MS: nothing on the device is waited for or destroyed after it (its
  // ai: memory stays until the process ends), so a hung kernel cannot block its user's teardown.
  uint32_t hangMs = 0;
  std::atomic<bool> lost{false}, abandoned{false};
#if LIZ_VK
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  // ai: the second queue (the family's second where it has one, else queue itself) and its own command pool: the
  // ai: receiver's camera thread samples a frame into the ring there without waiting behind a batch; the sender's
  // ai: presenter presents there and the stream copies a file's chunks there. Submits, presents and waits under queue2Mu().
  VkQueue queue2 = VK_NULL_HANDLE;
  VkCommandPool pool2 = VK_NULL_HANDLE;
#else
  void* mtl = nullptr;      // ai: id<MTLDevice>
  void* queue = nullptr;    // ai: id<MTLCommandQueue>, the batches'
  void* queue2 = nullptr;   // ai: id<MTLCommandQueue>, the second queue's work (copyOnQueue2)
  void* linear = nullptr;   // ai: id<MTLSamplerState>, Bind::Sampler's
  void* last2 = nullptr;    // ai: the last id<MTLCommandBuffer> committed to queue2 (waitQueue2), under pool2Mu
  void* libs = nullptr;     // ai: NSMutableDictionary: each module's MSL compiled once (a library), by its name
  std::mutex libMu;
#endif
  bool twoQueues = false;
  bool lowPriority = false;   // ai: the queues made at the global priority LOW (DeviceExtras::lowPriority, where offered)
#if LIZ_VK
  uint32_t family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkPipelineCache cache = VK_NULL_HANDLE;
  bool pipeStats = false;   // ai: LIZ_PIPE_STATS: pipelines created with statistics capture, printed (lab)
  // ai: the layouts' immutable sampler (Bind::Sampler): bilinear, clamped to the edge, normalised coordinates, one
  // ai: level; made at the first layout that names it, destroyed with the device
  VkSampler linear = VK_NULL_HANDLE;
  VkSampler linearSampler();
  VkPhysicalDeviceMemoryProperties mem{};
#endif
  std::string name, driver;
  uint32_t apiVersion = 0, vendorId = 0, deviceId = 0, driverVersion = 0;
  Kind kind = Kind::Other;
  Limits limits;
  Features features;
  double timestampPeriod = 1.0;   // ai: ns a tick
  uint64_t timestampMask = ~0ull;
  std::atomic<uint64_t> allocated{0};   // ai: bytes of device memory this device holds (the web's fh.allocated)
  // ai: guards writes, inits, staging and retired, and the queue's submits where queue2 is the queue
  std::mutex mu;
  // ai: queue2's submits, presents and waits where it is a queue of its own
  std::mutex mu2;
  std::mutex& queue2Mu() { return twoQueues ? mu2 : mu; }
  // ai: pool2's (copyOnQueue2's callers, the stream's workers, record there one at a time)
  std::mutex pool2Mu;
#if LIZ_VK
  // ai: command buffers of dropped Tickets, freed by the recording thread at its next encoder()
  std::vector<VkCommandBuffer> retired;
#endif
  std::function<void(const std::string&)> log;
  std::string cachePath;

  // ai: The queue's writes waiting for the next submit (WebGPU's queue.writeBuffer), with their staging.
  struct Write { Buffer* dst; uint64_t off; std::vector<uint8_t> data; };
  std::vector<Write> writes;
  // ai: fills and layout changes of new objects, by the object they are for (its destructor drops its own)
  // ai: phase 0 (fills, clears, layout changes) runs before phase 1 (uploads' copies), a barrier between.
  // ai: Cmd: Vulkan's command buffer, or Metal's blit encoder (an id<MTLBlitCommandEncoder>) the head records into
#if LIZ_VK
  using Cmd = VkCommandBuffer;
#else
  using Cmd = void*;
#endif
  struct Init { const void* owner; std::function<void(Cmd)> f; int phase = 0; };
  std::vector<Init> inits;

  // ai: want: a substring of the device name to pick (LIZ_VK_DEVICE), else the first discrete, else the first.
  // ai: extras (2026-10-07, the desktop sender presents from the painter's device): what a host asks beyond the
  // ai: kernels' needs, each enabled where offered and left out where not (`has` says which took).
  static std::unique_ptr<Device> create(const std::string& want = "", bool validate = false, std::function<void(const std::string&)> log = nullptr,
                                        const DeviceExtras& extras = DeviceExtras());
  // ai: the device extensions enabled (the kernels' and the extras' that the device had)
  std::vector<std::string> enabled;
  bool has(const char* ext) const;
  ~Device();

  // ai: zero: filled with zeros at the next submit (WebGPU's rule); false only where an upload covers it whole.
  std::shared_ptr<Buffer> createBuffer(uint64_t size, uint32_t usage, const std::string& id = "", bool zero = true);
  // ai: storage: also written by a compute shader (the camera's ingest into the ring), as an r8 storage image.
  std::shared_ptr<Texture> createTexture(uint32_t w, uint32_t h, uint32_t layers, const std::string& id = "", bool storage = false);
  std::shared_ptr<BindGroupLayout> createBindGroupLayout(const std::vector<Bind>& entries, const std::string& id = "");
  // ai: spec: values for specialisation constants 0, 1, ... (u32 each), the bench kernels' sizes (lizard_gpu_check bench)
  std::shared_ptr<Pipeline> createPipeline(const Module& m, std::shared_ptr<BindGroupLayout> bgl, const std::string& id = "", const std::string& label = "", const std::vector<uint32_t>& spec = {});
#if LIZ_VK
  std::shared_ptr<Pipeline> createPipeline(const std::vector<uint32_t>& spirv, std::shared_ptr<BindGroupLayout> bgl, const std::string& id = "", const std::string& label = "", const std::vector<uint32_t>& spec = {});
#endif
  // ai: entries: a resource a binding in order, none for a Sampler binding (the layout holds it)
  std::shared_ptr<BindGroup> createBindGroup(std::shared_ptr<BindGroupLayout> bgl, const std::vector<Resource>& entries, const std::string& id = "");
  std::shared_ptr<QuerySet> createQuerySet(uint32_t count, const std::string& id = "");

  void writeBuffer(Buffer& b, uint64_t off, const void* data, uint64_t size);
  // ai: A large write (the setup's tables): through a host-visible staging buffer, copied at the head of the next
  // ai: submit; the staging lives until that submit is done.
  void upload(Buffer& b, uint64_t off, const void* data, uint64_t size);
  std::vector<std::shared_ptr<Buffer>> staging;
  Encoder encoder();
  // ai: The pending writes and inits first (their own command buffer, behind a barrier), then enc's commands; with
  // ai: wait set (Vulkan's), the submit waits for that timeline semaphore to reach waitValue (the camera ingest's)
  // ai: before it copies anything.
#if LIZ_VK
  std::shared_ptr<Ticket> submit(Encoder& enc, VkSemaphore wait = VK_NULL_HANDLE, uint64_t waitValue = 0);
#else
  std::shared_ptr<Ticket> submit(Encoder& enc);
#endif
  // ai: A copy on the second queue, waited for (bounded by hangMs; past it the copy's objects are left to the device and
  // ai: this throws): the stream's chunks into its symbols (core/tx/gpu_stream.cpp), from any thread.
  void copyOnQueue2(const Buffer& src, uint64_t srcOff, const Buffer& dst, uint64_t dstOff, uint64_t size);
  // ai: Everything on the second queue done (the ring that the camera's ingest writes there let go).
  void waitQueue2();
  // ai: A host-written mapped buffer's bytes made the device's (Vulkan's non-coherent memory; nothing on Metal's
  // ai: shared memory).
  void flushMapped(Buffer& b);
  // ai: The pending writes and inits alone, waited for (the setup's uploads).
  void flush();
  void waitIdle();
  // ai: A MAP_READ buffer's bytes once its ticket is done (invalidated where the memory is not coherent).
  const uint8_t* read(Buffer& b, uint64_t off, uint64_t size);

  // ai: The pipeline cache (Vulkan's; Metal keeps its compiled kernels in the system's own cache, so none there)
  void loadCache(const std::string& path);
  void saveCache();

#if LIZ_VK
  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0);
#endif
  std::atomic<int> nextId{0};
  std::string autoId(char kind);
};

#if LIZ_VK
void check(VkResult r, const char* what);
#endif

}  // namespace wg
