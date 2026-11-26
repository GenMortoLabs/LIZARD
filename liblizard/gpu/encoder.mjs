// ai: The sender's encoder on the GPU (2026-09-29), for PCs with a good GPU, as the sender has lagged at LIZARD-1024;
// ai: bit exact only in the ring; batches of 16, since a sender's latency matters little and a dropped frame does.
// ai: Frames' blocks in, their symbols' pixels out, on the device, and onto a WebGPU canvas a frame at a time; the page
// ai: (lizard-web/send.mjs) keeps its timing and paces the encodes, and the worker keeps making the blocks.
// ai:
// ai: The kernels, one compute pass an encode of up to `sub` frames, a symbol a workgroup z (frames x codes, the codes
// ai: of a frame side by side):
// ai:   paint: a block a workgroup (slang/src/send_paint.slang, 2026-10-07): the CRC-64, its code's systematic codeword
// ai:     (every code of the format's rate profile, src/lizard_phy.h lizard_tiers_for; the codes' tables in TAB, the native
// ai:     painter's, core/tx/send_tables.cpp SendConsts through the wasm), the bit map and whitening into SD, each data coefficient's two bits; the test stream's payloads made there from their ids (BLAKE3 of the id,
// ai:     src/xfer.h stream_fill: encode's test, 2026-11-15), so its blocks come as ids;
// ai:   irows, ipic: the decoder's own butterflies as the inverse transforms (slang/src/send_irows.slang, send_ipic.slang) on
// ai:     the spectrum transposed (core/tx/send_tables.cpp sendTables' uv, INV its entries' frame coefficients, bit 31 conjugated):
// ai:     irows makes each entry's coefficient from paint's data bits (the pilots, the padding, QPSK; 2026-11-24, the
// ai:     disc no longer painted between), the inverse along y, Y two halves a word, then turn_pack's
// ai:     packing and the inverse along x, clipped to 0..1 in f32, the picture row-major;
// ai:     in the copy form (below) ipic writes the frame's bytes of the square itself, from workgroup memory (ipicFrame);
// ai:   rsh: the C's Lanczos-3 resample onto the ring's square with the C's own taps (sendTables' taps), a workgroup
// ai:     a frame row, its vertical pass in workgroup memory, grey as lizard_paint_rgba rounds it, and every pixel outside
// ai:     the square the border's byte (slang/src/send_rsh.slang);
// ai:     where every tap is the C's on-sample copy (pxm = n / span: 384 to 1536 in the 64 ring, 512, 768, 1024, 1280
// ai:     and 1536 in the 128, 512, 1024 and 1536 in the 256, 15 of the 24 pairs) no resample, ipic's frame form writing the
// ai:     square's words and rsh's border form every other word;
// ai:   present: a render pass onto the canvas, whole canvas pixels a frame pixel, a frame of a batch.
// ai: The border (ring, marks, band, word, quiet margin) is the C's paint, taken once a configuration from this
// ai: module's own Lizard on the importing thread's wasm, so it is the C's to the byte. The square
// ai: is not: the transform's rounding differs, within one grey level of the C's (scripts/gpu/encoder_check.mjs); `check`
// ai: holds a device to that on frames of its choosing (a driver that returns garbage fails it), not a conformance
// ai: rule: a sender's symbol need only decode (2026-09-29).
// ai:
// ai:   const enc = await GpuEncoder.create();                 // throws Unavailable: no WebGPU, no adapter
// ai:   const cfg = await enc.configure({ spec, codes, gap });   // spec { n, subch, span }; gap: modules between codes; again at a re-pick
// ai:   const ctx = enc.attach(canvas);                        // a canvas with no 2D context
// ai:   enc.encode(half, f0, blocks, count, cfg, parity, test); // frames f0 .. f0 + count - 1 of a half, count <= cfg.sub,
// ai:                                                          // each codes x blocks x 472 B (test: the ids alone read)
// ai:   enc.present(half * cfg.batch + f, ctx, whole, cfg);    // frame f of that half, at its due time
// ai:   enc.release(cfg);                                      // a configuration no frame still needs
// ai:   await enc.check(blocks, count, cfg); cfg.frameMs; enc.takeGpuMs(); enc.destroy();
// ai: A configuration holds two batches' frames (half 0 and 1, `batch` frames each: one presenting while the next is
// ai: made; a frame is encoded again only after its present was submitted) and the scratch of `sub` frames. An encode
// ai: is one submission, and WebGPU runs a queue in order, so a present waits behind every encode submitted before it:
// ai: one encode of a whole batch of 16 at LIZARD-1024 on the iGPU (85 ms) cost 6 of 60 frames a second (2026-09-29),
// ai: which is why the encodes are small and the page paces them (cfg.frameMs, the measured GPU ms a frame of that
// ai: configuration: timestamps, or a submission's time to done where there are none). A configuration is its own
// ai: buffers, so frames encoded before a re-pick present from theirs while the next one's are made; cfg left out is
// ai: the latest. The wasm holds one codec at a time, the latest configuration's Lizard (cfg.lizard): configure and
// ai: check run one at a time on a chain shared by every encoder. batch 16 and sub 4 unless asked, sub then batch halved
// ai: while the device cannot hold a configuration's buffers (cfg.batch, cfg.sub say what it got).
// ai: the kernels, WGSL slangc writes from the sender's Slang (slang/src/send_*.slang), by ../gen/gen.mjs; their constants
// ai: core/tx/send_plan.h's, and every table the native painter's (core/tx/send_tables.cpp), from the wasm (sim/ob.mjs
// ai: sendTables: the module's codec's, made by the C as the native painter makes them)
import { SEND_WGSL } from "./send_wgsl.mjs";
// ai: the squares' RMS difference from the C's paint a device's frames are held under (grey levels; check)
export const SQUARE_RMS = 0.5;
import { init as initOb, Lizard, streamBlock, sendTables, GAP_MODULES } from "../sim/ob.mjs";
import { ID_BYTES, idGet } from "../sim/ob.mjs";

// ai: A compute pipeline, compiled asynchronously with no error scope of its own, so any number of them can be in flight
// ai: at once (2026-09-29, the worker's start: STATUS "The ramp at the start"). An error scope is the device's, pushed and
// ai: popped in call order, and two compiles awaited across each other pop each other's; a failed creation rejects its
// ai: own promise instead (GPUPipelineError), and the module's own messages, where it has any, name the line.
async function computePipeline(device, { code, layout, label }) {
  const module = device.createShaderModule({ label, code });
  try {
    return await device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: "main" } });
  } catch (e) {
    const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (!errs.length) throw e;
    throw new Error(errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + String(code).split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
  }
}

// ai: Frames a half unless asked (32), and frames an encode at most: the sender paints ahead, so a deep batch adds no
// ai: latency (the page's pump still holds each encode to its share of the animation frame).
export const BATCH = 32, SUB = 32;
// ai: Timestamp readbacks in flight at most; a batch that finds none free goes untimed.
const TIMERS = 4;
// ai: The compute stages a batch, in order (a profile's names).
export const STAGES = ["paint", "irows", "ipic", "rsh"];

export class Unavailable extends Error {}

// ai: The wasm holds one codec for the whole module, whichever encoder set it up: the steps that use it (configure,
// ai: check) run one at a time on this chain across every GpuEncoder, and lizard is the Lizard that holds it
// ai: (null while it changes hands). A configuration paints its border only while its Lizard is this one.
const codec = { chain: Promise.resolve(), lizard: null };

export class GpuEncoder {
  // ai: adapter: one the caller asked for, else the one powerPreference names (high-performance: the sender is for a
  // ai: PC's good GPU). batch: frames a half; sub: frames an encode at most. profile: every stage its own compute pass
  // ai: and timestamp pair (takeStageMs), for a timing; the default is one pass an encode, timed whole (takeGpuMs).
  static async create({ adapter = null, powerPreference = "high-performance", batch = BATCH, sub = SUB, profile = false, log = () => {} } = {}) {
    if (!globalThis.navigator?.gpu) throw new Unavailable("no WebGPU in this browser");
    adapter ??= await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Unavailable("no WebGPU adapter");
    const L = adapter.limits, ts = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: ts ? ["timestamp-query"] : [],
      requiredLimits: { maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize },
    });
    const e = new GpuEncoder();
    const i = adapter.info ?? {};
    Object.assign(e, { adapter, device, batch, sub: Math.min(sub, batch), dead: false, log, ts, format: navigator.gpu.getPreferredCanvasFormat(), pipes: new Map(), cfg: null, cfgs: new Set(), lost: null, error: null, timers: [], timersMade: 0, gpuNs: [], profile: profile && ts, stageNs: [] });
    e.adapterName = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ").replace(/\s+/g, " ").trim() || "unknown adapter";
    device.lost.then((x) => { e.lost = x.message || x.reason || "lost"; log(`gpu encoder: device lost: ${e.lost}`); });
    device.addEventListener("uncapturederror", (x) => { e.error ??= x.error.message; log(`gpu encoder: ${x.error.message}`); });
    try { await e.buildShared(); } catch (err) { device.destroy(); throw err; }
    return e;
  }

  // ai: The layouts, and the pipelines no picture size shapes (paint, rsh twice, present), compiled at once.
  async buildShared() {
    const d = this.device, RO = "read-only-storage", RW = "storage", UN = "uniform";
    const layout = (types, visibility = GPUShaderStage.COMPUTE) => d.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility, buffer: { type } })) });
    this.bgl = {
      // ai: paint: BLOCKS, TAB, SD, P; irows: SD, TW, Y, ROWS, SU, TAB, INV, P (it makes the coefficients from the data bits)
      paint: layout([RO, RO, RW, UN]),
      irows: layout([RO, RO, RW, UN, UN, RO, RO, UN]),
      ipic: layout([RO, RO, RW, UN]),
      ipicFrame: layout([RO, RO, RW, UN, UN, RO, RO]),
      rsh: layout([RO, RO, RO, RW, UN]),
      present: layout([RO, UN], GPUShaderStage.FRAGMENT),
    };
    const vs = d.createShaderModule({ label: "send present vs", code: SEND_WGSL.presentVs }), fs = d.createShaderModule({ label: "send present fs", code: SEND_WGSL.presentFs });
    // ai: The pipelines under names of their own: `present` is the method.
    [this.paintPipe, this.rshPipe, this.rshBorderPipe, this.presentPipe] = await Promise.all([
      computePipeline(d, { code: SEND_WGSL.paint, layout: this.pl("paint"), label: "send paint" }),
      computePipeline(d, { code: SEND_WGSL.rsh, layout: this.pl("rsh"), label: "send rsh" }),
      computePipeline(d, { code: SEND_WGSL.rshBorder, layout: this.pl("rsh"), label: "send rsh border" }),
      d.createRenderPipelineAsync({ label: "send present", layout: this.pl("present"), vertex: { module: vs, entryPoint: "main" }, fragment: { module: fs, entryPoint: "main", targets: [{ format: this.format }] }, primitive: { topology: "triangle-list" } }),
    ]);
  }
  pl(k) { return this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl[k]] }); }

  // ai: The two inverse transforms of a picture size (ipic in both forms), built once and kept. ipic holds two exchange buffers of n
  // ai: words, a complex value two halves a word (12,288 B at 1536): a device that offers less has no GPU encoder at that size (Unavailable).
  pipesFor(n) {
    if (!this.pipes.has(n)) {
      const lim = this.device.limits, need = 8 * n;
      if (need > lim.maxComputeWorkgroupStorageSize) throw new Unavailable(`the ${n} picture's inverse needs ${need} B of workgroup memory, the device has ${lim.maxComputeWorkgroupStorageSize}`);
      if (n / 8 > lim.maxComputeInvocationsPerWorkgroup) throw new Unavailable(`the ${n} picture's inverse needs ${n / 8} invocations a workgroup`);
      if (!SEND_WGSL[`irows${n}`] || !SEND_WGSL[`ipic${n}`] || !SEND_WGSL[`ipicFrame${n}`]) throw new Unavailable(`no inverse built for the ${n} picture`);
      const d = this.device, job = Promise.all([
        computePipeline(d, { code: SEND_WGSL[`irows${n}`], layout: this.pl("irows"), label: `send irows ${n}` }),
        computePipeline(d, { code: SEND_WGSL[`ipic${n}`], layout: this.pl("ipic"), label: `send ipic ${n}` }),
        computePipeline(d, { code: SEND_WGSL[`ipicFrame${n}`], layout: this.pl("ipicFrame"), label: `send ipic frame ${n}` }),
      ]);
      job.catch(() => this.pipes.delete(n));
      this.pipes.set(n, job);
    }
    return this.pipes.get(n);
  }

  // ai: The encoder's steps that use the wasm's one codec run one at a time, in call order (a failed one does not stop
  // ai: the next).
  serial(fn) {
    const p = codec.chain.then(fn);
    codec.chain = p.catch(() => {});
    return p;
  }

  // ai: A configuration: the format's tables (the native painter's, made by the C: sim/ob.mjs sendTables), the border
  // ai: from the C's paint, every buffer a batch needs.
  // ai: spec as the page's (n, subch, span); codes symbols a frame, side by side.
  configure(opts) { return this.serial(() => this.configureNow(opts)); }
  async configureNow({ spec, codes = 1, gap = GAP_MODULES, clock = [60, 1] }) {
    const { n, subch, span = 0 } = spec;
    const alive = () => { if (this.dead) throw new Unavailable("the encoder was destroyed"); };
    await initOb();
    alive();
    const pipes = this.pipesFor(n);
    // ai: The codec changes hands from here (this configuration's Lizard), so the Lizard before it can no longer paint,
    // ai: whether or not this configuration comes to be.
    codec.lizard?.free();
    codec.lizard = null;
    const fc = new Lizard(n, subch, 1, { span });
    fc.setClock(...clock);   // ai: the sender's clock the border's word states (src/fmt.h): fps, the places' largest delay
    let c = null;
    try {
      alive();
      const [irows, ipic, ipicFrame] = await pipes;
      alive();
      // ai: The sizes. Over the device's limits: FRAME (two halves of batch frames) halves batch, the scratch (sub
      // ai: frames) halves sub. Out of memory, which buffer is unknown: sub first, then batch. Both error scopes, popped
      // ai: together at once: a buffer out of memory is invalid and the bind groups made with it are validation errors,
      // ai: which would otherwise reach uncapturederror (this.error, which gives the encoder up) from an attempt thrown
      // ai: away, and a scope left open across an animation frame would catch the running configuration's errors.
      const lim = this.device.limits, most = Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize), d = this.device;
      let batch = this.batch, sub = this.sub;
      while (!c) {
        sub = Math.min(sub, batch);
        if (batch < 1) throw new Unavailable(`no configuration of LIZARD-${subch} at ${n} fits this device's memory`);
        // ai: the tables for encodes of sub frames (the paint's uniform counts its symbols), while fc is the module's codec
        const tables = sendTables(sub, codes, gap);
        if (tables.why) throw new Unavailable(`LIZARD-${subch} at ${n}: ${tables.why}`);
        // ai: TAB, the encoder's for its life (the same for every format; release leaves it, destroy takes the device)
        if (!this.TAB) { this.TAB = this.device.createBuffer({ size: tables.tab.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(this.TAB, 0, tables.tab); }
        const t = this.shape({ spec, tables, fc, batch, sub, irows, ipic, ipicFrame }), z = t.sizes;
        if (z.FRAME > most) { batch = Math.floor(batch / 2); continue; }
        if (Math.max(z.BLOCKS, z.SD, z.Y, z.PIC) > most) {
          if (sub === 1) throw new Unavailable(`one frame of LIZARD-${subch} at ${n} needs a buffer over this device's limit`);
          sub = Math.floor(sub / 2); continue;
        }
        d.pushErrorScope("validation");
        d.pushErrorScope("out-of-memory");
        this.allocate(t, tables);
        const [oom, bad] = await Promise.all([d.popErrorScope(), d.popErrorScope()]);
        if (oom || bad || this.dead) {
          for (const b of t.bufs) b.destroy();
          alive();
          this.log(`gpu encoder: halves of ${batch}, encodes of ${sub}: ${(oom ?? bad).message}; smaller`);
          if (sub > 1) sub = Math.floor(sub / 2); else batch = Math.floor(batch / 2);
          continue;
        }
        c = t;
      }
    } catch (e) { fc.free(); throw e; }
    c.lizard = codec.lizard = fc;
    c.clock = [...clock];
    this.cfg = c;
    this.cfgs.add(c);
    this.paintBorder(c);
    this.log(`gpu encoder: ${c.label}, ${this.adapterName}, halves of ${c.batch} frames, encodes of up to ${c.sub}, square ${c.q} px from ${c.sq}${c.copy ? " (a copy)" : ""}, frame ${c.FW} x ${c.W}, ${(c.bufs.reduce((t, b) => t + b.size, 0) / 1048576).toFixed(1)} MB`);
    return c;
  }

  // ai: A configuration's numbers and its buffers' sizes: halves of `batch` frames, scratch for `sub` (R symbols). The
  // ai: frame (the C's): codes symbols W wide a gap apart (the margin's colour, slang/src/send_rsh.slang pixel), RW words
  // ai: a row, FS words a frame; a copy where every pixel takes one sample whole (the C's on-sample taps: no resample, no
  // ai: PIC, ipic's frame form writing the square's bytes).
  shape({ spec, tables: t, fc, batch, sub, irows, ipic, ipicFrame }) {
    const { span = 0 } = spec, { n, subch, codes, gap, W, FW, RW, FS, q, sq, Vr, npos: sStride, sdWords, copy, blocks: V } = t, B = fc.blockBytes;
    const R = sub * codes, words = RW * W, yStride = Vr * n;
    const sizes = {
      // ai: Y between the inverses a complex value two halves a word (slang/src/send_irows.slang)
      // ai: SD, the data coefficients' bits (paint's, 16 a word); irows makes the coefficients from them, no disc between
      BLOCKS: R * V * t.BLOCK_BYTES, SD: R * sdWords * 4, Y: R * yStride * 4,
      PIC: copy ? 16 : R * n * n * 4, FRAME: batch * FS * 4,
    };
    return { spec, n, subch, span, codes, gap, V, B, W, FW, RW, FS, q, sq, words, Vr, sStride, yStride, sdWords, batch, sub, R, copy, sizes, irows, ipic, ipicFrame, bufs: [], whole: 0, frameMs: null, lizard: null,
      blockStride: t.BLOCK_BYTES, pDims: t.P_DIMS, rshThreads: t.RSH_THREADS, rshBand: t.RSH_BAND,
      label: `LIZARD-${subch} at ${n}, ${W} px a symbol${codes > 1 ? ` x ${codes}` : ""}` };
  }

  // ai: The buffers and bind groups of a shaped configuration.
  allocate(c, t) {
    const d = this.device, U = GPUBufferUsage, { n, q, W } = c;
    const buf = (size, usage, data) => {
      const b = d.createBuffer({ size: Math.max(16, 4 * Math.ceil(size / 4)), usage });
      if (data) d.queue.writeBuffer(b, 0, data);
      c.bufs.push(b);
      return b;
    };
    // ai: the C's: the paint's uniform (the blocks, the slots they take and the coefficients the frame holds: the padding
    // ai: between; each code's tier, first block, first slot, blocks; dims; the CRC's powers), the inverse's sizes and
    // ai: clip (SU), the disc's rows, the taps (each one's sample mod n, then the weights, 6 a pixel), rsh's G (the
    // ai: geometry, then the anchors' runs: send_rsh.slang's G)
    c.g = new Uint32Array([...t.g, ...t.anch]);
    c.BLOCKS = buf(c.sizes.BLOCKS, U.STORAGE | U.COPY_DST);
    c.staging = new Uint8Array(c.sizes.BLOCKS);
    // ai: INV, a disc entry's frame coefficient (bit 31 conjugated): the size's UV turned round (an entry past the
    // ai: format's coefficients irows makes zero)
    const inv = new Uint32Array(t.uv.length);
    for (let i = 0; i < t.uv.length; i++) inv[t.uv[i] & 0x7fffffff] = (i | (t.uv[i] & 0x80000000)) >>> 0;
    const INV = buf(4 * inv.length, U.STORAGE | U.COPY_DST, inv);
    c.SD = buf(c.sizes.SD, U.STORAGE);
    const PU = buf(4 * t.pu.length, U.UNIFORM | U.COPY_DST, t.pu);
    const TW = buf(8 * n, U.STORAGE | U.COPY_DST, t.tw);
    const ROWS = buf(4 * t.rows.length, U.UNIFORM | U.COPY_DST, t.rows);
    const SU = buf(4 * t.su.length, U.UNIFORM | U.COPY_DST, t.su);
    const Y = buf(c.sizes.Y, U.STORAGE);
    const PIC = buf(c.sizes.PIC, U.STORAGE);
    const TAPS = buf(48 * q, U.STORAGE | U.COPY_DST, t.taps);
    c.BORDER = buf(W * W, U.STORAGE | U.COPY_DST);
    const GU = (c.GU = buf(4 * t.G_WORDS, U.UNIFORM | U.COPY_DST, c.g));
    c.PG = buf(4 * t.G_WORDS, U.UNIFORM | U.COPY_DST, c.g);
    c.FRAME = [buf(c.sizes.FRAME, U.STORAGE | U.COPY_SRC), buf(c.sizes.FRAME, U.STORAGE | U.COPY_SRC)];
    const group = (k, list) => d.createBindGroup({ layout: this.bgl[k], entries: list.map((r, binding) => ({ binding, resource: r.buffer ? r : { buffer: r } })) });
    c.paintGroup = group("paint", [c.BLOCKS, this.TAB, c.SD, PU]);
    c.PU = PU;
    c.irowsGroup = group("irows", [c.SD, TW, Y, ROWS, SU, this.TAB, INV, PU]);
    // ai: ipic's group: the copy form's a half each (it writes that half's FRAME)
    c.ipicGroups = c.copy ? c.FRAME.map((F) => group("ipicFrame", [Y, TW, F, SU, GU, c.BORDER, TAPS])) : [0, 1].map(() => group("ipic", [Y, TW, PIC, SU]));
    // ai: rsh's group, a half each: the resample reads PIC (the border form binds it unread)
    c.rshGroups = c.FRAME.map((F) => group("rsh", [PIC, TAPS, c.BORDER, F, GU]));
    // ai: A frame's present binds its own words of its half's FRAME, FS words apart (FS a multiple of 64 words: 256 B,
    // ai: the offset alignment WebGPU asks of a storage binding).
    c.presentGroups = [];
    for (const F of c.FRAME) for (let f = 0; f < c.batch; f++) c.presentGroups.push(group("present", [{ buffer: F, offset: 4 * c.FS * f, size: 4 * c.words }, c.PG]));
  }

  // ai: The sender's clock the latest configuration's word states (fps 0 to 240, delay 1 to 30: src/lizard_phy.h
  // ai: lizard_clock): its border painted again where it moved, the queue's next write (its encodes after it).
  setClock(fps, delay) {
    return this.serial(async () => {
      const c = this.cfg;
      if (!c || c.lizard !== codec.lizard || (c.clock[0] === fps && c.clock[1] === delay)) return;
      c.lizard.setClock(fps, delay);
      c.clock = [fps, delay];
      this.paintBorder(c);
    });
  }
  // ai: The C's paint of configuration c's symbol, its grey bytes (every pixel outside the square is read from them): at
  // ai: configure and at a clock moved (setClock). Only while c's Lizard is the wasm's codec.
  paintBorder(c) {
    if (!c.lizard || c.lizard !== codec.lizard) throw new Error(`${c.label}: its codec is no longer the wasm's`);
    const { rgba } = c.lizard.encodeRGBA(new Uint8Array(c.V * c.B)), grey = new Uint8Array(4 * Math.ceil((c.W * c.W) / 4));
    for (let i = 0; i < c.W * c.W; i++) grey[i] = rgba[4 * i];
    this.device.queue.writeBuffer(c.BORDER, 0, grey);
  }

  // ai: count frames (1 to c.sub) into frames f0 .. f0 + count - 1 of half h (0 or 1), one submission: each frame codes x
  // ai: V blocks of 472 bytes, each block its id then its payload (sim/ob.mjs frameBlocks), frame after frame. The
  // ai: frame offset goes in G before the submission, in queue order. parity: the first frame's count (the sender's
  // ai: count of painted pictures mod 4; each frame after it one more), which the pilots carry (SPEC 7.3). test: the
  // ai: blocks are the test stream's, each read for its id alone and its payload made by PAINT (BLAKE3 of the id, SPEC
  // ai: 9.3; Params' dims.w bit 31), so a block's bytes past its id may be anything (lizard-web/send-worker.mjs leaves
  // ai: them unwritten).
  encode(h, f0, blocks, count, c = this.cfg, parity = 0, test = false) {
    const d = this.device, per = c.codes * c.V, R = count * c.codes;
    if (count < 1 || count > c.sub || f0 + count > c.batch || blocks.length !== count * per * c.B) throw new Error(`an encode is 1 to ${c.sub} frames of ${per} blocks of ${c.B} B within a half of ${c.batch}, got ${count} frames from ${f0} in ${blocks.length} B`);
    // ai: under test each block's first ID_BYTES (12) alone, packed (PAINT reads the id's three words there), else the blocks at
    // ai: the plan's block stride (core/tx/send_plan.h BLOCK_BYTES)
    const at = test ? ID_BYTES : c.blockStride, take = test ? ID_BYTES : c.B;
    for (let i = 0; i < count * per; i++) c.staging.set(blocks.subarray(i * c.B, i * c.B + take), i * at);
    d.queue.writeBuffer(c.BLOCKS, 0, c.staging, 0, count * per * at);
    d.queue.writeBuffer(c.GU, 36, new Uint32Array([f0]));
    d.queue.writeBuffer(c.PU, 4 * (c.pDims + 3), new Uint32Array([((parity & 3) | (c.codes << 2) | (test ? 0x80000000 : 0)) >>> 0]));
    const enc = d.createCommandEncoder(), t = this.timer(), K = this.profile ? STAGES.length : 1;
    const stages = [
      [this.paintPipe, c.paintGroup, [c.V, 1, R]],
      [c.irows, c.irowsGroup, [c.Vr, 1, R]],
      [c.copy ? c.ipicFrame : c.ipic, c.ipicGroups[h], [c.n / 4, 1, R]],
      // ai: the copy form's border words, a lane a word; else the resample, a workgroup a strip of words by a band of rows
      [c.copy ? this.rshBorderPipe : this.rshPipe, c.rshGroups[h], [Math.ceil(c.RW / c.rshThreads), c.copy ? c.W : Math.ceil(c.W / c.rshBand), count]],
    ];
    // ai: One pass (timed whole), or with profile a pass a stage.
    let p = null;
    stages.forEach((st, k) => {
      if (!p || this.profile) p = enc.beginComputePass(t ? { timestampWrites: { querySet: t.qs, beginningOfPassWriteIndex: this.profile ? 2 * k : 0, endOfPassWriteIndex: this.profile ? 2 * k + 1 : 1 } } : {});
      if (st) { const [pipe, group, n] = st; p.setPipeline(pipe); p.setBindGroup(0, group); p.dispatchWorkgroups(...n); }
      if (this.profile || k === stages.length - 1) p.end();
    });
    if (t) { enc.resolveQuerySet(t.qs, 0, 2 * K, t.res, 0); enc.copyBufferToBuffer(t.res, 0, t.read, 0, 16 * K); }
    const t0 = performance.now();
    d.queue.submit([enc.finish()]);
    // ai: c's GPU ms a frame, a running mean the page paces by: the timestamps; without timestamp-query, the submission's
    // ai: time to done (an upper bound: it holds whatever the queue held before it).
    const seen = (ms) => { c.frameMs = c.frameMs == null ? ms : 0.8 * c.frameMs + 0.2 * ms; };
    if (t) t.read.mapAsync(GPUMapMode.READ).then(() => {
      const ns = new BigInt64Array(t.read.getMappedRange(0, 16 * K)).slice();
      t.read.unmap();
      const dt = (k) => Number(ns[2 * k + 1] - ns[2 * k]) / count;
      let ns1 = 0;
      if (this.profile) { const v = STAGES.map((_, k) => dt(k)); if (v.every((x) => x >= 0)) { this.stageNs.push(v); ns1 = v.reduce((a, b) => a + b, 0); } }
      else if (dt(0) > 0) ns1 = dt(0);
      if (ns1 > 0) { this.gpuNs.push(ns1); seen(ns1 / 1e6); }
      this.timers.push(t);
    }).catch(() => {});
    else if (!this.ts) d.queue.onSubmittedWorkDone().then(() => seen((performance.now() - t0) / count)).catch(() => {});
  }
  // ai: A free timestamp set, made up to TIMERS; null without timestamp-query or with every set in flight.
  timer() {
    if (!this.ts) return null;
    if (this.timers.length) return this.timers.pop();
    if (this.timersMade >= TIMERS) return null;
    this.timersMade++;
    const d = this.device, U = GPUBufferUsage, K = this.profile ? STAGES.length : 1;
    return { qs: d.createQuerySet({ type: "timestamp", count: 2 * K }), res: d.createBuffer({ size: 16 * K, usage: U.QUERY_RESOLVE | U.COPY_SRC }), read: d.createBuffer({ size: 16 * K, usage: U.MAP_READ | U.COPY_DST }) };
  }
  // ai: The mean GPU ms a frame over the encodes timed by timestamps since the last call (null: none timed).
  takeGpuMs() {
    const a = this.gpuNs.splice(0);
    return a.length ? a.reduce((t, x) => t + x, 0) / a.length / 1e6 : null;
  }
  // ai: With profile: each stage's mean ms a frame since the last call ({ paint, irows, ... }), else null.
  takeStageMs() {
    const a = this.stageNs.splice(0);
    return a.length ? Object.fromEntries(STAGES.map((k, i) => [k, a.reduce((t, v) => t + v[i], 0) / a.length / 1e6])) : null;
  }

  // ai: The canvas the frames are presented on (it must have no 2D context).
  attach(canvas) {
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Unavailable("the canvas gave no WebGPU context");
    ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    return ctx;
  }
  // ai: The code's colours the presents paint (2026-11-16, Send's Colours): its ink and its paper, 0xRRGGBB, the grey
  // ai: along the line between them, the canvas cleared to the paper; black and white the grey. The next present takes them.
  setColours(ink, paper) { this.ink = ink & 0xffffff; this.paper = paper & 0xffffff; }
  // ai: Slot h batch + f (frame f of half h) onto the canvas: whole canvas pixels a frame pixel (the page's layout).
  present(slot, ctx, whole = 1, c = this.cfg) {
    const d = this.device, ink = this.ink ?? 0x000000, paper = this.paper ?? 0xffffff;
    if (c.whole !== whole || c.ink !== ink || c.paper !== paper) {
      c.whole = whole; c.ink = ink; c.paper = paper;
      const u = c.g.slice(); u[8] = whole; u[9] = ink; u[11] = paper; d.queue.writeBuffer(c.PG, 0, u);
    }
    const enc = d.createCommandEncoder(), bg = { r: ((paper >> 16) & 255) / 255, g: ((paper >> 8) & 255) / 255, b: (paper & 255) / 255, a: 1 };
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: bg }] });
    pass.setPipeline(this.presentPipe); pass.setBindGroup(0, c.presentGroups[slot]); pass.draw(3); pass.end();
    d.queue.submit([enc.finish()]);
  }

  // ai: Slot h batch + f's pixels read back: FW x W grey bytes, the symbols side by side, row by row.
  async readback(slot, c = this.cfg) {
    const d = this.device, bytes = 4 * c.words, h = Math.floor(slot / c.batch), f = slot % c.batch;
    const rd = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(c.FRAME[h], 4 * c.FS * f, rd, 0, bytes);
    d.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const all = new Uint8Array(rd.getMappedRange()), out = new Uint8Array(c.FW * c.W);
    for (let y = 0; y < c.W; y++) out.set(all.subarray(4 * c.RW * y, 4 * c.RW * y + c.FW), c.FW * y);
    rd.unmap(); rd.destroy();
    return out;
  }

  // ai: This device against the C on count frames in half 0 (encodes of up to c.sub): each frame encoded here and by the
  // ai: C (its own paint, the same word), compared symbol by symbol. ok when every border and margin is equal to the byte
  // ai: and the squares' RMS difference is under SQUARE_RMS grey levels: the GPU's arithmetic is its own (Y in halves,
  // ai: 2026-11-24), so a square need not be the C's, and the blind read (scripts/gpu/encoder_check.mjs) is what holds
  // ai: it; the RMS catches a device that paints garbage. borderDiff: border pixels that differ; squareMax, squareDiff,
  // ai: squareRms: the squares' largest difference, the pixels that differ at all, the RMS; got: the frames' pixels.
  // ai: c: the configuration to check, which must still be the latest when the check's turn comes (a check queued
  // ai: behind another configure is refused, never run on the other configuration). test (the default, as the native
  // ai: sender's check, core/tx/gpu_painter.cpp): the device encodes from the blocks' ids alone, the test stream made
  // ai: on it, and the C paints each block with its id's stream (the module's wasm, src/xfer.h stream_fill), so a device
  // ai: whose BLAKE3 differs fails here; false: both from the blocks as given, a file's path.
  check(blocks, count = 1, c = this.cfg, test = true) { return this.serial(() => this.checkNow(blocks, count, c, test)); }
  async checkNow(blocks, count, c, test) {
    if (!c || c !== this.cfg || c.lizard !== codec.lizard) throw new Error(`check: ${c?.label ?? "no configuration"} is no longer the latest`);
    const per = c.V * c.B, fb = c.codes * per;
    // ai: frame f painted as the f-th picture (count f mod 4), the C told the same before its encode
    for (let f0 = 0; f0 < count; f0 += c.sub) {
      const k = Math.min(c.sub, count - f0);
      this.encode(0, f0, blocks.subarray(f0 * fb, (f0 + k) * fb), k, c, f0 & 3, test);
    }
    // ai: the C's blocks: under test each payload its id's stream
    const full = test ? Uint8Array.from(blocks) : blocks;
    if (test) for (let o = 0; o + c.B <= full.length; o += c.B) {
      full.set(streamBlock(idGet(full, o), c.B - ID_BYTES), o + ID_BYTES);   // ai: the bare id's stream (its tag aside)
    }
    let borderDiff = 0, squareMax = 0, squareDiff = 0, sq2 = 0, squarePx = 0;
    const got = new Uint8Array(count * c.FW * c.W);
    for (let f = 0; f < count; f++) {
      const px = await this.readback(f, c);
      got.set(px, f * c.FW * c.W);
      c.lizard.setParity(f & 3);
      for (let k = 0; k < c.codes; k++) {
        const { rgba } = c.lizard.encodeRGBA(full.subarray((f * c.codes + k) * per, (f * c.codes + k + 1) * per));
        for (let y = 0; y < c.W; y++) for (let x = 0; x < c.W; x++) {
          const dv = Math.abs(px[y * c.FW + k * (c.W + c.gap) + x] - rgba[4 * (y * c.W + x)]);
          if (x >= c.sq && x < c.sq + c.q && y >= c.sq && y < c.sq + c.q) { squarePx++; sq2 += dv * dv; if (dv) squareDiff++; if (dv > squareMax) squareMax = dv; } else if (dv) borderDiff++;
        }
      }
    }
    c.lizard.setParity(0);
    const squareRms = squarePx ? Math.sqrt(sq2 / squarePx) : 0;
    return { ok: !this.lost && !this.error && borderDiff === 0 && squareRms < SQUARE_RMS, borderDiff, squareMax, squareDiff, squareRms, pixels: count * c.codes * c.W * c.W, got };
  }

  // ai: A configuration's buffers freed (its Lizard stays the wasm's codec while it is the latest: another configure frees it).
  release(c) {
    if (!c || !this.cfgs.delete(c)) return;
    for (const b of c.bufs) b.destroy();
    if (this.cfg === c) this.cfg = null;
  }
  // ai: A configure still running on the chain throws at its next step (dead) and frees its own Lizard.
  destroy() {
    this.dead = true;
    if (this.cfg && this.cfg.lizard === codec.lizard) { codec.lizard?.free(); codec.lizard = null; }
    for (const c of [...this.cfgs]) this.release(c);
    this.device.destroy();
  }
}
