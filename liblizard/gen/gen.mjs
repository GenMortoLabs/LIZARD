// ai: The native decoder's setup, taken from the decoder's setup host: liblizard/gpu/decoder.mjs FrontHalf.create,
// ai: with the options the native receiver runs (main's call below), is built on a recording device (mockgpu.mjs) once
// ai: a variant, and what it made is written out: every shader module's Slang (gpu/slang/) compiled to SPIR-V by slangc
// ai: (`compileSlang`), every bind group layout and pipeline, every static buffer with the bytes the host wrote into it,
// ai: and the values the lanes and batches are made from (`exportTree`). The native host (core/dec/) builds exactly
// ai: these objects and is the one that runs lanes and batches. Nothing here decodes; the tables come from the codec's
// ai: wasm module (sim/ob.mjs over liblizard/build/ob.mjs).
//
// ai: The sender's GPU painter (gen/sender.mjs) is the sender's Slang (gpu/slang/send.mjs, src/send_*.slang), compiled
// ai: into the same out/spv; the web's encoder runs the WGSL slangc writes from the same Slang (webWgsl,
// ai: gpu/send_wgsl.mjs). The engine's own shaders (core/ingest/*.slang, core/tx/*.slang: the camera's ingest, the
// ai: presenter's stretch and expand, the stream files' packing) are Slang too, compiled from their files. Every kernel's
// ai: source is Slang since 2026-11-24 (the decoder's GLSL and the sender's WGSL before it; the decoder's WGSL twins went
// ai: 2026-10-31, archive/webgpu-decoder/).
//
// ai: A variant is (precision, subgroups): precision int8 (the int8 proposer, the classifier f16 with its int8 twin
// ai: built beside), f16 or f32, subgroups whether the classifier reduces by shuffles. The back half is built at every
// ai: batch size a device may need (B_CEIL down to 1: the host builds it at Bmax where memory holds fewer frames), all
// ai: at workgroup memory 32 KB, the floor every phone we know has. The anchors' stages (gpu/slang/anchors.mjs, SPEC
// ai: 6.13) are a setup's `anch` (exportAnch), built with the front half; pass 1, every back half's, reads their
// ai: records.
//
//   node liblizard/gen/gen.mjs [out]      (lizard-android/build.sh gen; out defaults to liblizard/out)
// ai: Writes out/setup/<variant>.json, out/blobs/<hash>.bin (the buffers' bytes), out/slang/<hash>.slang (a decoder
// ai: module's text), out/slang/send_<name>.slang (a sender's) and out/spv/<hash>.spv, send_<name>.spv and
// ai: ingest_<name>.spv, and out/gen.json (the variants, the modules, the gpu sources' hash that made them; the WebGPU
// ai: receiver's WGSL and its packed tree went 2026-11-22, archive/webgpu-receiver-2026-11-22/).
// ai: Needs, each checked before anything runs: ../build/ob.mjs and the module it names,
// ai: ../bindings/wasm/dist/lizard.wasm (liblizard's ./build.sh); slang 2026.22's slangc (lizard-android/build.sh tools;
// ai: liblizard/.tools/slang, else the PATH's). slangc validates the SPIR-V it writes.
import { installGlobals, Recorder, mockAdapter } from "./mockgpu.mjs";
import { senderSetup } from "./sender.mjs";
import { mkdirSync, mkdtempSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync, realpathSync, rmSync } from "node:fs";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

installGlobals();
const HERE = new URL("./", import.meta.url), LIB = new URL("../", import.meta.url);
const OUT = process.argv[2] ? pathToFileURL(`${resolve(process.argv[2])}/`) : new URL("../out/", HERE);
const imp = (p) => import(new URL(p, LIB).href);
// ai: Run as a script (not when a check imports stripCaps), by real path: through a symlink or a space
// ai: import.meta.url and argv[1] spell the one file two ways.
const MAIN = (() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })();

// ai: The tree's version, every setup's treeVersion, which the native host (core/dec/front.cpp TREE_VERSION) refuses a
// ai: tree of another of: raised in both places whenever what the host reads of a tree changes.
// ai: 23: the map past F7 the C's one model (REFIT's homography, ANCH_REFIT's lens over the border and the anchors on
// ai: every frame, ANCH_FIT's mesh): no REFIT_POST (the tree's front.refitPost), the record RESID the lens alone (5 vec4).
// ai: 22: the word's four counts, every layout (no CRC): the configuration two words, record word 29 the version << 24 |
// ai: the 6/7 count << 16 | the 3/4 << 8 | the 2/3, word 30 the clock | the 1/2 count << 16; LISTS a seventh word a frame.
// ai: 21: the soft pass's four steps a size (tnoise, each sub-channel's 1 / N into EP's fourth float, before talign,
// ai: which weights the pilots by it and binds EP at 6).
// ai: 20: versions 1 to 3 in the word's list (VERSION_MIN 1, the configurations' sums from version 1).
// ai: 19: the word's index and the clock (RS(n, 6), F8 solving five unknown bytes; record word 30 the clock, word 6 the
// ai: word's last 32 bits), the pilots apart from the blocks (PILOT a frame's four readings, then its shift; EP a vec4 a
// ai: sub-channel; L in data-slot order, dims.lStride 159 a sub-channel), the block's CRC-64/LIZARD (a 472-byte block,
// ai: its CRC in the two words after it; REC 120 words a record).
export const TREE_VERSION = 23;

const EXE = process.platform === "win32" ? ".exe" : "";
// ai: slangc for every module (gpu/slang/, the kernels' source since 2026-11-24, and the engine's own shaders):
// ai: liblizard/.tools/slang (lizard-android/build.sh tools) where it is, else the PATH's. Pinned (SLANG, build.sh's too):
// ai: a module is named by its source, so another slangc would write other SPIR-V under the same names; a run refuses any
// ai: but SLANG. One compiler, two targets: SPIR-V 1.3 for the native hosts, WGSL for the web's encoder. The SPIR-V
// ai: reaches the driver as slangc writes it (no spirv-opt: it unrolled the fused pass 1 and F8's word to three times
// ai: their size, 2026-09-29).
export const SLANG = "2026.22";
const slangTool = () => {
  const lab = fileURLToPath(new URL(`../.tools/slang/bin/slangc${EXE}`, HERE));
  if (existsSync(lab)) return lab;
  for (const dir of (process.env.PATH ?? "").split(delimiter)) if (dir && existsSync(join(dir, "slangc" + EXE))) return join(dir, "slangc" + EXE);
  return lab;
};
export const TOOLS = { slangc: slangTool() };

// ai: What a run needs, checked before the gpu modules load (sim/ob.mjs imports ../build/ob.mjs, which names the
// ai: codec's module): every one missing, each with where it comes from.
const WASM = new URL("bindings/wasm/dist/lizard.wasm", LIB);
if (MAIN) {
  const missing = [
    [fileURLToPath(new URL("build/ob.mjs", LIB)), "liblizard/build.sh"], [fileURLToPath(WASM), "liblizard/build.sh"],
    [TOOLS.slangc, `slang ${SLANG}: lizard-android/build.sh tools, or slangc on the PATH`],
  ].filter(([p]) => !existsSync(p));
  if (missing.length) { console.error(missing.map(([p, from]) => `no ${p} (${from})`).join("\n")); process.exit(1); }
  // ai: slangc prints its version to stderr
  const sv = (() => { try { const r = spawnSync(TOOLS.slangc, ["-version"], { encoding: "utf8" }); return `${r.stdout ?? ""}${r.stderr ?? ""}`.trim().split("\n")[0] || null; } catch { return null; } })();
  if (sv !== SLANG) { console.error(`${TOOLS.slangc} is slang ${sv ?? "of no version it states"}; the tree's modules are slang ${SLANG}'s (lizard-android/build.sh tools)`); process.exit(1); }
}

const { FrontHalf, CAP_MS, B_CEIL, LEVELS, COUNTERS, TAU, INSTALLED, receiverOptions, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, RING_SLACK, TWIN_FRAMES, TWIN_ROUNDS } =
  await imp("gpu/decoder.mjs");
const { formatTables, PICTURES } = await imp("gpu/tables.mjs");
const { loadNet } = await imp("gpu/cnn/netfile.mjs");
const { K, CELL, HYP, RING_COUNT, VOTE_READS, PEAK_LIST, R2_N } = await imp("gpu/plan/finder.mjs");
const { NODES_MAX, RESID } = await imp("gpu/plan/register.mjs");
const { ANCH_K, ANCH_REC, ANCH_GS } = await imp("gpu/plan/anchors.mjs");
const { PYRAMID_BLOCK } = await imp("gpu/plan/pyramid.mjs");
const { TILE } = await imp("gpu/plan/bank_fcn2.mjs");
const { SLOTS } = await imp("gpu/plan/common.mjs");
const { ARGS_WORDS, ARGS_SLOT, argsWords } = await imp("gpu/plan/back_transform.mjs");
const { VERSION_MIN, VERSION_MAX } = await imp("gpu/wordcode.mjs");
const { COUNTS } = await imp("gpu/back/back.mjs");
const { REC_BYTES } = await imp("gpu/back/ldpc.mjs");
const { PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");

// ai: The classifier the native receiver runs (gpu/decoder.mjs INSTALLED): cnn-v3 on every peak, a path under
// ai: liblizard/ (as the app's build needs it).
const NET = INSTALLED.classifier;
// ai: The device the setup is made for: 32 KB of workgroup memory (every phone we know, and what Chrome gives the
// ai: desktop's GPUs), the rest generous; plan()'s memory limits are the real device's, read at run time.
const LIMITS = { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31, maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 16, maxTextureArrayLayers: 2048, maxComputeInvocationsPerWorkgroup: 256 };
const FEATURES = ["timestamp-query", "shader-f16", "subgroups"];
export const VARIANTS = ["int8", "f16", "f32"].flatMap((precision) => [true, false].map((subgroups) => ({ precision, subgroups, name: `${precision}${subgroups ? "-sg" : ""}` })));
const BACK_B = [B_CEIL, 16, 8, 4, 2, 1];

// ai: A buffer, texture view or pipeline of the recorder as its id; everything else as it is.
const idOf = (o) => (o == null ? null : o.__id ?? null);
const pb = (x) => (x ? { p: idOf(x.p), bgl: idOf(x.bgl) } : null);
// ai: A classifier's pipe: P, the patches a workgroup of its kernel takes (its dispatch is ceil(slots / P)).
const pbNet = (x) => (x ? { ...pb(x), P: x.P } : null);
const perSlot = (arr) => Array.from({ length: SLOTS }, (_, s) => idOf(arr?.[s]));
const plain = (v) => JSON.parse(JSON.stringify(v, (k, x) => (ArrayBuffer.isView(x) ? Array.from(x) : x)));

// ai: What the lanes and batches read of the back half (gpu/back/back.mjs, transform.mjs, soft.mjs, ldpc.mjs).
function exportBack(bh) {
  const t = bh.transform, so = bh.soft, ld = bh.ldpc;
  return {
    B: bh.B, precision: bh.precision, dims: plain(bh.dims), bytes: bh.bytes, readBytes: bh.readBytes,
    transform: {
      built: t.built, cbytes: t.cbytes, yStride: t.yStride, sStride: t.sStride, partStride: t.partStride,
      gate: idOf(t.gate), gateBgl: idOf(t.gateBgl), p1fBgl: idOf(t.p1fBgl), redBgl: idOf(t.redBgl), p2Bgl: idOf(t.p2Bgl),
      pass1f: perSlot(t.pass1f), reduce: perSlot(t.reduce), pass2: perSlot(t.pass2),
      twBufs: perSlot(t.twBufs), x12Bufs: perSlot(t.x12Bufs), rowsBufs: perSlot(t.rowsBufs), picUnis: perSlot(t.picUnis), gateUni: idOf(t.gateUni),
      sizes: t.tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks, m2: tb.m2, sx2: tb.sx2, sq2: tb.sq2 } : null)),
    },
    // ai: the soft and LDPC stages' tables and lane buffers (back/soft.mjs, back/ldpc.mjs), which the rate profile's
    // ai: stage below decodes through
    soft: { uvBuf: idOf(so.uvBuf), params: perSlot(so.params), served: so.served, blocks: so.sizes.map((z) => (z ? z.blocks : 0)), bytes: so.bytes },
    ldpc: { bytes: ld.bytes, blocksMax: ld.blocksMax, recCap: ld.recCap },
    // ai: the format's rate profile's stage (gpu/back/tiers.mjs exportFor), in the soft and LDPC stages' place
    rule: bh.rule.exportFor(idOf),
  };
}

// ai: What the native host's lanes and batches read of the front half (core/dec/front.cpp), with the host's
// ai: constants, so the native code restates none of them.
function exportTree(fh, v) {
  const tf = fh.twinForms;
  const twin = (t) => (t ? { pipe: pbNet(t.pipe), buf: idOf(t.buf), precision: t.w.precision } : null);
  const f = fh.fcn2Form;
  return {
    treeVersion: TREE_VERSION, variant: v.name, precision: fh.precision, floatPrecision: fh.floatPrecision, subgroups: v.subgroups, features: fh.features, nets: fh.nets,
    consts: { LEVELS, COUNTERS, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, RING_SLACK, K, CELL, HYP, RING_COUNT, VOTE_READS, PEAK_LIST, R2_N, NODES_MAX, RESID, PYRAMID_BLOCK, TAU,
      TILE, B_CEIL, CAP_MS, TWIN_FRAMES, TWIN_ROUNDS, VERSION_MIN, VERSION_MAX, SLOTS, ARGS_WORDS, ARGS_SLOT, argsWords, COUNTS, REC_BYTES, PAYLOAD, ANCH_K, ANCH_REC, ANCH_GS },
    cap: fh.cap, nodesMax: fh.nodesMax, modules: Array.from(fh.modules), bankKeep: fh.bank.KEEP,
    weightsPrecision: fh.weights.precision,
    front: {
      pyramid: pb(fh.pyramid), classify: pbNet(fh.classify),
      vote: pb(fh.vote), peaks: pb(fh.peaks), gather: pb(fh.gather), complete: pb(fh.complete), score: pb(fh.score), pick: pb(fh.pick), trace: pb(fh.trace), nodesP: pb(fh.nodesP), refit: pb(fh.refit), wordP: pb(fh.wordP), select: pb(fh.select),
      // ai: the finder's second round (gpu/slang/finder.mjs REDO; gpu/bank_fcn2.mjs COARSE_FROM)
      redo: pb(fh.redo), coarseFrom: fh.bank.COARSE_FROM,
      fcn2: pb(fh.fcn2), fcn2X: pb(fh.fcn2X), fcn2X0: pb(fh.fcn2X0), fcn2Form: { TX: f.TX, TY: f.TY, XB: f.XB, walk: f.walk, xplane: f.xplane, int8: f.int8, store: f.store, name: f.name },
      buffers: Object.fromEntries(["planBuf", "planUni", "p4Uni", "pickUni", "nodesBuf", "tmplBuf", "wordPtsBuf", "wordUni", "wordLatUni", "fitUni", "weightsBuf"].map((k) => [k, idOf(fh[k])])),
      round: idOf(fh.round),
      twins: tf ? { large: twin(tf.large) } : null,
    },
    pictures: PICTURES.map(({ n, subch, lo, hi }) => ({ n, subch, lo, hi })),
    backs: {},
  };
}

// ai: The anchors' stages (gpu/slang/anchors.mjs), which the native host builds with the front half (core/dec/setup.cpp
// ai: build): ANCH_FIND and ANCH_FIT ({ p, bgl }), ANCH_SUMS (sums: on ANCH_FIND's layout and group, run between
// ai: ANCH_REFIT and ANCH_FIT) and their uniform (anchUni).
function exportAnch(fh) {
  return {
    // ai: refit: ANCH_REFIT ({ p, bgl }), which runs between ANCH_FIND and ANCH_FIT; predict: ANCH_PREDICT on its layout,
    // ai: and nodes: the border's second search (F7's NODES centred) on NODES' layout with its own round
    // ai: (buffers.round), both run before it
    find: pb(fh.anchFind), predict: pb(fh.anchPredict), nodes: pb(fh.nodesC), refit: pb(fh.anchRefit), sums: pb(fh.anchSums), fit: pb(fh.anchFit),
    buffers: { anchUni: idOf(fh.anchUni), round: idOf(fh.anchRound) },
  };
}

// ai: The recorder's objects a tree names, and every object they name in turn (a pipeline's layouts), as the native
// ai: loader takes them: buffers with the hash of the bytes the host wrote (out/blobs/<hash>.bin, shared by every
// ai: buffer and variant holding the same bytes: the back half's tables are the same at every B), layouts, pipelines
// ai: by module hash.
// ai: The web encoder's WGSL (gpu/send_wgsl.mjs): each sender kernel the web runs, and the present's vertex and fragment
// ai: stages, compiled by slangc from the same Slang as the native SPIR-V (gpu/slang/src/send_*.slang): one source, the
// ai: WGSL never written by hand. The web's build has no slangc, so the module is kept in the tree, written again only
// ai: where its text changed. (Until 2026-11-24 the GLSL through glslang, spirv-opt, naga and two rewrites for Tint's
// ai: uniformity analysis: slangc hands the entry's builtins to the body as parameters, which the analysis follows.)
// ai: slangc's optimisation level, every Slang module's (2026-11-24): its default -O1 inlined every function of COMPLETE
// ai: into one with each copy's loops (glslang's 43 loops 85), and the S26 ran the finder's vote stages +24% slower;
// ai: -O2 (smaller again) and -O0 (glslang's shape) both level with the GLSL there; -O2 level on the iGPU and in the
// ai: sender on both (research/build/slang-2026-11-24/)
const SLANG_OPT = ["-O2"];
function webWgsl(name, { code, stage }) {
  const dir = mkdtempSync(join(tmpdir(), "liz-wgsl-"));
  const [src, w] = [`${name}.slang`, `${name}.wgsl`].map((f) => join(dir, f));
  try {
    writeFileSync(src, code);
    run(TOOLS.slangc, [src, "-target", "wgsl", "-entry", "main", "-stage", stage, "-warnings-as-errors", "all", ...SLANG_OPT, "-o", w]);
    return readFileSync(w, "utf8");
  } finally { rmSync(dir, { recursive: true, force: true }); }
}

// ai: The device features a module needs beyond its variant's, by the capability its SPIR-V declares for each: "int16"
// ai: (Int16: shaderInt16; tldpc's i16 form alone). The host leaves out a pipeline whose device lacks one
// ai: (core/dec/setup.cpp).
const NEEDS = [["int16", 22]];   // ai: the capability's number, Int16
// ai: A stage's lanes and needs from its SPIR-V (compiled here once, the compile loop reusing it): its entry's LocalSize
// ai: (OpExecutionMode, opcode 16, mode 17), the workgroup's invocations, which the host holds to its device's
// ai: maxComputeWorkGroupInvocations (core/dec/setup.cpp), and the capabilities NEEDS names.
const early = new Map();
function slangFacts(h, code) {
  if (!early.has(h)) early.set(h, compileSlang(h, code));
  const path = fileURLToPath(new URL(`spv/${h}.spv`, OUT)), b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  let lanes = null;
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) if ((w[i] & 0xffff) === 16 && w[i + 2] === 17) lanes = w[i + 3] * w[i + 4] * w[i + 5];
  if (!lanes) throw new Error(`${h}: no LocalSize in its SPIR-V`);
  const caps = capsOf(path), needs = NEEDS.filter(([, cap]) => caps.has(cap)).map(([k]) => k);
  return { lanes, ...(needs.length ? { needs } : {}) };
}
function exportObjects(rec, tree, blobs) {
  const want = new Set();
  JSON.stringify(tree, (k, x) => { if (typeof x === "string" && /^[bLp]\d+$/.test(x)) want.add(x); return x; });
  for (const id of [...want]) { const o = rec.objects.get(id); if (o?.__kind === "p") for (const g of o.layout.groups) want.add(g.__id); }
  const buffers = {}, bgls = {}, pipelines = {};
  for (const id of [...want].sort((a, b) => a.localeCompare(b, "en", { numeric: true }))) {
    const o = rec.objects.get(id);
    if (!o) throw new Error(`the tree names ${id}, which the recorder never made`);
    if (o.__kind === "b") {
      const blob = o.data ? createHash("sha256").update(o.data).digest("hex").slice(0, 16) : null;
      if (blob && !blobs.has(blob)) blobs.set(blob, o.data);
      buffers[id] = { size: o.size, usage: o.usage, blob };
    } else if (o.__kind === "L") bgls[id] = o.entries;
    else if (o.__kind === "p") pipelines[id] = { module: o.module.hash, entry: o.entry, groups: o.layout.groups.map((g) => g.__id), label: o.label,
      ...slangFacts(o.module.hash, o.module.code) };
  }
  return { buffers, bgls, pipelines };
}

// ai: SPIR-V without the capabilities a compute module declares and never uses (naga's, then slangc's):
// ai: StorageInputOutput16 (4436) asks the device for 16-bit shader inputs and outputs, which a compute shader has none
// ai: of (its only inputs are u32 builtins) and some drivers lack. UniformAndStorageBuffer16BitAccess (4434): slangc
// ai: declares it with every module holding a 16-bit type in a buffer, the S26 Ultra's Adreno 840 lacks the feature (its
// ai: vkjson, 2026-09-29), and a compute module needs it only for a 16-bit type in a uniform block. Validation does not
// ai: check either (a module with f16 in a uniform validates without it: a negative control run 2026-09-29), so the types
// ai: are checked here: a capability is dropped only where no variable of its storage class (Uniform for 4434, Input or
// ai: Output for 4436) reaches a 16-bit type, and a module where one does throws.
const DROP_CAPS = new Map([[4436, [1, 3]], [4434, [2]]]);
export function stripCaps(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4), out = Array.from(w.subarray(0, 5));
  const types = new Map(), vars = [], bufferBlock = new Set();
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (!n) throw new Error(`${path}: a zero-length instruction at word ${i}`);
    if (op === 71 && w[i + 2] === 3) bufferBlock.add(w[i + 1]);                                                  // OpDecorate BufferBlock
    if (op === 21 || op === 22) types.set(w[i + 1], { small: w[i + 2] === 16, of: [] });                        // OpTypeInt, OpTypeFloat
    else if (op === 23 || op === 24 || op === 28 || op === 29) types.set(w[i + 1], { small: false, of: [w[i + 2]] });   // vector, matrix, array, runtime array
    else if (op === 30) types.set(w[i + 1], { small: false, of: Array.from(w.subarray(i + 2, i + n)) });          // OpTypeStruct
    else if (op === 32) types.set(w[i + 1], { small: false, of: [w[i + 3]], sc: w[i + 2] });                     // OpTypePointer
    else if (op === 59) vars.push({ type: w[i + 1], sc: w[i + 3] });                                             // OpVariable
    i += n;
  }
  const small = (id, seen = new Set()) => { if (seen.has(id)) return false; seen.add(id); const t = types.get(id); return !!t && (t.small || t.of.some((x) => small(x, seen))); };
  // ai: a Uniform variable of a BufferBlock struct is a storage buffer in SPIR-V 1.3's older form (slangc's at
  // ai: spirv_1_3), its 16-bit types StorageBuffer16BitAccess's (the capability's old name StorageUniformBufferBlock16)
  const storageLike = (v) => v.sc === 12 || (v.sc === 2 && bufferBlock.has(types.get(v.type)?.of[0]));
  const needed = new Set();
  for (const [cap, classes] of DROP_CAPS) if (vars.some((v) => classes.includes(v.sc) && !(cap === 4434 && storageLike(v)) && small(v.type))) needed.add(cap);
  if (needed.size) throw new Error(`${path}: a 16-bit type in a ${[...needed].map((c) => (c === 4434 ? "uniform block" : "shader input or output")).join(" and ")}: capability ${[...needed].join(", ")} needed, which the S26's Adreno lacks`);
  // ai: a storage buffer that reaches a 16-bit type needs StorageBuffer16BitAccess (4433), which UniformAndStorageBuffer16
  // ai: BitAccess implies: slangc declares the latter alone (2026-11-24), so where it is dropped the former is declared
  const caps = new Set();
  for (let i = 5; i < w.length; i += w[i] >>> 16) if ((w[i] & 0xffff) === 17) caps.add(w[i + 1]);
  const storage16 = caps.has(4434) && !caps.has(4433) && vars.some((v) => storageLike(v) && small(v.type));
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (op === 17 && w[i + 1] === 4434 && storage16) out.push((2 << 16) | 17, 4433);
    if (!(op === 17 && DROP_CAPS.has(w[i + 1]))) for (let k = 0; k < n; k++) out.push(w[i + k]);
    i += n;
  }
  writeFileSync(path, new Uint8Array(new Uint32Array(out).buffer));
}

function run(cmd, args, input) {
  try { return execFileSync(cmd, args, { encoding: "utf8", ...(input === undefined ? { stdio: ["ignore", "pipe", "pipe"] } : { input, stdio: ["pipe", "pipe", "pipe"] }) }); }
  catch (e) { const err = new Error(`${cmd.split("/").pop()} ${args.join(" ")}: ${(e.stderr || e.stdout || e.message).toString().trim().split("\n").slice(0, 12).join("\n")}`); err.tool = true; throw err; }
}

// ai: A module (gpu/slang/, out/slang/<h>.slang its text) to SPIR-V 1.3 (the host's Vulkan 1.1 floor), a compute entry
// ai: `main`, every warning an error. slangc validates its SPIR-V with its own SPIRV-Tools (the NDK's spirv-val predates
// ai: Slang's source-language operand); a workgroup variable has no initializer (Slang zeroes none, so MoltenVK's
// ai: SPIRV-Cross writes no zeroing loop over gl_WorkGroupSize).
function compileSlang(h, code) {
  const src = fileURLToPath(new URL(`slang/${h}.slang`, OUT));
  writeFileSync(src, code);
  return slangSpv(src, fileURLToPath(new URL(`spv/${h}.spv`, OUT)), h);
}
// ai: a Slang file to SPIR-V at spv, its capabilities held to the Adreno's (stripCaps)
function slangSpv(src, spv, name) {
  // ai: warning 41012 (the profile's capability atoms widened for what the module uses) is not an error: the SPIR-V's own
  // ai: capabilities are what a device is held to (NEEDS, stripCaps)
  run(TOOLS.slangc, [src, "-target", "spirv", "-profile", "spirv_1_3", "-entry", "main", "-stage", "compute", "-warnings-as-errors", "all", "-warnings-disable", "41012", ...SLANG_OPT, "-o", spv]);
  stripCaps(spv);
  const b = readFileSync(spv), version = b.readUInt32LE(4);
  if (version !== 0x10300) throw new Error(`${name}: slangc wrote SPIR-V ${version >> 16}.${(version >> 8) & 255}, not 1.3`);
  // ai: a Workgroup variable (OpVariable, opcode 59, storage class 4) with an initializer would need the device's
  // ai: shaderZeroInitializeWorkgroupMemory, which the host does not ask for (core/dec/setup.cpp variantsFor)
  const w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  for (let i = 5; i < w.length; i += w[i] >>> 16) if ((w[i] & 0xffff) === 59 && w[i + 3] === 4 && w[i] >>> 16 > 4) throw new Error(`${name}: a workgroup variable with an initializer (zero it in the kernel, then a barrier)`);
  return { bytes: b.byteLength };
}

// ai: A SPIR-V module's capabilities (OpCapability, opcode 17).
function capsOf(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4), caps = new Set();
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) if ((w[i] & 0xffff) === 17) caps.add(w[i + 1]);
  return caps;
}

// ai: The gpu sources whose output this is: a change to any of them makes the setup stale (gen.json's `sources`
// ai: names what it was made from; nothing checks it at run time, so a change to gpu/ wants gen run again).
function sourcesHash() {
  const h = createHash("sha256");
  const walk = (dir) => { for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isDirectory()) walk(p); else if (/\.(mjs|safetensors|slang)$/.test(f)) h.update(f).update(readFileSync(p)); } };
  // ai: send_wgsl.mjs, gen's own output, left out
  for (const d of ["gpu/plan", "gpu/slang", "gpu/back", "gpu/cnn", "gpu"]) { const dir = fileURLToPath(new URL(d, LIB)); for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isFile() && /\.(mjs|safetensors)$/.test(f) && f !== "send_wgsl.mjs") h.update(`${d}/${f}`).update(readFileSync(p)); } }
  walk(fileURLToPath(new URL("gpu/cnn/proposer", LIB))); walk(fileURLToPath(new URL("gpu/back/stop", LIB))); walk(fileURLToPath(new URL("gpu/slang/src", LIB)));
  h.update(readFileSync(WASM));
  return h.digest("hex").slice(0, 16);
}

async function main() {
  for (const d of ["setup", "slang", "spv", "blobs"]) mkdirSync(new URL(`${d}/`, OUT), { recursive: true });
  const tables = await formatTables();
  const net = await loadNet(new URL(NET, LIB));
  const modules = new Map(), summary = [], blobs = new Map(), used = {};
  for (const v of VARIANTS) {
    const rec = new Recorder(), log = [];
    const adapter = mockAdapter(rec, { features: FEATURES, limits: LIMITS });
    const fh = await FrontHalf.create(adapter, tables, {
      ...receiverOptions(net), precision: v.precision, subgroups: v.subgroups, backN: 1536, log: (m) => log.push(m), twins: null,
      // ai: LIZ_GEN_CAP=<n> (lab, 2026-11-24): SELECT's cap in place of decoder.mjs CAP, for a tree written elsewhere (argv[2])
      ...(process.env.LIZ_GEN_CAP ? { cap: +process.env.LIZ_GEN_CAP } : {}),
    });
    const tree = exportTree(fh, v);
    tree.backs[fh.bh.B] = exportBack(fh.bh);
    for (const B of BACK_B.filter((b) => b !== fh.bh.B)) { await fh.buildBack(B, true); tree.backs[B] = exportBack(fh.bh); }
    tree.anch = exportAnch(fh);
    tree.log = log;
    const objects = exportObjects(rec, tree, blobs);
    tree.objects = objects;
    for (const p of Object.values(objects.pipelines)) if (!modules.has(p.module)) modules.set(p.module, rec.modules.get(p.module));
    used[v.name] = new Set(Object.values(objects.pipelines).map((p) => p.module));
    writeFileSync(new URL(`setup/${v.name}.json`, OUT), JSON.stringify(tree));
    summary.push({ variant: v.name, pipelines: Object.keys(objects.pipelines).length, buffers: Object.keys(objects.buffers).length, nets: fh.nets });
    console.log(`${v.name}: ${Object.keys(objects.pipelines).length} pipelines, ${Object.keys(objects.buffers).length} buffers, nets ${JSON.stringify(fh.nets)}`);
  }
  let blobBytes = 0;
  for (const [h, data] of blobs) { writeFileSync(new URL(`blobs/${h}.bin`, OUT), data); blobBytes += data.byteLength; }
  console.log(`${blobs.size} blobs, ${(blobBytes / 1024).toFixed(0)} KB`);
  const compiled = {}, failed = [];
  for (const [h, code] of modules) {
    try { compiled[h] = early.get(h) ?? compileSlang(h, code.code); } catch (e) { if (!e.tool) throw e; failed.push({ h, message: e.message }); }
  }
  // ai: The sender's GPU encoder (gen/sender.mjs, 2026-10-02): its kernels as out/spv/send_<name>.spv, its manifest
  // ai: setup/send.json and its paint's TAB blobs/<hash>.bin (the codes and the whitening), for core/tx/gpu_painter.cpp.
  const send = await senderSetup();
  writeFileSync(new URL(`blobs/${send.tab.hash}.bin`, OUT), send.tab.bytes);
  for (const [k, code] of Object.entries(send.modules)) {
    try { compiled[`send_${k}`] = compileSlang(`send_${k}`, code); } catch (e) { if (!e.tool) throw e; failed.push({ h: `send_${k}`, message: e.message }); }
  }
  writeFileSync(new URL("setup/send.json", OUT), JSON.stringify(send.manifest));
  // ai: the web encoder's WGSL from the same Slang (webWgsl), into the tree beside the encoder
  const wgsl = {};
  for (const [k, src] of Object.entries(send.web)) {
    try { wgsl[k] = webWgsl(`send_${k}`, src); } catch (e) { if (!e.tool) throw e; failed.push({ h: `web send_${k}`, message: e.message }); }
  }
  if (Object.keys(wgsl).length === Object.keys(send.web).length) {
    const text = `// The web encoder's WGSL (encoder.mjs): compiled by ../gen/gen.mjs from the sender's Slang (slang/src/send_*.slang,\n// slangc ${SLANG}). Generated: never edited by hand.\nexport const SEND_WGSL = ${JSON.stringify(wgsl, null, 1)};\n`;
    const at = new URL("../gpu/send_wgsl.mjs", HERE);
    if (!existsSync(at) || readFileSync(at, "utf8") !== text) writeFileSync(at, text);
    console.log(`web: ${Object.keys(wgsl).length} WGSL modules, ${(Buffer.byteLength(text) / 1024).toFixed(0)} KB (gpu/send_wgsl.mjs)`);
  }
  console.log(`sender: ${Object.keys(send.modules).length} kernels, TAB ${(send.tab.bytes.byteLength / 1024).toFixed(0)} KB`);
  for (const f of failed) console.log(`FAIL ${f.h}\n${f.message}\n`);
  // ai: The engine's own shaders, Slang from their files to out/spv/<prefix><name>.spv: core/ingest/ (ingest_: the camera's
  // ai: frame through the YCbCr sampler the host builds for its format, and the raw develop) and core/tx/ (send_: the
  // ai: presenter's expand and Lanczos stretch into RGBA8 storage images, the stream files' packing). Native alone: no WGSL.
  const engine = [["../core/ingest/", "ingest_"], ["../core/tx/", "send_"]].flatMap(([d, pre]) => {
    const dir = new URL(d, HERE);
    return readdirSync(dir).filter((f) => f.endsWith(".slang")).map((f) => ({ src: fileURLToPath(new URL(f, dir)), name: `${pre}${f.replace(/\.slang$/, "")}` }));
  });
  for (const { src, name } of engine) {
    try { const c = slangSpv(src, fileURLToPath(new URL(`spv/${name}.spv`, OUT)), name); console.log(`engine ${name}: ${c.bytes} B of SPIR-V`); }
    catch (e) { if (!e.tool) throw e; failed.push({ h: name, message: e.message }); }
  }
  const spvBytes = Object.values(compiled).reduce((a, c) => a + c.bytes, 0);
  console.log(`${modules.size} modules: ${Object.keys(compiled).length} compiled (${(spvBytes / 1024).toFixed(0)} KB of SPIR-V), ${failed.length} failed`);
  // ai: What an earlier run wrote and this one did not (a module or a blob no setup names now) let go, so a tree
  // ai: copied whole (the desktop's bundle, an install) carries no stale file. Only gen's own names: a file put in
  // ai: spv/ by hand (bench_dot.spv) stays.
  const kept = new Set([...VARIANTS.map((v) => `setup/${v.name}.json`), "setup/send.json", ...[...blobs.keys(), send.tab.hash].map((h) => `blobs/${h}.bin`)]);
  for (const h of modules.keys()) for (const f of [`slang/${h}.slang`, `spv/${h}.spv`]) kept.add(f);
  for (const k of Object.keys(send.modules)) for (const f of [`slang/send_${k}.slang`, `spv/send_${k}.spv`]) kept.add(f);
  for (const { name } of engine) kept.add(`spv/${name}.spv`);
  const ours = { setup: /\.json$/, blobs: /^[0-9a-f]{16}\.bin$/, wgsl: /\.wgsl$/, naga: /\.wgsl$/, glsl: /\.comp$/, slang: /\.slang$/, spv: /^([0-9a-f]{16}|send_\w+|ingest_\w+)(\.raw)?\.spv$/ };
  let gone = 0;
  // ai: wgsl/, naga/ and glsl/ an earlier run's alone (the sender's WGSL and every GLSL module went 2026-11-24, with
  // ai: spv/'s .raw.spv copies), each let go once empty
  for (const [d, re] of Object.entries(ours)) for (const f of existsSync(new URL(`${d}/`, OUT)) ? readdirSync(new URL(`${d}/`, OUT)) : []) if (re.test(f) && !kept.has(`${d}/${f}`)) { rmSync(new URL(`${d}/${f}`, OUT)); gone++; }
  for (const d of ["wgsl", "naga", "glsl"]) if (existsSync(new URL(`${d}/`, OUT)) && !readdirSync(new URL(`${d}/`, OUT)).length) rmSync(new URL(`${d}/`, OUT), { recursive: true });
  if (gone) console.log(`${gone} files of an earlier run let go`);
  console.log(`${Object.keys(compiled).length + engine.length} SPIR-V modules by slangc ${SLANG}: ${Object.keys(compiled).length} the decoder's and the sender's, ${engine.length} the engine's`);
  writeFileSync(new URL("gen.json", OUT), JSON.stringify({ sources: sourcesHash(), writer: `slangc ${SLANG} (SPIR-V 1.3, ${SLANG_OPT.join(" ")}) for every module, and the web encoder's WGSL (gpu/send_wgsl.mjs) from the same Slang`, variants: summary, modules: Object.keys(compiled), engine: engine.map((e) => e.name), failed: failed.map((f) => f.h) }, null, 1));
  if (failed.length) process.exit(1);
}

if (MAIN) await main();
