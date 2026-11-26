// ai: The GPU tree the native apps carry: every decoder module the host's setup names (core/dec/build.cpp, which makes
// ai: the setup itself: every buffer, layout and pipeline from the C codec and the nets' files) compiled by slangc from
// ai: its recipe (a kernel's file under gpu/slang/src and its constants), the recipes listed by the host's own code with
// ai: no device (lizard_gpu_tree recipes, core/tools/gpu_tree.cpp), a variant's index written for the host to name its
// ai: modules through. Nothing here decodes or packs a table.
//
// ai: The sender's GPU painter's modules are the sender's Slang (src/send_*.slang), their recipes listed by the host too
// ai: (lizard_gpu_tree send, core/tx/send_plan.h), compiled into the same out/spv; the web's encoder runs the WGSL
// ai: slangc writes from the same recipes (webWgsl, gpu/send_wgsl.mjs). The engine's own shaders (core/ingest/*.slang, core/tx/*.slang: the camera's ingest, the
// ai: presenter's stretch and expand, the stream files' packing) are Slang too, compiled from their files. Every kernel's
// ai: source is Slang since 2026-11-24 (the decoder's GLSL and the sender's WGSL before it; the decoder's WGSL twins went
// ai: 2026-10-31, archive/webgpu-decoder/).
//
// ai: A variant is (precision, subgroups): precision int8 (the int8 proposer, the classifier f16 with its int8 twin
// ai: built beside), f16 or f32, subgroups whether the classifier reduces by shuffles; each holds the back half at
// ai: every batch size a device may need (32 down to 1), all at workgroup memory 32 KB, the floor every phone we know
// ai: has (core/dec/build.h VARIANTS, BACK_SIZES).
//
//   node liblizard/gen/gen.mjs [out]      (lizard-android/build.sh gen; out defaults to liblizard/out)
// ai: Writes out/setup/<variant>.json, the variant's index the native host makes its setup through: each module's
// ai: recipe, name, lanes and needs, and the nets' files it reads (the classifier, the proposer, the LDPC's stops) as
// ai: out/blobs/<hash>.bin; out/setup/send.json, the sender's (the tree's version and each module's recipe, which the
// ai: painter holds to its own: core/tx/gpu_painter.cpp checkSendTree); out/slang/<name>.json (a module's recipe: its kernel's file under gpu/slang/src and its
// ai: constants), out/spv/<name>.spv (a decoder module's <name> 16 hex of its recipe's key and its SPIR-V, so a module
// ai: keeps its name while its SPIR-V does; a sender's send_<k>), ingest_<name>.spv and send_<name>.spv (the engine's);
// ai: the same modules as Metal Shading Language for the Apple hosts, out/msl/<name>.metal with its binding map
// ai: out/msl/<name>.json (`slangMsl`); out/gen.json (the variants, the modules, the gpu sources' hash that made them).
// ai: The lab's scripts read a made tree from lizard_gpu_tree dump.
// ai: Needs, each checked before anything runs: the host's tool lizard_gpu_tree (LIZ_GPU_TREE, else
// ai: lizard-android/build/linux's: lizard-android/build.sh gen builds it first); slang 2026.22's slangc
// ai: (lizard-android/build.sh tools; liblizard/.tools/slang, else the PATH's). slangc validates the SPIR-V it writes.
import { mkdirSync, mkdtempSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync, realpathSync, rmSync, renameSync } from "node:fs";
import { execFile, execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { availableParallelism, tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = new URL("./", import.meta.url), LIB = new URL("../", import.meta.url);
const OUT = process.argv[2] ? pathToFileURL(`${resolve(process.argv[2])}/`) : new URL("../out/", HERE);
// ai: Every file gen puts in out/ (and gpu/send_wgsl.mjs), slangc's outputs too, is written whole beside its place and
// ai: renamed over it: each run makes new files, so a hard link to one of out/'s (a tree copied by links) keeps what it
// ai: held and is never rewritten by a later run, and no reader sees a file half written. A crash's leftover temp file
// ai: (<name>.tmp-<pid>-<n>) is let go by the next run's cleanup.
let putCount = 0;
function put(target, data) {
  const path = target instanceof URL ? fileURLToPath(target) : target;
  const tmp = `${path}.tmp-${process.pid}-${putCount++}`;
  try {
    writeFileSync(tmp, data);
    renameSync(tmp, path);
  } catch (e) { rmSync(tmp, { force: true }); throw e; }
}
const TMP_FILE = /\.tmp-\d+-\d+$/;
// ai: Run as a script (not when a check imports stripCaps), by real path: through a symlink or a space
// ai: import.meta.url and argv[1] spell the one file two ways.
const MAIN = (() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })();

const EXE = process.platform === "win32" ? ".exe" : "";
// ai: slangc for every module (gpu/slang/, the kernels' source since 2026-11-24, and the engine's own shaders):
// ai: liblizard/.tools/slang (lizard-android/build.sh tools) where it is, else the PATH's. Pinned (SLANG, build.sh's too):
// ai: a module is named by its source, so another slangc would write other SPIR-V under the same names; a run refuses any
// ai: but SLANG. One compiler, three targets: SPIR-V 1.3 for the Vulkan hosts, MSL for the Metal ones (macOS), WGSL for
// ai: the web's encoder. The SPIR-V
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

// ai: What a run needs: every one missing, each with where it comes from.
const GPU_TREE = process.env.LIZ_GPU_TREE ?? fileURLToPath(new URL(`../../lizard-android/build/linux/lizard_gpu_tree${EXE}`, HERE));
if (MAIN) {
  const missing = [
    [GPU_TREE, "lizard-android/build.sh gen (it builds the tool first), or LIZ_GPU_TREE"],
    [TOOLS.slangc, `slang ${SLANG}: lizard-android/build.sh tools, or slangc on the PATH`],
  ].filter(([p]) => !existsSync(p));
  if (missing.length) { console.error(missing.map(([p, from]) => `no ${p} (${from})`).join("\n")); process.exit(1); }
  // ai: slangc prints its version to stderr
  const sv = (() => { try { const r = spawnSync(TOOLS.slangc, ["-version"], { encoding: "utf8" }); return `${r.stdout ?? ""}${r.stderr ?? ""}`.trim().split("\n")[0] || null; } catch { return null; } })();
  if (sv !== SLANG) { console.error(`${TOOLS.slangc} is slang ${sv ?? "of no version it states"}; the tree's modules are slang ${SLANG}'s (lizard-android/build.sh tools)`); process.exit(1); }
}

// ai: slangc's optimisation level, every Slang module's (2026-11-24): its default -O1 inlined every function of COMPLETE
// ai: into one with each copy's loops (glslang's 43 loops 85), and the S26 ran the finder's vote stages +24% slower;
// ai: -O2 (smaller again) and -O0 (glslang's shape) both level with the GLSL there; -O2 level on the iGPU and in the
// ai: sender on both (research/build/slang-2026-11-24/)
const SLANG_OPT = ["-O2"];
// ai: A recipe's sources as slangc takes them: the kernel's file under gpu/slang/src, -I that folder (its #include
// ai: "lib/..." lines slangc's own), -D each constant in the recipe's order
const SLANG_SRC = new URL("gpu/slang/src/", LIB);
function recipeArgs(recipe) {
  const args = [fileURLToPath(new URL(recipe.src, SLANG_SRC)), "-I", fileURLToPath(SLANG_SRC)];
  for (const [k, v] of Object.entries(recipe.defs)) args.push("-D", `${k}=${v}`);
  return args;
}
function webWgsl(name, { code, stage }) {
  const dir = mkdtempSync(join(tmpdir(), "liz-wgsl-"));
  const w = join(dir, `${name}.wgsl`);
  try {
    run(TOOLS.slangc, [...recipeArgs(code), "-target", "wgsl", "-entry", "main", "-stage", stage, "-warnings-as-errors", "all", ...SLANG_OPT, "-o", w]);
    return readFileSync(w, "utf8");
  } finally { rmSync(dir, { recursive: true, force: true }); }
}

// ai: The device features a module needs beyond its variant's, by the capability its SPIR-V declares for each: "int16"
// ai: (Int16: shaderInt16; tldpc's i16 form alone). The host leaves out a pipeline whose device lacks one
// ai: (core/dec/setup.cpp).
const NEEDS = [["int16", 22]];   // ai: the capability's number, Int16
// ai: A module's lanes and needs from its SPIR-V: its entry's LocalSize (OpExecutionMode, opcode 16, mode 17), the
// ai: workgroup's invocations, which the host holds to its device's maxComputeWorkGroupInvocations (core/dec/setup.cpp),
// ai: and the capabilities NEEDS names.
function slangFacts(name) {
  const path = fileURLToPath(new URL(`spv/${name}.spv`, OUT)), b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  let lanes = null;
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) if ((w[i] & 0xffff) === 16 && w[i + 2] === 17) lanes = w[i + 3] * w[i + 4] * w[i + 5];
  if (!lanes) throw new Error(`${name}: no LocalSize in its SPIR-V`);
  const caps = capsOf(path), needs = NEEDS.filter(([, cap]) => caps.has(cap)).map(([k]) => k);
  return { lanes, ...(needs.length ? { needs } : {}) };
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
  try { return execFileSync(cmd, args, { encoding: "utf8", maxBuffer: 1 << 28, ...(input === undefined ? { stdio: ["ignore", "pipe", "pipe"] } : { input, stdio: ["pipe", "pipe", "pipe"] }) }); }
  catch (e) { const err = new Error(`${cmd.split("/").pop()} ${args.join(" ")}: ${(e.stderr || e.stdout || e.message).toString().trim().split("\n").slice(0, 12).join("\n")}`); err.tool = true; throw err; }
}

// ai: A module's recipe (the host's: { src, defs }) to SPIR-V 1.3 (the host's Vulkan 1.1 floor), a compute entry `main`, every
// ai: warning an error: compiled, then named (fixed: the sender's send_<k>; else 16 hex of its recipe's key and its
// ai: SPIR-V) and placed: out/spv/<name>.spv and out/slang/<name>.json (the recipe, which the MSL's compile and anyone
// ai: compiling it again read).
// ai: slangc validates its SPIR-V with its own SPIRV-Tools (the NDK's spirv-val predates Slang's source-language
// ai: operand); a workgroup variable has no initializer (Slang zeroes none).
function compileRecipe(key, recipe, fixed = null) {
  const tmp = mkdtempSync(join(tmpdir(), "liz-mod-"));
  try {
    const spvTmp = join(tmp, "m.spv");
    const c = slangSpv(recipeArgs(recipe), spvTmp, fixed ?? recipe.src);
    const bytes = readFileSync(spvTmp);
    const name = fixed ?? createHash("sha256").update(key).update(bytes).digest("hex").slice(0, 16);
    put(new URL(`spv/${name}.spv`, OUT), bytes);
    put(fileURLToPath(new URL(`slang/${name}.json`, OUT)), JSON.stringify({ src: recipe.src, defs: recipe.defs }));
    return { ...c, name, recipe };
  } finally { rmSync(tmp, { recursive: true, force: true }); }
}
// ai: a Slang source (slangc's arguments up to the target: a recipe's, or an engine shader's file) to SPIR-V at spv, its
// ai: capabilities held to the Adreno's (stripCaps); its parameters' Vulkan bindings from slangc's reflection (`vk`: the
// ai: name's binding, "push" for push constants), which slangMsl joins to Metal's
function slangSpv(srcArgs, spv, name) {
  // ai: warning 41012 (the profile's capability atoms widened for what the module uses) is not an error: the SPIR-V's own
  // ai: capabilities are what a device is held to (NEEDS, stripCaps)
  const refl = `${spv}.refl.json`;
  run(TOOLS.slangc, [...srcArgs, "-target", "spirv", "-profile", "spirv_1_3", "-entry", "main", "-stage", "compute", "-warnings-as-errors", "all", "-warnings-disable", "41012", ...SLANG_OPT, "-o", spv, "-reflection-json", refl]);
  const vk = {};
  let params = [];
  try {
    params = JSON.parse(readFileSync(refl, "utf8")).parameters ?? [];
    for (const p of params) {
      const b = p.binding;
      if (b?.kind === "descriptorTableSlot") vk[p.name] = b.index;
      else if (b?.kind === "pushConstantBuffer") vk[p.name] = "push";
      else throw new Error(`${name}: parameter ${p.name} bound as ${JSON.stringify(p.binding ?? p.bindings)}, not a descriptor or push constants`);
    }
  } finally { rmSync(refl, { force: true }); }
  stripCaps(spv);
  const b = readFileSync(spv), version = b.readUInt32LE(4);
  if (version !== 0x10300) throw new Error(`${name}: slangc wrote SPIR-V ${version >> 16}.${(version >> 8) & 255}, not 1.3`);
  // ai: a Workgroup variable (OpVariable, opcode 59, storage class 4) with an initializer would need the device's
  // ai: shaderZeroInitializeWorkgroupMemory, which the host does not ask for (core/dec/setup.cpp variantsFor)
  const w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  for (let i = 5; i < w.length; i += w[i] >>> 16) if ((w[i] & 0xffff) === 59 && w[i + 3] === 4 && w[i] >>> 16 > 4) throw new Error(`${name}: a workgroup variable with an initializer (zero it in the kernel, then a barrier)`);
  return { bytes: b.byteLength, vk, params };
}

// ai: A reflected type's memory layout as the host writes it: every field's offset and size by its path, each array's
// ai: stride, a buffer's element size (alignment left out: MSL's packed vectors align to 4 where std430's align to 16,
// ai: at the same offsets and sizes). Two targets' must be the same, or the host's bytes land elsewhere on one of them
// ai: (MSL's float3 is 16 bytes, std140's and std430's 12).
function layoutOf(t, path = "", base = 0, out = []) {
  if (!t) return out;
  const size = (x) => x?.sizes?.find((s) => s.kind === "uniform")?.value;
  if (t.kind === "constantBuffer" || t.kind === "parameterBlock") return layoutOf(t.elementType, path, base, out);
  if (t.kind === "resource") {
    if (t.baseShape === "structuredBuffer") { out.push(`${path}[] ${size(t.resultType)}`); layoutOf(t.resultType, `${path}[]`, 0, out); }
    return out;
  }
  if (t.kind === "struct") {
    for (const f of t.fields ?? []) {
      const at = base + (f.binding?.offset ?? 0);
      out.push(`${path}.${f.name}@${at}:${f.binding?.size}`);
      layoutOf(f.type, `${path}.${f.name}`, at, out);
    }
  } else if (t.kind === "array") {
    out.push(`${path} x${t.elementCount} by ${t.uniformStride}`);
    layoutOf(t.elementType, `${path}[]`, base, out);
  }
  return out;
}

// ai: A Slang file to Metal Shading Language (the Apple hosts' kernels, core/wg/wg_metal.mm; compiled by the system's
// ai: Metal compiler at run time) and its binding map: Metal binds by argument index, a buffer, texture and sampler
// ai: index space each, which slangc assigns in declaration order whatever the Vulkan binding says, and takes the
// ai: workgroup's size at the dispatch, not from the kernel. out/msl/<name>.json: entry (slangc renames main, a word MSL
// ai: keeps), local (the workgroup's size), bind ({Vulkan binding: [[kind, index], ...]}, kind "b" a buffer, "t" a
// ai: texture, "s" a sampler) and push (the push constants' [kind, index], or null). Every index the kernel's signature
// ai: names must be in the map, so nothing slangc adds unasked (a hidden argument) goes unbound. Warning 56005 (MSL
// ai: has no `precise`, so the learned stop's and the develop's sums may contract into fused multiply-adds there) is no
// ai: error: float noise, as the GPU is not held bit exact; 40100 is the entry's rename.
const MSL_KIND = { constantBuffer: "b", shaderResource: "t", unorderedAccess: "t", samplerState: "s" };
// ai: No #line directives: they would name the build machine's paths in the text the apps carry.
async function slangMsl(srcArgs, name, vk, spvParams = []) {
  const dir = mkdtempSync(join(tmpdir(), "liz-msl-"));
  const [msl, refl] = [`${name}.metal`, `${name}.json`].map((f) => join(dir, f));
  try {
    const args = [...srcArgs, "-target", "metal", "-entry", "main", "-stage", "compute", "-warnings-as-errors", "all", "-warnings-disable", "41012,40100,56005", "-line-directive-mode", "none", ...SLANG_OPT, "-o", msl, "-reflection-json", refl];
    await new Promise((ok, no) => execFile(TOOLS.slangc, args, { encoding: "utf8" }, (e, out, err) => {
      if (!e) return ok();
      const x = new Error(`slangc ${args.join(" ")}: ${(err || out || e.message).toString().trim().split("\n").slice(0, 12).join("\n")}`);
      x.tool = true;
      no(x);
    }));
    const text = readFileSync(msl, "utf8"), r = JSON.parse(readFileSync(refl, "utf8"));
    const entry = /\[\[kernel\]\]\s+void\s+(\w+)\s*\(/.exec(text)?.[1];
    if (!entry) throw new Error(`${name}: no kernel in slangc's MSL`);
    const local = r.entryPoints?.[0]?.threadGroupSize;
    if (!local || local.length !== 3) throw new Error(`${name}: no workgroup size in slangc's reflection`);
    const bind = {}, named = new Set();
    let push = null;
    for (const p of r.parameters ?? []) {
      const at = (p.bindings ?? [p.binding]).map((b) => {
        const k = MSL_KIND[b?.kind];
        if (!k) throw new Error(`${name}: parameter ${p.name} bound on Metal as ${JSON.stringify(b)}`);
        named.add(`${k}${b.index}`);
        return [k, b.index];
      });
      if (!(p.name in vk)) throw new Error(`${name}: parameter ${p.name} on Metal has no Vulkan binding`);
      const twin = spvParams.find((q) => q.name === p.name);
      const [a, b] = [layoutOf(twin?.type), layoutOf(p.type)];
      const off = a.findIndex((x, i) => x !== b[i]);
      if (off >= 0 || a.length !== b.length) throw new Error(`${name}: parameter ${p.name} lies out differently on Metal: ${a[off] ?? "(none)"} on SPIR-V, ${b[off] ?? "(none)"} on Metal`);
      if (vk[p.name] === "push") push = at[0];
      else bind[vk[p.name]] = at;
    }
    const sig = text.slice(text.indexOf(`void ${entry}(`), text.indexOf("{", text.indexOf(`void ${entry}(`)));
    for (const [, kind, i] of sig.matchAll(/\[\[(buffer|texture|sampler)\((\d+)\)\]\]/g))
      if (!named.has(`${kind[0]}${i}`)) throw new Error(`${name}: the kernel binds ${kind}(${i}), which no parameter is`);
    put(new URL(`msl/${name}.metal`, OUT), text);
    put(new URL(`msl/${name}.json`, OUT), JSON.stringify({ entry, local, bind, push }));
    return { bytes: Buffer.byteLength(text) };
  } finally { rmSync(dir, { recursive: true, force: true }); }
}
// ai: the MSL jobs, run together at the end (a slangc a core)
async function pool(jobs, n = availableParallelism()) {
  const out = new Array(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, jobs.length) }, async () => { while (next < jobs.length) { const k = next++; try { out[k] = { ok: await jobs[k]() }; } catch (e) { out[k] = { e }; } } }));
  return out;
}

// ai: A SPIR-V module's capabilities (OpCapability, opcode 17).
function capsOf(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4), caps = new Set();
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) if ((w[i] & 0xffff) === 17) caps.add(w[i + 1]);
  return caps;
}

// ai: What the tree is made from: the kernels' Slang (gpu/slang/src), the engine's shaders, the nets' files the setups
// ai: read and the host's listings (every recipe, the sender's setup): a change to any of them makes the tree stale
// ai: (gen.json's `sources` names what it was made from; nothing checks it at run time, so a change wants gen run again).
function sourcesHash(listings, nets, engine) {
  const h = createHash("sha256");
  const walk = (dir) => { for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isDirectory()) walk(p); else if (f.endsWith(".slang")) h.update(f).update(readFileSync(p)); } };
  walk(fileURLToPath(SLANG_SRC));
  for (const { src } of engine) h.update(src.slice(src.lastIndexOf("/") + 1)).update(readFileSync(src));
  for (const f of [...nets].sort()) h.update(f).update(readFileSync(new URL(f, LIB)));
  for (const x of listings) h.update(x);
  return h.digest("hex").slice(0, 16);
}

async function main() {
  for (const d of ["setup", "slang", "spv", "msl", "blobs"]) mkdirSync(new URL(`${d}/`, OUT), { recursive: true });
  // ai: the recipes each variant's setup names, from the host's own code (LIZ_GEN_CAP=<n>: SELECT's cap, the lab's, for a
  // ai: tree written elsewhere)
  const listingText = run(GPU_TREE, ["recipes", fileURLToPath(LIB)]), listing = JSON.parse(listingText);
  const compiled = {}, failed = [], byKey = new Map(), netBlobs = new Map(), summary = [];
  const netHash = new Map();
  for (const [variant, L] of Object.entries(listing.variants)) {
    const nets = {};
    for (const f of L.nets) {
      if (!netHash.has(f)) {
        const bytes = readFileSync(new URL(f, LIB)), h = createHash("sha256").update(bytes).digest("hex").slice(0, 16);
        netHash.set(f, h);
        netBlobs.set(h, bytes);
      }
      nets[f] = netHash.get(f);
    }
    const index = [];
    for (const r of L.recipes) {
      if (!byKey.has(r.key)) {
        try { byKey.set(r.key, compileRecipe(r.key, { src: r.src, defs: r.defs })); } catch (e) { if (!e.tool) throw e; failed.push({ h: `${r.src} ${r.key.split("\n").slice(1).join(" ").slice(0, 200)}`, message: e.message }); byKey.set(r.key, null); }
      }
      const c = byKey.get(r.key);
      if (!c) continue;
      compiled[c.name] = c;
      index.push({ src: r.src, defs: r.defs, name: c.name, ...slangFacts(c.name) });
    }
    index.sort((x, y) => (x.name < y.name ? -1 : x.name > y.name ? 1 : 0));
    put(new URL(`setup/${variant}.json`, OUT), JSON.stringify({ treeVersion: listing.treeVersion, variant, modules: index, nets }));
    summary.push({ variant, pipelines: L.pipelines, buffers: L.buffers, nets: L.nets_used });
    console.log(`${variant}: ${L.recipes.length} modules, ${L.pipelines} pipelines, ${L.buffers} buffers, nets ${JSON.stringify(L.nets_used)}`);
  }
  const modules = new Set(Object.keys(compiled));
  let netBytes = 0;
  for (const [h, data] of netBlobs) { put(new URL(`blobs/${h}.bin`, OUT), data); netBytes += data.byteLength; }
  console.log(`${netBlobs.size} nets' files, ${(netBytes / 1024).toFixed(0)} KB`);

  // ai: The sender's GPU encoder (2026-10-02): its modules as the host lists them (lizard_gpu_tree send, core/tx/
  // ai: send_plan.h), each native one as out/spv/<name>.spv under its fixed name, their index setup/send.json (the
  // ai: tree's version, each module's recipe: core/tx/gpu_painter.cpp checkSendTree), the web's as WGSL.
  const sendText = run(GPU_TREE, ["send"]), send = JSON.parse(sendText);
  const sendNative = send.modules.filter((m) => m.native);
  for (const m of sendNative) {
    try { compiled[m.name] = compileRecipe(m.name, { src: m.src, defs: m.defs }, m.name); } catch (e) { if (!e.tool) throw e; failed.push({ h: m.name, message: e.message }); }
  }
  put(new URL("setup/send.json", OUT), JSON.stringify({ treeVersion: send.treeVersion, modules: sendNative.map(({ name, src, defs }) => ({ name, src, defs })) }));
  // ai: the web encoder's WGSL from the same recipes (webWgsl), into the tree beside the encoder, keyed by the name
  // ai: less send_ (gpu/encoder.mjs SEND_WGSL)
  const web = send.modules.filter((m) => m.web), wgsl = {};
  for (const m of web) {
    try { wgsl[m.name.replace(/^send_/, "")] = webWgsl(m.name, { code: { src: m.src, defs: m.defs }, stage: m.web }); } catch (e) { if (!e.tool) throw e; failed.push({ h: `web ${m.name}`, message: e.message }); }
  }
  if (Object.keys(wgsl).length === web.length) {
    const text = `// The web encoder's WGSL (encoder.mjs): compiled by ../gen/gen.mjs from the sender's Slang (slang/src/send_*.slang,\n// slangc ${SLANG}). Generated: never edited by hand.\nexport const SEND_WGSL = ${JSON.stringify(wgsl, null, 1)};\n`;
    const at = new URL("../gpu/send_wgsl.mjs", HERE);
    if (!existsSync(at) || readFileSync(at, "utf8") !== text) put(at, text);
    console.log(`web: ${Object.keys(wgsl).length} WGSL modules, ${(Buffer.byteLength(text) / 1024).toFixed(0)} KB (gpu/send_wgsl.mjs)`);
  }
  console.log(`sender: ${sendNative.length} kernels`);
  // ai: The engine's own shaders, Slang from their files to out/spv/<prefix><name>.spv: core/ingest/ (ingest_: the camera's
  // ai: frame through the YCbCr sampler the host builds for its format, and the raw develop) and core/tx/ (send_: the
  // ai: presenter's expand and Lanczos stretch into RGBA8 storage images, the stream files' packing). Native alone: no WGSL.
  const engine = [["../core/ingest/", "ingest_"], ["../core/tx/", "send_"]].flatMap(([d, pre]) => {
    const dir = new URL(d, HERE);
    return readdirSync(dir).filter((f) => f.endsWith(".slang")).map((f) => ({ src: fileURLToPath(new URL(f, dir)), name: `${pre}${f.replace(/\.slang$/, "")}` }));
  });
  // ai: MSL_ONLY: an engine shader the Metal hosts alone run (the Metal presenter's bilinear stretch, which Vulkan's blit
  // ai: makes): its SPIR-V made in a temporary folder for its bindings, and not kept
  const MSL_ONLY = new Set(["send_scale_linear"]);
  const engineVk = {};
  for (const { src, name } of engine) {
    // ai: compiled in a temporary folder, then put in out/spv (the MSL-only shader's not kept)
    const tmp = mkdtempSync(join(tmpdir(), "liz-spv-")), at = join(tmp, `${name}.spv`);
    try {
      const c = slangSpv([src], at, name);
      if (!MSL_ONLY.has(name)) put(new URL(`spv/${name}.spv`, OUT), readFileSync(at));
      engineVk[name] = c;
      console.log(`engine ${name}: ${MSL_ONLY.has(name) ? "MSL alone" : `${c.bytes} B of SPIR-V`}`);
    } catch (e) { if (!e.tool) throw e; failed.push({ h: name, message: e.message }); }
    finally { rmSync(tmp, { recursive: true, force: true }); }
  }
  // ai: every module and engine shader that compiled to SPIR-V again as MSL (slangMsl), from the same Slang
  const mslJobs = [
    ...Object.entries(compiled).map(([h, c]) => ({ name: h, args: recipeArgs(c.recipe), vk: c.vk, params: c.params })),
    ...engine.filter(({ name }) => engineVk[name]).map(({ src, name }) => ({ name, args: [src], vk: engineVk[name].vk, params: engineVk[name].params })),
  ];
  const mslDone = await pool(mslJobs.map((j) => () => slangMsl(j.args, j.name, j.vk, j.params)));
  let mslBytes = 0;
  mslDone.forEach((r, k) => { if (r.e) { if (!r.e.tool && !/^\w+: /.test(r.e.message)) throw r.e; failed.push({ h: `msl ${mslJobs[k].name}`, message: r.e.message }); } else mslBytes += r.ok.bytes; });
  console.log(`MSL: ${mslDone.filter((r) => r.ok).length} of ${mslJobs.length} modules, ${(mslBytes / 1024).toFixed(0)} KB`);
  for (const f of failed) console.log(`FAIL ${f.h}\n${f.message}\n`);
  const spvBytes = Object.values(compiled).reduce((a, c) => a + c.bytes, 0);
  console.log(`${modules.size} modules: ${Object.keys(compiled).length} compiled (${(spvBytes / 1024).toFixed(0)} KB of SPIR-V), ${failed.length} failed`);
  // ai: What an earlier run wrote and this one did not (a module or a blob no setup names now) let go, so a tree
  // ai: copied whole (the desktop's bundle, an install) carries no stale file. Only gen's own names: a file put in
  // ai: spv/ by hand (bench_dot.spv) stays.
  const kept = new Set([...Object.keys(listing.variants).map((v) => `setup/${v}.json`), "setup/send.json", ...[...netBlobs.keys()].map((h) => `blobs/${h}.bin`)]);
  for (const h of [...modules, ...sendNative.map((m) => m.name)])
    for (const f of [`slang/${h}.json`, `slang/${h}`, `spv/${h}.spv`]) kept.add(f);
  for (const { name } of engine) if (!MSL_ONLY.has(name)) kept.add(`spv/${name}.spv`);
  for (const j of mslJobs) for (const f of [`msl/${j.name}.metal`, `msl/${j.name}.json`]) kept.add(f);
  const ours = { setup: /\.json$/, blobs: /^[0-9a-f]{16}\.bin$/, record: /(\.json|^blobs)$/, "record/blobs": /^[0-9a-f]{16}\.bin$/, wgsl: /\.wgsl$/, naga: /\.wgsl$/, glsl: /\.comp$/, slang: /^([0-9a-f]{16}|send_\w+)(\.json|\.slang)?$/, spv: /^([0-9a-f]{16}|send_\w+|ingest_\w+)(\.raw)?\.spv$/,
    msl: /^([0-9a-f]{16}|send_\w+|ingest_\w+)\.(metal|json)$/ };
  let gone = 0;
  // ai: wgsl/, naga/ and glsl/ an earlier run's alone (the sender's WGSL and every GLSL module went 2026-11-24, with
  // ai: spv/'s .raw.spv copies), record/ too (the JS setup's recording, gone 2026-11-25), each let go once empty
  for (const [d, re] of Object.entries(ours)) for (const f of existsSync(new URL(`${d}/`, OUT)) ? readdirSync(new URL(`${d}/`, OUT)) : []) if ((re.test(f) && !kept.has(`${d}/${f}`)) || TMP_FILE.test(f)) { rmSync(new URL(`${d}/${f}`, OUT), { recursive: true, force: true }); gone++; }
  for (const d of ["wgsl", "naga", "glsl", "record/blobs", "record"]) if (existsSync(new URL(`${d}/`, OUT)) && !readdirSync(new URL(`${d}/`, OUT)).length) rmSync(new URL(`${d}/`, OUT), { recursive: true });
  if (gone) console.log(`${gone} files of an earlier run let go`);
  console.log(`${Object.keys(compiled).length + engine.length - MSL_ONLY.size} SPIR-V modules by slangc ${SLANG}: ${Object.keys(compiled).length} the decoder's and the sender's, ${engine.length - MSL_ONLY.size} the engine's`);
  const nets = new Set(Object.values(listing.variants).flatMap((L) => L.nets));
  put(new URL("gen.json", OUT), JSON.stringify({ sources: sourcesHash([listingText, sendText], nets, engine), writer: `slangc ${SLANG} (SPIR-V 1.3 and MSL, ${SLANG_OPT.join(" ")}) for every module, and the web encoder's WGSL (gpu/send_wgsl.mjs) from the same Slang`, variants: summary, modules: Object.keys(compiled), engine: engine.map((e) => e.name), failed: failed.map((f) => f.h) }, null, 1));
  if (failed.length) process.exit(1);
}

if (MAIN) await main();
