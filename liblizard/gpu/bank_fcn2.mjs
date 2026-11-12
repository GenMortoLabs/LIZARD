// The filter bank as the trained proposer, version 2 (glsl/bank_fcn2.mjs, plan/bank_fcn2.mjs; scripts/gpu/cnn/proposer/README.md
// "Version 2"), to the contract in glsl/bank.mjs: levels 1 to 4 (level 0's slots stay empty), one workgroup a region of
// 16 x 16 tiles. The map does not depend on the region, so the region is the largest the device's workgroup memory
// holds (32 x 32 in f16 or halves where it gives 64 KB, 32 x 16 in 32 KB). An X pass writes each level's standardised
// map once, halves where the activations are halves, and the network kernel reads it. The shader runs in f16 when the
// decoder's precision is f16, else f32 storage where it fits, else f32 with halves stored. A device too small for all
// of them gets FitError.
// ai: FitError leaves the device with no GPU decoder: the C decodes (since 2026-09-26, no DoG bank to fall to).
// ai: The decoder's precision "int8" runs the int8 form on a q8 file (cnn/q8.mjs), WEIGHTS_INT8; a float precision
// ai: runs a float form on a float file. Either refuses the other's file, and a precision it does not know.
// ai: A lane's planes and bind groups and the dispatches (the WebGPU host's ensure, dispatch and grids2) are the native
// ai: host's (core/dec/front.cpp); the fused one-kernel form and the lab's ?fcnstore= forms went with the WGSL
// ai: decoder (archive/webgpu-decoder/).
import { fcn2Plan, fcn2Forms, fcn2Bytes, KEEP } from "./plan/bank_fcn2.mjs";
import { fcn2Source } from "./glsl/bank_fcn2.mjs";
import { loadNet } from "./cnn/netfile.mjs";
import { LEVELS } from "./constants.mjs";   // ai: not ./decoder.mjs, which imports this module (constants.mjs)

export { KEEP };
// ai: The installed proposer since 2026-11-12: v2's net (8 channels, 3 layers, the pairwise ranking term RANK=1) trained
// ai: on today's format by a curriculum on the GPU renderer's and PySETCAM's scenes: easy scenes, then half-blown
// ai: exposures and cluttered rooms, then PySETCAM's drawn cameras distilled from the second stage's net (seed 2;
// ai: research/build/curriculum-2026-11-12, w8d3-D2), its floor 0.15 (research/results/gpu/prop_d2/d2-f15.safetensors,
// ai: its f64 reference d2.ref there). Through the decoder in int8 on 24 real replays: blocks +0.26%, level and +0.09%
// ai: in the 64, 128 and 256 rings against the one before (research/STATUS.md "The proposer retrained on today's format
// ai: installed"). The one before it, 2026-11-09 to 11-12: rank-s1 (research/results/gpu/prop_rank/, kept in
// ai: research/results/gpu/installed-before-2026-11-12/); before it pb-s1 (installed-before-2026-11-09/). A reference
// ai: holds the net's values, not its peaks: the checks take the floor from the file.
export const WEIGHTS = "gpu/cnn/proposer/weights2.safetensors";
export const REFERENCE = "../research/results/gpu/prop_d2/d2.ref";
// ai: The installed int8 proposer (a q8 file) and its PQ81 reference: the post-training quantisation of WEIGHTS on its
// ai: last stage's sets (research/results/gpu/prop_d2/d2-q8-f15.safetensors, the floor 0.15), installed 2026-11-12; the
// ai: one before it rank-s1-q8-f15 (research/results/gpu/prop_rank/, reference rank-s1-q8.ref there).
export const WEIGHTS_INT8 = "gpu/cnn/proposer/weights2-q8.safetensors";
export const REFERENCE_INT8 = "../research/results/gpu/prop_d2/d2-q8.ref";

export class FitError extends Error {}

export async function build(fh) {
  if (!["int8", "f16", "f32"].includes(fh.precision)) throw new Error(`fcn2 bank: precision ${fh.precision}: int8, f16 or f32`);
  const int8 = fh.precision === "int8", file = int8 ? WEIGHTS_INT8 : WEIGHTS;
  const json = await loadNet(new URL(`../${file}`, import.meta.url));
  const f16 = !int8 && fh.floatPrecision === "f16";
  const limit = fh.device.limits.maxComputeWorkgroupStorageSize;
  let forms;
  try { forms = fcn2Forms(json, { f16, int8 }).filter((t) => t.xplane); } catch (e) { throw new Error(`fcn2 bank ${fh.precision}, ${file}: ${e.message}`); }
  const form = forms.find((t) => t.bytes <= limit);
  // ai: int8 too big for the device is a failed int8 build, not a small device: the caller's precision chain then
  // ai: tries f16 or f32, whose halves-stored forms are smaller.
  if (!form && int8) throw new Error(`fcn2 bank int8: ${Math.min(...forms.map((t) => t.bytes))} bytes of workgroup memory at the least, the device gives ${limit}`);
  if (!form) throw new FitError(`fcn2 bank: ${Math.min(...forms.map((t) => t.bytes))} bytes of workgroup memory at the least, the device gives ${limit}`);
  const plan = fcn2Plan(json);
  // ai: the plane form's two kernels (glsl/bank_fcn2.mjs): the network on the plane, and the X pass
  const src = fcn2Source(json, { ...form, stage: "net" });
  fh.fcn2 = await fh.pipeline(src, ["ro", "ro", "rw", "rw", "uniform"]);
  fh.fcn2X = await fh.pipeline(fcn2Source(json, { ...form, stage: "x" }), ["ro", "ro", "rw", "uniform"]);
  fh.fcn2Form = form;
  fh.bankStore = form.name;
  fh.log(`fcn2 bank ${json.spec ?? ""} (${file}) ${form.name}: ${plan.macs} MACs a map pixel, box ${plan.K}, floor ${plan.floor}, ${form.bytes} bytes of workgroup memory (network ${fcn2Bytes(src)}; the device gives ${limit}), levels 1 to ${LEVELS - 1}`);
}

// ai: The finder's first round runs levels COARSE_FROM up on every frame; levels below it run in its second round
// ai: (core/dec/front.cpp's batch, glsl/finder.mjs REDO), on the frames the first did not find. On the S26's 2:1 replays level 1
// ai: was 72% of this stage's time; without it the 128 ring's replay read every frame as before and the 256 ring's
// ai: found 165 of 192 (2026-10-21).
export const COARSE_FROM = 2;
