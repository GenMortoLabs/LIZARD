// F3 SELECT in GLSL: the strongest cap peaks of a frame from its tiles' slots, one workgroup a frame, by two cuts and
// the slots' order: a histogram of the peaks' scores (64 bins, 2^(1/8) apart), the bin where the count above reaches
// cap; a histogram of that bin's peaks (256 places in it, 2^(1/2048) apart), the place where the count above reaches
// what is left; then one append pass: everything above the second cut into the list's first places, the second cut's
// own place after them, all of it where it fits, else in the slots' order (a scan of each lane's count over its own
// run of slots) while the cap has room. Which peaks are kept is the scores', ties at a place the lower slot's, never the lanes' arrival (2026-11-16:
// a cut bin taken by its atomics' order kept a peak or its 9% weaker neighbour as the lanes happened to arrive, and
// on a real replay 7 of 1,165 frames were found or lost by that order alone). The list's order is the appends', which
// nothing reads.
// ai: Bindings: 0 frames, 1 raw, 2 peaks, 3 counts, 4 the uniform (slots, cap, tau, p0). hist, hist2, used and
// ai: used2 are read before any lane writes them (the histograms' counts, the appends' places), so they are zeroed by
// ai: their initializers (GL_EXT_null_initializer, the same OpConstantNull naga gives every workgroup variable); cut,
// ai: above, cut2, above2 and tied are written by lane 0, and part by every lane, before the barriers that precede their
// ai: reads; tied is read as gen.mjs makes a read straight after a barrier (WGSL's workgroupUniformLoad), so the scan's
// ai: barriers may sit under it.
// ai: The atomics are common.mjs's, at naga's scopes (wgAdd, wgLoad, wgStore, devAdd, devStore).
import { VERSION, ATOMICS, NULL_INIT, DIMS, wgAdd, wgLoad, wgStore, devAdd, devStore, uniformLoad } from "./common.mjs";

export const SELECT = VERSION + ATOMICS + NULL_INIT + DIMS + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Raw { vec4 raw[]; };
layout(set = 0, binding = 2, std430) buffer Peaks { vec4 peaks[]; };
layout(set = 0, binding = 3, std430) buffer Counts { uint counts[]; };
// ai: slots: a frame's over every level; p0 the first a round's levels hold
layout(set = 0, binding = 4, std140) uniform Sel { uint slots; uint cap; float tau; uint p0; } P;
shared uint hist[64] = {};
shared uint hist2[256] = {};
shared uint part[256];
shared uint cut;
shared uint above;
shared uint cut2;
shared uint above2;
shared uint tied;
shared uint used = {};
shared uint used2 = {};
float lv(float v) { return 8.0 * log2(v / P.tau); }
uint bin(float v) { return uint(clamp(floor(lv(v)), 0.0, 63.0)); }
// ai: the place within bin b (a clamped bin's out-of-range scores at its ends, where the slots' order decides)
uint place(float v, uint b) { return uint(clamp(floor((lv(v) - float(b)) * 256.0), 0.0, 255.0)); }
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  // ai: a frame record's word (read-only storage), so the whole group returns before any barrier
  if (frames[f].valid != 1u) { return; }
  uint base = f * P.slots;
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    float v = raw[base + i].w;
    if (v > 0.0) { ${wgAdd("hist[bin(v)]", "1u")}; }
  }
  barrier();
  if (li == 0u) {
    uint total = 0u;
    int b = 63;
    while (true) {
      uint h = ${wgLoad("hist[b]")};
      if (total + h >= P.cap || b == 0) { break; }
      total += h;
      b--;
    }
    cut = uint(b);
    above = total;
  }
  // ai: WGSL's workgroupUniformLoad(&cut): a barrier, the load, a barrier
  barrier();
  uint t = cut;
  uint a0 = above;
  barrier();
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    float v = raw[base + i].w;
    if (v > 0.0 && bin(v) == t) { ${wgAdd("hist2[place(v, t)]", "1u")}; }
  }
  barrier();
  if (li == 0u) {
    // ai: above < cap by the first cut, so room > 0
    uint room = P.cap - a0, total = 0u;
    int s = 255;
    while (true) {
      uint h = ${wgLoad("hist2[s]")};
      if (total + h >= room || s == 0) { break; }
      total += h;
      s--;
    }
    cut2 = uint(s);
    above2 = a0 + total;
    // ai: the second cut's own place overflows what is left: its peaks taken in the slots' order
    tied = total + ${wgLoad("hist2[s]")} > room ? 1u : 0u;
  }
  ${uniformLoad("uint tie", "tied")}
  uint s = cut2;
  uint a1 = above2;
  if (tie == 0u) {
    // ai: everything from the second cut on fits: the cut bin's in by their own count, which the order cannot change
    for (uint i = P.p0 + li; i < P.slots; i += 256u) {
      vec4 e = raw[base + i];
      if (e.w > 0.0) {
        uint bb = bin(e.w);
        if (bb > t) { peaks[f * P.cap + ${wgAdd("used", "1u")}] = e; }
        else if (bb == t && place(e.w, t) >= s) { peaks[f * P.cap + a0 + ${wgAdd("used2", "1u")}] = e; }
      }
    }
    barrier();
    if (li == 0u) { ${devStore("counts[f * 16u + 2u]", `min(a1 + ${wgLoad("hist2[s]")}, P.cap)`)}; }
    return;
  }
  // ai: the second cut's own place in the slots' order: each lane's run of slots, its count, the counts' scan
  uint per = (P.slots - P.p0 + 255u) / 256u;
  uint lo = min(P.p0 + li * per, P.slots), hi = min(lo + per, P.slots);
  uint mine = 0u;
  for (uint i = lo; i < hi; i++) {
    float v = raw[base + i].w;
    if (v > 0.0 && bin(v) == t && place(v, t) == s) { mine++; }
  }
  part[li] = mine;
  barrier();
  for (uint d = 1u; d < 256u; d <<= 1u) {
    uint add = li >= d ? part[li - d] : 0u;
    barrier();
    part[li] += add;
    barrier();
  }
  uint at = a1 + part[li] - mine;
  // ai: One pass: above the first cut into the list's first 'above' places, the cut bin above the second cut into the
  // ai: next (both fewer than the cap by construction), the second cut's own place after them while the cap has room.
  for (uint i = lo; i < hi; i++) {
    vec4 e = raw[base + i];
    if (e.w > 0.0) {
      uint bb = bin(e.w);
      if (bb > t) { peaks[f * P.cap + ${wgAdd("used", "1u")}] = e; }
      else if (bb == t) {
        uint pp = place(e.w, t);
        if (pp > s) { peaks[f * P.cap + a0 + ${wgAdd("used2", "1u")}] = e; }
        else if (pp == s) { if (at < P.cap) { peaks[f * P.cap + at] = e; } at++; }
      }
    }
  }
  barrier();
  if (li == 0u) { ${devStore("counts[f * 16u + 2u]", `min(a1 + ${wgLoad("hist2[s]")}, P.cap)`)}; }
}
`;
