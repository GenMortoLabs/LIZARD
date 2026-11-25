// F3 SELECT: the strongest cap peaks of a frame from its tiles' slots, one workgroup a frame, by two cuts and the
// slots' order: a histogram of the peaks' scores (64 bins, 2^(1/8) apart), the bin where the count above reaches cap; a
// histogram of that bin's peaks (256 places in it, 2^(1/2048) apart), the place where the count above reaches what is
// left; then one append pass: everything above the second cut into the list's first places, the second cut's own
// place after them, all of it where it fits, else in the slots' order (a scan of each lane's count over its own run of
// slots) while the cap has room. Which peaks are kept is the scores', ties at a place the lower slot's, never the
// lanes' arrival (a cut bin taken by its atomics' order kept a peak or its 9% weaker neighbour as the lanes happened to
// arrive). The list's order is the appends', which nothing reads.
// Bindings: 0 frames, 1 raw, 2 peaks, 3 counts, 4 the uniform (slots, cap, tau, p0). hist, hist2, used and used2 are
// counted into from zero (the histograms' counts, the appends' places), so the kernel zeroes them first, before a
// barrier; cut, above, cut2, above2 and tied are written by lane 0, and part by every lane, before the barriers that
// precede their reads; tied is read straight after a barrier (lib/sync.slang UNIFORM_LOAD), so the scan's barriers may
// sit under it. Atomic<uint> for the workgroup's atomics.
#include "lib/sync.slang"
#include "lib/frame.slang"
[[vk::binding(0, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(1, 0)]] StructuredBuffer<float4> raw;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> peaks;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counts;
// slots: a frame's over every level; p0 the first a round's levels hold
struct SelBlock { uint slots; uint cap; float tau; uint p0; };
[[vk::binding(4, 0)]] ConstantBuffer<SelBlock> P;
groupshared Atomic<uint> hist[64];
groupshared Atomic<uint> hist2[256];
groupshared uint part[256];
groupshared uint cut;
groupshared uint above;
groupshared uint cut2;
groupshared uint above2;
groupshared uint tied;
groupshared Atomic<uint> used;
groupshared Atomic<uint> used2;
float lv(float v) { return 8.0 * log2(v / P.tau); }
uint bin(float v) { return uint(clamp(floor(lv(v)), 0.0, 63.0)); }
// the place within bin b (a clamped bin's out-of-range scores at its ends, where the slots' order decides)
uint place(float v, uint b) { return uint(clamp(floor((lv(v) - float(b)) * 256.0), 0.0, 255.0)); }
[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint lix : SV_GroupIndex) {
  uint3 wg = wid;
  uint li = lix;
  uint f = wg.x;
  // a frame record's word (read-only storage), so the whole group returns before any barrier
  if (frames[f].valid != 1u) { return; }
  uint base = f * P.slots;
  // the histograms and the places' counters from zero (the workgroup's memory is not zeroed for it)
  hist2[li].store(0u);
  if (li < 64u) { hist[li].store(0u); }
  if (li == 0u) { used.store(0u); used2.store(0u); }
  GroupMemoryBarrierWithGroupSync();
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    float v = raw[base + i].w;
    if (v > 0.0) { hist[bin(v)].add(1u); }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    uint total = 0u;
    int b = 63;
    while (true) {
      uint h = hist[b].load();
      if (total + h >= P.cap || b == 0) { break; }
      total += h;
      b--;
    }
    cut = uint(b);
    above = total;
  }
  // cut and above to every lane: a barrier, the load, a barrier
  GroupMemoryBarrierWithGroupSync();
  uint t = cut;
  uint a0 = above;
  GroupMemoryBarrierWithGroupSync();
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    float v = raw[base + i].w;
    if (v > 0.0 && bin(v) == t) { hist2[place(v, t)].add(1u); }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) {
    // above < cap by the first cut, so room > 0
    uint room = P.cap - a0, total = 0u;
    int s = 255;
    while (true) {
      uint h = hist2[s].load();
      if (total + h >= room || s == 0) { break; }
      total += h;
      s--;
    }
    cut2 = uint(s);
    above2 = a0 + total;
    // the second cut's own place overflows what is left: its peaks taken in the slots' order
    tied = total + hist2[s].load() > room ? 1u : 0u;
  }
  UNIFORM_LOAD(uint tie, tied)
  uint s = cut2;
  uint a1 = above2;
  if (tie == 0u) {
    // everything from the second cut on fits: the cut bin's in by their own count, which the order cannot change
    for (uint i = P.p0 + li; i < P.slots; i += 256u) {
      float4 e = raw[base + i];
      if (e.w > 0.0) {
        uint bb = bin(e.w);
        if (bb > t) { peaks[f * P.cap + used.add(1u)] = e; }
        else if (bb == t && place(e.w, t) >= s) { peaks[f * P.cap + a0 + used2.add(1u)] = e; }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) { counts[f * 16u + 2u] = min(a1 + hist2[s].load(), P.cap); }
    return;
  }
  // the second cut's own place in the slots' order: each lane's run of slots, its count, the counts' scan
  uint per = (P.slots - P.p0 + 255u) / 256u;
  uint lo = min(P.p0 + li * per, P.slots), hi = min(lo + per, P.slots);
  uint mine = 0u;
  for (uint i = lo; i < hi; i++) {
    float v = raw[base + i].w;
    if (v > 0.0 && bin(v) == t && place(v, t) == s) { mine++; }
  }
  part[li] = mine;
  GroupMemoryBarrierWithGroupSync();
  for (uint d = 1u; d < 256u; d <<= 1u) {
    uint add = li >= d ? part[li - d] : 0u;
    GroupMemoryBarrierWithGroupSync();
    part[li] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  uint at = a1 + part[li] - mine;
  // One pass: above the first cut into the list's first 'above' places, the cut bin above the second cut into the
  // next (both fewer than the cap by construction), the second cut's own place after them while the cap has room.
  for (uint i = lo; i < hi; i++) {
    float4 e = raw[base + i];
    if (e.w > 0.0) {
      uint bb = bin(e.w);
      if (bb > t) { peaks[f * P.cap + used.add(1u)] = e; }
      else if (bb == t) {
        uint pp = place(e.w, t);
        if (pp > s) { peaks[f * P.cap + a0 + used2.add(1u)] = e; }
        else if (pp == s) { if (at < P.cap) { peaks[f * P.cap + at] = e; } at++; }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (li == 0u) { counts[f * 16u + 2u] = min(a1 + hist2[s].load(), P.cap); }
}
