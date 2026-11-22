// One shape for every code under test, so the harness cannot treat them differently.
//
//   const phy = await makePhy(spec)
//   phy.frame(seq, baseId)         -> { w, h, quiet, dark | drive, truth }   what the screen shows
//     baseId is the first block id in the frame, defaulting to seq * blocksPerFrame. A live
//     sender passes its own monotonic counter instead: blocksPerFrame moves when the layout is
//     re-picked mid-stream, and an id derived from it would walk backwards onto ids already sent.
//   phy.decode(img, iw, ih, truth, rgba, axis) -> one result per phy.variants entry (rgba: img is RGBA, not luma;
//        axis: [x, y], where the camera's optical axis falls in the image, null for its centre):
//        { fresh, seen, bad, found, ber, mi, ms: { detect, sample, fec, total } }
//
// A variant is a decoder setting applied to the same capture, which is how ablations stay
// paired. `fresh` counts fountain blocks not seen before, `seen` every valid block in the
// capture. usefulBytes is what one block hands the fountain: headers and ids are not payload.
import { init as initOb, Lizard, LizardAny, prof, streamBlock, streamMatches } from "./ob.mjs";
import { isControlId, PAYLOAD, ID_BYTES, idGet, toiGet, idPut } from "./xfer.mjs";
import { PICTURE_SIZES } from "./lizard_pick.mjs";

// What the sender paints outside a LIZARD frame, in SAMPLES: the codec's margin (fc.quiet modules, src/lizard_phy.h
// LIZARD_QUIET), or spec.quiet modules where an experiment overrides it. Specs give modules; the painters take samples,
// and until 2026-09-23 this file handed the margin over unconverted, so the rig painted half a module where the
// harnesses painted two.
const marginSamples = (spec, fc) => (spec.quiet ?? fc.quiet) * fc.cell;

// ai: LDPC rate indices to names (src/ldpc.h): the format's four
const RATE = { 2: "1/2", 3: "2/3", 4: "3/4", 6: "6/7" };

// Block contents are a function of the block id, so a receiver can check any block it decodes
// without being told which frame it came from.
//
// BLAKE3 of the id (sim/ob.mjs, src/xfer.h stream_fill; SPEC 9.3; SHAKE256 until 2026-11-15), the one test stream: what a live sender paints and what the
// simulator's harnesses paint and check. The xorshift32 walk it replaced had a single orbit of 2^32 - 1, so two ids
// whose seeds landed within a block of each other on it emitted byte strings that were shifts of one another (119 such
// pairs in 40k ids), and one id seeded the absorbing zero state; neither happens under a hash. The walk went on
// 2026-10-29 with the last thing that read it.
export function blockBytesStream(id, len, out = new Uint8Array(len)) {
  out.set(streamBlock(id, len));
  return out;
}

// The spec a RECORDING was made with, fit to replay: every recording kept names its span, mode 1 and the bit map
// (src/lizard_phy.h LIZARD_BITMAP), so it replays as written (the rules for the specs of 2026-09-23, before the map and the
// spans were written down, went with those recordings).
export const recordedSpec = (spec) => spec;

// ai: The largest picture a receiver builds: the ladder's top (1536, LIZARD-896 to -1024), so it reads any version the
// ai: word names, with no flag (2026-09-24; 1024 until 2026-09-29, the receiver's #sizes menu choosing 1536, the menu
// ai: deleted that day since the ring's word names the size). A blind C worker builds a picture's
// ai: codec only when a word names it (src/wasm.c pic_for, a few ms since the same day's counting sort).
export const BLIND_NMAX = PICTURE_SIZES.at(-1);
// ai: The blind spec, the one a receiver decodes LIZARD with: any ring, any picture to nmax, told nothing.
export const blindSpec = (nmax = BLIND_NMAX) => ({ phy: "lizard", blind: 1, nmax });
// ai: A recording read blind, as a receiver would have read it; stream is the recording's generator, for blockJudge.
// ai: Only a recording on today's border reads: research/captures v0.1 and v0.2 were painted before the three rings
// ai: (2026-09-27), and scripts/exp/recordings.mjs lists the current version's runs alone.
export const recordedBlind = (spec) => ({ ...blindSpec(Math.max(BLIND_NMAX, spec.n ?? 0)), stream: spec.stream });

// ai: name -> generator: BLAKE3 alone (SPEC 9.3), which a blind receiver judges from the light alone (blockJudge); a
// ai: spec that names no stream gets it.
export const STREAMS = { blake3: blockBytesStream };
// ai: name -> the judge's test of a payload against its id's stream (STREAMS' bytes, compared as they are made)
export const MATCHERS = { blake3: streamMatches };
export const sourceFor = (stream) => {
  const gen = STREAMS[stream ?? "blake3"];
  if (!gen) throw new Error(`unknown test stream ${stream}`);
  return (id, out) => gen(id, out.length, out);
};

// ai: A blind receiver's judge of a frame's verified blocks, laid out as the codec lays them (the 4-byte id, then the
// ai: payload), from the light alone (the whole receiver blind since 2026-09-26). The test stream's payload
// ai: is a function of its id (BLAKE3 of it, SPEC 9.3), so a frame is the test stream's when one of its verified data
// ai: blocks carries its id's bytes (a file's block does so with probability 2^-3752); every other verified data block
// ai: of that frame is then bad (the CRC let a wrong block through), counted and dropped. A frame with none is a file's,
// ai: or a test frame whose every verified block is wrong, the one case not counted: the CRC alone judges it.
// ai: frame(ok, blocks): ok[k] set for each verified block k, at k * (payload + ID_BYTES). Returns got, the frame's every block
// ai: not judged bad, control blocks (a transfer's session header) included, its bare id, its file's TOI and its bytes; seen (got's count),
// ai: bad; test, whether the frame was judged the test stream's; judged, its data blocks held to the stream.
// ai: The judge keeps nothing from frame to frame (2026-09-29): which blocks are new is the receiver's page's to say, the
// ai: one set of ids that a new transfer or Clear lets go (lizard-web/recv.mjs). Until then each decode worker's judge kept the
// ai: ids it had reported and sent only new ones, a set nothing let go, so after Clear, or at a second file in one
// ai: session (whose ids start again from 0), the blocks the workers had already reported never reached the page again.
// ai: stream: the generator (sim/phy.mjs STREAMS), a replay's recording naming its own.
export function blockJudge(payload = PAYLOAD, stream = "blake3") {
  const B = payload + ID_BYTES, match = MATCHERS[stream ?? "blake3"];
  if (!match) throw new Error(`unknown test stream ${stream}`);
  return {
    frame(ok, blocks) {
      let test = false;
      const got = [], data = [];
      let bad = 0;
      for (let k = 0; k < ok.length; k++) {
        if (!ok[k]) continue;
        const o = k * B, id = idGet(blocks, o), toi = toiGet(blocks, o), bytes = blocks.slice(o + ID_BYTES, o + B);
        if (isControlId(id)) { got.push({ id, toi, bytes }); continue; }
        const same = match(id, bytes);
        if (same) test = true;
        data.push({ id, toi, bytes, same });
      }
      for (const d of data) if (test && !d.same) bad++; else got.push({ id: d.id, toi: d.toi, bytes: d.bytes });
      return { got, seen: got.length, bad, test, judged: test ? data.length : 0 };
    },
  };
}

// ai: The blind receiver in the C (sim/ob.mjs LizardAny): it registers against every ring, reads the word there and
// ai: decodes the picture the word names, to spec.nmax. A frame whose word does not read is not decoded (2026-11-18: the
// ai: word before the frame); nothing carries from one frame to the next. Blocks are judged from the light
// ai: (blockJudge); spec.stream names a replay's generator, BLAKE3 when left out.
async function makeBlind(spec) {
  await initOb();
  const nmax = spec.nmax ?? BLIND_NMAX, fa = new LizardAny(nmax);
  const judge = blockJudge(fa.blockBytes - ID_BYTES, spec.stream ?? "blake3");
  let builds = 0;
  return {
    label: `lizard blind to ${nmax}`, family: "lizard", variants: ["blind"], usefulBytes: fa.blockBytes - ID_BYTES, nmax,
    decode(img, iw, ih, truth, rgba = false, axis = null) {
      const t = performance.now(), r = fa.decode(img, iw, ih, { rgba, axis }), total = performance.now() - t;
      // ai: built: this frame built a picture's codec (its first sight of that ring and picture; src/wasm.c pic_for).
      const built = r.builds > builds;
      builds = r.builds;
      return [{ ...judge.frame(r.ok, r.blocks), found: r.found, n: r.pictureN, ring: r.ring, fmt: r.fmt, quad: r.quad, built, ber: NaN, mi: NaN, ms: { detect: r.msDetect, total } }];
    },
    reset() {},
    prof, free() { fa.free(); },
  };
}
// variants: [{ name, gamma }]. Each block starts with its 12-byte id (src/xfer.h: the bare id, the source block and symbol, then the 48-bit TOI).
async function makeLizard(spec) {
  await initOb();
  const { n = 512, subch = 32, mode = 1, span = 0, tiers = null, bitmap, variants = [{ name: "ldpc" }] } = spec;
  const fc = new Lizard(n, subch, mode, { span, tiers, bitmap }), B = fc.blockBytes, T = fc.blocks, seen = variants.map(() => new Set()), buf = new Uint8Array(T * B), want = new Uint8Array(B - ID_BYTES);
  const quiet = marginSamples(spec, fc);
  // The page's pixels come from the codec's own paint, which has only its margin: an experiment's override has no way there.
  const noRGBA = spec.quiet !== undefined && spec.quiet !== fc.quiet ? `spec.quiet ${spec.quiet} is the simulator's override; the codec paints ${fc.quiet}` : null;
  // ai: TOI: every block's (src/xfer.h); the test stream's spec.toi (1 unless given: the format has no test stream, TOI 0
  // ai: is none, so it goes under a TOI as any send's; a live sender draws one, lizard-web/send-worker.mjs), or a sender's
  // ai: (sim/xfer.mjs XferSender: toiFor(id) a block's), setSource's
  const testToi = Math.floor(spec.toi ?? 1) % 2 ** 48 || 1;   // ai: the test stream's TOI, 48 bits
  let source = sourceFor(spec.stream), toi = testToi, ink = 0x000000, paper = 0xffffff;
  const toiOfId = (id) => (typeof toi === "number" ? toi : toi.toiFor(id));
  return {
    label: `lizard ${n} ${tiers ? tiers.map(([r, b]) => `${b}x${RATE[r]}`).join("+") : `${subch}sub`}${span ? " span" + span : ""}`, family: "lizard", variants: variants.map((v) => v.name), usefulBytes: B - ID_BYTES, blocksPerFrame: T, modules: fc.side, bitmap: fc.bitmap,
    // ai: the paint's samples a module and the codec's margin in modules (src/lizard_phy.h LIZARD_QUIET), for a painter
    // ai: that places symbols in modules (lizard-web/send-worker.mjs: two codes a gap apart, 2026-09-30)
    cell: fc.cell, quiet: fc.quiet,
    // baseId may instead be the frame's T ids, for a sender whose ids are not a run (sim/xfer.mjs: source block and symbol).
    frame(seq, baseId = seq * T) {
      for (let t = 0; t < T; t++) { const id = typeof baseId === "number" ? baseId + t : baseId[t], o = t * B; idPut(buf, o, id, toiOfId(id)); source(id, buf.subarray(o + ID_BYTES, o + B)); }
      fc.setParity(seq & 3);
      return { w: fc.side, h: fc.side, quiet, drive: fc.encode(buf), truth: null };
    },
    // The same frame as the pixels a page paints, margin included (lizard-web/send-worker.mjs): { w, rgba }, the rgba a view
    // into the wasm heap that the next call overwrites. ai: seq is the painted count, so seq mod 4 is the pilots'
    // (SPEC 7.3), as frame() above paints them and the senders do (gpu/encoder.mjs, core/tx/sender.cpp).
    frameRGBA(seq, baseId = seq * T) {
      if (noRGBA) throw new Error(noRGBA);
      fc.setParity(seq & 3);
      return fc.encodeRGBA(this.frameBlocks(seq, baseId), ink, paper);
    },
    // ai: the code's colours frameRGBA paints in (0xRRGGBB; 2026-11-16, Send's Colours): its ink and its paper
    setColours(i, p) { ink = i & 0xffffff; paper = p & 0xffffff; },
    // ai: the sender's clock the word states (src/lizard_phy.h lizard_clock): fps 0 to 240 (0 a still), delay 1 to 30
    setClock(fps, delay = 1) { fc.setClock(fps, delay); },
    // ai: The frame's T blocks alone, each its id (little-endian) then its payload, B bytes a block, for an encoder
    // ai: elsewhere (the sender's GPU encoder, gpu/encoder.mjs). A view reused by the next call.
    frameBlocks(seq, baseId = seq * T) {
      for (let t = 0; t < T; t++) { const id = typeof baseId === "number" ? baseId + t : baseId[t], o = t * B; idPut(buf, o, id, toiOfId(id)); source(id, buf.subarray(o + ID_BYTES, o + B)); }
      return buf;
    },
    // One decoder result turned into the harness's, shared by the one-call path and the split one so the two
    // cannot report differently.
    tally(r, vi, total) {
      const got = [];
      let fresh = 0, n2 = 0, bad = 0;
      for (let k = 0; k < T; k++) {
        if (!r.ok[k]) continue;
        const o = k * B, id = idGet(r.blocks, o);
        // A transfer's session header (sim/xfer.mjs) is no test stream's, so it is not compared: its CRC is
        // the judge, as for a file. Not fresh, since it is not payload, and handed on every time it is read, since its
        // id is the same in every transfer and only its bytes say whether it is a new one.
        if (isControlId(id)) { n2++; got.push({ id, bytes: r.blocks.slice(o + ID_BYTES, o + B) }); continue; }
        // With no source to compare against (the receiving end of a file), the block's own CRC,
        // checked in the decoder, is the only judge.
        let same = true;
        if (source) { source(id, want); for (let q = 0; q < B - ID_BYTES; q++) if (r.blocks[o + ID_BYTES + q] !== want[q]) { same = false; break; } }
        if (!same) { bad++; continue; }
        n2++;
        if (!seen[vi].has(id)) { seen[vi].add(id); fresh++; got.push({ id, bytes: r.blocks.slice(o + ID_BYTES, o + B) }); }
      }
      return { got, fresh, seen: n2, bad, found: r.found, fmt: r.fmt, quad: r.quad, mark: r.mark, ber: NaN, mi: NaN, ms: { detect: r.msDetect, sample: r.msSample, fec: r.msDecode, total } };
    },
    decode(img, iw, ih, truth, rgba = false, axis = null) {
      return variants.map((v, vi) => {
        const t = performance.now(), r = fc.decode(img, iw, ih, { ...v, rgba, axis }), total = performance.now() - t;
        return this.tally(r, vi, total);
      });
    },
    reset() { seen.forEach((s) => s.clear()); },
    seenCount: () => seen.reduce((t, x) => t + x.size, 0),
    // ai: a transfer's blocks (fn(id, out) its payloads) under a TOI: a number (every block's), or the sender itself
    // ai (each block under its own TOI, the file's or its session header's: sim/xfer.mjs XferSender toiFor, 2026-11-20)
    setSource(fn, t = testToi) { source = fn; toi = typeof t === "number" ? t >>> 0 : t; },
    prof, free() { fc.free(); },
  };
}

export async function makePhy(spec) {
  if (spec.phy === "lizard") return spec.blind ? makeBlind(spec) : makeLizard(spec);
  throw new Error(`unknown phy ${spec.phy}`);
}
