// ai: A worker of the sender's pool (xfer-pool.mjs, 2026-11-06): hashes the source blocks it is handed (their chaining values,
// ai: the pool's first pass), makes each source block it is handed after (sim/xfer.mjs makeSourceBlock: its fountain), keeps the
// ai: fountain and answers the blocks a frame asks of it; a trial source block for the pool's timing of the cores is made and
// ai: let go. One message at a time, in order (the pool waits on its answers in the order it asked).
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { makeSourceBlock, sbCv, PAYLOAD, idSbn, idEsi } from "../liblizard/sim/xfer.mjs";

const enc = new Map();
let M = null, chain = Promise.resolve();

async function handle(m) {
  if (!M && m.type !== "blocks") [M] = await Promise.all([initOb(), initFountain()]);
  if (m.type === "cv") postMessage({ cv: sbCv(M, new Uint8Array(m.buf), m.c, m.sbLog2) });
  else if (m.type === "make") {
    const k = makeSourceBlock(new Uint8Array(m.buf), m.c, m.sbLog2, false, { M, Encoder });
    if (k.enc) enc.set(m.c, k.enc);
    postMessage({ K: k.K });
  } else if (m.type === "trial") {
    // ai: the pool's timing of the cores (xfer-pool.mjs): a source block made and let go
    makeSourceBlock(new Uint8Array(m.buf), 0, m.sbLog2, true, { M, Encoder }).enc.free();
    postMessage({});
  } else if (m.type === "blocks") {
    const out = new Uint8Array(m.ids.length * PAYLOAD);
    m.ids.forEach((id, i) => enc.get(idSbn(id)).block(idEsi(id), out.subarray(i * PAYLOAD, (i + 1) * PAYLOAD)));
    postMessage({ buf: out.buffer }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ error: String(e?.message ?? e) }));
};
