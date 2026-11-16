// RaptorQ (RFC 6330), the transfer's fountain, for end-to-end runs: liblizard/src/rq.c inside the codec's module
// (liblizard/build.sh), the instance sim/ob.mjs's default build shares. A chunk is one source block of 471-byte symbols;
// a symbol's ESI is its id's 24 bits.
import { baseModule } from "./ob.mjs";

let M;
export async function init() {
  if (!M) M = await baseModule();
  return M;
}

export class Encoder {
  constructor(message, blockBytes) {
    const p = M._malloc(Math.max(1, message.length));
    M.HEAPU8.set(message, p);
    this.h = M._rq_enc_new(p, message.length, blockBytes);
    M._free(p);
    if (!this.h) throw new Error(`RaptorQ encoder: ${message.length} B`);
    this.blockBytes = blockBytes; this.buf = M._malloc(blockBytes);
    this.blocks = Math.ceil(message.length / blockBytes);
  }
  // Symbol `id`, blockBytes long (a source symbol past the message zero padded).
  block(id, out = new Uint8Array(this.blockBytes)) {
    M._rq_enc_symbol(this.h, id, this.buf);
    out.set(M.HEAPU8.subarray(this.buf, this.buf + this.blockBytes));
    return out;
  }
  free() { M._rq_enc_free(this.h); M._free(this.buf); }
}

export class Decoder {
  // ESIs below 2^24, the transfer's symbol field (RaptorQ's whole ESI).
  constructor(messageBytes, blockBytes) {
    this.h = M._rq_dec_new(messageBytes, blockBytes, 1 << 24);
    if (!this.h) throw new Error(`RaptorQ decoder: ${messageBytes} B`);
    Object.assign(this, { messageBytes, blockBytes, buf: M._malloc(blockBytes) });
  }
  // true once the block is whole, false for more. It takes every symbol, never refusing one.
  add(id, bytes) {
    M.HEAPU8.set(bytes.subarray(0, this.blockBytes), this.buf);
    const rc = M._rq_dec_add(this.h, id, this.buf);
    if (rc < 0) throw new Error(`RaptorQ decode: symbol ${id}`);
    return rc === 1;
  }
  recover() {
    const p = M._malloc(this.messageBytes);
    if (M._rq_dec_recover(this.h, p)) { M._free(p); throw new Error("RaptorQ recover before the block was whole"); }
    const out = M.HEAPU8.slice(p, p + this.messageBytes); M._free(p);
    return out;
  }
  free() { M._rq_dec_free(this.h); M._free(this.buf); }
}
