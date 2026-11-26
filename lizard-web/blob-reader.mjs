// ai: A picked file read for the C (2026-11-25; liblizard/bindings/wasm/send.cpp, the web sender's transfer): this worker
// ai: holds the file (a Blob) and the threaded module's shared memory, and answers the module's threads through the
// ai: prep's slot in it, a request at a time: the slice's bytes read with FileReaderSync and put where the request says,
// ai: then the answer's count raised and the waiting thread woken. Between two requests it returns to its event loop (a
// ai: task, MessageChannel's, where the next was already asked), so each slice's buffer is collected: a module thread
// ai: that reads for itself never returns to one, and a 4.5 GB file's reads stood as gigabytes of the renderer's memory.
// ai:   sender's worker to this one: { memory, slot, blob } (the slot's address in the memory)
// ai: The slot (send.cpp Slot): req (int32, the requests made), done (int32, the last answered), off (float64 at +8),
// ai: n (int32 at +16), into (uint32 at +20), got (int32 at +24, -1 a failure).
const tick = new MessageChannel(), ticks = [];
tick.port1.onmessage = () => ticks.shift()();
const yieldTask = () => new Promise((r) => { ticks.push(r); tick.port2.postMessage(0); });

onmessage = async ({ data: { memory, slot, blob } }) => {
  const at = slot >> 2, reader = new FileReaderSync();
  for (let seen = 0;;) {
    const i32 = new Int32Array(memory.buffer), r = Atomics.load(i32, at);
    if (r === seen) { await Atomics.waitAsync(i32, at, seen).value; continue; }
    seen = r;
    const off = new Float64Array(memory.buffer)[(slot + 8) >> 3], n = i32[at + 4], into = new Uint32Array(memory.buffer)[at + 5];
    let got = -1;
    try {
      const ab = reader.readAsArrayBuffer(blob.slice(off, off + n));
      if (ab.byteLength === n) { new Uint8Array(memory.buffer).set(new Uint8Array(ab), into); got = n; }
      ab.transfer(0);   // ai: its bytes let go now, not at the collector's next turn
    } catch {}
    const j32 = new Int32Array(memory.buffer);
    j32[at + 6] = got;
    Atomics.store(j32, at + 1, r);
    Atomics.notify(j32, at + 1);
    await yieldTask();
  }
};
