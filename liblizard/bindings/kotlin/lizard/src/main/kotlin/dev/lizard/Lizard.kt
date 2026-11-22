// ai: liblizard's Kotlin binding (2026-10-03): the C API (include/lizard.h) through JNI (api/jni.cpp), one plain
// ai: Kotlin/JVM library for desktops and Android alike. The same layers: format arithmetic (Lizard), the per-frame
// ai: codec (Encoder, Decoder), the transfer (Tx, Rx), two conveniences over them on the caller's thread, FrameSender
// ai: (a file to painted frames) and FrameReceiver (camera frames to a file), and the engines (Engines.kt: Receiver and
// ai: Sender, on threads of their own, the GPU where it runs). Handles are AutoCloseable: close() each (use { }). A
// ai: handle is used from one thread at a time. A failed call throws IllegalArgumentException, IllegalStateException,
// ai: UnsupportedOperationException, OutOfMemoryError or LizardException.
package dev.lizard

import java.nio.ByteBuffer

class LizardException(val code: Int, message: String) : RuntimeException(message)

enum class PixelFormat(internal val code: Int, val bytes: Int) { GREY(1, 1), RGBX(2, 4), RGBA(3, 4), BGRA(4, 4) }

// ai: blocks, the format's size (4 to 64: LIZARD-64 to -1024, 16 x blocks; a symbol carries the rate profile's blocks, the
// ai: geometry's frameBlocks over codes); ring 0 to RINGS - 1 (the 64, 128 or 256 ring) or RING_DEFAULT; the
// ai: display rate its word states; codes, 1 or 2 symbols side by side
data class Format(val blocks: Int, val ring: Int = Lizard.RING_DEFAULT, val fps: Int = 60, val codes: Int = 1)

data class Geometry(val n: Int, val span: Int, val pxm: Int, val side: Int, val width: Int, val height: Int, val gap: Int, val frameBlocks: Int)

data class Rect(val x: Int, val y: Int, val w: Int, val h: Int)

// ai: where a block of a symbol lies (lizard.h liz_block_place): its code rate (0 6/7, 1 3/4, 2 2/3, 3 1/2; the 6/7 and
// ai: 3/4 blocks carry the pilots), its first slot and slots (640 a sub-channel), the sub-channels it spans
data class BlockPlace(val rate: Int, val firstSlot: Int, val slots: Int, val firstSubch: Int, val lastSubch: Int)

object Lizard {
  const val BLOCK = 472
  const val ID_BYTES = 12
  const val PAYLOAD = 460
  const val MIN_BLOCKS = 4
  const val MAX_BLOCKS = 64
  const val RINGS = 3
  const val RING_DEFAULT = -1
  const val GAP_MODULES = 12

  val version: String get() = Native.version()
  // ai: whether the codec's vector paths (NEON) are in this build
  val simd: Boolean get() = Native.simd()

  fun ringCells(ring: Int = RING_DEFAULT): Int = Native.ringCells(ring)
  fun geometry(f: Format): Geometry = IntArray(8).also { Native.geometry(f.blocks, f.ring, f.fps, f.codes, it) }.toGeometry()
  // ai: the display pixels a side a symbol of this many blocks needs (the senders' room)
  fun roomFor(blocks: Int, ring: Int = RING_DEFAULT): Double = Native.roomFor(blocks, ring)
  // ai: the largest size (Format.blocks) a symbol may have in a w x h room of display pixels (the senders' pick)
  fun pick(w: Double, h: Double, codes: Int = 1, ring: Int = RING_DEFAULT, top: Int = MAX_BLOCKS): Int = Native.pick(w, h, codes, ring, top)
  // ai: where a camera frame's symbols are looked for: layout 1 the centre square, 2 the two squares of a 2:1 region
  fun layoutRects(w: Int, h: Int, layout: Int = 1): List<Rect> {
    val r = IntArray(8)
    val n = Native.layoutRects(w, h, layout, r)
    return List(n) { Rect(r[4 * it], r[4 * it + 1], r[4 * it + 2], r[4 * it + 3]) }
  }
  // ai: block b of a symbol of a format's size (Format.blocks)
  fun blockPlace(blocks: Int, b: Int): BlockPlace = IntArray(5).also { Native.blockPlace(blocks, b, it) }.let { BlockPlace(it[0], it[1], it[2], it[3], it[4]) }
  const val REFRESH_MAX = 30
  // ai: the test stream's payload for an id
  fun streamFill(id: Long): ByteArray = ByteArray(PAYLOAD).also { Native.streamFill(id, it) }
}

internal fun IntArray.toGeometry() = Geometry(this[0], this[1], this[2], this[3], this[4], this[5], this[6], this[7])

abstract class Handle internal constructor(protected var h: Long, private val release: (Long) -> Unit) : AutoCloseable {
  protected fun live(): Long = if (h != 0L) h else throw IllegalStateException("closed")
  override fun close() { if (h != 0L) { release(h); h = 0L } }
}

class Encoder(format: Format) : Handle(Native.encoderNew(format.blocks, format.ring, format.fps, format.codes), Native::encoderFree) {
  val format = format
  val geometry: Geometry = Lizard.geometry(format)
  // ai: one frame: blocks, frameBlocks x 472 bytes; picture, the frame's count (mod 4 the pilots); out, width x height
  // ai: pixels of fmt (made if not given)
  fun paint(blocks: ByteArray, picture: Int, out: ByteArray? = null, fmt: PixelFormat = PixelFormat.GREY): ByteArray {
    val o = out ?: ByteArray(geometry.width * geometry.height * fmt.bytes)
    Native.encoderPaint(live(), blocks, picture, o, geometry.width * fmt.bytes, fmt.code)
    return o
  }
  // ai: into a direct buffer (a surface's, a texture upload's), rows stride bytes apart
  fun paint(blocks: ByteArray, picture: Int, out: ByteBuffer, stride: Int, fmt: PixelFormat) {
    require(out.isDirect) { "a direct buffer" }
    Native.encoderPaintBuffer(live(), blocks, picture, out, stride, fmt.code)
  }
}

class Decoded(
  val count: Int, val blocks: ByteArray, val found: Boolean, val ring: Int, val n: Int, val word: Boolean,
  val blocksPerSymbol: Int, val total: Int, val quad: FloatArray,
  val pilotSubch: Int, val pilotR: FloatArray, val pilotSd: FloatArray,
)

// ai: Each frame decoded on its own (lizard.h liz_decode): nothing carries from one frame to the next.
class Decoder(nmax: Int = 0) : Handle(Native.decoderNew(nmax), Native::decoderFree) {
  val maxBlocks: Int = Native.decoderMaxBlocks(live())
  private val out = ByteArray(maxBlocks * Lizard.BLOCK)
  private val ints = IntArray(8)
  private val floats = FloatArray(12)

  fun decode(px: ByteArray, w: Int, h: Int, fmt: PixelFormat = PixelFormat.GREY, stride: Int = w * fmt.bytes, offset: Int = 0): Decoded =
    result(Native.decode(live(), px, offset, w, h, stride, fmt.code, out, ints, floats))
  // ai: from a direct buffer (a camera plane), offset bytes in
  fun decode(px: ByteBuffer, w: Int, h: Int, fmt: PixelFormat, stride: Int, offset: Int = 0): Decoded {
    require(px.isDirect) { "a direct buffer" }
    return result(Native.decodeBuffer(live(), px, offset, w, h, stride, fmt.code, out, ints, floats))
  }
  private fun result(n: Int) = Decoded(
    n, out.copyOf(n * Lizard.BLOCK), ints[0] != 0, ints[1], ints[2], ints[3] != 0, ints[4], ints[5],
    floats.copyOfRange(0, 8), ints[7], floats.copyOfRange(8, 10), floats.copyOfRange(10, 12),
  )
}

data class TxInfo(val test: Boolean, val length: Long, val sourceBlocks: Int, val cycle: Int, val root: String)

class Tx private constructor(h: Long) : Handle(h, Native::txFree) {
  companion object {
    // ai: a file's bytes (copied in) and its name as its description carries it
    fun file(bytes: ByteArray, name: String = "") = Tx(Native.txNew(bytes, name))
    fun path(path: String, name: String? = null) = Tx(Native.txNewPath(path, name))
    // ai: the test stream from firstId (0: a random one)
    fun test(firstId: Int = 0) = Tx(Native.txNewTest(firstId))
  }
  // ai: the next frame's n blocks (n x 472 bytes)
  fun next(n: Int, out: ByteArray = ByteArray(n * Lizard.BLOCK)): ByteArray { Native.txNext(live(), n, out); return out }
  val info: TxInfo get() {
    val l = LongArray(4)
    val root = ByteArray(32)
    Native.txInfo(live(), l, root)
    return TxInfo(l[0] != 0L, l[1], l[2].toInt(), l[3].toInt(), root.joinToString("") { "%02x".format(it) })
  }
}

data class Verdict(val seen: Int, val bad: Int, val judged: Int, val fresh: Int, val test: Boolean)
data class Progress(val header: Boolean, val done: Boolean, val sourceBlocks: Int, val verified: Int, val rejected: Int,
                    val length: Long, val bytesIn: Long, val fraction: Double, val solveMs: Double)

// ai: dir null: in memory, at most maxBytes (0: no cap); else files under dir/lizard-xfer (that folder emptied first)
class Rx(dir: String? = null, maxBytes: Long = 0) : Handle(Native.rxNew(dir, maxBytes), Native::rxFree) {
  private val v = IntArray(5)
  fun frame(blocks: ByteArray, count: Int = blocks.size / Lizard.BLOCK): Verdict {
    Native.rxFrame(live(), blocks, count, v)
    return Verdict(v[0], v[1], v[2], v[3], v[4] != 0)
  }
  val progress: Progress get() {
    val i = IntArray(5); val l = LongArray(4); val d = DoubleArray(2)
    Native.rxProgress(live(), i, l, d)
    return Progress(i[0] != 0, i[1] != 0, i[2], i[3], i[4], l[0], l[1], d[0], d[1])
  }
  // ai: each source block: 0 to 99 the share of its source symbols in, 255 verified
  val sourceBlocks: ByteArray get() = Native.rxSourceBlocks(live())
  val name: String get() = Native.rxMeta(live(), 0)
  val root: String get() = Native.rxMeta(live(), 1)
  val path: String get() = Native.rxMeta(live(), 2)
  val error: String get() = Native.rxMeta(live(), 3)
  // ai: the finished file's bytes in a memory store, null before it is done
  fun data(): ByteArray? = Native.rxData(live())
  fun clear() = Native.rxClear(live())
}

// ai: A file (or the test stream) to painted frames, in order, the pilots counted, on the caller's thread.
class FrameSender(private val tx: Tx, format: Format) : AutoCloseable {
  val encoder = Encoder(format)
  val geometry get() = encoder.geometry
  val info get() = tx.info
  var picture = 0
    private set
  class Frame(val data: ByteArray, val width: Int, val height: Int, val picture: Int)
  fun frame(fmt: PixelFormat = PixelFormat.GREY, out: ByteArray? = null): Frame {
    val g = geometry
    val p = picture++
    return Frame(encoder.paint(tx.next(g.frameBlocks), p, out, fmt), g.width, g.height, p)
  }
  override fun close() { tx.close(); encoder.close() }
}

// ai: Camera frames to a file on the caller's thread: each frame's regions (layout 1 or 2) decoded, their blocks into
// ai: the transfer.
class FrameReceiver(val layout: Int = 1, nmax: Int = 0, dir: String? = null, maxBytes: Long = 0) : AutoCloseable {
  val decoder = Decoder(nmax)
  val rx = Rx(dir, maxBytes)
  class Pushed(val decoded: List<Decoded>, val verdict: Verdict, val progress: Progress)
  fun push(px: ByteArray, w: Int, h: Int, fmt: PixelFormat = PixelFormat.GREY, stride: Int = w * fmt.bytes): Pushed {
    val decoded = Lizard.layoutRects(w, h, layout).map { r -> decoder.decode(px, r.w, r.h, fmt, stride, r.y * stride + r.x * fmt.bytes) }
    val all = ByteArray(decoded.sumOf { it.blocks.size })
    var o = 0
    for (d in decoded) { d.blocks.copyInto(all, o); o += d.blocks.size }
    return Pushed(decoded, rx.frame(all), rx.progress)
  }
  class File(val name: String, val root: String, val bytes: ByteArray)
  // ai: the file once whole (a memory store), else null
  val file: File? get() = rx.data()?.let { File(rx.name, rx.root, it) }
  fun clear() = rx.clear()
  override fun close() { decoder.close(); rx.close() }
}
