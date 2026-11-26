package dev.lizard.lib

import java.io.File
import java.io.FileOutputStream
import java.io.FilterOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

// ai: liblizard's C as both apps reach it (2026-11-25; compiled with their shared UI, lizard-ui, whose commonMain takes
// ai: liblizard/kotlin as a source folder): the format's numbers the senders show and paint by (Ladder), and the senders'
// ai: bundle and the receivers' tar reader (Bundle, tarEach, untar), each the C's (lizard.h; api/format.cpp, api/tar.cpp)
// ai: through LibNative, whose methods the app's library registers as it loads (liblizard/api/jni_lib.cpp, from the
// ai: Android app's jni.cpp and the desktop's jni_util.cpp JNI_OnLoad). Nothing here computes what the C does; the app
// ai: loads its library before it reads any of it.

internal object LibNative {
    @JvmStatic external fun loaded(): Boolean
    @JvmStatic external fun consts(): IntArray
    @JvmStatic external fun bundleMake(paths: Array<String>, dirs: BooleanArray, sizes: LongArray, mtimes: DoubleArray, now: Double, info: LongArray): Long
    @JvmStatic external fun bundleName(h: Long): String
    @JvmStatic external fun bundleEntry(h: Long, e: Int, out: LongArray): ByteArray
    @JvmStatic external fun bundleFree(h: Long)
    @JvmStatic external fun untarNew(): Long
    @JvmStatic external fun untarFree(h: Long)
    @JvmStatic external fun untarWant(h: Long): Long
    @JvmStatic external fun untarTake(h: Long, bytes: ByteArray, n: Int): Int
    @JvmStatic external fun untarEntry(h: Long, out: LongArray): String
}

// ai: The format's numbers (lizard.h, read from the C once): a size (a version) is 1 to 67 of 16 sub-channels each
// ai: (LIZARD-16 to -1072), the senders' size unless one is chosen LIZARD-480; a block carries 460 bytes; the rings are
// ai: indices into RINGS (their cells a side: 64, 128, 256), RING_DEFAULT the senders' (the 128); QUIET the margin the
// ai: codec paints round a symbol, modules; span(r) the picture's modules a side (2 x the ring's cells), modules(r) the
// ai: symbol's (the span and the border), its margin not counted.
object Ladder {
    private val c: IntArray by lazy { LibNative.consts() }
    val STEP get() = c[0]
    val VERSION_MIN get() = c[1]
    val VERSION_MAX get() = c[2]
    val PAYLOAD get() = c[3]
    val QUIET get() = c[4]
    val VERSION_DEFAULT get() = c[5]
    val GAP_MODULES get() = c[6]
    val RINGS: IntArray by lazy { c.copyOfRange(9, 9 + c[7]) }
    val RING_DEFAULT get() = c[8]
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    fun modules(ring: Int = RING_DEFAULT) = c[9 + c[7] + ring]
    internal val TAR_END get() = c[9 + 2 * c[7]]   // ai: the archive's end, its two zero blocks' bytes
}

// ai: a bundle (several files or a folder sent as one tar, Bundle's <name>.tar) told by its name: the description carries
// ai: no media type, so a received file is unpacked as a tar where its name ends .tar
fun isTar(name: String) = name.endsWith(".tar", ignoreCase = true)

/**
 * An output stream that stops at its next write once `live` says no (a make the app let go, 2026-11-16: a tar or a
 * copy into the cache, under the buffer, so it is asked once a buffer), with [Stopped].
 */
class Stoppable(out: OutputStream, private val live: () -> Boolean) : FilterOutputStream(out) {
    override fun write(b: Int) { check(); out.write(b) }
    override fun write(b: ByteArray, off: Int, len: Int) { check(); out.write(b, off, len) }
    private fun check() { if (!live()) throw Stopped() }
}
class Stopped : IOException("cancelled")

/** One item of a choice: a folder (no `open`), or a file of `size` bytes that `open` reads, at a relative path. */
class TarEntry(val path: String, val size: Long = 0, val mtimeSec: Long? = null, val open: (() -> InputStream)? = null) {
    val dir get() = open == null
}

/**
 * A choice of files and folders as one transfer, the C's (lizard.h liz_bundle_*: one file alone is itself, anything else
 * one tar named after its one folder or its first item and how many more). The archive is laid out by the C and
 * written here: each entry's head, its file's bytes read from `open`, its padding, the end.
 */
class Bundle(choice: List<TarEntry>) {
    private val items = choice.toList()
    private val now = (System.currentTimeMillis() / 1000).toDouble()
    val single: TarEntry?
    val name: String
    val files: Int
    /** The archive's bytes, or the one file's. */
    val size: Long

    init {
        require(items.isNotEmpty()) { "nothing chosen" }
        val info = LongArray(4)
        val h = make(info)
        try { name = LibNative.bundleName(h) } finally { LibNative.bundleFree(h) }
        size = info[0]
        single = if (info[1] >= 0) items[info[1].toInt()] else null
        files = info[2].toInt()
    }

    private fun make(info: LongArray): Long = LibNative.bundleMake(Array(items.size) { items[it].path }, BooleanArray(items.size) { items[it].dir },
        LongArray(items.size) { if (items[it].dir) 0 else items[it].size }, DoubleArray(items.size) { items[it].mtimeSec?.toDouble() ?: Double.NaN }, now, info)

    /** The archive into `out` (never called for a single file, which is sent as itself); its bytes. */
    fun write(out: OutputStream): Long {
        check(single == null) { "one file is sent as itself" }
        val info = LongArray(4)
        val h = make(info)
        var written = 0L
        try {
            val at = LongArray(3)
            val buf = ByteArray(1 shl 20)
            for (k in 0 until info[3].toInt()) {
                val head = LibNative.bundleEntry(h, k, at)
                out.write(head); written += head.size
                val item = at[1].toInt()
                if (item >= 0) {
                    val e = items[item]
                    e.open!!().use { src ->
                        var left = at[0]
                        while (left > 0) {
                            val n = src.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                            if (n < 0) throw IOException("${e.path} ended $left bytes short of its ${at[0]}")
                            out.write(buf, 0, n); written += n; left -= n
                        }
                        if (src.read() >= 0) throw IOException("${e.path} is longer than its ${at[0]} bytes (it changed while bundled)")
                    }
                }
                if (at[2] > 0) { out.write(ByteArray(at[2].toInt())); written += at[2] }
            }
            val end = ByteArray(Ladder.TAR_END)
            out.write(end); written += end.size
            out.flush()
        } finally {
            LibNative.bundleFree(h)
        }
        return written
    }

}

/**
 * A tar read entry by entry, the C's reader (lizard.h liz_untar_*): each plain file and folder handed to
 * `each(parts, dir, size, data)`, `parts` its path's parts, `data` the entry's bytes, exactly `size` of them, read or not
 * by `each` (what is left is skipped, by the stream's own skip where it seeks). Anything the C refuses (a link, a device,
 * a path out of the archive, a damaged header, a tar cut short) throws its IOException.
 */
fun tarEach(src: InputStream, each: (parts: List<String>, dir: Boolean, size: Long, data: InputStream) -> Unit) {
    val u = LibNative.untarNew()
    try {
        var buf = ByteArray(512)
        val at = LongArray(2)
        while (true) {
            val want = LibNative.untarWant(u).toInt()
            if (want > buf.size) buf = ByteArray(want)
            when (LibNative.untarTake(u, buf, readUpTo(src, buf, want))) {
                0 -> return
                2 -> {
                    val path = LibNative.untarEntry(u, at)
                    val body = Body(src, at[1])
                    each(path.split('/'), at[0] != 0L, at[1], body)
                    body.drain()
                }
            }
        }
    } finally {
        LibNative.untarFree(u)
    }
}

// ai: up to n bytes into b, fewer only where the stream ends
private fun readUpTo(src: InputStream, b: ByteArray, n: Int): Int {
    var off = 0
    while (off < n) { val k = src.read(b, off, n - off); if (k < 0) break; off += k }
    return off
}

// ai: An entry's bytes, `size` of them off the tar's stream: reads stop there, drain skips the rest (a cut short throws)
private class Body(private val src: InputStream, private var left: Long) : InputStream() {
    override fun read(): Int {
        if (left <= 0) return -1
        val b = src.read(); if (b < 0) throw IOException("a tar cut short"); left--; return b
    }
    override fun read(b: ByteArray, off: Int, len: Int): Int {
        if (left <= 0) return -1
        val n = src.read(b, off, minOf(len.toLong(), left).toInt()); if (n < 0) throw IOException("a tar cut short"); left -= n; return n
    }
    // ai: the last byte read, not skipped: a file's skip may go past its end (FileInputStream's), so a tar cut short
    // ai: inside an entry is found here and not taken for its end
    fun drain() {
        val buf = ByteArray(1 shl 16)
        while (left > 1) {
            val k = src.skip(left - 1)
            if (k > 0) { left -= k; continue }
            val n = src.read(buf, 0, minOf(buf.size.toLong(), left - 1).toInt()); if (n < 0) throw IOException("a tar cut short"); left -= n
        }
        if (left == 1L) { if (src.read() < 0) throw IOException("a tar cut short"); left = 0 }
    }
}

/**
 * A tar, unpacked under `dest` (made if missing), over `tarEach`: plain files and folders made; anything `tarEach`
 * refuses throws, leaving what was written so far for the caller to delete. Returns the files made, in archive order.
 */
fun untar(src: InputStream, dest: File): List<File> {
    val made = ArrayList<File>()
    val root = dest.canonicalFile.also { it.mkdirs() }
    tarEach(src) { parts, dir, size, data ->
        val target = File(root, parts.joinToString(File.separator)).canonicalFile
        if (!target.path.startsWith(root.path + File.separator)) throw IOException("tar path refused: ${parts.joinToString("/")}")
        if (dir) target.mkdirs()
        else {
            target.parentFile?.mkdirs()
            FileOutputStream(target).use { out ->
                val buf = ByteArray(1 shl 20); var left = size
                while (left > 0) { val n = data.read(buf, 0, minOf(buf.size.toLong(), left).toInt()); if (n < 0) throw IOException("a tar cut short"); out.write(buf, 0, n); left -= n }
            }
            made.add(target)
        }
    }
    return made
}
