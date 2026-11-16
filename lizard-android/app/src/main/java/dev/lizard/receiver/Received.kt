package dev.lizard.receiver

import android.content.ContentValues
import android.content.Context
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Log
import android.webkit.MimeTypeMap
import androidx.core.content.FileProvider
import dev.lizard.Session
import dev.lizard.readout.Readout
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.tarEach
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import dev.lizard.ui.*

// ai: The received files (2026-10-31; the app's own library in filesDir/library before, the desktop's Received.kt the
// ai: model): a verified file is taken out of the receiver's store by the receive session into its data folder's kept/
// ai: (liblizard/core/rx/session.h: kept/<id>/ with filing.json and file/<name>; Session.kept lists them), then placed
// ai: here into the phone's Downloads through MediaStore (RELATIVE_PATH Download/, no permission from Android 10) and
// ai: its take let go (Session.placed). A file the default client packed is kept by the session as it came
// ai: (keepPacked, 2026-11-15) and read back here through the native Unpacker as it is placed, one pass from the take's
// ai: file into Downloads, the bytes counted for Receive's Saving row (onSaving). A tar (several files or a folder sent
// ai: as one, Tar.kt) is unpacked into a folder there whole as it is read (one folder alone at its top is the bundle's,
// ai: else the tar's name less .tar; a free name, the folder's files hidden as pending until the last is in); one that
// ai: does not read whole (a packed frame damaged, a tar damaged or cut short) is placed as the file it came as. What
// ai: Downloads does not take stays in the data folder, listed and opened from there (the FileProvider's "session"
// ai: path), and is placed again at the next pass (retry): a file is never only in a store that lets it go. A take is
// ai: placed when its file is in Downloads, made durable, and listed: its row written before its take is let go (a
// ai: copy whose row cannot be written is taken back out of Downloads), so a placing cut short leaves the take, which
// ai: the next pass lets go once its root is listed. The list is <data>/received.json (entries name, type, size, root,
// ai: at, uri for a file, folder and items {path, uri, size} for a bundle, local for one Downloads did not take), newest
// ai: first, written durably whole; an entry whose file MediaStore says is gone (the user deleted it there) is dropped
// ai: from it, but not one MediaStore did not answer for, and a list that does not read is never written over (one that
// ai: does not parse is set aside whole as received.json.bad); the same root placed again replaces its entry. Every
// ai: read and write of the list is under one lock; list and retry query MediaStore, so they run off the main thread.
// ai: The files received before 2026-10-31 (filesDir/library, Library.kt's <id>/<name> and .meta.json) are moved into
// ai: <data>/library at the first start (MainActivity.carryOver) and placed by the same pass, each removed from there
// ai: once it is in Downloads.
class Received(private val ctx: Context, private val data: File) {
    // ai: a bundle's file: its path in the folder, its MediaStore URI (or a local file's), its size
    data class Item(val path: String, val uri: String, val size: Long)
    data class Entry(val name: String, val type: String, val size: Long, val root: String, val at: Long, val uri: String, val folder: String,
                     val items: List<Item>, val local: String) {
        val bundle get() = folder.isNotEmpty() || items.isNotEmpty()
        val files get() = items.size
        // ai: the key a row is kept by: the root, else the file's place
        val id get() = root.ifEmpty { uri.ifEmpty { local } }
        // ai: the header's media type, else the name's extension's, else bytes
        val mime: String get() = type.takeIf { it.isNotEmpty() && it != TAR_TYPE }.orEmpty().ifEmpty { mimeOf(name) }
    }

    private val mu = Any()
    private val listFile get() = File(data, "received.json")
    private val keptDir get() = File(data, "kept")
    private val oldLibrary get() = File(data, "library")
    private val resolver get() = ctx.contentResolver

    // ai: the URI another app opens a file by: MediaStore's for a file in Downloads, the FileProvider's for one kept in
    // ai: the data folder
    fun uriOf(u: String): Uri = if (u.startsWith("content:")) Uri.parse(u) else FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", File(u))

    // ai: the entries whose files are there, newest first; a list that does not read gives none and is left as it is
    fun list(): List<Entry> = synchronized(mu) {
        val all = runCatching { read() }.getOrElse { Log.w(TAG, "received: ${listFile.path}: ${it.message}"); return emptyList() }
        val here = all.filter { present(it) }
        if (here.size != all.size) runCatching { write(here) }.onFailure { Log.w(TAG, "received: ${listFile.path}: ${it.message}") }
        here.sortedByDescending { it.at }
    }

    // ai: whether an entry's file may still be there (a MediaStore row of this app's, or the local file): gone only where
    // ai: MediaStore answers with no row; a query that throws or gives no answer (the media provider restarting) keeps it
    private fun present(e: Entry): Boolean = when {
        e.local.isNotEmpty() -> File(e.local).exists()
        e.uri.isNotEmpty() -> exists(e.uri)
        else -> e.items.any { exists(it.uri) }
    }
    private fun exists(u: String): Boolean = if (!u.startsWith("content:")) File(u).exists() else runCatching {
        resolver.query(Uri.parse(u), arrayOf(MediaStore.MediaColumns._ID), null, null, null)?.use { it.moveToFirst() } ?: true
    }.getOrDefault(true)

    // ai: the list as written: none where there is no list yet; a read that fails throws (nothing is written over the list
    // ai: then); a list that does not parse (a zero-length one a power loss left, before its writes were made durable) is
    // ai: set aside whole beside it as received.json.bad, said in the log, and a new list begun
    private fun read(): List<Entry> {
        val f = listFile
        if (!f.exists()) return emptyList()
        val text = f.readText()
        val a = try { JSONArray(text) } catch (x: JSONException) {
            val bad = File(data, "received.json.bad")
            bad.delete()
            if (!f.renameTo(bad)) throw IOException("${f.path} does not parse and could not be set aside")
            Log.w(TAG, "received: ${f.path} did not parse (${text.length} B): set aside as ${bad.path}")
            return emptyList()
        }
        return (0 until a.length()).mapNotNull { i ->
            val o = a.optJSONObject(i) ?: return@mapNotNull null
            val its = o.optJSONArray("items")
            val items = if (its == null) emptyList() else (0 until its.length()).mapNotNull { k ->
                its.optJSONObject(k)?.let { Item(it.optString("path"), it.optString("uri"), it.optLong("size")) } }
            Entry(o.optString("name"), o.optString("type"), o.optLong("size"), o.optString("root"), o.optLong("at"), o.optString("uri"),
                o.optString("folder"), items, o.optString("local"))
        }
    }

    // ai: written whole beside it, made durable, and renamed over it, so a crash or a power loss leaves the list as it was
    private fun write(es: List<Entry>) {
        val a = JSONArray()
        for (e in es) a.put(JSONObject().put("name", e.name).put("type", e.type).put("size", e.size).put("root", e.root).put("at", e.at)
            .put("uri", e.uri).put("folder", e.folder).put("local", e.local)
            .put("items", JSONArray().apply { for (it in e.items) put(JSONObject().put("path", it.path).put("uri", it.uri).put("size", it.size)) }))
        data.mkdirs()
        val tmp = File(data, "received.json.tmp")
        FileOutputStream(tmp).use { o -> o.write(a.toString().toByteArray()); o.fd.sync() }
        if (!tmp.renameTo(listFile)) throw IOException("${listFile.path} could not be written")
    }

    private fun add(e: Entry) = synchronized(mu) { write(read().filter { (it.root != e.root || e.root.isEmpty()) && !(e.local.isNotEmpty() && it.local == e.local) } + e) }
    // ai: an entry placed into Downloads listed; where its row cannot be written, its files taken back out of Downloads
    // ai: (placed means listed: the take stays, placed again by a later pass)
    private fun listed(e: Entry) {
        try { add(e) } catch (x: Exception) {
            if (e.uri.startsWith("content:")) runCatching { resolver.delete(Uri.parse(e.uri), null, null) }
            for (it in e.items) if (it.uri.startsWith("content:")) runCatching { resolver.delete(Uri.parse(it.uri), null, null) }
            throw x
        }
    }

    // ai: What a placing came to: the root and name, and "" where the file is in Downloads, else why it is not (it is in
    // ai: the data folder then, listed from there)
    class Filed(val root: String, val name: String, val why: String)

    // ai: A placing's progress (2026-11-15, Receive's Saving row): the take's root and Readout's figures (the bytes read
    // ai: of all, the folder under Downloads); null once it is over. Called on the placing's thread, at most every 100 ms.
    @Volatile var onSaving: ((String, Readout.Saving?) -> Unit)? = null

    // ai: a take as the session lists it (Session.kept): packed, the name and type of the file a packed take holds
    private class Take(val id: String, val name: String, val type: String, val root: String, val file: File, val packed: Pair<String, String>?)
    private fun takes(): List<Take> = runCatching {
        val a = JSONArray(Session.kept(data.path))
        (0 until a.length()).mapNotNull { i ->
            val o = a.optJSONObject(i) ?: return@mapNotNull null
            val p = o.optJSONObject("packed")?.let { it.optString("name") to it.optString("type") }?.takeIf { it.first.isNotEmpty() }
            Take(o.optString("id"), o.optString("name"), o.optString("type"), o.optString("root"), File(o.optString("path")), p)
                .takeIf { it.id.isNotEmpty() && it.name.isNotEmpty() }
        }
    }.getOrDefault(emptyList())
    private fun placed(id: String) {
        val why = Session.placed(data.path, id)
        if (why.isNotEmpty()) Log.w(TAG, "received: kept/$id: $why")
    }

    // ai: Every take in kept/ placed, oldest first, then the files received before 2026-10-31 (<data>/library): what each
    // ai: came to. One pass at a time (the caller's one thread, MainActivity's filer).
    fun retry(): List<Filed> {
        val out = ArrayList<Filed>()
        for (k in takes()) out += runCatching { file(k) }.getOrElse { Log.w(TAG, "received: ${k.id} not placed: ${it.message}"); Filed(k.root, k.name, it.message.orEmpty()) }
        for (d in oldLibrary.listFiles().orEmpty().filter { it.isDirectory }) out += runCatching { carried(d) }.getOrElse { Filed("", d.name, it.message.orEmpty()) }
        return out
    }

    // ai: A take placed: a root a list entry holds in Downloads is not placed again (the same file received again, or a
    // ai: placing cut short after its row: its take let go); else into Downloads (place: a packed file read back as it
    // ai: goes, a tar unpacked there whole; one that does not read whole placed as the file it came as), the entry
    // ai: listed, then the take let go (its file with it). Where Downloads does not take it the file stays in the take's
    // ai: folder, listed from there as it came.
    private fun file(k: Take): Filed {
        val name = k.packed?.first ?: k.name
        if (k.root.isNotEmpty() && list().any { it.root == k.root && it.local != k.file.path }) {
            placed(k.id)
            return Filed(k.root, name, "")
        }
        val now = System.currentTimeMillis()
        // ai: a packed file as it came: its type without the original's parameter (application/zstd), which MediaStore
        // ai: and the apps that open it know
        val came = if (k.packed != null) k.type.substringBefore(';').trim() else k.type
        return try {
            val t0 = android.os.SystemClock.elapsedRealtime()
            val took = k.file.length()
            val e = (try { place(k, now) } finally { onSaving?.invoke(k.root, null) })
                ?: placeFile(k.file, k.name, came).let { (uri, size) -> Entry(k.name, came, size, k.root, now, uri, "", emptyList(), "") }
            listed(e)
            placed(k.id)
            Log.i(TAG, "received: ${e.name} placed in Downloads in ${android.os.SystemClock.elapsedRealtime() - t0} ms (${e.size} B${if (e.items.isNotEmpty()) ", ${e.items.size} files" else ""}${if (k.packed != null) ", unpacked from $took B" else ""})")
            Filed(k.root, name, "")
        } catch (e: Exception) {
            val why = e.message ?: e.javaClass.simpleName
            Log.w(TAG, "received: $name not placed in Downloads: $why; kept in ${k.file.path}")
            if (k.file.isFile) synchronized(mu) {
                if (read().none { it.local == k.file.path }) add(Entry(k.name, came, k.file.length(), k.root, now, "", "", emptyList(), k.file.path))
            }
            Filed(k.root, name, why)
        }
    }

    // ai: A take's file into Downloads in one pass from where it lies (2026-11-15; the session unpacked a packed file into
    // ai: kept/ and a tar was read twice, its headers first, before): a packed file read back through the native
    // ai: Unpacker (Unpacked), a tar unpacked as it is read (untar), the bytes read counted for the Saving row. Null where
    // ai: the file does not read whole (a packed frame damaged, a tar damaged or cut short: nothing of it left in
    // ai: Downloads; the caller places the file as it came); a throw where Downloads does not take it.
    private fun place(k: Take, at: Long): Entry? {
        val name = k.packed?.first ?: k.name
        val type = k.packed?.second ?: k.type
        val raw: InputStream = try { if (k.packed != null) Unpacked(k.file.path) else k.file.inputStream() }
            catch (e: IOException) { Log.w(TAG, "received: ${k.name}: ${e.message}; placed as it came"); return null }
        return raw.use {
            val total = (raw as? Unpacked)?.size ?: k.file.length()
            var folder = ""
            var last = 0L
            val say = onSaving
            val src = Read(raw) { n ->
                val t = android.os.SystemClock.elapsedRealtime()
                if (say != null && t - last >= 100) { last = t; say(k.root, Readout.Saving(n, total, folder)) }
            }.buffered(1 shl 16)
            say?.invoke(k.root, Readout.Saving(0, total, ""))
            try {
                if (type == TAR_TYPE) untar(src, name, k.root, at) { folder = it; say?.invoke(k.root, Readout.Saving(0, total, it)) }
                else placeStream(src, name, type).let { (uri, size) -> Entry(name, type, size, k.root, at, uri, "", emptyList(), "") }
            } catch (e: Broken) {
                Log.w(TAG, "received: ${k.name}: ${e.message}; placed as it came")
                null
            }
        }
    }

    // ai: a source's read refused (a packed frame damaged, a tar damaged or cut short): the file is placed as it came
    private class Broken(cause: IOException) : IOException(cause.message, cause)
    // ai: Downloads refused what untar asked of it (a row, a write, a move): untar's caller keeps the take
    private class Refused(cause: Throwable) : IOException(cause.message, cause)

    // ai: A source as placing reads it: its read errors Broken, the bytes read so far to `count` after each read
    private class Read(private val src: InputStream, private val count: ((Long) -> Unit)? = null) : InputStream() {
        private var n = 0L
        private inline fun <T> guard(f: () -> T): T = try { f() } catch (e: IOException) { throw e as? Broken ?: Broken(e) }
        override fun read(): Int = guard { src.read() }.also { if (it >= 0) { n++; count?.invoke(n) } }
        override fun read(b: ByteArray, off: Int, len: Int): Int = guard { src.read(b, off, len) }.also { if (it > 0) { n += it; count?.invoke(n) } }
        override fun close() = src.close()
    }

    // ai: A packed file read back as the file it holds (Native.unpackOpen, the core's Unpacker): size its bytes
    private class Unpacked(path: String) : InputStream() {
        private var h = Native.unpackOpen(path)
        val size = Native.unpackSize(h)
        override fun read(): Int { val b = ByteArray(1); return if (read(b, 0, 1) < 1) -1 else b[0].toInt() and 0xff }
        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (h == 0L) throw IOException("closed")
            if (off < 0 || len < 0 || len > b.size - off) throw IndexOutOfBoundsException()
            return if (len == 0) 0 else Native.unpackRead(h, b, off, len)
        }
        override fun close() { if (h != 0L) { Native.unpackClose(h); h = 0L } }
    }

    // ai: A file received before 2026-10-31 (Library.kt's folder: the file or the bundle's folder beside .meta.json)
    // ai: placed into Downloads as a take is, its folder removed once it is there; one Downloads does not take listed from
    // ai: where it is
    private fun carried(d: File): Filed {
        val m = runCatching { JSONObject(File(d, ".meta.json").readText()) }.getOrNull()
        val stored = m?.optString("stored").orEmpty()
        val f = if (stored.isNotEmpty()) File(d, stored) else d.listFiles()?.firstOrNull { !it.name.startsWith(".") }
        if (f == null || !f.exists()) { d.deleteRecursively(); return Filed("", d.name, "") }
        val name = m?.optString("name").orEmpty().ifEmpty { f.name }
        val root = m?.optString("root").orEmpty()
        val type = m?.optString("type").orEmpty()
        val at = m?.optLong("at")?.takeIf { it > 0 } ?: f.lastModified()
        if (root.isNotEmpty() && list().any { it.root == root && it.local.isEmpty() }) { d.deleteRecursively(); return Filed(root, name, "") }
        return try {
            val e = if (f.isDirectory) placeFolder(f, f.name, root, at, type)
                else placeFile(f, name, type).let { (uri, size) -> Entry(name, type, size, root, at, uri, "", emptyList(), "") }
            listed(e)
            d.deleteRecursively()
            Filed(root, name, "")
        } catch (x: Exception) {
            val why = x.message ?: x.javaClass.simpleName
            Log.w(TAG, "received: ${f.name} (received before) not placed in Downloads: $why")
            synchronized(mu) {
                if (read().none { it.local == f.path }) {
                    val items = if (f.isDirectory) f.walkTopDown().filter { it.isFile }.map { Item(it.relativeTo(f).invariantSeparatorsPath, it.path, it.length()) }.toList() else emptyList()
                    add(Entry(name, type, if (f.isDirectory) items.sumOf { it.size } else f.length(), root, at, "", if (f.isDirectory) f.name else "", items, f.path))
                }
            }
            Filed(root, name, why)
        }
    }

    // ai: One file into Downloads through MediaStore: inserted pending (hidden from other apps), its bytes copied and made
    // ai: durable, then shown; MediaStore gives a name taken a free one ("name (1).ext"). Its URI and size; a throw where
    // ai: Downloads does not take it (nothing of it left there). The source is the caller's to let go once it is listed.
    private fun placeFile(src: File, name: String, type: String, sub: String = "", pending: MutableList<Uri>? = null): Pair<String, Long> =
        src.inputStream().use { placeStream(it, name, type, sub, pending) }

    // ai: placeFile's work from a stream (2026-11-14: a tar's entry straight into Downloads, unpack): the URI and the
    // ai: bytes written
    private fun placeStream(src: java.io.InputStream, name: String, type: String, sub: String = "", pending: MutableList<Uri>? = null): Pair<String, Long> {
        val v = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, safeName(name))
            put(MediaStore.MediaColumns.MIME_TYPE, type.ifEmpty { mimeOf(name) })
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + sub)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v) ?: throw IOException("$name could not be written")
        var n = 0L
        try {
            val fd = resolver.openFileDescriptor(uri, "w") ?: throw IOException("$name could not be written")
            ParcelFileDescriptor.AutoCloseOutputStream(fd).use { o -> n = src.copyTo(o, 1 shl 20); o.fd.sync() }
            if (pending == null) resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            else pending += uri
        } catch (e: Throwable) {
            runCatching { resolver.delete(uri, null, null) }
            throw e
        }
        return uri.toString() to n
    }

    // ai: A folder's files into Downloads/<a free name>/..., each pending until the last is in, then all shown (the
    // ai: bundle there whole, or none of it: what was inserted is deleted at a failure). The folder's name is free where
    // ai: no file of this app's is under it and (Android 11 and up, where the folder is seen) no folder of that name is
    // ai: there. The entry; the source folder is left to the caller.
    private fun placeFolder(dir: File, name: String, root: String, at: Long, type: String): Entry {
        val folder = freeFolder(safeName(name))
        val files = dir.walkTopDown().filter { it.isFile }.sortedBy { it.relativeTo(dir).invariantSeparatorsPath }.toList()
        val pending = ArrayList<Uri>()
        val items = ArrayList<Item>()
        try {
            for (f in files) {
                val rel = f.relativeTo(dir).invariantSeparatorsPath
                val sub = (folder + "/" + rel.substringBeforeLast('/', "")).trimEnd('/') + "/"
                val (uri, size) = placeFile(f, f.name, "", sub, pending)
                items += Item(rel, uri, size)
            }
            for (u in pending) resolver.update(u, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
        } catch (e: Throwable) {
            for (u in pending) runCatching { resolver.delete(u, null, null) }
            throw e
        }
        return Entry(folder, type.ifEmpty { TAR_TYPE }, items.sumOf { it.size }, root, at, "", Environment.DIRECTORY_DOWNLOADS + "/" + folder, items, "")
    }

    private fun freeFolder(name: String): String {
        @Suppress("DEPRECATION") val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        fun taken(n: String): Boolean {
            // ai: a folder there already is taken unless it is one this app can list and it holds nothing
            if (Build.VERSION.SDK_INT >= 30) File(downloads, n).let { if (it.exists() && !(it.isDirectory && it.list()?.isEmpty() == true)) return true }
            return runCatching {
                resolver.query(MediaStore.Downloads.EXTERNAL_CONTENT_URI, arrayOf(MediaStore.MediaColumns._ID),
                    "${MediaStore.MediaColumns.RELATIVE_PATH} LIKE ?", arrayOf(Environment.DIRECTORY_DOWNLOADS + "/" + n + "/%"), null)?.use { it.count > 0 } ?: false
            }.getOrDefault(false)
        }
        if (!taken(name)) return name
        var k = 1
        while (taken("$name ($k)")) k++
        return "$name ($k)"
    }

    // ai: A tar unpacked into Downloads as it is read (2026-11-15: one pass, from the take's file or straight out of its
    // ai: packed stream; 2026-11-14 two passes over the tar, its headers first, and into a folder of its own beside it
    // ai: then copied in before that): each file's bytes into Downloads as placeFolder places them, each pending until
    // ai: the last is in. One folder alone at its top is the bundle's, else the tar's name less .tar: the first entry
    // ai: under a folder names the bundle by it, and a second top, or a file at the top, after it moves what is in under
    // ai: the tar's name (their rows' RELATIVE_PATH, still pending). Tar.kt tarEach refuses links, paths out of it,
    // ai: damaged headers and a tar cut short: Broken, nothing of it left in Downloads; a throw where Downloads does not
    // ai: take it (what was inserted deleted). folderIs hears the folder each time it is chosen.
    private fun untar(src: InputStream, name: String, root: String, at: Long, folderIs: (String) -> Unit): Entry {
        val tarName = safeName(name.removeSuffix(".tar").ifEmpty { "files" })
        val pending = ArrayList<Uri>()
        val items = ArrayList<Item>()
        var one: String? = null
        var folder: String? = null
        fun sub(f: String, rel: String) = (f + "/" + rel.substringBeforeLast('/', "")).trimEnd('/') + "/"
        fun <T> refused(f: () -> T): T = try { f() } catch (x: Broken) { throw x } catch (x: Exception) { throw Refused(x) }
        try {
            tarEach(src) { parts, dir, _, data ->
                val atTop = parts.size == 1 && !dir
                val now = folder
                val top = one
                if (now == null) {
                    one = parts[0].takeIf { !atTop }
                    folder = freeFolder(safeName(parts[0].takeIf { !atTop } ?: tarName)).also(folderIs)
                } else if (top != null && (atTop || parts[0] != top)) {
                    val f = freeFolder(tarName)
                    refused {
                        for (i in items.indices) {
                            val it = items[i]
                            val rel = top + "/" + it.path
                            val v = ContentValues().apply { put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + sub(f, rel)) }
                            if (resolver.update(Uri.parse(it.uri), v, null, null) < 1) throw IOException("${it.path} could not be moved")
                            items[i] = Item(rel, it.uri, it.size)
                        }
                    }
                    dropEmpty(now)
                    one = null
                    folder = f.also(folderIs)
                }
                if (!dir) {
                    val rel = (if (one != null) parts.drop(1) else parts).joinToString("/")
                    val (uri, size) = refused { placeStream(Read(data), parts.last(), "", sub(folder!!, rel), pending) }
                    items += Item(rel, uri, size)
                }
            }
            if (folder == null) folder = freeFolder(tarName).also(folderIs)
            refused { for (u in pending) resolver.update(u, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null) }
        } catch (e: Throwable) {
            for (u in pending) runCatching { resolver.delete(u, null, null) }
            folder?.let { dropEmpty(it) }
            throw when (e) {
                is Refused -> e.cause ?: e
                is Broken -> e
                is IOException -> Broken(e)
                else -> e
            }
        }
        items.sortBy { it.path }
        return Entry(folder!!, TAR_TYPE, items.sumOf { it.size }, root, at, "", Environment.DIRECTORY_DOWNLOADS + "/" + folder, items, "")
    }

    // ai: A folder under Downloads that MediaStore left behind (it deletes and moves files, never folders): the folders
    // ai: in it left empty go, deepest first; one still holding anything (a file put there since) stays, and its parents
    // ai: with it. Each one gone is scanned, which drops MediaStore's row for it: a folder's row outlives a File API
    // ai: delete (2026-11-15, seen on the S26 after untar's move), and freeFolder's query would count a subfolder's row
    // ai: as its parent's name taken. Android 11 and up, where the File API reaches Downloads.
    private fun dropEmpty(name: String) {
        if (Build.VERSION.SDK_INT < 30 || name.isEmpty() || name.contains('/') || name == "." || name == "..") return
        @Suppress("DEPRECATION") val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val gone = File(downloads, name).takeIf { it.isDirectory }?.walkBottomUp()?.filter { it.isDirectory && it.delete() }?.map { it.path }?.toList().orEmpty()
        if (gone.isNotEmpty()) MediaScannerConnection.scanFile(ctx, gone.toTypedArray(), null, null)
    }

    // ai: An entry gone: its files deleted from Downloads (this app's own rows), or its local file or folder; one listed
    // ai: from a take's folder takes that folder with it. Its row goes only once its files have: a delete MediaStore
    // ai: refused keeps it (a throw, said by the caller). Off the main thread.
    fun delete(e: Entry) {
        val uris = (listOf(e.uri) + e.items.map { it.uri }).filter { it.startsWith("content:") }
        var left = uris.count { u -> !(runCatching { resolver.delete(Uri.parse(u), null, null) > 0 }.getOrDefault(false) || !exists(u)) }
        if (e.local.isNotEmpty()) {
            val f = File(e.local)
            if (!f.deleteRecursively()) left++
            else {
                val take = f.absoluteFile.parentFile?.parentFile
                if (take?.parentFile?.absolutePath == keptDir.absolutePath) placed(take.name)
                val lib = f.absoluteFile.parentFile
                if (lib?.parentFile?.absolutePath == oldLibrary.absolutePath) lib.deleteRecursively()
            }
        }
        if (left > 0) throw IOException(e.name)
        // ai: A bundle's folder: MediaStore deletes files, never folders, so each bundle deleted left Download/<name>/
        // ai: (and its subfolders) behind, empty, and the next bundle of that name was placed as "<name> (n)" (2026-11-15)
        if (e.folder.startsWith(Environment.DIRECTORY_DOWNLOADS + "/")) dropEmpty(e.folder.removePrefix(Environment.DIRECTORY_DOWNLOADS + "/"))
        synchronized(mu) { write(read().filter { it.id != e.id || it.local != e.local }) }
    }

    companion object {
        // ai: a name's media type by its extension, else bytes
        fun mimeOf(name: String): String =
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase()) ?: "application/octet-stream"
        // ai: a name MediaStore takes: no separator, no control character, not "." or ".."
        fun safeName(name: String): String {
            val n = name.map { if (it == '/' || it == '\\' || it.code < 0x20 || it.code == 0x7f) '_' else it }.joinToString("")
            return if (n.isEmpty() || n == "." || n == "..") "file" else n
        }
    }
}
