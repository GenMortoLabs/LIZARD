package dev.lizard.ui

import dev.lizard.readout.Readout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import java.io.File
import kotlin.math.floor

// ai: What both apps read out of the receive session's JSON (2026-10-31; each app's own parsing before, the desktop's
// ai: kotlinx and the Android app's org.json): the stats a quarter second (liblizard/core/rx/session.h stats(): the
// ai: receiver's fields, the session's phase, its camera, its runs), the store's unfinished transfers (Session.partials).
// ai: Pure: no drawing here.

private val json = Json { ignoreUnknownKeys = true }
fun parse(s: String): JsonObject? = runCatching { json.parseToJsonElement(s).jsonObject }.getOrNull()
fun JsonObject.num(k: String) = (this[k] as? JsonPrimitive)?.doubleOrNull ?: 0.0
fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.contentOrNull ?: ""
fun JsonObject.flag(k: String) = (this[k] as? JsonPrimitive)?.booleanOrNull ?: false
fun JsonObject.obj(k: String) = this[k] as? JsonObject
fun JsonObject.objs(k: String) = (this[k] as? JsonArray)?.mapNotNull { it as? JsonObject }

// ai: A stream file's meta.json as Send reads it (2026-11-18, liblizard/core/tx/stream_export.h): the format it was
// ai: painted in (blocks a version's, the ring 0 to 2, codes, fps), its pictures and their shape (a code's side and the gap
// ai: between two, pixels), and the file it carries (its name, bytes, whether the test stream); null where it is not a
// ai: stream's
class StreamMeta(val blocks: Int, val ring: Int, val codes: Int, val fps: Int, val pictures: Int, val name: String, val length: Long, val test: Boolean,
                 val side: Int = 0, val gap: Int = 0) {
    val label get() = SendRows.streamLabel(blocks, codes, fps)
    // ai: a picture's width over its height (its codes and the gap between them)
    val aspect get() = if (side > 0) (codes * side + gap).toDouble() / side else codes.toDouble()
}
fun streamMetaOf(m: JsonObject?): StreamMeta? {
    val f = m?.obj("format") ?: return null
    if (m.str("kind") != "stream") return null
    val file = m.obj("file")
    return StreamMeta(f.num("blocks").toInt(), f.num("ring").toInt(), f.num("codes").toInt().coerceIn(1, 2), m.num("fps").toInt(),
        m.num("pictures").toInt(), file?.str("name").orEmpty().ifEmpty { m.str("run") }, file?.num("length")?.toLong() ?: 0L, m.flag("test"),
        m.num("h").toInt(), m.num("gap").toInt())
}

// ai: The receiver's phase as a screen shows it (the session's "phase" and "why"; Starting while an app's own start
// ai: order runs before the session's start, `opening`)
sealed interface RxPhase {
    data object Idle : RxPhase
    data object Loading : RxPhase
    data object Starting : RxPhase
    data object On : RxPhase
    data class Error(val why: String) : RxPhase
}

fun rxPhaseOf(j: JsonObject?, opening: Boolean = false): RxPhase = when (j?.str("phase")) {
    "starting" -> RxPhase.Starting
    "loading" -> RxPhase.Loading
    "on" -> RxPhase.On
    "error" -> RxPhase.Error(j.str("why"))
    else -> if (opening) RxPhase.Starting else RxPhase.Idle
}

// ai: The stats' fields the screens read (liblizard/core/rx/receiver.h stats() and receiver.cpp name them)
fun rxOf(j: JsonObject?): Readout.Rx {
    if (j == null) return Readout.Rx()
    val f = j.obj("file")
    return Readout.Rx(state = j.str("state"), error = j.str("error"), goodputKBs = j.num("goodputKBs"), foundShare = j.num("foundShare"),
        bandVersion = j.num("bandVersion").toInt(),
        hasFile = f != null, name = f?.str("name").orEmpty().ifEmpty { "a file" }, size = f?.num("size")?.toLong() ?: 0,
        received = f?.num("received")?.toLong() ?: 0, verified = f?.flag("verified") ?: false, secs = f?.num("secs") ?: 0.0,
        sourceBlocks = f?.num("sourceBlocks")?.toInt() ?: 0, sourceBlocksVerified = f?.num("sourceBlocksVerified")?.toInt() ?: 0, inHand = f?.flag("inHand") ?: false)
}

// ai: Save replays' runs (the stats' "replays", session.h): a run's .lizard of the newest frames the decoder takes
enum class RunState { Recording, Ending, Ready, Failed }
data class Run(val run: String, val file: File, val state: RunState, val frames: Long = 0, val bytes: Long = 0, val why: String = "")

// ai: the runs the stats list, null where they say nothing of them
fun runsOf(j: JsonObject?): List<Run>? = j?.objs("replays")?.map { o ->
    val st = when (o.str("state")) { "recording" -> RunState.Recording; "saving" -> RunState.Ending; "ready" -> RunState.Ready; else -> RunState.Failed }
    Run(o.str("run"), File(o.str("path")), st, o.num("frames").toLong(), o.num("bytes").toLong(), o.str("error"))
}

// ai: a run's line in Developer Tools
fun runLine(r: Run): String = when (r.state) {
    RunState.Recording -> "${r.run}: recording"
    RunState.Ending -> "${r.run}: saving"
    RunState.Ready -> "${r.run}: ${r.frames} frames, ${String.format(java.util.Locale.ROOT, "%.0f", r.bytes / 1e6)} MB" + if (r.why.isEmpty()) "" else " (${r.why})"
    RunState.Failed -> "${r.run}: ${r.why}"
}

// ai: Transfers begun and not finished (2026-10-22): the receive session's store, one folder a transfer by its root's
// ai: first 16 hex digits with its header and meta.json; Session.partials lists them, newest first. A receiver resumes
// ai: one when its file is in the light again; Home lists them to continue or delete. The web keeps the same in its
// ai: origin's OPFS (lizard-web/library.mjs unfinished).
data class Partial(val id: String, val name: String, val length: Long, val fraction: Double, val root: String, val at: Long)

fun partialsOf(listing: String): List<Partial> = runCatching {
    (json.parseToJsonElement(listing) as? JsonArray).orEmpty().mapNotNull { e ->
        (e as? JsonObject)?.let { o -> Partial(o.str("id"), o.str("name"), o.num("length").toLong(), o.num("fraction"), o.str("root"), o.num("at").toLong()) }
    }
}.getOrDefault(emptyList())

