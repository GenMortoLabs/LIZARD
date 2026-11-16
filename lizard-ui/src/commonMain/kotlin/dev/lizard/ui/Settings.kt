package dev.lizard.ui

import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

// ai: A fold's content as an app describes it (2026-10-31): each setting what it shows and what a change does, the app
// ai: deciding which there are and their order (a camera's own controls, a platform's own fields), the shared screens
// ai: drawing them in the fields' rhythm (Fields).
sealed interface Setting

// ai: How a setting is drawn (2026-11-09): a few fixed choices as chips, a spectrum of many as a slider, a menu only
// ai: where neither fits (a list as long as the device makes it: a camera, a webcam's resolutions)

// ai: a menu (Select)
class Choice<T>(val title: String, val value: T, val options: List<Pair<T, String>>, val blank: String = "", val enabled: Boolean = true,
                val enabledFor: (T) -> Boolean = { true }, val onOpen: () -> Unit = {}, val choose: (T) -> Unit) : Setting

// ai: a choice among a few as chips under its title (Chips: Send's FPS, Ring, Codes, Encoder, VSync, Payload; Receive's
// ai: Decoder, Crop (1:1, 2:1; named Codes, 1 and 2, from 2026-11-09 to 11-13), Format, Phase lock, Frame rate, Save replays; Show statistics on both; the desktop's Exclusive
// ai: fullscreen; the phone's Resolution and Compression); `enabledFor` greys a choice the device does not offer
class ChipChoice<T>(val title: String, val value: T, val options: List<Pair<T, String>>, val enabled: Boolean = true,
                    val enabledFor: (T) -> Boolean = { true }, val choose: (T) -> Unit) : Setting

// ai: a slider (Bar) under its title and the value as it reads: each value while dragged, `done` at the release
class Slide(val title: String, val label: String, val value: Float, val range: ClosedFloatingPointRange<Float>, val steps: Int,
            val done: (() -> Unit)? = null, val choose: (Float) -> Unit) : Setting

// ai: a line of text (TextInput), done at Enter or as the focus leaves it changed
class Typed(val title: String, val value: String, val placeholder: String, val done: (String) -> Unit) : Setting

// ai: a field of lines (the PC link's: its state on the right, its lines muted under it, one action under them)
class Lines(val title: String, val value: String, val lines: List<String>, val act: Act? = null) : Setting

// ai: the lab's text (CodeBlock); a button across the fold (a run's Download, Show in folder)
class Code(val text: String, val size: Int = 13) : Setting
class Press(val act: Act, val enabled: Boolean = true) : Setting

// ai: The settings folds by topic (2026-11-12; one Settings fold a screen before): each a fold's key and title in the
// ai: column's order, before Experimental and Developer Tools. Receive's: the camera (its choice, its mode, Phase lock, Crop
// ai: after Frame rate, its controls; Crop in a Code fold of its own, named Codes, until 2026-11-13), the decoder's performance
// ai: (Decoder, Batch size, the phone's PC link and Compression; titled Decoder until 2026-11-13).
// ai: Send's: the code and its place on the screen (Density or Blocks, Ring, Codes, then Size or the phone's Brightness,
// ai: Spacing, Margin; a Layout fold of their own until 2026-11-13), its refresh (FPS, Rate refresh), the painter
// ai: (Encoder, VSync). The web pages the same (lizard-web).
val RECV_FOLDS = listOf(FoldKey.RecvCamera to "Camera", FoldKey.RecvPerformance to "Performance", FoldKey.RecvFiles to "Files")
val SEND_FOLDS = listOf(FoldKey.SendCode to "Code", FoldKey.SendRefresh to "Refresh", FoldKey.SendPerformance to "Performance")

// ai: each fold of `order` that the app gives a setting, open as `folds` keeps it
@Composable
fun SettingFolds(folds: Folds, order: List<Pair<FoldKey, String>>, settings: Map<FoldKey, List<Setting>>) {
    for ((k, title) in order) {
        val items = settings[k].orEmpty()
        if (items.isNotEmpty()) Fold(title, folds.open(k), { folds.toggle(k) }) { Settings(items) }
    }
}

// ai: The receivers' Phase lock (2026-11-08, both apps; the web's #lockRow the same row): On or Off where the camera
// ai: can delay a capture (the session's camera "lockBar" ""), else one greyed chip "Not supported"; above Frame rate. With it
// ai: on such a camera runs at most LOCK_FPS captures a second (session.h phaseLock), so Frame rate offers no more
// ai: (lockedRates); off, every rate the camera lists.
const val LOCK_FPS = 60.0

fun phaseLockSetting(on: Boolean, supported: Boolean, choose: (Boolean) -> Unit): Setting =
    if (supported) ChipChoice("Phase lock", on, listOf(true to "On", false to "Off")) { choose(it) }
    else ChipChoice("Phase lock", 1, listOf(0 to "Not supported"), enabled = false) {}  // ai: none chosen: the chip greyed

// ai: Save replays' Replay size (2026-11-09, both apps; the session's replayBytes): a run's cap, the raw bytes of the
// ai: frames it keeps (the newest that fit, the earliest dropped as new ones come), a slider from REPLAY_MB_MIN to
// ai: REPLAY_MB_MAX in REPLAY_MB_STEP; the apps show it while Save replays is on. MB are MiB, as the session counts.
const val REPLAY_MB_MIN = 256
const val REPLAY_MB_MAX = 8192
const val REPLAY_MB_STEP = 256
const val REPLAY_MB_DEFAULT = 1024

fun replayMb(v: Int): Int = (v.coerceIn(REPLAY_MB_MIN, REPLAY_MB_MAX) + REPLAY_MB_STEP / 2) / REPLAY_MB_STEP * REPLAY_MB_STEP

// ai: 256 MB, 512 MB, 768 MB, 1 GB, 1.25 GB ... 8 GB
fun replaySizeText(mb: Int): String = if (mb < 1024) "$mb MB"
    else "${mb / 1024}" + when (mb % 1024) { 256 -> ".25"; 512 -> ".5"; 768 -> ".75"; else -> "" } + " GB"

// ai: Send's Seed (Developer Tools, both apps, 2026-11-14): a file's schedule drawn from it (lizard.h liz_sender_seed:
// ai: each chunk's first symbol, each round's order of the chunks, each frame's shuffle), so a send can be repeated;
// ai: empty, a random one each send. Kept as typed: seedText takes what is a seed or empty and refuses the rest (null);
// ai: seedOf reads it, an unsigned 32-bit value carried in an Int, null for none.
fun seedSetting(text: String, choose: (String) -> Unit): Setting = Typed("Seed", text, "Random") { t -> seedText(t)?.let(choose) }
fun seedText(t: String): String? = t.trim().let { v -> if (v.isEmpty() || v.toUIntOrNull() != null) v else null }
fun seedOf(text: String): Int? = text.trim().toUIntOrNull()?.toInt()

// ai: Receive's Largest file (2026-11-15, both apps, the Files fold: look A of
// ai: research/build/ui-proposal-2026-11-15-filelimit/mock.png; session.h fileLimit): a transfer whose file is over it is
// ai: refused at its header, the camera stopped, the popup saying so. GB of 10^9 bytes, as the apps' figures; the stops a
// ai: slider's, FILE_GB_NONE (0) the last, "No limit" (the format's own, 65,535 chunks of 16 MiB); 50 GB by default.
const val FILE_GB_NONE = 0
const val FILE_GB_DEFAULT = 50
val FILE_GB_STOPS = listOf(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, FILE_GB_NONE)
// ai: a kept value held to a stop (one that is none of them: the default)
fun fileGb(v: Int): Int = if (v in FILE_GB_STOPS) v else FILE_GB_DEFAULT
fun fileGbText(gb: Int): String = when (gb) { FILE_GB_NONE -> "No limit"; 1000 -> "1 TB"; else -> "$gb GB" }
// ai: the bytes the session takes (0 none)
fun fileLimitBytes(gb: Int): Long = gb.toLong() * 1_000_000_000L
fun fileLimitSetting(gb: Int, choose: (Int) -> Unit): Setting {
    val at = FILE_GB_STOPS.indexOf(fileGb(gb))
    return Slide("Largest file", fileGbText(fileGb(gb)), at.toFloat(), 0f..(FILE_GB_STOPS.size - 1).toFloat(), FILE_GB_STOPS.size - 2) { v ->
        val g = FILE_GB_STOPS[v.roundToInt().coerceIn(0, FILE_GB_STOPS.size - 1)]
        if (g != gb) choose(g)
    }
}

fun replaySizeSetting(mb: Int, choose: (Int) -> Unit): Setting =
    Slide("Replay size", replaySizeText(mb), mb.toFloat(), REPLAY_MB_MIN.toFloat()..REPLAY_MB_MAX.toFloat(),
        (REPLAY_MB_MAX - REPLAY_MB_MIN) / REPLAY_MB_STEP - 1) { v -> val m = replayMb(v.roundToInt()); if (m != mb) choose(m) }

// ai: the bytes the session takes for Replay size's MB
fun replayBytes(mb: Int): Long = replayMb(mb).toLong() shl 20

// ai: Send's Rate refresh (2026-11-12, both apps; liblizard/core/tx/sender.h Sender::setRefresh): for each code rate,
// ai: from the picture's inside out (REFRESH_RATES), a block
// ai: of that rate takes a new id in every n-th frame and repeats the frame before's otherwise, n 1 (each frame, the
// ai: default) to REFRESH_MAX. A slider a rate in the Refresh fold, under FPS (a heading over them until the folds by
// ai: topic, 2026-11-12), each value applied as it moves (the sender takes it from its next frame, no configure); kept
// ai: as "a,b,c,d" (refreshText, refreshOf), as the sender's stats carry it ("refresh").
const val REFRESH_MAX = 4
val REFRESH_RATES = listOf("7/8", "3/4", "2/3", "1/2")

// ai: a kept text to the four values: a missing or unreadable one 1, each held to 1..REFRESH_MAX
fun refreshOf(text: String?): IntArray {
    val v = text?.split(',')?.map { it.trim().toIntOrNull() }
    return IntArray(REFRESH_RATES.size) { r -> (v?.getOrNull(r) ?: 1).coerceIn(1, REFRESH_MAX) }
}

fun refreshText(every: IntArray): String = every.joinToString(",")

fun rateRefreshSettings(every: IntArray, choose: (rate: Int, value: Int) -> Unit): List<Setting> =
    REFRESH_RATES.mapIndexed { r, title ->
        val v = every[r]
        Slide(title, "$v", v.toFloat(), 1f..REFRESH_MAX.toFloat(), REFRESH_MAX - 2) { x ->
            val n = x.roundToInt().coerceIn(1, REFRESH_MAX); if (n != v) choose(r, n) }
    }

// ai: Send's Header, the animated header (2026-11-15, titled Animated header in Experimental until that afternoon; the
// ai: Refresh fold after Rate refresh; liblizard/core/tx/sender.h
// ai: Sender::setCarousel, SPEC 5.5): Off (0, the static word, the default) or the frames each carousel symbol of the
// ai: format's own header is shown, 1 to CAROUSEL_MAX; each move handed to the sender at once (its next frame made; the
// ai: GPU paints each code's symbol over its word cells, gpu/wgsl/send.mjs WORDS, the CPU where it cannot)
const val CAROUSEL_MAX = 30

fun carouselSetting(frames: Int, choose: (Int) -> Unit): Setting =
    Slide("Header", if (frames == 0) "Off" else if (frames == 1) "1 frame" else "$frames frames", frames.toFloat(),
        0f..CAROUSEL_MAX.toFloat(), CAROUSEL_MAX - 1) { x -> val n = x.roundToInt().coerceIn(0, CAROUSEL_MAX); if (n != frames) choose(n) }

// ai: Receive's Focus (2026-11-16, both apps' Camera fold, after Zoom; the camera's "focus" control, the session's set
// ai: and cameras.json): Auto first, the camera's own autofocus and the default (the session turns it on at each open
// ai: where no focus is kept: the phone's continuous autofocus on the code, a UVC camera's focus_automatic_continuous),
// ai: then the lens by hand over the camera's range in FOCUS_STEPS steps, low to high: a phone's focus distance in
// ai: dioptres (0 infinity) shown as a distance, a PC camera's focus in its own units as they are (a UVC camera's
// ai: focus_absolute; which end is near is the camera's). Moved live. A camera with no automatic focus starts at its low end.
const val FOCUS_STEPS = 100

fun focusSetting(lo: Double, hi: Double, value: Double?, canAuto: Boolean, dioptres: Boolean, choose: (Double?) -> Unit): Setting {
    val span = maxOf(hi - lo, 1e-9)
    val first = if (canAuto) 1 else 0
    val v = value ?: if (canAuto) null else lo
    val x = if (v == null) 0 else first + ((v.coerceIn(lo, hi) - lo) / span * (FOCUS_STEPS - first)).roundToInt()
    val label = when {
        v == null -> "Auto"
        !dioptres -> String.format(java.util.Locale.ROOT, "%.0f", v)
        v <= 0.005 -> "Infinity"
        v <= 1.0 -> String.format(java.util.Locale.ROOT, "%.1f m", 1 / v)
        else -> String.format(java.util.Locale.ROOT, "%.0f cm", 100 / v)
    }
    return Slide("Focus", label, x.toFloat(), 0f..FOCUS_STEPS.toFloat(), FOCUS_STEPS - 1) { f ->
        val k = f.roundToInt()
        choose(if (k < first) null else lo + span * (k - first) / (FOCUS_STEPS - first))
    }
}

// ai: the rates Frame rate offers: none past LOCK_FPS while a camera that can lock has the lock on
fun lockedRates(rates: List<Double>, locks: Boolean): List<Double> = if (locks) rates.filter { it <= LOCK_FPS * 1.005 } else rates

@Composable
fun Settings(items: List<Setting>) = Fields { for (s in items) SettingView(s) }

@Composable
private fun ColumnScope.SettingView(s: Setting) {
    when (s) {
        is Choice<*> -> ChoiceView(s)
        is ChipChoice<*> -> ChipsView(s)
        is Slide -> Field(s.title, s.label) { Bar(s.value, s.range, s.steps, s.done, s.choose) }
        is Typed -> Field(s.title) { TextInput(s.value, s.placeholder, s.done) }
        is Lines -> Field(s.title, s.value) {
            for (l in s.lines) Text(l, style = MaterialTheme.typography.bodyMedium, color = Muted)
            s.act?.let { a -> Btn(a.label, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) { a.run() } }
        }
        is Code -> CodeBlock(s.text, s.size)
        is Press -> Btn(s.act.label, enabled = s.enabled, modifier = Modifier.fillMaxWidth()) { s.act.run() }
        is Colours -> ColoursView(s)
    }
}

@Composable
private fun <T> ChoiceView(c: Choice<T>) =
    Select(c.title, c.value, c.options, c.blank, c.enabled, c.enabledFor, c.onOpen, c.choose)

@Composable
private fun <T> ChipsView(c: ChipChoice<T>) = Field(c.title) {
    Chips { for ((k, label) in c.options) Chip(label, k == c.value, c.enabled && c.enabledFor(k)) { if (k != c.value) c.choose(k) } }
}

// ai: A count's slider whose leftmost step is auto (send.html #blocks and #subchAuto, 2026-10-05): the setting `value` (0
// ai: auto), `shown` what auto takes now (0 while unknown), `range` the counts by hand; its value "Auto, <text>" while
// ai: what auto takes is known (`auto` names the leftmost step: the phone's Brightness "System")
fun autoSlide(title: String, value: Int, shown: Int, range: IntRange, text: (Int) -> String, done: (() -> Unit)?, auto: String = "Auto",
              choose: (Int) -> Unit): Slide {
    val isAuto = value == 0
    val b = if (isAuto) shown else value
    val lo = range.first - 1
    return Slide(title, if (isAuto) (if (b > 0) "$auto, ${text(b)}" else auto) else text(b), (if (isAuto) lo else value).toFloat(),
        lo.toFloat()..range.last.toFloat(), range.last - lo - 1, done) { val v = it.roundToInt(); choose(if (v < range.first) 0 else v) }
}
