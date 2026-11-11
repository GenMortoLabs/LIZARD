package dev.lizard.ui

import androidx.compose.ui.draw.clip
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.text.style.TextOverflow
import dev.lizard.readout.Readout

// ai: The folds of the tool screens and Home's lists (HomeUnfinished, HomeFiles: the phone's Home, 2026-11-07), kept by
// ai: each app under its own keys, closed until opened. Collapsed: the tool screens' column at its rail, one for Receive
// ai: and Send (2026-11-07; each its own before)
enum class FoldKey { RecvSettings, RecvExperimental, RecvTools, SendSettings, SendExperimental, SendTools, Collapsed, HomeUnfinished, HomeFiles }

interface Folds {
    fun open(k: FoldKey): Boolean
    fun toggle(k: FoldKey)
}

// ai: The camera's permission while the app has none: the one button that gets it (the phone's; why is the state line)
class Access(val act: Act)

// ai: The file camera's choice (2026-11-09): Send's upload (Send.kt Upload) in the camera's place while the camera is
// ai: File and idle: the tiles (File, Folder, and where the platform has one Media), the items played in order at Start (a
// ai: folder's files in their names' order), their size after the count; `enabled` false while a pick is read in.
class ReceiveUpload(val tiles: List<Act>, val items: List<UploadItem>, val total: String, val enabled: Boolean,
                    val onRemove: (String) -> Unit, val onClear: () -> Unit)

// ai: What Receive shows and does, the app's: the receive session's state as the app reads it, the camera's settings as
// ai: the camera offers them, the platform's picture of the camera
interface ReceiveHost {
    val folds: Folds
    val width: ColumnWidth
    val canPortrait: Boolean           // ai: an upright screen takes the portrait layout (the phone's)
    val swapInColumn: Boolean          // ai: the swap in the landscape column's bar (the desktop's; the phone's column has room for the collapser alone)
    fun goHome()
    fun goSend()
    val line: Readout.Line
    val showStats: Boolean
    val heat: Warn?                    // ai: the heat warning (Throttling, the camera's rate where it fell), null while nothing throttles
    val hint: Warn?                    // ai: the camera's hint (Sender too fast, its most), null for none
    val note: String                   // ai: a filing's or a Save's failure; "" none
    fun clearNote() {}                 // ai: its popup's OK
    val running: Boolean
    val starting: Boolean              // ai: Start inert while the camera opens
    val startable: Boolean get() = true   // ai: Start can run: false while the file camera holds nothing
    val upload: ReceiveUpload? get() = null   // ai: the file camera's choice, drawn in the camera's place; null otherwise
    fun toggleCamera()
    val access: Access?
    val kept: List<Act>?               // ai: the received file's actions, the first the solid Open (each with its icon: a tile's
                                       // ai: and the rail's); null while none is in
    val keptName: String
    val railRate: Double?
    val railPct: Double?
    fun cropRatio(portrait: Boolean): Float   // ai: the preview's box: the crop's width over its height
    val twoToOne: Boolean
    val off: String                    // ai: over the preview: "No camera yet", "Camera off", or "" while it shows
    val waiting: Boolean               // ai: "Getting ready" until the receiver first answers
    @Composable fun Picture(modifier: Modifier, portrait: Boolean)
    val settings: List<Setting>
    val experimental: List<Setting>    // ai: the Experimental fold's lab arms; no fold where there are none
    val tools: List<Setting>
}

// ai: Receive (the web's recv.html; both apps' since 2026-10-31): the camera's crop, the transfer (the actions: Open and
// ai: the platform's others once a file is in, Start or Stop (the camera); then the state line, meter, figures, the
// ai: heat warning, the hint: the desktop's order; a filing's failure is the preview's to say), Settings, Experimental, Developer Tools, About.
// ai: Landscape: the column on the left, its edge dragged, the camera on the right as big as the rest allows; the rail
// ai: when collapsed (play or pause, the rate, the percent, the green check and the file's actions, Home at its foot).
// ai: Portrait (a phone): the bar over one column, the camera at its top.
@Composable
fun ReceiveScreen(h: ReceiveHost) = BoxWithConstraints {
    val portrait = h.canPortrait && maxHeight > maxWidth
    val ratio = h.cropRatio(portrait)
    // ai: one preview for both layouts: a turn of the phone moves it rather than making another (a second camera view
    // ai: would end the first one's surface, and the camera with it)
    val preview = remember(h) { movableContentOf<Modifier, Boolean> { m, p -> PreviewBox(h, m, p) } }
    val swap: @Composable () -> Unit = { SwapBtn("Sender") { h.goSend() } }
    // ai: the received file in the camera's place once it is in and the camera is off (2026-11-06, the web's #done)
    val got = h.kept != null && !h.running && !h.starting
    // ai: an error said once, in a popup (errorOf): OK closes it until another comes; the phase keeps its error until
    // ai: the next Start, so the screen keeps which it has said
    val err = errorOf(h)
    var said by remember(h) { mutableStateOf<String?>(null) }
    LaunchedEffect(err) { if (err == null) said = null }
    if (err != null && err != said) NoteDialog(err) { said = err; if (err == h.note) h.clearNote() }
    if (portrait) {
        // ai: the camera the column's width, up to 55% of the screen's height
        val tall = maxHeight * 0.55f
        val up = h.upload
        ToolPortrait("Receive", { h.goHome() }, { swap() }) {
            if (got) ReceivedPanel(h, Modifier.fillMaxWidth(), column = null)
            // ai: the file camera's choice as Send's phone draws it: a row's room, the items one row scrolling sideways
            else if (up != null) Box(Modifier.fillMaxWidth().height(UPLOAD_ONE_ROW)) { Upload(up.tiles, up.items, up.enabled, row = true, onRemove = up.onRemove, onClear = up.onClear, total = up.total) }
            else preview(Modifier.fillMaxWidth().heightIn(max = tall), true)
            Side(h)
        }
    } else {
        // ai: the column's floor, Send's too (columnFloor), so the column is the same on both screens
        ToolLandscape("Receive", h.folds.open(FoldKey.Collapsed), { h.folds.toggle(FoldKey.Collapsed) }, { h.goHome() },
            if (h.swapInColumn) swap else null, h.width, columnFloor(h.width), rail = { RailSquares(h) }, side = { Side(h, home = true) }) {
            val up = h.upload
            if (got) ReceivedPanel(h, Modifier.fillMaxSize(), column = 576.dp)
            // ai: the file camera's choice as Send's desktop draws it: a 576 dp column centred in the area
            else if (up != null) Upload(up.tiles, up.items, up.enabled, row = false, onRemove = up.onRemove, onClear = up.onClear, column = 576.dp, total = up.total)
            else {
                val w = minOf(maxHeight * ratio, maxWidth)
                preview(Modifier.width(w).height(w / ratio), false)
            }
        }
    }
}

// ai: The received file (2026-11-06; the web's recv.html #done), in the camera's place once the camera is off, in the
// ai: sender's upload geometry (Send.kt Upload): its actions as tiles in one row in equal
// ai: cells, Open the solid one, then a card, "Received" with what came and the file as an item tile (how it went is
// ai: the column's block with Show statistics on, 2026-11-07: the card holds things to act on). `column`: in
// ai: landscape a column at most that wide centred in the camera's area (the sender's 576 dp); null, the column's width
// ai: at the top (portrait, a phone's: the tiles across it).
@Composable
private fun ReceivedPanel(h: ReceiveHost, modifier: Modifier, column: Dp?) {
    val acts = h.kept ?: return
    val ln = h.line
    BoxWithConstraints(modifier) {
        val w = if (column != null) minOf(maxWidth - 32.dp, column) else maxWidth
        Column(if (column != null) Modifier.fillMaxSize() else Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = if (column != null) Arrangement.spacedBy(16.dp, Alignment.CenterVertically) else Arrangement.spacedBy(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                acts.forEachIndexed { i, a -> Tile(a.icon ?: Icons.file, a.label, (w - 12.dp * (acts.size - 1)) / acts.size, solid = i == 0, onClick = a.run) }
            }
            Column(Modifier.width(w).clip(RoundedCornerShape(12.dp)).background(Card).padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp)) {
                // ai: the title counts what came, "Received 10 files", "Received 1 file" (the size is the tile's)
                val good = ln.tone == Readout.Tone.Good
                Column(Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp, top = 8.dp, bottom = 12.dp)) {
                    Text(if (good && ln.files > 0) "Received ${countOf(ln.files)}" else "Received",
                        style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg)
                }
                ItemTile(h.keptName, if (ln.tone == Readout.Tone.Good) ln.size else "", Icons.file, TILE.dp, onRemove = null)
            }
        }
    }
}

// ai: The rail's squares (the web's #rail): the heat's red triangle above the play while the phone throttles, play or
// ai: pause, the rate and the percent while a file or the test stream is read; once the file is kept a green check (or
// ai: the heat's triangle) alone: its actions are the camera area's tiles (ReceivedPanel), not the rail's (2026-11-07)
@Composable
private fun RailSquares(h: ReceiveHost) {
    val kept = h.kept
    val hot = h.heat
    if (hot != null && kept == null) AlertSquare(hot.text)
    Square {
        IconBtn(if (h.running) Icons.pause else Icons.play, if (h.running) "Stop" else "Start",
            enabled = h.access == null && !h.starting && (h.running || h.startable)) { h.toggleCamera() }
    }
    if (kept == null) {
        RateSquare(h.railRate)
        h.railPct?.let { PctSquare(it) }
    } else {
        if (hot != null) AlertSquare(hot.text)
        else Square { Icon(Icons.check, contentDescription = "Received ${h.keptName}", Modifier.size(24.dp), tint = Good) }
    }
}

@Composable
private fun Side(h: ReceiveHost, home: Boolean = false) {
    TransferPanel(h)
    Fold("Settings", h.folds.open(FoldKey.RecvSettings), { h.folds.toggle(FoldKey.RecvSettings) }) { Settings(h.settings) }
    if (h.experimental.isNotEmpty()) Fold("Experimental", h.folds.open(FoldKey.RecvExperimental), { h.folds.toggle(FoldKey.RecvExperimental) }) { Settings(h.experimental) }
    Fold("Developer Tools", h.folds.open(FoldKey.RecvTools), { h.folds.toggle(FoldKey.RecvTools) }) { Settings(h.tools) }
    if (home) HomeRow { h.goHome() }
    Spacer(Modifier.height(24.dp))
}

// ai: The transfer, the desktop's order and gaps: 12 dp under the bar the actions (once the file is kept, its actions on
// ai: one row, Open the solid one filling it, the others their own width; Start or Stop alone under them,
// ai: the full width; on the desktop every button its label's width, LocalButtonsFill; without the camera's permission
// ai: the one button that gets it, why on the state line), 12 dp under
// ai: them the transfer's block (Details), then 4 dp apart the heat warning and one hint (a filing's failure over the
// ai: preview), 16 dp before the folds. A file in: its block only with Show statistics on.
@Composable
private fun TransferPanel(h: ReceiveHost) {
    Column(Modifier.fillMaxWidth().padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val access = h.access
        val kept = h.kept
        val wide = if (LocalButtonsFill.current) Modifier.fillMaxWidth() else Modifier
        if (access != null) Btn(access.act.label, Kind.Primary, modifier = wide) { access.act.run() }
        else {
            // ai: one solid button at a time: Start while nothing waits, else Open (the camera's area's, ReceivedPanel)
            Btn(if (h.running) "Stop" else "Start", if (!h.running && kept == null) Kind.Primary else Kind.Tonal,
                enabled = !h.starting && (h.running || h.startable), modifier = wide) { h.toggleCamera() }
        }
    }
    val ln = h.line
    // ai: an error is the preview's to say (errorOf), not the column's. The transfer as elements (2026-11-07, Details:
    // ai: the name, the meter and its percentage, a row a fact; "Receiving x" over "49%, 147 of 300 KB, 2 s left"
    // ai: before); a file in is the card's to say, its block here only with Show statistics on and the camera off
    val good = ln.tone == Readout.Tone.Good
    val shown = ln.text.isNotEmpty() && ln.tone != Readout.Tone.Bad && (!good || (h.showStats && !h.running))
    val now = if (shown) Details(ln.name ?: ln.text, test = ln.test, end = ln.pct, frac = ln.frac, rows = ln.rows) else null
    // ai: Within a run (Start until it stops) the block once shown keeps its room until the run ends, drawn clear
    // ai: when it falls silent (2026-11-02: Receiving, its meter and figures, and the hint each moved the folds under them
    // ai: as they came and went); none is held before it has had words, so a camera starting shows no gap (a run's whole
    // ai: room held from Start stood empty under Stop). The run's file in ends it at once, the kept row and the
    // ai: block's going in one frame (the camera stops a poll later); a run started over a file already kept holds on.
    val keptAtStart = remember(h.running) { h.kept != null }
    val run = h.running && (h.kept == null || keptAtStart)
    val last = remember(h.running) { arrayOfNulls<Details>(1) }
    val lastHint = remember(h.running) { arrayOfNulls<Warn>(1) }
    if (run && now != null) last[0] = now
    (now ?: last[0].takeIf { run })?.let { DetailsBlock(it, Modifier.padding(top = 12.dp), clear = now == null) }
    // ai: the warnings as rows under the block (2026-11-07; red lines and a sentence before): the heat, then the camera's
    // ai: hint, held still while the camera runs (steady), its room kept as the block's is, gone with it
    h.heat?.let { WarnRow(it, Modifier.padding(top = 4.dp)) }
    val hintKey = steady(h.hint?.let { "${it.label}\t${it.value}" }.orEmpty(), on = 0).takeIf { run }.orEmpty()
    val hint = hintKey.takeIf { it.isNotEmpty() }?.split('\t')?.let { Warn(it[0], it.getOrElse(1) { "" }, hint = true) }
    if (run && hint != null) lastHint[0] = hint
    (hint ?: lastHint[0].takeIf { run })?.let { WarnRow(it, Modifier.padding(top = if (h.heat == null) 4.dp else 0.dp), clear = hint == null) }
    Spacer(Modifier.height(16.dp))
}

// ai: The camera's crop as the decoder reads it, in its box on the soft fill, square corners (the web's #view): the
// ai: platform's picture; in 2:1 a red 1 dp line across the middle of the crop's long side, where the receiver cuts the
// ai: two halves; over it while the camera is off why, with its icon, and "Getting ready" with a spinner until the
// ai: receiver first answers (Readout.gettingReady)
@Composable
private fun PreviewBox(h: ReceiveHost, modifier: Modifier, portrait: Boolean) {
    val ratio = h.cropRatio(portrait)
    Box(modifier.clipToBounds().background(Soft), contentAlignment = Alignment.Center) {
        Box(Modifier.aspectRatio(ratio, matchHeightConstraintsFirst = true).clipToBounds(), contentAlignment = Alignment.Center) {
            h.Picture(Modifier.fillMaxSize(), portrait)
            if (h.twoToOne) Canvas(Modifier.matchParentSize()) {
                val wide = size.width >= size.height
                val a = if (wide) Offset(size.width / 2, 0f) else Offset(0f, size.height / 2)
                val b = if (wide) Offset(size.width / 2, size.height) else Offset(size.width, size.height / 2)
                drawLine(Color(0xFFE53935), a, b, strokeWidth = 1.dp.toPx())
            }
            // ai: the camera's access a state the preview says (its button the column's), no popup
            val err = h.line.takeIf { h.access != null && it.tone == Readout.Tone.Bad }?.text?.takeIf { it.isNotEmpty() }
            if (err != null || h.off.isNotEmpty() || h.waiting) Column(Modifier.matchParentSize().background(Soft).padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
                when {
                    err != null -> Icon(Icons.alert, contentDescription = null, Modifier.size(24.dp), tint = Bad)
                    h.off.isNotEmpty() -> Icon(Icons.cameraOff, contentDescription = null, Modifier.size(24.dp), tint = Muted)
                    else -> Spinner()
                }
                Text(err ?: h.off.ifEmpty { "Getting ready" }, style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
            }
        }
    }
}

// ai: What went wrong, said in a popup (NoteDialog, ReceiveScreen; since 2026-11-07: a fatal error pops up, a warning
// ai: that lets the run go on stays a row; a cover over the preview with the rail's red triangle from 2026-11-02, never
// ai: the column): the line's error (the phase's why, "Camera stopped: ...", "Reading
// ai: stopped") or a filing's refusal or a Save's failure (the note). The camera's access is no error: the preview says
// ai: it, its button in the column. The warnings (Throttling, Sender too fast) stay rows.
private fun errorOf(h: ReceiveHost): String? = h.line.takeIf { h.access == null && it.tone == Readout.Tone.Bad }?.text?.takeIf { it.isNotEmpty() }
    ?: h.note.takeIf { it.isNotEmpty() }

// ai: the red triangle in a rail's square, its words the description: the phone's heat, a receiver's error
@Composable
fun AlertSquare(text: String) = Square { Icon(Icons.alert, contentDescription = text, Modifier.size(24.dp), tint = Bad) }
