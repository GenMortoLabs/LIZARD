package dev.lizard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout
import kotlinx.coroutines.delay

// ai: What Send shows and does, the app's: the run (the sender's state), the platform's code area and how it is shown;
// ai: the column, its settings and full screen are ToolHost's (Tool.kt), as Receive's
interface SendHost : ToolHost {
    override val kind: ToolKind get() = SEND
    val busy: Boolean                  // ai: a send runs or is being prepared
    val on: Boolean                    // ai: a send runs
    val paused: Boolean
    val canStart: Boolean
    fun start()
    fun stop()
    fun pause(on: Boolean)
    val pauseShown: Boolean            // ai: Pause or Resume beside Stop (the desktop's while sending; the phone's paused)
    // ai: what goes, from Start until Stop (null otherwise): the file's or the test stream's block (Details: the name, a
    // ai: spinner while it is prepared, a row a fact), its rows the same from Start, so nothing moves as figures arrive
    val details: Details?
    // ai: what went wrong ("" none), said in a popup (NoteDialog, 2026-11-07: a cover over the code area that morning hid
    // ai: File, Folder and Media after a failed pick, and a cover cannot be closed; every error stops the stream, so the
    // ai: upload is what is under it); OK closes it (clearError, and the screen keeps that it said it: an error that
    // ai: stays, the desktop's missing native library, is said again each time Send opens)
    val error: String
    fun clearError() {}
    val reading: Boolean get() = false // ai: the pick being read (a dropped folder walked): a spinner, no words
    val railCapacity: Double?
    // ai: (`alone`, ToolHost's: the code alone on the screen, the phone's send running, the desktop's full screen) the code
    // ai: alone drawn as the landscape layout bare (ToolLandscape's `bare`), the code area never moved in the tree:
    // ai: the desktop's, whose code area is a heavyweight canvas a move destroys and makes again, its presenter stopped
    // ai: and begun anew (2026-11-02, every full-screen switch); the phone's CodeAlone otherwise
    val codeAloneInPlace: Boolean get() = false
    val edgeTap: Boolean               // ai: a tap near the left edge shows the rail over the code alone (the phone's)
    fun leaveCodeAlone()               // ai: the rail's collapser over the code alone (the phone's; the desktop's edge strip opens SideOver)
    val codePadding: Dp                // ai: the light round the code area in the page (the phone's 16 dp; the desktop's canvas fills)
    val codeFill: Color get() = Bg     // ai: the code area's fill at its left edge, which the column's margin beside it takes
    val edgeRoom: Dp get() = EdgeRoom  // ai: the column's margin past its edge (ToolLandscape's `room`; the desktop's the edge's whole strip)
    fun codeSize(maxWidth: Dp, maxHeight: Dp, portrait: Boolean): DpSize?   // ai: the code area's size in its room, null to fill it
    val rootModifier: Modifier         // ai: the platform's on the whole screen (the desktop's drops)
    @Composable fun CodeArea(modifier: Modifier)
}

// ai: Send (the web's send.html; both apps' since 2026-10-31): a column (Start, or Stop and Pause or Resume while sending,
// ai: beside Fullscreen where the platform has one; the state line naming the file; the figures; the heat warning;
// ai: Settings in two groups, Experimental, Developer Tools, About) and the code area. Landscape: the column on the left,
// ai: its edge dragged, the code area on the right; collapsed, the rail (play or pause, the capacity, Home at its foot).
// ai: Portrait (a phone): the bar over one column, the code at its top. The code alone: the code area the whole screen.
@Composable
fun SendScreen(h: SendHost) = Box(Modifier.fillMaxSize().then(h.rootModifier)) {
    // ai: one code area for every layout: a turn of the phone moves its surface rather than making another
    val code = remember(h) { movableContentOf<Modifier> { m -> h.CodeArea(m) } }
    BoxWithConstraints {
        val portrait = h.canPortrait && maxHeight > maxWidth
        if (h.alone && !(h.codeAloneInPlace && !portrait)) CodeAlone(h, code)
        else if (portrait) ToolPortrait(h.kind.title, { h.goHome() }, { ToolSwap(h) }) {
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val size = h.codeSize(maxWidth, maxHeight, true)
                code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxWidth())
            }
            SendSide(h)
        } else ToolLandscape(h.kind.title, h.folds.open(FoldKey.Collapsed), { h.folds.toggle(FoldKey.Collapsed) }, { h.goHome() },
            if (h.swapInColumn) ({ ToolSwap(h) }) else null, h.width,
            columnFloor(h.width),
            mainPadding = h.codePadding, room = h.edgeRoom, roomFill = h.codeFill, bare = h.alone, mainModifier = h.areaModifier,
            rail = { SendRail(h) }, side = { SendSide(h, home = true) }) {
            val size = h.codeSize(maxWidth, maxHeight, false)
            code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxSize())
        }
    }
    val err = h.error
    var said by remember(h) { mutableStateOf("") }
    LaunchedEffect(err) { if (err.isEmpty()) said = "" }
    if (err.isNotEmpty() && err != said) NoteDialog(err) { said = err; h.clearError() }
}

// ai: The code alone (2026-10-26 on the phone, as the desktop's full screen): on the page's white, as large as the screen
// ai: allows, clear of a display's cutout; where the platform taps (the phone), a tap within EDGE of the left edge shows
// ai: the collapsed column over the code (its collapser, Pause or Resume, the rate, Home), a tap elsewhere or RAIL_MS
// ai: untouched hides it
@Composable
private fun CodeAlone(h: SendHost, code: @Composable (Modifier) -> Unit) {
    var rail by remember { mutableStateOf(false) }
    var touched by remember { mutableStateOf(0) }
    LaunchedEffect(rail, touched) { if (rail) { delay(RAIL_MS); rail = false } }
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg).then(if (h.edgeTap) Modifier.pointerInput(Unit) {
        detectTapGestures { at -> rail = at.x < EDGE.toPx(); touched++ }
    }.windowInsetsPadding(WindowInsets.displayCutout) else Modifier), contentAlignment = Alignment.Center) {
        val size = h.codeSize(maxWidth, maxHeight, false)
        code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxSize())
        if (rail) Row(Modifier.align(Alignment.CenterStart).fillMaxHeight().background(Bg).pointerInput(Unit) { detectTapGestures { touched++ } }) {
            Rail(onClick = { rail = false; h.leaveCodeAlone() }, foot = { Square { IconBtn(Icons.back, "Home") { h.goHome() } } }) { SendRail(h) }
            VerticalDivider(color = Line)
        }
    }
}

// ai: how near the left edge a tap shows the rail over the code alone, and how long it stays untouched
private val EDGE = 72.dp
private const val RAIL_MS = 4000L

// ai: The rail's squares: the heat's red triangle while the phone throttles, play to start (or to resume), pause while
// ai: sending, and the code's capacity
@Composable
fun SendRail(h: SendHost) {
    h.heat?.let { AlertSquare(it.text) }
    Square {
        IconBtn(if (h.on && !h.paused) Icons.pause else Icons.play, if (!h.on) "Start" else if (h.paused) "Resume" else "Pause",
            enabled = h.on || h.canStart) { if (h.on) h.pause(!h.paused) else h.start() }
    }
    RateSquare(h.railCapacity)
}

@Composable
internal fun SendSide(h: SendHost, home: Boolean = false) {
    Spacer(Modifier.height(12.dp))
    val fill = LocalButtonsFill.current
    Row(if (fill) Modifier.fillMaxWidth() else Modifier, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        val grow = if (fill) Modifier.weight(1f) else Modifier
        if (!h.busy) Btn("Start", Kind.Primary, enabled = h.canStart, modifier = grow) { h.start() }
        else {
            Btn("Stop", modifier = grow, pad = 16.dp) { h.stop() }
            if (h.pauseShown) {
                if (h.paused) Btn("Resume", Kind.Primary, pad = 16.dp) { h.pause(false) }
                else Btn("Pause", enabled = h.on, pad = 16.dp) { h.pause(true) }
            }
        }
        FullscreenBtn(h)
    }
    // ai: what goes as elements (2026-11-07, Details; "Sending x, 2.4 GB (4.4 GB raw)" over "1.69 MB/s, pass 2" before),
    // ai: or a spinner while the pick is read; an error a popup (SendScreen), never the column's
    val d = h.details
    if (d != null) DetailsBlock(d, Modifier.padding(top = 12.dp))
    else if (h.reading) CircularProgressIndicator(Modifier.padding(top = 12.dp).size(16.dp), color = Fg, strokeWidth = 2.dp, trackColor = Line)
    h.heat?.let { WarnRow(it, Modifier.padding(top = 8.dp)) }
    Spacer(Modifier.height(16.dp))
    ToolFolds(h, home)
}

// ai: The senders' file upload (2026-10-26, LocalSend's geometry; the web's ui.css .tile and .itile): the tiles (File,
// ai: Folder and where the platform has one Media) in one row stretched across the box in equal cells, and under them
// ai: the selection's card as wide, always (2026-11-07; while it held something before), "0 items" and a row's empty room
// ai: while it holds nothing, Clear dimmed: the count with Clear, then a tile an item (its icon, its
// ai: name with the stem cut and the ending kept, the line under it, an x that removes it), in rows stretched across the
// ai: card (TileRows, TILE dp the least), or with `row` (the phone's, held either way since 2026-11-07; in landscape
// ai: alone before, its portrait card scrolling down) one row scrolling sideways across the screen as it is held;
// ai: nothing changes while sending. The phone's at the top of its box, 16 dp from its edges, stretched across;
// ai: `column` (the desktop's, 576 dp) a column
// ai: at most that wide, centred, and the whole centred in its height where it fits, the card at least a row of items
// ai: tall; only the card scrolls. `total`: the selection's size, after the count in the card's head ("2 items · 4.4 GB",
// ai: 2026-11-07; a line under the card named what goes and its size before).
// ai: Both apps draw it in Compose (the desktop's with AWT on its code canvas until 2026-11-02).
class UploadItem(val key: String, val name: String, val sub: String, val dir: Boolean)

@Composable
fun Upload(tiles: List<Act>, items: List<UploadItem>, enabled: Boolean, row: Boolean, onRemove: (String) -> Unit, onClear: () -> Unit,
           column: Dp? = null, total: String = "") {
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg)) {
        val w = if (column != null) minOf(maxWidth - 32.dp, column) else maxWidth - 32.dp
        Column(Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = if (column != null) Arrangement.spacedBy(16.dp, Alignment.CenterVertically) else Arrangement.spacedBy(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                tiles.forEach { t -> Tile(t.icon ?: Icons.file, t.label, (w - 12.dp * (tiles.size - 1)) / tiles.size, enabled = enabled, onClick = t.run) }
            }
            Column(Modifier.width(w).weight(1f, fill = false), verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally) {
                Column(Modifier.fillMaxWidth().weight(1f, fill = false).then(if (column != null) Modifier.heightIn(min = (4 + 48 + 4 + TILE + 12).dp) else Modifier)
                    .clip(RoundedCornerShape(12.dp)).background(Card).padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp)) {
                    Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Row(Modifier.weight(1f)) {
                            Text("${items.size} ${if (items.size == 1) "item" else "items"}", style = MaterialTheme.typography.labelMedium)
                            if (total.isNotEmpty()) Text(" · $total", style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1)
                        }
                        Btn("Clear", Kind.Text, enabled = enabled && items.isNotEmpty()) { onClear() }
                    }
                    val item: @Composable (Int, Dp) -> Unit = { i, iw ->
                        val it = items[i]
                        ItemTile(it.name, it.sub, if (it.dir) Icons.folder else Icons.file, iw, enabled = enabled) { onRemove(it.key) }
                    }
                    if (row && items.isEmpty()) Spacer(Modifier.padding(top = 4.dp).height(TILE.dp))   // ai: a row's room, empty
                    else if (row) {
                        // ai: one row at the columns' width (TileRows' rule), scrolling sideways past what fits
                        val gap = 8.dp; val inner = w - 24.dp
                        val cols = maxOf(1, ((inner + gap) / (TILE.dp + gap)).toInt())
                        Row(Modifier.padding(top = 4.dp).horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(gap)) {
                            items.indices.forEach { item(it, (inner - gap * (cols - 1)) / cols) }
                        }
                    } else Box(Modifier.padding(top = 4.dp).weight(1f, fill = false).verticalScroll(rememberScrollState())) { TileRows(items.size, w - 24.dp, 8.dp, item) }
                }
            }
        }
    }
}

// ai: the test stream's note where the file upload would be (both apps): the file icon and "Test stream" on the soft fill
@Composable
fun TestStreamNote() {
    Column(Modifier.fillMaxSize().background(Soft), verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.file, contentDescription = null, Modifier.size(24.dp), tint = Muted)
        Text("Test stream", style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 16.dp))
    }
}

// ai: the upload's height with one row of items: its padding, the tiles, the gap, the card's head and one row
val UPLOAD_ONE_ROW = (16 + 104 + 16 + 4 + 48 + 4 + 104 + 12 + 16).dp

const val TILE = 104

// ai: a Tile is a button TILE dp tall, an icon over one word; as wide as its row gives it
@Composable
fun Tile(icon: ImageVector, label: String, width: Dp, enabled: Boolean = true, solid: Boolean = false, onClick: () -> Unit) {
    // ai: solid: the screen's one solid button as a tile (Receive's Open, 2026-11-06), the foreground's fill
    val tint = if (!enabled) Fg.copy(alpha = .38f) else if (solid) Bg else Fg
    Column(Modifier.width(width).height(TILE.dp).clip(RoundedCornerShape(12.dp)).background(if (solid) Fg else Soft).clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, contentDescription = null, Modifier.size(32.dp), tint = tint)
        Text(label, style = MaterialTheme.typography.labelLarge, color = tint)
    }
}

// ai: an ItemTile is one item on the selection's card: its icon, its name in medium with the stem cut and the ending
// ai: kept (Readout.nameParts), the line under it regular, and an x at its corner that removes it
@Composable
fun ItemTile(name: String, sub: String, icon: ImageVector, width: Dp, enabled: Boolean = true, onRemove: (() -> Unit)?) {
    val (stem, end) = Readout.nameParts(name)
    val name13 = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium)
    Box(Modifier.width(width).height(TILE.dp).clip(RoundedCornerShape(12.dp)).background(Bg)) {
        Column(Modifier.fillMaxSize().padding(8.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, contentDescription = null, Modifier.padding(bottom = 6.dp).size(28.dp), tint = Muted)
            Row {
                Text(stem, style = name13, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                Text(end, style = name13, color = Fg, maxLines = 1)
            }
            Text(sub, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Normal), color = Muted, maxLines = 1)
        }
        // ai: no x where nothing removes it (Receive's received file, 2026-11-06)
        if (onRemove != null) Box(Modifier.align(Alignment.TopEnd).padding(2.dp).size(28.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onRemove)
            .semantics { contentDescription = "Remove $name" }, contentAlignment = Alignment.Center) {
            Icon(Icons.x, contentDescription = null, Modifier.size(16.dp), tint = if (enabled) Fg else Fg.copy(alpha = .38f))
        }
    }
}

// ai: n things in rows across `width`, each `gap` apart: as many columns as fit at TILE dp or more, each column stretched
// ai: to share the width, so a full row fills it and a short last row keeps the columns' width; cell(i, w) draws the i-th
// ai: at width w (the web's grid: repeat(auto-fill, minmax(104px, 1fr)))
@Composable
fun TileRows(n: Int, width: Dp, gap: Dp, cell: @Composable (Int, Dp) -> Unit) {
    val cols = maxOf(1, ((width + gap) / (TILE.dp + gap)).toInt())
    val w = (width - gap * (cols - 1)) / cols
    Column(verticalArrangement = Arrangement.spacedBy(gap)) {
        for (r in 0 until (n + cols - 1) / cols) Row(horizontalArrangement = Arrangement.spacedBy(gap)) {
            for (i in r * cols until minOf(n, (r + 1) * cols)) cell(i, w)
        }
    }
}
