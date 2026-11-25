package dev.lizard.desktop

import dev.lizard.pick.Pick
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.lizard.readout.Readout
import dev.lizard.tar.Bundle
import dev.lizard.tar.Stoppable
import dev.lizard.tar.Stopped
import dev.lizard.tar.TarEntry
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import java.awt.Canvas
import java.awt.datatransfer.DataFlavor
import java.awt.dnd.DnDConstants
import java.awt.dnd.DropTarget
import java.awt.dnd.DropTargetAdapter
import java.awt.dnd.DropTargetDropEvent
import javax.swing.SwingUtilities
import java.awt.EventQueue
import java.awt.Graphics
import java.awt.event.ComponentAdapter
import java.awt.event.ComponentEvent
import java.awt.event.HierarchyEvent
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.io.File
import java.nio.file.Files
import javax.swing.Timer
import kotlin.concurrent.thread
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asComposeImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import dev.lizard.ui.*

// ai: The code area's window while codes play: a heavyweight AWT canvas, a native window of its own inside the Compose
// ai: window (Screen.kt's SwingPanel), on which the presenter's swapchain draws. In the window only while a presenter
// ai: runs (`want`, 2026-11-02, as the phone's SurfaceView): idle, the area is Compose's, the file upload or the test
// ai: stream's note, under no window of its own (that look painted here by AWT went, its every resize a frame of bare
// ai: fill before the paint, a blink of the test stream's note on each step of a dragged edge; a hidden canvas would not
// ai: do, its SwingPanel's container painting AWT's grey over Compose's area). White, nothing of AWT painted on it.
// ai: Before its window goes
// ai: (removeNotify: a dispose, a window made again) it stops the presenter on it. A click on it does nothing; files and
// ai: folders dropped on it are taken (a run stops first).
class CodeCanvas(private val s: SendState) : Canvas() {
    // ai: how near the left edge the pointer shows full screen's edge strip, the window's units (FullScreen.kt EDGE_NEAR)
    private val near = 24f
    // ai: wanted in the window (a presenter is to draw into it; Screen.kt composes its SwingPanel while it is)
    var shown by mutableStateOf(false)
        private set

    init {
        background = java.awt.Color.WHITE
        isFocusable = false
        addComponentListener(object : ComponentAdapter() {
            override fun componentResized(e: ComponentEvent) = s.canvasMoved()
            override fun componentMoved(e: ComponentEvent) = s.canvasMoved()
        })
        if (Native.mac) addHierarchyBoundsListener(object : java.awt.event.HierarchyBoundsAdapter() {
            override fun ancestorMoved(e: HierarchyEvent) = s.canvasMoved()
            override fun ancestorResized(e: HierarchyEvent) = s.canvasMoved()
        })
        // ai: full screen's edge strip near the left edge, heard here over the canvas as the main area hears it elsewhere
        // ai: (FullScreen.kt edgeModifier; the column's edge is Compose's: the canvas begins past its strip, SendHost.edgeRoom)
        addMouseMotionListener(object : MouseAdapter() {
            override fun mouseMoved(e: MouseEvent) = s.full.pointerAt(e.x.toFloat(), near)
        })
        // ai: Files and folders dropped are added (a run stops first, as the web's does; refused while sending until
        // ai: 2026-10-23); the window's column and the idle area take them too (Screen.kt)
        dropTarget = DropTarget(this, object : DropTargetAdapter() {
            override fun dragEnter(e: java.awt.dnd.DropTargetDragEvent) = s.dragging(true)
            override fun dragExit(e: java.awt.dnd.DropTargetEvent) = s.dragging(false)
            override fun drop(e: DropTargetDropEvent) {
                s.dragging(false)
                e.acceptDrop(DnDConstants.ACTION_COPY)
                e.dropComplete(s.dropped(e.transferable))
            }
        })
    }

    // ai: put in the window for a presenter (its window made at addNotify, which begins it: SendState.canvasMade), taken
    // ai: out at the presenter's end
    fun want(on: Boolean) { shown = on }

    // ai: its X background the area's colour (SendState.codeFill: white round the presenter's code, which clears to
    // ai: white), so what X fills on a map or a resize before anything draws is that (AWT sets the desktop theme's grey:
    // ai: a dark frame where the canvas came, 2026-11-02)
    private fun look() { if (s.loadError.isEmpty() && isDisplayable) runCatching { Native.windowBackground(this, s.codeFill.toArgb() and 0xFFFFFF) } }

    // ai: nothing painted by AWT: the presenter's pictures are the canvas's (Canvas's own paint clears it)
    override fun update(g: Graphics) {}
    override fun paint(g: Graphics) {}

    // ai: its window made (a presenter wanted): its X background the area's colour (Native.windowBackground), then the
    // ai: presenter begun on it
    override fun addNotify() {
        super.addNotify()
        look()
        EventQueue.invokeLater { s.canvasMade() }
    }
    override fun removeNotify() { s.canvasGone(); super.removeNotify() }

    // ai: macOS: where the presenter's layer goes (native/macos.h Place): the canvas's bounds in points from the top left
    // ai: of its window's content (the frame's insets are its title bar), and the screen's scale
    fun place(): FloatArray {
        val win = SwingUtilities.getWindowAncestor(this) ?: return FloatArray(5)
        val at = SwingUtilities.convertPoint(this, 0, 0, win)
        val ins = win.insets
        val k = graphicsConfiguration?.defaultTransform?.scaleX ?: 1.0
        return floatArrayOf((at.x - ins.left).toFloat(), (at.y - ins.top).toFloat(), width.toFloat(), height.toFloat(), k.toFloat())
    }
}

// ai: The run (the web's send.mjs, lizard-android/.../Send.kt SendState): the settings, kept (Prefs), the sender and
// ai: the presenter, and what the column shows. Start makes the sender (txCreate; txPrepare off the UI thread whatever
// ai: the encoder: the presenter shows the code from the painter's device), then, once the code area has a size and is
// ai: shown, configures the format, and starts the presenter on the canvas. A change of the area's
// ai: size, Size, Spacing, Margin, Ring, Codes or Blocks re-picks, and configures where the format changed (a configure
// ai: drops the frames painted ahead; the transfer goes on, as the web's re-pick); FPS configures (the word states it)
// ai: and tells the presenter. Every Native call here is on the UI thread but txCreate and txPrepare; the native side
// ai: holds one lock for configure and the presenter's take. `full`: the window's full screen, Send's and Receive's
// ai: (FullScreen.kt, 2026-11-13; this class's own before), the code alone in it as the web's and the app's.
class SendState(val full: FullScreen) {
    sealed interface Phase { data object Idle : Phase; data object Preparing : Phase; data object On : Phase; data class Error(val why: String) : Phase }
    // ai: what is painted: the format configured (its picture the sender's own, lizard_n_for, in its stats)
    data class Format(val subch: Int, val span: Int, val fps: Int, val codes: Int, val gap: Int, val painter: Int)
    companion object {
        // ai: Scaling's choices in the presenter's order (presenter.h Scaling: 0 Bilinear, 1 Lanczos, 2 None)
        val SCALINGS = listOf("bilinear", "lanczos", "none")
    }

    // ai: What Start sends (2026-10-16): one file as itself, or several files or a folder as one tar
    // ai: (../liblizard/kotlin/Tar.kt Bundle, the web sender's rule), made from the selection (items); `reading` while
    // ai: picks wanted are walked or the bundle is made
    var choice by mutableStateOf<Bundle?>(null)
    private var singleFile: File? = null   // ai: the one file the choice is, where it is one
    var reading by mutableStateOf(false)
    // ai: The selection (2026-10-23; until then a pick replaced the last): items added by each pick, drop or path handed
    // ai: in (Main.kt), in order, each a file or a folder under its own name, files and folders mixed. An item already
    // ai: listed (its canonical path) is not added again. Links: one picked is followed; those inside a folder are
    // ai: skipped (never followed, so a loop of them cannot make a bundle without end) and counted on its row. A folder
    // ai: that cannot be listed is an error, the list left as it was. Every change stops a run first, then the bundle is
    // ai: made off the UI thread and its transfer made again (make), as the web's does.
    class Item(val key: String, val file: File, val name: String, val dir: Boolean, val files: Int, val size: Long, val links: Int, val entries: List<TarEntry>)
    val items = mutableStateListOf<Item>()
    // ai: A pick's file or folder being walked (2026-11-16): its tile on the card at once, what the walk has found so
    // ai: far (its thread's fields, which the card's ticking reads), its x cancelling it (the walk stops at its next
    // ai: entry, the item dropped). Start waits only on the walks still wanted: a whole disk picked by mistake held it
    // ai: until the walk ended before, with no way to stop it.
    class Walking(val key: String, val name: String) {
        @Volatile var files = 0
        @Volatile var bytes = 0L
        @Volatile var cancelled = false
    }
    val walking = mutableStateListOf<Walking>()
    private var walkSeq = 0
    private var bundling = 0        // ai: the newest bundle asked for; bundleNow while it is being made
    var bundleNow by mutableStateOf(false)
        private set
    // ai: the items the choice was made from: one added since shows loading while the bundle is made
    var bundled by mutableStateOf<Set<String>>(emptySet())
        private set
    private fun readingNow() { reading = walking.isNotEmpty() || bundleNow }
    // ai: a drag over the window (the canvas's drop target, Screen.kt's for the rest): the idle area's outline
    var dragOver by mutableStateOf(false)
        private set
    fun dragging(on: Boolean) { dragOver = on }
    // ai: what a drop hands over: its files and folders added (the test stream off); false where it holds none
    fun dropped(t: java.awt.datatransfer.Transferable): Boolean {
        @Suppress("UNCHECKED_CAST")
        val files = runCatching { t.getTransferData(DataFlavor.javaFileListFlavor) as List<File> }.getOrNull().orEmpty()
        if (files.isEmpty()) return false
        // ai: a stream file dropped plays (Source Stream); anything else is a pick
        if (files.size == 1 && streamOf(files[0]) != null) { openStream(files[0]); return true }
        if (source == "stream") { chooseSource("files") }
        if (test) chooseTest(false)
        pick(files)
        return true
    }
    var fileSize by mutableStateOf(0L)
    // ai: The transfer made at Start (2026-11-01; at the pick and again at each stop from 2026-10-19, every source block
    // ai: compressed for picks never sent): the bundle's tar written and the file packed where that pays (txCreate,
    // ai: liblizard/core/tx/pack.h) on a thread of its own, the line "Preparing" meanwhile. compressed: the bytes the transfer
    // ai: carries (the packed file's, else the file's), 0 until a run makes it, kept through the run and after it until
    // ai: the choice changes. A new pick or the window's close lets a made one go.
    var compressed by mutableStateOf(0L)
    private var made = 0L                 // ai: the transfer made for the choice, not yet sending (0: none)
    private var madeTar: File? = null     // ai: its tar, where the choice is a bundle
    @Volatile private var making = 0      // ai: the make under way's number: a newer make or pick orphans it (its tar's write reads it)
    private var makingNow = false
    private var startWhenMade = false     // ai: Start pressed while the transfer was being made
    private var closing = false
    var test by mutableStateOf(Prefs.payload == "test")
    var enc by mutableStateOf(Prefs.enc.let { if (it == "gpu" || it == "cpu") it else "auto" })
    // ai: Send's Control fold (2026-11-18, lizard-ui controlSettings): Source, files (the picks, painted as they go) or
    // ai: stream (a stream file played, `stream`: chosen in the code area, dropped, or opened with LIZARD, App.handIn);
    // ai: Export (Start writes the frames into a stream file in Downloads, named after what goes, "<name>.lizard" under a
    // ai: free name), Laps, Maximum size, Hide codes (On: no screen, the frames made as fast as the painter goes, the run ending with the
    // ai: file). exportNow: the run's export as the sender says it (txExportState), kept after the run until the next
    // ai: Start or change, so its block says where the file went.
    class StreamPick(val file: File, val meta: StreamMeta)
    var source by mutableStateOf(if (Prefs.source == "stream") "stream" else "files")
    var stream by mutableStateOf<StreamPick?>(null)
    var export by mutableStateOf(Prefs.export)
    var cycles by mutableStateOf(cyclesOf(Prefs.cycles))
    var hideCodes by mutableStateOf(Prefs.hideCodes)
    var exportLimit by mutableStateOf(fileGb(Prefs.exportLimit))
    var exportNow by mutableStateOf<JsonObject?>(null)
    private var exportFile: File? = null
    private var runPlay = false     // ai: the run plays `stream`
    private var runExport = false   // ai: the run exports
    private var runDisplay = true   // ai: the run shows its frames
    // ai: the painter the sender runs on now, "gpu" or "cpu" ("" before a send): the Encoder chip auto marks (2026-10-05)
    val painterNow: String get() = sender?.str("painter") ?: ""
    // ai: a version 4 to 64 (16 sub-channels a step, Pick.STEP), Pick.VERSION_DEFAULT unless chosen; no auto since
    // ai: 2026-11-07 (a kept 0, auto's, reads as the default)
    var blocks by mutableStateOf(Prefs.blocks.let { if (it in Pick.VERSION_MIN..Pick.VERSION_MAX) it else Pick.VERSION_DEFAULT })
    var fps by mutableStateOf(if (Prefs.vsync) Pick.fpsOf(Prefs.fps) else Prefs.fps.coerceIn(1, 60))
    // ai: VSync (2026-10-23, on by default): while the code is shown its monitor holds a refresh the rate divides, the
    // ai: nearest 60 Hz, where its own does not (a 165 Hz laptop at 60: pictures of 3, 3, 2 and 3 refreshes), and is put
    // ai: back at the stop (presenter.h setVsync); off, the monitor is left as it is. FPS is Pick.FPS's chips under VSync,
    // ai: a 1 to 60 slider without it; VSync turned on takes a slider's rate to the nearest chip.
    var vsync by mutableStateOf(Prefs.vsync)
    // ai: Scaling (2026-11-17, Performance): "bilinear", the blit's linear filter (the default); "lanczos", Lanczos-3 on the
    // ai: grey picture; "none", the picture at its own pixels (presenter.h Scaling), handed to the presenter at once
    var scaling by mutableStateOf(Prefs.scaling.takeIf { it in SCALINGS } ?: "bilinear")
    var size by mutableStateOf(Prefs.size.coerceIn(25, 100))
    var spacing by mutableStateOf(Prefs.spacing.coerceIn(Room.SPACING_MIN, Room.SPACING_MAX))
    var margin by mutableStateOf(Prefs.margin.coerceIn(Pick.QUIET, Room.MARGIN_MAX))
    var ring by mutableStateOf(Prefs.ring.let { if (it in Pick.RINGS.indices) it else -1 })
    var codes by mutableStateOf(if (Prefs.codes == 2) 2 else 1)
    // ai: Refresh (2026-11-17, lizard-ui rateRefreshSettings; Rate refresh of 2026-11-12 before): each code rate's delay,
    // ai: its blocks new every n-th picture, 6/7, 3/4, 2/3, 1/2, 1 to REFRESH_MAX; handed to the sender
    // ai: as it is made and at each change (refreshNow), kept
    var refresh by mutableStateOf(refreshOf(Prefs.refresh))
    // ai: Developer Tools' Seed (2026-11-14, lizard-ui seedSetting): a file's schedule drawn from it at each Start (txSeed,
    // ai: before the first frame), so a send can be repeated; "" a random one each send; kept
    var seed by mutableStateOf(Prefs.seed)
    // ai: Code's Colours (2026-11-16, lizard-ui Colours): the code's ink and its paper, 0xRRGGBB, kept; handed to each
    // ai: sender made and at each change (the sender's colours are atomics: no lock, any time), the presenter painting
    // ai: them from the sender (expand.slang, the area round the code the paper)
    var ink by mutableStateOf(Prefs.ink)
    var paper by mutableStateOf(Prefs.paper)
    var toolsOpen by mutableStateOf(Prefs.toolsOpen)
    var experimentalOpen by mutableStateOf(Prefs.experimentalOpen)
    var collapsed by mutableStateOf(Prefs.collapsed)

    var phase by mutableStateOf<Phase>(Phase.Idle)
    var paused by mutableStateOf(false)
    // ai: the window maximized, as a title bar's double click does (Main.kt sets it; the test hook's `max` step)
    var maximize: (() -> Unit)? = null
    // ai: the window brought to the front, as paths handed in by a later launch arrive (Main.kt)
    var raise: (() -> Unit)? = null
    var areaW by mutableStateOf(0)
    var areaH by mutableStateOf(0)
    var fmt by mutableStateOf<Format?>(null)
    var present by mutableStateOf<JsonObject?>(null)
    var sender by mutableStateOf<JsonObject?>(null)
    // ai: the stats as the native side wrote them, the last second's (the test hook prints them)
    var presentRaw = ""
        private set
    var senderRaw = ""
        private set
    var loadError = ""

    val canvas = CodeCanvas(this)
    private var h = 0L
    private var p = 0L
    private var gen = 0
    private var gpuWhy = ""
    // ai: the GPU painter made for this run (txPrepare), and a prepare under way on a thread of its own: while one is,
    // ai: stop leaves the sender for that thread to destroy (it holds the sender's lock inside txPrepare)
    private var prepared = false
    private var preparing = 0L
    private var runTest = false
    private var runName = ""
    private val timer = Timer(1000) { poll() }

    val presenting get() = p != 0L
    // ai: the pictures a second the run shows: a stream's own, else FPS
    private val runFps get() = if (runPlay) stream?.meta?.fps ?: fps else fps
    // ai: the code area's colour at its left edge (the column's margin beside it takes it, SendHost.codeFill; the canvas's
    // ai: X background): the soft fill under the test stream's note, white under the file upload, the code's paper round
    // ai: the presenter's code (it clears to the paper) and round a paused run's still
    val codeFill: Color get() = if (canvas.shown || stamp != null) colourOfRgb(paper) else if (test) Soft else Bg
    val sending get() = phase == Phase.On || phase == Phase.Preparing
    // ai: a stream file is what plays: Source Stream with the test stream off (Control's Test stream, 2026-11-19, sends
    // ai: the test stream whatever Source says)
    val playing get() = source == "stream" && !test
    val canStart get() = loadError.isEmpty() && !sending && (if (playing) stream != null else (test || (choice != null && !reading)))
    // ai: Export offered: files painted as they go (a stream is painted already), not the test stream (it has no cycle)
    val canExport get() = source != "stream" && !test
    // ai: the run exports with no screen: the code area blank, no presenter, no Pause
    val blind get() = sending && runExport && !runDisplay
    val ringIndex get() = if (ring in Pick.RINGS.indices) ring else Pick.RING_DEFAULT
    // ai: the margin past the quiet zone the presenter keeps at the area's edges, for the ring configured
    private val edge get() = edgeShare
    // ai: the light the presenter keeps past the frame at each edge, a share of its height (presenter.h PresentOptions::edge)
    val edgeShare get() = Room.edgeShare(fmt?.span ?: Pick.span(ringIndex), margin).toFloat()
    // ai: what the code offers at most, KB/s: codes x blocks a frame x 460 B x the pictures a second it can show, the
    // ai: asked rate or the measured refresh where lower, so a limit outside the sender shows in the figure
    // ai: (2026-10-04), x the share of blocks new in a frame (freshShare); null before a configure
    val capacityKBs: Double? get() {
        val f = fmt ?: return null
        val hz = present?.num("hz") ?: 0.0
        return f.codes * sentBlocks * Pick.PAYLOAD.toDouble() * (if (hz > 0) min(f.fps.toDouble(), hz) else f.fps.toDouble()) * freshShare / 1000
    }
    // ai: the share of a frame's blocks new in each frame under Rate refresh (the sender's stats' fresh, 2026-11-12), 1
    // ai: before the stats or with none held: a held rate's repeats carry nothing new
    private val freshShare: Double get() = sender?.num("fresh")?.takeIf { it > 0 } ?: 1.0
    // ai: the blocks a code of the painted format carries, from the sender's stats, the configured format's before the
    // ai: first stats; its profile, "" none
    val sentBlocks: Int get() = sender?.num("blocks")?.toInt()?.takeIf { it > 0 } ?: fmt?.let { Native.blocksFor(it.subch) } ?: 0
    val tiers: String get() = sender?.str("tiers").orEmpty()

    // ai: The chooser (Chooser.kt): several files, or several folders, over the sender's window
    // ai: the window (App.kt sets it): the choosers' owner (the canvas has a window only while codes play)
    var frame: java.awt.Component? = null
    // ai: the app's folds (App.kt sets it), for the test hook's fold step
    var folds: Folds? = null
    fun chooseFile() = frame?.let { Chooser.choose(it, "Choose files", false, true) { f -> pick(f) } }
    fun chooseFolder() = frame?.let { Chooser.choose(it, "Choose folders", true, true) { f -> pick(f) } }

    // ai: Files and folders added (picked, dropped, handed in), walked off the UI thread, a tile each meanwhile
    fun pick(files: List<File>) {
        if (sending) stop()
        if (phase is Phase.Error) phase = Phase.Idle
        val ws = files.map { Walking("walk:${++walkSeq}", rootName(it)) }
        walking.addAll(ws); readingNow()
        thread(name = "lizard-send-pick", isDaemon = true) {
            val got = runCatching { files.zip(ws).mapNotNull { (f, w) -> itemOf(f, w) } }
            EventQueue.invokeLater {
                walking.removeAll(ws); readingNow()
                got.onSuccess { list ->
                    val have = items.mapTo(HashSet()) { it.key }
                    val fresh = list.filter { have.add(it.key) }
                    if (fresh.isNotEmpty()) { items.addAll(fresh); changed() }
                }.onFailure { phase = Phase.Error("Files unreadable: ${it.message}") }
            }
        }
    }
    // ai: an item removed, or a walk under way cancelled (its tile's x)
    fun remove(key: String) {
        val w = walking.firstOrNull { it.key == key }
        if (w != null) { w.cancelled = true; walking.remove(w); readingNow(); return }
        items.removeAll { it.key == key }; changed()
    }
    fun clearPicks() { walking.forEach { it.cancelled = true }; walking.clear(); items.clear(); changed() }
    // ai: a pick's name: its own; a drive's root, which has none ("/", "C:\"), its letters or "root" (2026-11-16: the
    // ai: empty name failed the bundle)
    private fun rootName(f: File) = f.name.ifEmpty { f.path.filter { it.isLetterOrDigit() }.ifEmpty { "root" } }
    // ai: a pick walked into its entries, counted into w as they are found; null where w was cancelled
    private fun itemOf(f: File, w: Walking): Item? {
        val top = f.canonicalFile
        var links = 0
        val out = ArrayList<TarEntry>()
        fun walk(x: File, rel: String) {
            if (w.cancelled) throw Stopped()
            if (x.isDirectory) {
                out.add(TarEntry(rel, mtimeSec = x.lastModified() / 1000))
                val kids = x.listFiles() ?: throw java.io.IOException("$rel could not be read")
                for (k in kids.sortedBy { it.name }) if (Files.isSymbolicLink(k.toPath())) links++ else walk(k, "$rel/${k.name}")
            } else if (x.isFile) {
                val n = x.length()
                out.add(TarEntry(rel, n, x.lastModified() / 1000) { x.inputStream() })
                w.files++; w.bytes += n
            }
        }
        try { walk(top, w.name) } catch (e: Exception) { if (w.cancelled) return null; throw e }
        if (out.isEmpty()) throw java.io.IOException("${w.name} is not a file or a folder")
        return Item(top.path, top, w.name, out[0].dir, out.count { !it.dir }, out.filter { !it.dir }.sumOf { it.size }, links, out)
    }
    // ai: the selection changed: the run stopped, the made transfer let go, the bundle made again off the UI thread (its
    // ai: transfer waits for Start); a bundle that fails is said (2026-11-16: its thread died silently before, Start
    // ai: greyed for good)
    private fun changed() {
        if (sending) stop()
        dropMade()
        if (phase is Phase.Error) phase = Phase.Idle
        choice = null; singleFile = null; fileSize = 0; compressed = 0
        val list = items.toList()
        val my = ++bundling
        bundleNow = list.isNotEmpty(); readingNow()
        if (list.isEmpty()) { bundled = emptySet(); return }
        thread(name = "lizard-send-bundle", isDaemon = true) {
            val got = runCatching { Bundle(list.flatMap { it.entries }) }
            EventQueue.invokeLater {
                if (my != bundling) return@invokeLater
                bundleNow = false; readingNow()
                bundled = list.mapTo(HashSet()) { it.key }
                got.onSuccess { b -> choice = b; singleFile = if (b.single != null) list[0].file else null; fileSize = b.size }
                    .onFailure { phase = Phase.Error("Not sendable: ${it.message}") }
            }
        }
    }
    // ai: a bundle's tar, written for the sender to map (txCreate maps a file) and deleted when the run ends
    private var tarFile: File? = null
    private fun dropTar() { tarFile?.delete(); tarFile = null }

    // ai: The choice's transfer made off the UI thread: its tar written where it is a bundle, txCreate (the file packed
    // ai: where that pays into the cache's send-pack, the files an earlier make left there let go first; its root hashed),
    // ai: the bytes it carries (txBytes). A failure is the state line's at once. One let go (dropMade: Stop, a change,
    // ai: the close) stops where it is (2026-11-16; it ran to its end before, a disk's worth of tar, pack and hash): the
    // ai: tar's write at its next buffer (Stoppable), the pack and the hash at their next piece (txCancelMakes).
    private fun make() {
        val b = choice ?: return
        dropMade()
        val my = making
        val ep = Native.txCancelMakes()
        makingNow = true
        val one = if (b.single == null) null else singleFile
        val tar = if (one != null) null else File(Native.cache, "send-${ProcessHandle.current().pid()}-$my.tar")
        val path = one?.path ?: tar!!.path
        thread(name = "lizard-send-make", isDaemon = true) {
            val wrote = if (tar == null) null else runCatching {
                tar.parentFile.mkdirs(); tar.deleteOnExit()
                Stoppable(tar.outputStream()) { my == making }.buffered(1 shl 20).use { b.write(it) }
            }.exceptionOrNull()
            // ai: a packed file still mapped by a transfer goes on being read after its name is removed (POSIX); Windows
            // ai: refuses the remove, and the transfer's own release takes it. A make let go meanwhile touches nothing
            // ai: there (a newer make's pack may be being written)
            val live = wrote == null && my == making
            val packDir = File(Native.cache, "send-pack")
            if (live) packDir.apply { mkdirs(); listFiles()?.forEach { it.delete() } }
            val hnd = if (!live) 0L else Native.txCreate(path, b.name, packDir.path, ep)
            val err = if (wrote != null) "Files unreadable: ${wrote.message}" else if (hnd == 0L) "Not sendable: ${Native.txError()}" else ""
            val sent = if (hnd != 0L) Native.txBytes(hnd) else 0L
            EventQueue.invokeLater {
                if (my != making || closing) { if (hnd != 0L) Native.txDestroy(hnd); tar?.delete(); return@invokeLater }   // ai: let go meanwhile
                makingNow = false
                if (hnd == 0L) { tar?.delete(); startWhenMade = false; phase = Phase.Error(err); return@invokeLater }
                made = hnd; madeTar = tar; compressed = sent
                if (startWhenMade) { startWhenMade = false; go() }
            }
        }
    }
    // ai: the made transfer let go, and any make under way stopped
    private fun dropMade() {
        if (makingNow) Native.txCancelMakes()
        making++; makingNow = false; startWhenMade = false
        if (made != 0L) { Native.txDestroy(made); made = 0L }
        madeTar?.delete(); madeTar = null
    }

    // ai: Start (the column's, the rail's play, the hook's); no tip before it since 2026-11-03
    fun start() {
        if (!canStart) return
        runPlay = playing
        runExport = !runPlay && export && !test
        runDisplay = !runExport || !hideCodes
        exportNow = null; exportFile = null
        if (runPlay) {
            val st = stream!!
            runTest = st.meta.test; runName = st.meta.name
            phase = Phase.Preparing; paused = false
            goPlay(st)
            return
        }
        // ai: Largest file: an export of a file past it refused, by the file's own size (before compression)
        if (runExport) exportTooLarge(choice!!.size, exportLimit)?.let { phase = Phase.Error(it); return }
        runTest = test
        runName = if (test) "" else choice!!.name
        dropTar()
        phase = Phase.Preparing; paused = false
        // ai: the file's transfer still being made (or its make let go): Start waits for it
        if (!test && made == 0L) { if (!makingNow) make(); startWhenMade = true; return }
        go()
    }
    // ai: Sending begins: the made transfer taken (the test stream's made here), the GPU painter's device and pipelines
    // ai: made off the UI thread (txPrepare, whatever the encoder: the presenter shows the code from that device,
    // ai: 2026-10-07), then the pick and the presenter once the canvas has a size
    private fun go() {
        val pre = if (runTest) 0L else made
        if (!runTest) { made = 0L; tarFile = madeTar; madeTar = null }
        val my = ++gen
        val sd = if (runTest) null else seedOf(seed)
        thread(name = "lizard-send-prep", isDaemon = true) {
            val hnd = if (pre != 0L) pre else Native.txCreate("", "", "", -1L)
            if (hnd != 0L && sd != null) Native.txSeed(hnd, sd)   // ai: before the first frame, which the configure makes
            val err = if (hnd == 0L) Native.txError() else ""
            // ai: the GPU painter's device and pipelines made here, off the UI thread that configures
            val why = if (hnd != 0L) Native.txPrepare(hnd, Native.assets) else ""
            EventQueue.invokeLater {
                if (my != gen) { if (hnd != 0L) Native.txDestroy(hnd); return@invokeLater }   // ai: stopped meanwhile
                if (hnd == 0L) { phase = Phase.Error("Not sendable: $err"); return@invokeLater }
                h = hnd; gpuWhy = why; prepared = true; fmt = null; phase = Phase.On
                if (why.isNotEmpty()) System.err.println("send: no GPU painter: $why")
                refreshNow()
                areaNow()
            }
        }
    }

    // ai: A stream played (Source Stream): its sender made off the UI thread (txPlay, the GPU's device for the frame ring
    // ai: and the presenter, txPrepare), then the presenter once the canvas has a size, as go's
    private fun goPlay(st: StreamPick) {
        val my = ++gen
        thread(name = "lizard-send-play", isDaemon = true) {
            val hnd = Native.txPlay(st.file.path)
            val err = if (hnd == 0L) Native.txError() else ""
            val why = if (hnd != 0L) Native.txPrepare(hnd, Native.assets) else ""
            EventQueue.invokeLater {
                if (my != gen) { if (hnd != 0L) Native.txDestroy(hnd); return@invokeLater }
                if (hnd == 0L) { phase = Phase.Error("Not playable: $err"); return@invokeLater }
                h = hnd; gpuWhy = why; prepared = true; fmt = null; phase = Phase.On
                refreshNow()
                areaNow()
            }
        }
    }

    // ai: Source (lizard-ui controlSettings): a run stops first, as a pick does
    fun chooseSource(v: String) { if (sending) stop(); source = if (v == "stream") "stream" else "files"; Prefs.source = source; exportNow = null }
    fun chooseExport(on: Boolean) { export = on; Prefs.export = on; exportNow = null }
    fun chooseCycles(v: Double) { cycles = cyclesOf(v); Prefs.cycles = cycles }
    // ai: Hide codes (lizard-ui controlSettings): kept, and mid-export the screen turned off or on at once (txExportShow:
    // ai: the sender's intake switched, the export going on): off, the presenter stopped first and the canvas out; on,
    // ai: the sender switched first, then the presenter begun as at a Start
    fun chooseHideCodes(on: Boolean) {
        hideCodes = on; Prefs.hideCodes = on
        if (!sending || !runExport || runDisplay == !on) return
        if (exportFile == null || exportNow?.flag("done") == true) { runDisplay = !on; return }
        if (on) {
            if (p != 0L) { Native.presentStop(p); p = 0L }
            canvas.want(false)
            stamp = null
        }
        val e = Native.txExportShow(h, !on)
        if (e.isNotEmpty()) { fail("Export failed: $e"); return }
        runDisplay = !on
        if (!on) repick()
    }
    fun chooseExportLimit(gb: Int) { exportLimit = fileGb(gb); Prefs.exportLimit = exportLimit }
    // ai: the stream file chooser (one .lizard) over the sender's window
    fun chooseStream() = frame?.let { Chooser.choose(it, "Choose a stream", false, false) { f -> f.firstOrNull()?.let { openStream(it) } } }
    // ai: what a .lizard is (its kind and meta.json, read by the native side): a stream's StreamMeta, else null
    fun streamOf(f: File): StreamMeta? = if (!ReceiveState.isReplay(f) || loadError.isNotEmpty()) null else
        parse(Native.lizardInfo(f.path))?.takeIf { it.str("kind") == "stream" }?.obj("meta")?.let { streamMetaOf(it) }
    // ai: a stream file chosen, dropped or opened with LIZARD: Source Stream with it; a replay or another file refused
    fun openStream(f: File) {
        if (sending) stop()
        val m = streamOf(f)
        if (m == null) { phase = Phase.Error("Not a stream: ${f.name}${if (ReceiveState.isReplay(f)) " is a replay (Receive's Upload decodes it)" else ""}"); return }
        if (phase is Phase.Error) phase = Phase.Idle
        stream = StreamPick(f, m)
        source = "stream"; Prefs.source = source
        if (test) chooseTest(false)
        exportNow = null
    }
    fun clearStream() { if (sending) stop(); stream = null }
    // ai: the export's block (lizard-ui SendRows.exportDetails), from the run's Start until the next Start or change
    val exportDetails: Details? get() {
        val e = exportNow ?: return null
        val f = exportFile ?: return null
        return SendRows.exportDetails(f.name, e.num("written").toInt(), e.num("frames").toInt(), e.num("bytes").toLong(),
            f.parentFile?.name.orEmpty(), e.flag("done"))
    }
    // ai: Laps' value: the stream file's size before it is made (SendRows.exportEstimate), its bytes once written
    val exportSize: String get() {
        exportNow?.takeIf { it.flag("done") }?.let { return Readout.bytes(it.num("bytes").toLong()) }
        val bytes = if (compressed > 0) compressed else choice?.size ?: 0L
        val subch = Pick.STEP * blocks
        val side = runCatching { Native.frameSide(subch, Pick.span(ringIndex)) }.getOrDefault(0)
        val est = SendRows.exportEstimate(cycles, bytes, codes * runCatching { Native.blocksFor(subch) }.getOrDefault(0), codes.toLong() * side * side)
        return if (est > 0) Readout.bytes(est) else ""
    }

    fun stop() {
        gen++
        // ai: an export stopped short leaves no block (a whole one's stays, saying where it went)
        if (exportNow?.flag("done") != true) exportNow = null
        timer.stop()
        settle.stop()
        if (p != 0L) { Native.presentStop(p); p = 0L }
        canvas.want(false)
        stamp = null
        if (h != 0L && h != preparing) Native.txDestroy(h)
        h = 0L
        dropTar()
        if (makingNow) dropMade()   // ai: Stop while the file is made stops the make (2026-11-16; it ran on before)
        startWhenMade = false
        if (sending) phase = Phase.Idle
        paused = false; fmt = null; present = null; sender = null
    }
    // ai: the window closing: the run stopped and the made transfer let go
    fun close() { closing = true; stop(); dropMade() }
    // ai: Send left for another screen (App.go, 2026-10-25; the Android app's go()): the run stopped (the presenter's canvas
    // ai: leaves the window with the screen); the made transfer kept for a return. Full screen is the window's (App.go)
    fun leave() { if (sending) stop() }

    private fun fail(why: String) { stop(); phase = Phase.Error(why) }

    // ai: play and pause as a video player's (the rail's button, Space and K): Start where nothing runs, else
    // ai: Pause or Resume
    fun playPause() { if (phase == Phase.On) pause(!paused) else if (!sending) start() }
    // ai: Paused: the last picture stays; a receiver holds what it has. In the window (2026-11-02) it becomes a still:
    // ai: the picture on screen read back (`stamp`), the presenter stopped and the canvas out of the window, so the area
    // ai: is Compose's (DesktopSend.Stamped) and a resize moves nothing native; Resume makes both again, the format picked
    // ai: for the area as it is then.
    class Stamp(val w: Int, val h: Int, val image: ImageBitmap)
    var stamp by mutableStateOf<Stamp?>(null)
        private set
    fun pause(on: Boolean) {
        if (phase != Phase.On) return
        paused = on
        if (on) {
            if (p != 0L) {
                Native.presentPause(p, true)
                stamp = snapshot(p)
                Native.presentStop(p); p = 0L
            }
            canvas.want(false)
        } else {
            begin()
        }
    }
    private fun snapshot(p: Long): Stamp? {
        val wh = IntArray(2)
        val px = Native.presentSnapshot(p, wh) ?: return null
        val (w, h) = wh[0] to wh[1]
        if (w <= 0 || h <= 0 || px.size < w * h) return null
        if (px.size < w * h * 4) return null
        val bm = org.jetbrains.skia.Bitmap()
        if (!bm.installPixels(org.jetbrains.skia.ImageInfo(w, h, org.jetbrains.skia.ColorType.RGBA_8888, org.jetbrains.skia.ColorAlphaType.OPAQUE), px, w * 4)) return null
        bm.setImmutable()
        return Stamp(w, h, bm.asComposeImageBitmap())
    }
    // ai: The code area resized by hand while the codes play in the window (the column's edge dragged, the window
    // ai: resized; 2026-11-02): paused, a still until Resume. Not a full screen's own sizes (FullScreen.settling), nor a
    // ai: fold of the column to the rail (a re-pick, as before).
    fun resizing() { if (phase == Phase.On && !paused && !full.on && !full.settling) pause(true) }

    fun toggleRail() { collapsed = !collapsed; Prefs.collapsed = collapsed }
    fun toggleTools() { toolsOpen = !toolsOpen; Prefs.toolsOpen = toolsOpen }
    fun toggleExperimental() { experimentalOpen = !experimentalOpen; Prefs.experimentalOpen = experimentalOpen }

    // ai: the settings: each kept as it moves; a slider's committed on release (commit), a chip's at once
    fun chooseTest(on: Boolean) { test = on; Prefs.payload = if (on) "test" else "file" }
    fun chooseSeed(t: String) { seed = t; Prefs.seed = t }
    // ai: The encoder moved while sending: to the CPU, or to the GPU or auto once its painter is made, configured at once;
    // ai: else the GPU painter made first on a thread of its own (txPrepare), then configured. Made inside configure, on
    // ai: the UI thread, it froze the window for the driver's compile and held the sender's lock (2026-10-04).
    fun chooseEnc(v: String) {
        enc = v; Prefs.enc = v
        if (phase != Phase.On) return
        if (v == "cpu" || prepared) { configure(); return }
        if (preparing != 0L) return   // ai: the one under way configures with the encoder as it is then
        val hnd = h
        val my = gen
        preparing = hnd
        thread(name = "lizard-send-prep", isDaemon = true) {
            val why = Native.txPrepare(hnd, Native.assets)
            EventQueue.invokeLater {
                preparing = 0L
                if (my != gen || h != hnd) { Native.txDestroy(hnd); return@invokeLater }   // ai: stopped meanwhile: it is this thread's
                prepared = true; gpuWhy = why
                if (why.isNotEmpty()) System.err.println("send: no GPU painter: $why")
                refreshNow()   // ai: a change made while it prepared
                configure()
            }
        }
    }
    fun chooseBlocks(b: Int) { if (b in Pick.VERSION_MIN..Pick.VERSION_MAX) { blocks = b; Prefs.blocks = b } }
    fun chooseFps(v: Int) { fps = v.coerceIn(1, 60); Prefs.fps = fps }
    fun chooseVsync(on: Boolean) {
        vsync = on; Prefs.vsync = on
        if (on && fps !in Pick.FPS) { chooseFps(Pick.fpsOf(fps)); commitFps() }
        if (p != 0L) Native.presentVsync(p, on)
    }
    fun chooseScaling(v: String) {
        if (v !in SCALINGS) return
        scaling = v; Prefs.scaling = v
        if (p != 0L) Native.presentScaling(p, SCALINGS.indexOf(v))
    }
    fun chooseSize(v: Int) { size = v.coerceIn(25, 100); Prefs.size = size }
    fun chooseSpacing(v: Int) { spacing = v.coerceIn(Room.SPACING_MIN, Room.SPACING_MAX); Prefs.spacing = spacing }
    fun chooseMargin(v: Int) { margin = v.coerceIn(Pick.QUIET, Room.MARGIN_MAX); Prefs.margin = margin }
    fun chooseRing(v: Int) { ring = if (v in Pick.RINGS.indices) v else -1; Prefs.ring = ring; repick() }
    fun chooseCodes(v: Int) { codes = if (v == 2) 2 else 1; Prefs.codes = codes; repick() }
    // ai: a group's delay (its code rates), applied at once (the sender's next frame; no configure): not while a prepare
    // ai: holds the sender's lock (the UI thread would wait out the driver's compile), which hands it on as it ends
    fun chooseRefresh(rate: Int, v: Int) {
        if (refresh[rate] == v) return
        refresh = refresh.copyOf().also { e -> e[rate] = v.coerceIn(1, REFRESH_MAX) }
        Prefs.refresh = refreshText(refresh)
        if (h != preparing) refreshNow()
    }
    private fun refreshNow() { if (h != 0L) { Native.txRefresh(h, refresh[0], refresh[1], refresh[2], refresh[3]); Native.txColours(h, ink, paper) } }
    // ai: the code's colours, applied at once (the next frame shown; the sender's atomics, so even while it prepares)
    fun chooseColours(i: Int, p: Int) {
        ink = i and 0xFFFFFF; paper = p and 0xFFFFFF; Prefs.ink = ink; Prefs.paper = paper
        if (h != 0L) Native.txColours(h, ink, paper)
    }
    fun commitFps() {
        if (phase != Phase.On) return
        configure()
        if (p != 0L) Native.presentFps(p, fps)
    }
    fun commitSize() = repick()
    fun repick() {
        if (phase != Phase.On) return
        if (p == 0L) begin() else if (configure()) Native.presentSize(p, size / 100f, edge)
    }

    // ai: the code area's size in device px as Compose lays it out (Screen.kt's onSizeChanged: the canvas, hidden while
    // ai: idle, is laid out by no one then); sending, a re-pick, or the start once it has one. While presenting
    // ai: the re-pick waits for the area to settle, 300 ms (the web's roomChanged): full screen passes through two or three
    // ai: sizes (the column folded, the manager's first size, the screen's), and a re-pick at each restarted the painters
    // ai: (2026-10-04, on a real screen: about 0.5 s of stalled presents a transition); the presenter meanwhile
    // ai: scales the picture it has into the new area.
    private var seenW = 0
    private var seenH = 0
    fun areaChanged(w: Int, hh: Int) { seenW = w; seenH = hh; areaNow() }
    private fun areaNow() {
        // ai: an area not yet laid out keeps the area it had (Auto's label flickered on each entry to Send, 2026-11-02)
        if (seenW <= 0 || seenH <= 0) return
        areaW = seenW; areaH = seenH
        if (p == 0L) repick() else settle.restart()
    }
    private val settle = Timer(300) { repick() }.apply { isRepeats = false }

    // ai: macOS: the presenter's layer follows the canvas (a move or a resize of it or of the window)
    fun canvasMoved() { if (p != 0L) Native.presentPlace(p, canvas.place()) }

    // ai: the canvas's window going: the presenter stopped first (its surface is that window); begun again on the next
    fun canvasGone() {
        if (p != 0L) { Native.presentStop(p); p = 0L }
    }
    // ai: the canvas's window made: the presenter begun where one is wanted
    fun canvasMade() { if (phase == Phase.On && p == 0L) repick() }

    // ai: the presenter needs the canvas's window made (wanted first: canvasMade comes back here), not shown: a Start at
    // ai: the launch (the hook's) comes before AWT counts the window as showing. None while paused (a still; Resume).
    private fun begin() {
        if (h == 0L || p != 0L || paused || areaW <= 0 || areaH <= 0) return
        // ai: an export with no screen: configured (the export started there), polled, nothing presented
        if (runExport && !runDisplay) { if (configure()) timer.start(); return }
        if (!canvas.isDisplayable) { canvas.want(true); return }
        if (!configure()) return
        val pr = Native.presentStart(canvas, h, runFps, size / 100f, edge, vsync, canvas.place())
        if (pr != 0L) Native.presentScaling(pr, SCALINGS.indexOf(scaling))
        if (pr == 0L) { fail("Display failed: ${Native.txError()}"); return }
        p = pr
        stamp = null
        if (paused) Native.presentPause(p, true)
        timer.start()
    }

    // ai: The format chosen (Blocks), configured where it changed. false where it was refused (the run then ends with
    // ai: why).
    private fun configure(): Boolean {
        if (h == 0L) return false
        // ai: a stream's format is its own (the native side takes it from the file; these say it to the presenter's room)
        val sm = if (runPlay) stream?.meta else null
        val f = if (sm != null) Format(Pick.STEP * sm.blocks, Pick.span(sm.ring), sm.fps, sm.codes, 0, 0)
            else Format(Pick.STEP * blocks, Pick.span(ringIndex), fps, codes, if (codes > 1) Room.gapOf(spacing) else 0,
                when (enc) { "cpu" -> 0; "gpu" -> 1; else -> 2 })
        if (f == fmt) return true
        val threads = min(8, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, f.subch, f.span, f.fps, f.codes, f.gap, threads, f.painter, Native.assets)
        if (err.isNotEmpty()) { fail(err); return false }
        fmt = f
        // ai: Export: started at the run's first configure, into Downloads (a later configure, a change of format, ends
        // ai: it: the sender's error says so)
        if (runExport && exportFile == null) {
            val dir = Received.downloads.apply { mkdirs() }
            val out = Received.free(dir, (File(runName).nameWithoutExtension.ifEmpty { runName }.ifEmpty { "LIZARD" }) + ".lizard")
            val e = Native.txExport(h, out.path, cycles, runDisplay, 0, runName, blocks, ringIndex, fps, codes, codes * sentBlocks)
            if (e.isNotEmpty()) { fail("Export failed: $e"); return false }
            exportFile = out
            exportNow = parse(Native.txExportState(h))
        }
        val st = parse(Native.txStats(h))
        System.err.println("send: LIZARD-${f.subch}${if (f.codes > 1) " x ${f.codes}, spacing $spacing" else ""}, margin $margin" +
            " at ${f.fps} a second, ring ${f.span / 2}, painting on the " +
            (if (st?.str("painter") == "gpu") "GPU (${st.str("device")})" else "CPU, $threads painters") +
            (st?.str("gpuWhy").orEmpty().let { if (it.isNotEmpty()) ", not the GPU: $it" else "" }))
        return true
    }

    // ai: once a second while sending: the sender's and the presenter's stats (each a window of about a second, reset as
    // ai: read); an error in either ends the run with it
    fun poll() {
        if (h == 0L) return
        senderRaw = Native.txStats(h); sender = parse(senderRaw)
        if (p != 0L) { presentRaw = Native.presentStats(p); present = parse(presentRaw) }
        val e = sender?.str("error").orEmpty().ifEmpty { if (p != 0L) present?.str("error").orEmpty() else "" }
        if (e.isNotEmpty()) { fail(e); return }
        // ai: the export's state; finished with no screen, the run ends (its block kept); with one, the send goes on
        if (runExport && exportFile != null) {
            val es = parse(Native.txExportState(h)) ?: return
            exportNow = es
            val ee = es.str("error")
            if (ee.isNotEmpty()) { fail("Export failed: $ee"); return }
            if (es.flag("done") && !runDisplay) stop()
        }
    }

    // ai: What goes, as elements from Start until Stop (lizard-ui SendRows.details, 2026-11-07; the state line "Sending
    // ai: <name>, <compressed> (<raw> raw)" over the figures before): the name, a spinner while it is made, Size, Files,
    // ai: Speed, and with Show statistics on Compressed (the bytes that go), Pass and Pass time; null otherwise
    fun details(stats: Boolean): Details? {
        exportDetails?.let { return it }
        if (phase != Phase.Preparing && phase != Phase.On) return null
        val sd = sender
        // ai: a stream's file is the one it carries (its meta.json's)
        if (runPlay) stream?.meta?.let { m ->
            return SendRows.details(m.name, m.test, phase == Phase.Preparing, m.length, 0, 1, speedKBs, sd?.num("pass") ?: 0.0, sd?.num("cycle") ?: 0.0, stats)
        }
        return SendRows.details(runName, runTest, phase == Phase.Preparing, fileSize, compressed,
            choice?.takeIf { it.single == null }?.files ?: 1, speedKBs, sd?.num("pass") ?: 0.0, sd?.num("cycle") ?: 0.0, stats)
    }
    // ai: the rate the pictures actually presented carry (the presenter's fresh pictures a second x codes x blocks a frame
    // ai: x 460 B, so a limit outside the sender shows in it; 2026-10-04; x the share of blocks new in a frame, Rate
    // ai: refresh's repeats none, 2026-11-12), 0 while paused or before the first present
    private val speedKBs: Double get() {
        if (phase != Phase.On || paused) return 0.0
        val f = fmt ?: return 0.0
        val pr = present ?: return 0.0
        return pr.num("shownFps") * f.codes * sentBlocks * Pick.PAYLOAD * freshShare / 1000.0
    }

    // ai: The readout (the web's #tx, Developer Tools): the format and why, the file, the frame on the surface, a frame's
    // ai: bytes, the pictures shown against the asked rate and the screen's, the refreshes each held, the encoder.
    val readout: String get() {
        val f = fmt ?: return ""
        val sd = sender ?: return ""
        val pr = present
        val b = sentBlocks
        val l = mutableListOf<String>()
        l += "LIZARD-${f.subch} ($b blocks${if (tiers.isNotEmpty()) ": $tiers" else ""}), picture ${sd.num("n").toInt()} samples in the ${f.span / 2}-cell ring"
        if (!runTest) l += "file $fileSize B${if (compressed in 1 until fileSize) ", $compressed B packed (zstd)" else ""}: ${sd.num("sourceBlocks").toInt()} chunk${if (sd.num("sourceBlocks").toInt() == 1) "" else "s"}, " +
            "${sd.num("cycle").toInt()} data blocks a pass, BLAKE3 ${sd.str("root").take(16)}..., seed ${sd.num("seed").toLong()}" +
            (sd.num("pass").let { p -> if (p >= 1) "\npass ${floor(p).toInt()}" + (sd.num("offeredKBs").let { k -> if (k > 0 && sd.num("cycle") > 0) ", ${Fmt.f1(sd.num("cycle") / (k * 1000 / Pick.PAYLOAD))} s a pass" else "" }) else "" })
        l += "frame ${sd.num("width").toInt()} x ${sd.num("side").toInt()} px" + (if (f.codes > 1) ", ${f.codes} side by side $spacing modules apart" else "") + ", margin $margin modules" +
            (pr?.let { " on a ${it.str("surface").replace("x", " x ")} px surface" } ?: "")
        l += "${if (f.codes > 1) "${f.codes} x " else ""}$b x ${Pick.PAYLOAD} B = ${f.codes * b * Pick.PAYLOAD} B per frame"
        if (pr != null) {
            l += "shown ${Fmt.f1(pr.num("shownFps"))}/s of ${f.fps} asked, ${Fmt.f1(pr.num("presentsPerSec"))} presents a second, the screen " +
                "${Fmt.f1(pr.num("hz"))} Hz${if (pr.num("modeHz") > 0) " (its mode ${Fmt.f1(pr.num("modeHz"))})" else ""}" +
                (if (pr.num("modeFrom") > 0) ", held by VSync from ${Fmt.f1(pr.num("modeFrom"))} Hz" else "") + ", times from ${if (pr.flag("presentWait")) "present_wait" else "the acquires"}"
            val held = (pr["held"] as? JsonObject)?.entries?.sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
                ?.joinToString(", ") { "${it.key}: ${(it.value as? JsonPrimitive)?.contentOrNull ?: "?"}" }.orEmpty()
            l += "refreshes a picture: ${held.ifEmpty { "none" }}; missed ${pr.num("missed").toInt()}, behind ${pr.num("behind").toInt()}"
            l += "presenter: Vulkan on ${pr.str("device")}${if (pr.flag("paused")) ", paused" else ""}"
        }
        l += (if (sd.str("painter") == "gpu") "encoder: GPU, ${sd.str("device")}, ${Fmt.f1(sd.num("paintMs"))} ms a frame"
            else "encoder: CPU, ${sd.num("painters").toInt()} painters, ${Fmt.f1(sd.num("paintMs"))} ms a frame on one" +
                (if (enc != "cpu" && sd.str("gpuWhy").ifEmpty { gpuWhy }.isNotEmpty()) " (not the GPU: ${sd.str("gpuWhy").ifEmpty { gpuWhy }})" else "")) +
            ", ${sd.num("ahead").toInt()} painted ahead"
        return l.joinToString("\n")
    }

    // ai: the build's time for About (build.gradle.kts buildInfo), as the web's: "Build: 2026-10-04 12:00 UTC"
    val about: String by lazy {
        val t = javaClass.getResourceAsStream("/lizard-build.txt")?.bufferedReader()?.readText()?.trim().orEmpty()
        if (t.isEmpty()) "" else "Build: " + t.replace("T", " ").replace(Regex(":\\d\\d(\\.\\d+)?Z$"), " UTC")
    }
}

