@file:OptIn(ExperimentalFoundationApi::class, ExperimentalComposeUiApi::class)

package dev.lizard.desktop

import dev.lizard.pick.Pick
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.draganddrop.dragAndDropTarget
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.draganddrop.DragAndDropEvent
import androidx.compose.ui.draganddrop.DragAndDropTarget
import androidx.compose.ui.draganddrop.awtTransferable
import java.awt.datatransfer.DataFlavor
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.SwingPanel
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt
import dev.lizard.ui.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.onSizeChanged
import dev.lizard.readout.Readout
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize

// ai: Send on the desktop (the screen is ../lizard-ui's SendScreen): the sender's state (SendState), the desktop's
// ai: settings (Ring, Size), Fullscreen beside Start, the code area the presenter's
// ai: heavyweight AWT canvas (CodeCanvas: its idle look and the upload painted by AWT, since nothing of Compose draws
// ai: over it), drops anywhere in the window (2026-10-23); full screen and its edge strip are FullScreen.kt's
class DesktopSend(private val app: App) : SendHost {
    private val s get() = app.send
    override val folds = app.folds
    override val width = app.column
    override val canPortrait = false
    override val swapInColumn = true
    override fun goHome() = app.go(Screen.Home)
    override fun goOther() = app.go(Screen.Receive)
    override val busy get() = s.sending
    override val on get() = s.phase == SendState.Phase.On
    override val paused get() = s.paused
    override val canStart get() = s.canStart
    override fun start() = s.start()
    override fun stop() = s.stop()
    override fun pause(on: Boolean) = s.pause(on)
    override val pauseShown = true
    override fun toggleFull() = app.full.toggle()
    // ai: what goes as elements (SendState.details: Show statistics, Receive's own setting, names the rows it adds), the
    // ai: error a popup, and the spinner while a pick is walked or a bundle made (2026-11-07; the state line before)
    override val details get() = s.details(app.recv.showStats)
    // ai: no native library: nothing sends; said in the popup each time Send opens (2026-11-07; a cover before)
    override val error get() = (s.phase as? SendState.Phase.Error)?.why ?: s.loadError
    override fun clearError() { if (s.phase is SendState.Phase.Error) s.phase = SendState.Phase.Idle }
    override val reading get() = s.phase == SendState.Phase.Idle && !s.test && s.reading
    override val heat: Warn? = null
    override val railCapacity get() = s.capacityKBs
    // ai: the settings by topic (2026-11-12, lizard-ui SEND_FOLDS; one list from 2026-11-01, headed Code and Display from
    // ai: 2026-10-23): what the symbol carries and where it sits (Layout's own fold until 2026-11-13), how often its blocks
    // ai: change, how this machine paints it
    override val settings: Map<FoldKey, List<Setting>> get() = mapOf(
        FoldKey.SendCode to listOfNotNull(
            // ai: Blocks a frame (send.html #blocks): a version 4 to 64, committed on release (its first step auto until
            // ai: 2026-11-07); Blocks or Density and the ring's names by Show statistics (lizard-ui SendRows, 2026-11-07)
            Slide(SendRows.blocksTitle(stats), SendRows.blocksText(Native.blocksFor(Pick.STEP * s.blocks), stats), s.blocks.toFloat(),
                Pick.VERSION_MIN.toFloat()..Pick.VERSION_MAX.toFloat(), Pick.VERSION_MAX - Pick.VERSION_MIN - 1, done = { s.repick() }) { s.chooseBlocks(it.roundToInt()) },
            // ai: the rings alone (2026-10-05): none chosen shows the default
            ChipChoice("Ring", if (s.ring == -1) Pick.RING_DEFAULT else s.ring, SendRows.rings(stats)) { s.chooseRing(it) },
            // ai: two codes side by side for the receiver's 2:1 crop (send.html #codes)
            ChipChoice("Codes", s.codes, SendRows.CODES) { s.chooseCodes(it) },
            Slide("Size", "${s.size}%", s.size.toFloat(), 25f..100f, 74, done = { s.commitSize() }) { s.chooseSize(it.roundToInt()) },
            if (s.codes == 2) Slide("Spacing", "${s.spacing} modules", s.spacing.toFloat(), Room.SPACING_MIN.toFloat()..Room.SPACING_MAX.toFloat(),
                Room.SPACING_MAX - Room.SPACING_MIN - 1, done = { s.repick() }) { s.chooseSpacing(it.roundToInt()) } else null,
            Slide("Margin", "${s.margin} modules", s.margin.toFloat(), Pick.QUIET.toFloat()..Room.MARGIN_MAX.toFloat(), Room.MARGIN_MAX - Pick.QUIET - 1,
                done = { s.repick() }) { s.chooseMargin(it.roundToInt()) }),
        // ai: FPS (chips under VSync, a slider without it), then Rate refresh, a slider a code rate (lizard-ui rateRefreshSettings)
        FoldKey.SendRefresh to listOf(
            if (s.vsync) ChipChoice("FPS", s.fps, Pick.FPS.map { it to "$it" }) { s.chooseFps(it); s.commitFps() }
            else Slide("FPS", "${s.fps}", s.fps.toFloat(), 1f..60f, 58, done = { s.commitFps() }) { s.chooseFps(it.roundToInt()) }) +
            rateRefreshSettings(s.refresh) { r, v -> s.chooseRefresh(r, v) },
        FoldKey.SendPerformance to listOf(
            // ai: GPU and CPU alone: with neither chosen the encoder is auto's and the painter running is the chip marked
            // ai: (the GPU before a first send); a choice takes that one outright
            ChipChoice("Encoder", if (s.enc == "auto") s.painterNow.ifEmpty { "gpu" } else s.enc, listOf("gpu" to "GPU", "cpu" to "CPU")) { s.chooseEnc(it) },
            ChipChoice("VSync", s.vsync, listOf(true to "On", false to "Off")) { s.chooseVsync(it) }))
    // ai: lab arms a user may try: none (Exclusive fullscreen, the last, deleted 2026-11-13), so no fold
    override val experimental: List<Setting> get() = emptyList()
    // ai: Show statistics, Receive's setting (ReceiveState.showStats, Prefs.Rx.stats), one switch for both screens (2026-11-07)
    private val stats get() = app.recv.showStats
    override val tools: List<Setting> get() = listOf(
        ChipChoice("Show statistics", app.recv.showStats, listOf(true to "On", false to "Off")) { app.recv.chooseStats(it) },
        ChipChoice("Payload", s.test, listOf(false to "File", true to "Test stream"), enabled = !s.sending) { s.chooseTest(it) },
        seedSetting(s.seed) { s.chooseSeed(it) })
    override val readout get() = s.readout.takeIf { it.isNotEmpty() }?.let { Code(it, 12) }
    // ai: full screen: the code alone (2026-10-23, as the web's and the app's; FullScreen.kt, Receive's too since
    // ai: 2026-11-13), the edge strip's collapser opening the column over it (FullScreen.sideOver)
    override val alone get() = app.full.on
    override val areaModifier get() = edgeModifier(app.full)
    override val codeAloneInPlace = true
    override val edgeTap = false
    override fun leaveCodeAlone() { app.full.sideOver = true }
    override val codePadding = 0.dp
    // ai: the column's margin the edge's whole strip: the canvas begins past it, so a drag of the edge is Compose's from
    // ai: its start (a run paused at it takes the canvas out of the window, which ended a drag it had)
    override val edgeRoom = EdgeStrip
    override val codeFill get() = s.codeFill
    override fun codeSize(maxWidth: Dp, maxHeight: Dp, portrait: Boolean): DpSize? = null
    // ai: files and folders dropped anywhere in the window are added as on the code area (SendState.dropped), the area's
    // ai: outline shown while a drag is over the window (2026-10-23)
    private val drops = object : DragAndDropTarget {
        override fun onEntered(event: DragAndDropEvent) = s.dragging(true)
        override fun onExited(event: DragAndDropEvent) = s.dragging(false)
        override fun onEnded(event: DragAndDropEvent) = s.dragging(false)
        override fun onDrop(event: DragAndDropEvent): Boolean { s.dragging(false); return s.dropped(event.awtTransferable) }
    }
    override val rootModifier: Modifier = Modifier.dragAndDropTarget({ it.awtTransferable.isDataFlavorSupported(DataFlavor.javaFileListFlavor) }, drops)

    // ai: The code area (2026-11-02, as the phone's): idle, Compose's, the file upload in a column at most 576 dp wide,
    // ai: centred, what goes named under its card (the AWT canvas painted it until then), or the test stream's note; a
    // ai: drag's outline over it. While codes play the canvas (a window of its own, the presenter's) in its place, from
    // ai: the presenter's start to its end; paused in the window, the picture as a still (`Stamped`) until the canvas is
    // ai: back. Its size, laid out here, is the room the format is picked from (SendState.areaChanged); in full screen
    // ai: the pointer near its left edge shows the edge strip (areaModifier; the canvas hears that while shown). The
    // ai: column's edge dragged pauses a run (SendState.resizing).
    @Composable
    override fun CodeArea(modifier: Modifier) {
        val dragging = width.dragging
        LaunchedEffect(dragging) { if (dragging) s.resizing() }
        Box(modifier.onSizeChanged { s.areaChanged(it.width, it.height) }) {
            val stamp = s.stamp
            if (s.canvas.shown) SwingPanel(background = s.codeFill, factory = { s.canvas }, modifier = Modifier.fillMaxSize())
            else if (stamp != null) Stamped(stamp, s.size / 100f, s.edgeShare)
            else {
                if (s.test) TestStreamNote()
                else Upload(listOf(Act("File", icon = Icons.file) { s.chooseFile() }, Act("Folder", icon = Icons.folder) { s.chooseFolder() }),
                    s.items.map { UploadItem(it.key, it.name, if (it.dir) "${it.files} ${if (it.files == 1) "file" else "files"}" else Readout.bytes(it.size), it.dir) },
                    enabled = !s.sending, row = false, onRemove = { s.remove(it) }, onClear = { s.clearPicks() },
                    column = 576.dp, total = s.choice?.takeIf { !s.reading }?.let { Readout.bytes(it.size) } ?: "")
                if (s.dragOver) DropOutline()
            }
        }
    }
}

// ai: the picture a run was paused on in the window (SendState.stamp), placed as the presenter places it: on white,
// ai: centred, the largest that fits with the edge's light round it (edge, a share of its height, each side), times
// ai: the Size share, stretched linearly
@Composable
private fun Stamped(st: SendState.Stamp, share: Float, edge: Float) = Canvas(Modifier.fillMaxSize().background(Bg)) {
    val e = 2 * edge * st.h
    val scale = minOf(size.width / (st.w + e), size.height / (st.h + e)) * share.coerceIn(0.25f, 1f)
    val dw = minOf(size.width, st.w * scale).roundToInt()
    val dh = minOf(size.height, st.h * scale).roundToInt()
    if (dw <= 0 || dh <= 0) return@Canvas
    drawImage(st.image, srcSize = IntSize(st.w, st.h), dstOffset = IntOffset(((size.width - dw) / 2).toInt(), ((size.height - dh) / 2).toInt()),
        dstSize = IntSize(dw, dh), filterQuality = FilterQuality.Low)
}

// ai: a drag over the window: the web's dashed outline, 12 dp in
@Composable
private fun DropOutline() = Canvas(Modifier.fillMaxSize()) {
    val inset = 12.dp.toPx()
    drawRect(Fg, topLeft = Offset(inset, inset), size = Size(size.width - 2 * inset, size.height - 2 * inset),
        style = Stroke(3.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(9.dp.toPx(), 6.dp.toPx()))))
}
