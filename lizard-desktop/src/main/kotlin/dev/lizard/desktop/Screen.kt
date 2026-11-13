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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.awt.SwingPanel
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogWindow
import kotlin.math.roundToInt
import kotlinx.coroutines.delay
import androidx.compose.ui.awt.ComposeDialog
import java.awt.Dialog
import java.awt.MouseInfo
import java.awt.Window
import dev.lizard.ui.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.onPointerEvent
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import dev.lizard.readout.Readout
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize

// ai: Send on the desktop (the screen is ../lizard-ui's SendScreen): the sender's state (SendState), the desktop's
// ai: settings (Ring, Size; Exclusive fullscreen on Linux), Fullscreen beside Start, the code area the presenter's
// ai: heavyweight AWT canvas (CodeCanvas: its idle look and the upload painted by AWT, since nothing of Compose draws
// ai: over it), drops anywhere in the window (2026-10-23), and full screen's edge rail, a window of its own
class DesktopSend(private val app: App) : SendHost {
    private val s get() = app.send
    // ai: the window (LizardWindow sets it): the owner of the edge rail
    var window: ComposeWindow? = null
    override val folds = app.folds
    override val width = app.column
    override val canPortrait = false
    override val swapInColumn = true
    override fun goHome() = app.go(Screen.Home)
    override fun goReceive() = app.go(Screen.Receive)
    override val busy get() = s.sending
    override val on get() = s.phase == SendState.Phase.On
    override val paused get() = s.paused
    override val canStart get() = s.canStart
    override fun start() = s.start()
    override fun stop() = s.stop()
    override fun pause(on: Boolean) = s.pause(on)
    override val pauseShown = true
    override fun toggleFull() = s.toggleFull()
    // ai: what goes as elements (SendState.details: Show statistics, Receive's own setting, names the rows it adds), the
    // ai: error a popup, and the spinner while a pick is walked or a bundle made (2026-11-07; the state line before)
    override val details get() = s.details(app.recv.showStats)
    // ai: no native library: nothing sends; said in the popup each time Send opens (2026-11-07; a cover before)
    override val error get() = (s.phase as? SendState.Phase.Error)?.why ?: s.loadError
    override fun clearError() { if (s.phase is SendState.Phase.Error) s.phase = SendState.Phase.Idle }
    override val reading get() = s.phase == SendState.Phase.Idle && !s.test && s.reading
    // ai: the control window says what plays where in its area (LeasedNote): no state line in its column
    override val lineShown get() = !s.leased
    override val heat: Warn? = null
    override val railCapacity get() = s.capacityKBs
    // ai: the settings by topic (2026-11-12, lizard-ui SEND_FOLDS; one list from 2026-11-01, headed Code and Display from
    // ai: 2026-10-23): what the symbol carries, how often its blocks change, where it sits, how this machine paints it
    override val settings: Map<FoldKey, List<Setting>> get() = mapOf(
        FoldKey.SendCode to listOf(
            // ai: Blocks a frame (send.html #blocks): a version 4 to 64, committed on release (its first step auto until
            // ai: 2026-11-07); Blocks or Density and the ring's names by Show statistics (lizard-ui SendRows, 2026-11-07)
            Slide(SendRows.blocksTitle(stats), SendRows.blocksText(Native.blocksFor(Pick.STEP * s.blocks), stats), s.blocks.toFloat(),
                Pick.VERSION_MIN.toFloat()..Pick.VERSION_MAX.toFloat(), Pick.VERSION_MAX - Pick.VERSION_MIN - 1, done = { s.repick() }) { s.chooseBlocks(it.roundToInt()) },
            // ai: the rings alone (2026-10-05): none chosen shows the default
            ChipChoice("Ring", if (s.ring == -1) Pick.RING_DEFAULT else s.ring, SendRows.rings(stats)) { s.chooseRing(it) },
            // ai: two codes side by side for the receiver's 2:1 crop (send.html #codes)
            ChipChoice("Codes", s.codes, SendRows.CODES) { s.chooseCodes(it) }),
        // ai: FPS (chips under VSync, a slider without it), then Rate refresh, a slider a code rate (lizard-ui rateRefreshSettings)
        FoldKey.SendRefresh to listOf(
            if (s.vsync) ChipChoice("FPS", s.fps, Pick.FPS.map { it to "$it" }) { s.chooseFps(it); s.commitFps() }
            else Slide("FPS", "${s.fps}", s.fps.toFloat(), 1f..60f, 58, done = { s.commitFps() }) { s.chooseFps(it.roundToInt()) }) +
            rateRefreshSettings(s.refresh) { r, v -> s.chooseRefresh(r, v) },
        FoldKey.SendLayout to listOfNotNull(
            Slide("Size", "${s.size}%", s.size.toFloat(), 25f..100f, 74, done = { s.commitSize() }) { s.chooseSize(it.roundToInt()) },
            if (s.codes == 2) Slide("Spacing", "${s.spacing} modules", s.spacing.toFloat(), Room.SPACING_MIN.toFloat()..Room.SPACING_MAX.toFloat(),
                Room.SPACING_MAX - Room.SPACING_MIN - 1, done = { s.repick() }) { s.chooseSpacing(it.roundToInt()) } else null,
            Slide("Margin", "${s.margin} modules", s.margin.toFloat(), Pick.QUIET.toFloat()..Room.MARGIN_MAX.toFloat(), Room.MARGIN_MAX - Pick.QUIET - 1,
                done = { s.repick() }) { s.chooseMargin(it.roundToInt()) }),
        FoldKey.SendPerformance to listOf(
            // ai: GPU and CPU alone: with neither chosen the encoder is auto's and the painter running is the chip marked
            // ai: (the GPU before a first send); a choice takes that one outright
            ChipChoice("Encoder", if (s.enc == "auto") s.painterNow.ifEmpty { "gpu" } else s.enc, listOf("gpu" to "GPU", "cpu" to "CPU")) { s.chooseEnc(it) },
            ChipChoice("VSync", s.vsync, listOf(true to "On", false to "Off")) { s.chooseVsync(it) }))
    // ai: lab arms a user may try: Exclusive fullscreen (2026-10-31), the monitor taken for the code while sending, shown
    // ai: on Linux alone (X11's lease)
    override val experimental: List<Setting> get() = listOfNotNull(
        if (!Native.mac && !Native.windows) ChipChoice("Exclusive fullscreen", s.exclusiveFull, listOf(true to "On", false to "Off")) { s.chooseExclusiveFull(it) } else null)
    // ai: Show statistics, Receive's setting (ReceiveState.showStats, Prefs.Rx.stats), one switch for both screens (2026-11-07)
    private val stats get() = app.recv.showStats
    override val tools: List<Setting> get() = listOf(
        ChipChoice("Show statistics", app.recv.showStats, listOf(true to "On", false to "Off")) { app.recv.chooseStats(it) },
        ChipChoice("Payload", s.test, listOf(false to "File", true to "Test stream"), enabled = !s.sending) { s.chooseTest(it) })
    override val readout get() = s.readout.takeIf { it.isNotEmpty() }?.let { Code(it, 12) }
    // ai: full screen: the code alone (2026-10-23, as the web's and the app's); while a monitor is taken for the code the
    // ai: window is the control: the column as when not full
    override val codeAlone get() = s.full && !s.leased
    override val codeAloneInPlace = true
    override val edgeTap = false
    // ai: the column over the code in full screen (2026-11-07; the rail's collapser left full screen before, the column
    // ai: open): open until its collapser, a change of full screen closing it (Windows)
    var sideOver by mutableStateOf(false)
    override fun leaveCodeAlone() { sideOver = true }
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
    // ai: the pointer near its left edge shows the rail (the canvas hears that while shown). The control window while a
    // ai: monitor of its own shows the code: "Playing on <monitor>" or "Paused on" (LeasedNote). The column's edge dragged
    // ai: pauses a run (SendState.resizing).
    @OptIn(ExperimentalComposeUiApi::class)
    @Composable
    override fun CodeArea(modifier: Modifier) {
        val dragging = width.dragging
        LaunchedEffect(dragging) { if (dragging) s.resizing() }
        val near = with(LocalDensity.current) { EDGE_NEAR.toPx() }
        Box(modifier.onSizeChanged { s.areaChanged(it.width, it.height) }.onPointerEvent(PointerEventType.Move) { e ->
            if (s.full && !s.leased && !s.edgeRail && (e.changes.firstOrNull()?.position?.x ?: near) < near) s.edgeRail = true
        }) {
            val stamp = s.stamp
            if (s.canvas.shown) SwingPanel(background = s.codeFill, factory = { s.canvas }, modifier = Modifier.fillMaxSize())
            else if (s.leased) LeasedNote("${if (s.paused) "Paused" else "Playing"} on ${s.leasedOn}")
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

    @Composable
    override fun Windows() {
        LaunchedEffect(s.full) { sideOver = false }
        val w = window ?: return
        if (s.full && !s.leased) EdgeRail(this, s, w)
    }
}

// ai: how near the code area's left edge the pointer shows full screen's rail
private val EDGE_NEAR = 24.dp

// ai: the control window's area while the code plays on a monitor of its own (2026-11-02; painted on the code canvas by
// ai: AWT until then): the line centred on the soft fill
@Composable
private fun LeasedNote(text: String) = Box(Modifier.fillMaxSize().background(Soft), contentAlignment = Alignment.Center) {
    Text(text, style = MaterialTheme.typography.bodyLarge, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(horizontal = 16.dp))
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

// ai: Full screen's rail (2026-10-23): the collapsed column over the screen's left edge while the pointer is within
// ai: EDGE_NEAR of it (the code area's, or the canvas's while it shows: CodeCanvas) or over the rail, in a window of its own (nothing of Compose draws over the
// ai: code's heavyweight canvas), over the code rather than beside it so the area and the format stay as they are. A
// ai: popup (override-redirect on X11): the window manager neither places it (a dialog it centres on its owner at
// ai: first) nor shades it, nor gives it the focus full screen holds. Mapped for as long as full screen lasts, parked
// ai: off every screen while hidden, so a show is a move (a window mapped and drawn at each show took 0.5 to 1.1 s).
// ai: Its collapser opens the column over the code (SendSideOver, 2026-11-07; it left full screen with the column open
// ai: before): the window as wide as the column, the code under it keeping its size and format, open until the column's
// ai: own collapser or full screen left. The rail alone is hidden once the pointer is off it (read a few times a second:
// ai: the canvas below hears nothing while the pointer is over the rail, nor once it goes past the screen's edge).
@Composable
private fun EdgeRail(h: DesktopSend, s: SendState, owner: ComposeWindow) {
    val shown = s.edgeRail || h.sideOver
    val over = h.sideOver
    // ai: the column's width as the landscape layout takes it on this screen (sendSideWidth)
    val sideDp = sendSideWidth(h, (owner.contentPane.width / owner.graphicsConfiguration.defaultTransform.scaleX).toFloat().dp).value
    DialogWindow(visible = true,
        create = { ComposeDialog(owner, Dialog.ModalityType.MODELESS).apply { type = Window.Type.POPUP; isUndecorated = true; isResizable = false; focusableWindowState = false } },
        dispose = ComposeDialog::dispose,
        update = { d ->
            // ai: full screen's code area is the window's whole content; the window's width the rail's or the column's,
            // ai: and its divider, in the screen's pixels
            val c = owner.contentPane
            val h = maxOf(1, c.height)
            val scale = owner.graphicsConfiguration.defaultTransform.scaleX
            val w = if (over) ((sideDp + 1) * scale).roundToInt() else 57
            if (shown && c.isShowing) c.locationOnScreen.let { d.setBounds(it.x, it.y, w, h) } else d.setBounds(-10000, -10000, w, h)
        }) {
        LaunchedEffect(shown, over) {
            while (shown && !over) {
                delay(150)
                val p = MouseInfo.getPointerInfo()?.location
                if (p == null || !window.bounds.contains(p)) { s.edgeRail = false; break }
            }
        }
        if (over) SendSideOver(h, sideDp.dp) { h.sideOver = false }
        else Row(Modifier.fillMaxSize().background(Bg)) {
            Rail(onClick = { h.leaveCodeAlone() }) { SendRail(h) }
            VerticalDivider(color = Line)
        }
    }
}

