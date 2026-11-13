package dev.lizard.desktop

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeDialog
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.composed
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.onPointerEvent
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogWindow
import androidx.compose.ui.window.WindowPlacement
import androidx.compose.ui.window.WindowState
import dev.lizard.ui.Bg
import dev.lizard.ui.Line
import dev.lizard.ui.Rail
import dev.lizard.ui.SideOver
import dev.lizard.ui.ToolHost
import dev.lizard.ui.ToolRail
import dev.lizard.ui.sideWidthOf
import java.awt.Dialog
import java.awt.Frame
import java.awt.MouseInfo
import java.awt.Window
import kotlin.math.roundToInt
import kotlinx.coroutines.delay

// ai: The window's full screen, one for Send and Receive (2026-11-13; Send's alone before, in SendState): the main area
// ai: alone (Send's code, Receive's camera), Send ↔ Receive keeping it, Home leaving it (App.go). Asked by the
// ai: Fullscreen button, F or F11, left by them or Escape (App.kt's keys); the manager's own word followed (its shortcut,
// ai: a leave of its own). Its edge strip (`edgeRail`, EdgeRail) and the column over the main area (`sideOver`, lizard-ui
// ai: SideOver) are the screen's own rail and column.
// ai:
// ai: The window manager's own full screen as other apps have it (2026-10-23; Native.windowState, EWMH _NET_WM_STATE),
// ai: which gives the window back as it was, maximized or not; Windows the window's frame off over its monitor
// ai: (win32Fullscreen, 2026-10-24); macOS Compose's placement. The window is found through itself.
// ai:
// ai: On X11 a window the app makes full screen is maximized first, where it is not already, and unmaximized after it
// ai: leaves (2026-11-13; a manager refuses to maximize a full-screen window, Muffin's recalc_features). Leaving full
// ai: screen, Muffin updates the frame's extents before the restored size (meta_window_unmake_fullscreen); AWT, which
// ai: counts it as Mutter, answers an extents change by requesting the bounds it last knew
// ai: (XDecoratedPeer.handleCorrectInsets), the full-screen rectangle, and Muffin makes a window asking for exactly its
// ai: monitor's rectangle full screen again ("Treating resize request of legacy application ... as a fullscreen
// ai: request"; window-x11.c, no opt-out): every leave came back full screen. AWT skips that request for a maximized
// ai: frame, maximized read from _NET_WM_STATE whether or not it is also full screen (XNETProtocol.getStateImpl), and
// ai: Muffin keeps the window's own rectangle through the maximize and the full screen (meta_window_save_rect), so the
// ai: unmaximize after the leave gives the window back its size. The unmaximize waits for the window to say it left
// ai: (seen, at its resize), the app's own leave and the manager's alike: AWT reads _NET_WM_STATE as it handles each
// ai: change, so with the unmaximize asked right behind the leave it read the window unmaximized as the extents came
// ai: (JDK 25 on Xvfb) and asked for the full-screen rectangle anyway. A full screen the manager makes itself (its own
// ai: shortcut) gets no maximize: the manager refuses it there. Setting the window's bounds after the leave instead
// ai: (b53a843, a9c11cf) raced AWT's toolkit thread and lost.
class FullScreen {
    var on by mutableStateOf(false)
        private set
    // ai: the edge strip shown: the pointer came within EDGE_NEAR of the main area's left edge (edgeModifier; Send's code
    // ai: canvas hears it over itself)
    var edgeRail by mutableStateOf(false)
    // ai: the column over the main area, open from the strip's collapser until its own collapser or a change of full screen
    var sideOver by mutableStateOf(false)
    // ai: a full screen's own sizes come for a while after a change: not resizes by hand (SendState.resizing)
    private var settleUntil = 0L
    val settling get() = System.nanoTime() < settleUntil

    private var window: ComposeWindow? = null
    private var ws: WindowState? = null
    private val x11 = !Native.windows && !Native.mac
    // ai: the window maximized by this class under its full screen (to be unmaximized as it leaves)
    private var maxedUnder = false
    private var before = WindowPlacement.Floating

    // ai: the window, once made (LizardWindow)
    fun attach(window: ComposeWindow, ws: WindowState) { this.window = window; this.ws = ws }

    fun toggle() = set(!on)

    // ai: asked on or off: on X11 on with the window maximized first where it is not (one connection, in that order); off
    // ai: full screen alone, the maximize removed once the window says it left (seen)
    fun set(v: Boolean) {
        changed(v)
        val w = window ?: return
        val under = v && x11 && w.extendedState and Frame.MAXIMIZED_BOTH != Frame.MAXIMIZED_BOTH
        if (runCatching { Native.windowState(top(w), v, true, under) }.getOrDefault(false)) { if (v) maxedUnder = under }
        else ws?.let { s -> if (v) { before = s.placement; s.placement = WindowPlacement.Fullscreen } else s.placement = before }
    }

    // ai: at each resize of the window: the manager's word followed (nothing asked); a window the app maximized under its
    // ai: full screen unmaximized once it is no longer full screen, whoever left (the app, the manager's own shortcut)
    fun seen() {
        val w = window ?: return
        val t = top(w)
        if (t == 0L) return
        val full = Native.windowStateOf(t) and WM_FULL != 0
        if (!full && maxedUnder) { Native.windowState(t, false, false, true); maxedUnder = false }
        if (full != on) changed(full)
    }

    // ai: the pointer at x (px) in the main area, `near` being EDGE_NEAR there: the strip shown in full screen
    fun pointerAt(x: Float, near: Float) { if (on && !edgeRail && x < near) edgeRail = true }

    private fun changed(v: Boolean) {
        on = v
        edgeRail = false
        sideOver = false
        settleUntil = System.nanoTime() + 1_500_000_000L
    }
    private fun top(w: ComposeWindow) = runCatching { Native.windowOf(w) }.getOrDefault(0L)

    companion object {
        // ai: Native.windowStateOf's full-screen bit (presenter.h WM_FULL)
        const val WM_FULL = 1
    }
}

// ai: how near the main area's left edge the pointer shows full screen's edge strip
val EDGE_NEAR = 24.dp

// ai: The main area's own (both screens' ToolHost.areaModifier): the pointer near its left edge shows the edge strip
@OptIn(ExperimentalComposeUiApi::class)
fun edgeModifier(full: FullScreen): Modifier = Modifier.composed {
    val near = with(LocalDensity.current) { EDGE_NEAR.toPx() }
    Modifier.onPointerEvent(PointerEventType.Move) { e -> full.pointerAt(e.changes.firstOrNull()?.position?.x ?: near, near) }
}

// ai: Full screen's edge strip (2026-10-23 Send's, 2026-11-13 both screens'): the screen's collapsed rail (lizard-ui
// ai: ToolRail) over the window's left edge while the pointer is within EDGE_NEAR of it or over the strip, in a window of
// ai: its own (nothing of Compose draws over Send's heavyweight code canvas), over the main area rather than beside it
// ai: so the area and Send's format stay as they are. A popup (override-redirect on X11): the window manager neither
// ai: places it (a dialog it centres on its owner at first) nor shades it, nor gives it the focus full screen holds.
// ai: Mapped for as long as full screen lasts, parked off every screen while hidden, so a show is a move (a window mapped
// ai: and drawn at each show took 0.5 to 1.1 s). Its collapser opens the column over the main area (lizard-ui SideOver,
// ai: 2026-11-07): the window as wide as the column, open until the column's own collapser or full screen left. The
// ai: strip alone is hidden once the pointer is off it (read a few times a second: the area below hears nothing while
// ai: the pointer is over the strip, nor once it goes past the screen's edge).
@Composable
fun EdgeRail(full: FullScreen, h: ToolHost, owner: ComposeWindow) {
    val shown = full.edgeRail || full.sideOver
    val over = full.sideOver
    // ai: the column's width as the landscape layout takes it on this screen (sideWidthOf)
    val sideDp = sideWidthOf(h, (owner.contentPane.width / owner.graphicsConfiguration.defaultTransform.scaleX).toFloat().dp).value
    DialogWindow(visible = true,
        create = { ComposeDialog(owner, Dialog.ModalityType.MODELESS).apply { type = Window.Type.POPUP; isUndecorated = true; isResizable = false; focusableWindowState = false } },
        dispose = ComposeDialog::dispose,
        update = { d ->
            // ai: full screen's main area is the window's whole content; the window's width the rail's or the column's,
            // ai: and its divider, in the screen's pixels
            val c = owner.contentPane
            val hgt = maxOf(1, c.height)
            val scale = owner.graphicsConfiguration.defaultTransform.scaleX
            val w = if (over) ((sideDp + 1) * scale).roundToInt() else 57
            if (shown && c.isShowing) c.locationOnScreen.let { d.setBounds(it.x, it.y, w, hgt) } else d.setBounds(-10000, -10000, w, hgt)
        }) {
        LaunchedEffect(shown, over) {
            while (shown && !over) {
                delay(150)
                val p = MouseInfo.getPointerInfo()?.location
                if (p == null || !window.bounds.contains(p)) { full.edgeRail = false; break }
            }
        }
        if (over) SideOver(h, sideDp.dp) { full.sideOver = false }
        else Row(Modifier.fillMaxSize().background(Bg)) {
            Rail(onClick = { full.sideOver = true }) { ToolRail(h) }
            VerticalDivider(color = Line)
        }
    }
}
