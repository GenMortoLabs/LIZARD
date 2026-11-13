package dev.lizard.desktop

import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.window.WindowPlacement
import androidx.compose.ui.window.WindowState
import java.awt.Frame

// ai: The window's full screen as other apps have it (2026-10-23): the window manager's own (Native.windowState, EWMH
// ai: _NET_WM_STATE), which gives the window back as it was, maximized or not; Windows the window's frame off over its
// ai: monitor (win32Fullscreen, 2026-10-24); macOS Compose's placement. The window is found through itself.
// ai:
// ai: On X11 a window the app makes full screen is maximized first, where it is not already, and unmaximized after it
// ai: leaves (2026-11-13; a manager refuses to maximize a full-screen window, Muffin's recalc_features). Leaving full
// ai: screen, Muffin updates the frame's extents before the restored size (meta_window_unmake_fullscreen); AWT, which
// ai: counts it as Mutter, answers an extents change by
// ai: requesting the bounds it last knew (XDecoratedPeer.handleCorrectInsets), the full-screen rectangle, and Muffin
// ai: makes a window asking for exactly its monitor's rectangle full screen again ("Treating resize request of legacy
// ai: application ... as a fullscreen request"; window-x11.c, no opt-out): every leave came back full screen. AWT skips
// ai: that request for a maximized frame, maximized read from _NET_WM_STATE whether or not it is also full screen
// ai: (XNETProtocol.getStateImpl), and Muffin keeps the window's own rectangle through the maximize and the full screen
// ai: (meta_window_save_rect), so the unmaximize after the leave gives the window back its size. A full screen the
// ai: manager makes itself (its own shortcut) gets no maximize: the manager refuses it there. Setting the window's
// ai: bounds after the leave instead (b53a843, a9c11cf) raced AWT's toolkit thread and lost.
class FullScreen(private val window: ComposeWindow, private val ws: WindowState) {
    private val x11 = !Native.windows && !Native.mac
    // ai: the window maximized by this class under its full screen (to be unmaximized as it leaves)
    private var maxedUnder = false
    private var before = WindowPlacement.Floating
    private fun top() = runCatching { Native.windowOf(window) }.getOrDefault(0L)

    // ai: full screen asked on or off: on X11 on with the window maximized first where it is not, off with that maximize
    // ai: removed after it (one connection, in that order)
    fun set(on: Boolean) {
        val under = x11 && if (on) window.extendedState and Frame.MAXIMIZED_BOTH != Frame.MAXIMIZED_BOTH else maxedUnder
        if (runCatching { Native.windowState(top(), on, true, under) }.getOrDefault(false)) maxedUnder = on && under
        else if (on) { before = ws.placement; ws.placement = WindowPlacement.Fullscreen } else ws.placement = before
    }

    // ai: at each resize of the window: whether the manager has it full screen (null where the window cannot be asked);
    // ai: a full screen the app made and the manager left (its own shortcut) unmaximized as the app's own leave does
    fun seen(): Boolean? {
        val t = top()
        if (t == 0L) return null
        val full = Native.windowStateOf(t) and WM_FULL != 0
        if (!full && maxedUnder) { Native.windowState(t, false, false, true); maxedUnder = false }
        return full
    }

    companion object {
        // ai: Native.windowStateOf's bits (presenter.h WM_FULL, WM_MAXIMIZED)
        const val WM_FULL = 1
    }
}
