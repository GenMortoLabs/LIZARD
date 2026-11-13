package dev.lizard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

// ai: What Send and Receive share, each app's host for either (2026-11-13, one implementation for both screens: the
// ai: column, its folds and its swap, full screen): SendHost and ReceiveHost add what is each one's own.
interface ToolHost {
    val kind: ToolKind
    val folds: Folds
    val width: ColumnWidth
    val canPortrait: Boolean           // ai: an upright screen takes the portrait layout (the phone's)
    val swapInColumn: Boolean          // ai: the swap in the landscape column's bar (the desktop's; the phone's column has room for the collapser alone)
    fun goHome()
    fun goOther()                      // ai: the swap: Send's to Receive, Receive's to Send
    val heat: Warn?                    // ai: the heat warning (the phone's Throttling), null while nothing throttles
    val settings: Map<FoldKey, List<Setting>>   // ai: each of its kind's folds' settings; a fold given none is not drawn
    val experimental: List<Setting>    // ai: the Experimental fold's lab arms; no fold where there are none
    val tools: List<Setting>
    val readout: Code? get() = null    // ai: the lab's text under Developer Tools' fields (CodeBlock's own 6 dp under them)
    val fullscreen: Boolean get() = width.fullscreen   // ai: a Fullscreen button (ColumnWidth.fullscreen: the desktop's)
    fun toggleFull() {}
    // ai: the main area alone on the screen: the desktop's full screen on both screens; Send's on the phone while sending
    val alone: Boolean get() = false
    // ai: the platform's on the main area (the desktop's: in full screen the pointer near its left edge shows the edge
    // ai: strip, App.kt)
    val areaModifier: Modifier get() = Modifier
}

// ai: What tells Send's column from Receive's: its title, the swap's label, its settings folds and its Experimental
// ai: and Developer Tools folds' keys
class ToolKind(val title: String, val other: String, val folds: List<Pair<FoldKey, String>>, val experimental: FoldKey, val tools: FoldKey)
val SEND = ToolKind("Send", "Receiver", SEND_FOLDS, FoldKey.SendExperimental, FoldKey.SendTools)
val RECEIVE = ToolKind("Receive", "Sender", RECV_FOLDS, FoldKey.RecvExperimental, FoldKey.RecvTools)

// ai: The swap to the other screen as the column's bar draws it
@Composable
fun ToolSwap(h: ToolHost) = SwapBtn(h.kind.other) { h.goOther() }

// ai: Fullscreen beside Start where the platform has it (ColumnWidth.fullscreen, the desktop's), both screens'
@Composable
fun FullscreenBtn(h: ToolHost) { if (h.fullscreen) Btn("Fullscreen", pad = 16.dp) { h.toggleFull() } }

// ai: The column's lower part, both screens': the settings folds by topic (SettingFolds), Experimental where it holds
// ai: a setting, Developer Tools (the readout under its fields), Back where `home` (the landscape column and the column
// ai: over the main area; portrait's bar has its own arrow)
@Composable
fun ToolFolds(h: ToolHost, home: Boolean) {
    val k = h.kind
    SettingFolds(h.folds, k.folds, h.settings)
    if (h.experimental.isNotEmpty()) Fold("Experimental", h.folds.open(k.experimental), { h.folds.toggle(k.experimental) }) { Settings(h.experimental) }
    Fold("Developer Tools", h.folds.open(k.tools), { h.folds.toggle(k.tools) }) {
        Settings(h.tools)
        h.readout?.let { CodeBlock(it.text, it.size) }
    }
    if (home) HomeRow { h.goHome() }
    Spacer(Modifier.height(24.dp))
}

// ai: The column's width on a screen `screen` wide, as the landscape layout takes it (SideWidth: its default or the
// ai: width its edge was dragged to, within its floor, one for both screens (columnFloor), and three fifths of the screen)
@Composable
fun sideWidthOf(h: ToolHost, screen: Dp): Dp = SideWidth(screen, h.width.dp, h.width.default(screen), columnFloor(h.width))

// ai: The column over the main area alone (the desktop's full screen; Send's 2026-11-07, both screens' 2026-11-13): the
// ai: edge strip's collapser opens it over the area's left part, full screen kept, the area keeping its size under it;
// ai: the column as the landscape layout draws it (the bar with the swap, the screen's column scrolling, Back at its
// ai: foot), its own collapser closing it. Fullscreen, its keys or Escape still leave full screen.
@Composable
fun SideOver(h: ToolHost, width: Dp, onClose: () -> Unit) = Row(Modifier.fillMaxHeight().background(Bg)) {
    Column(Modifier.width(width).fillMaxHeight()) {
        Box(Modifier.padding(horizontal = SidePad)) {
            TopBar(h.kind.title) { if (h.swapInColumn) ToolSwap(h); CollapseBtn(false, onClose) }
        }
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val scroll = rememberScrollState()
            Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = SidePad)) {
                CompositionLocalProvider(LocalButtonsFill provides h.width.stretch) {
                    when (h) {
                        is SendHost -> SendSide(h, home = true)
                        is ReceiveHost -> ReceiveSide(h, home = true)
                    }
                }
            }
            LocalScrollbar.current(scroll, Modifier.align(Alignment.CenterEnd).fillMaxHeight())
        }
    }
    VerticalDivider(color = Line)
}

// ai: The collapsed rail's squares of either screen (the desktop's edge strip in full screen)
@Composable
fun ToolRail(h: ToolHost) {
    when (h) {
        is SendHost -> SendRail(h)
        is ReceiveHost -> ReceiveRail(h)
    }
}
