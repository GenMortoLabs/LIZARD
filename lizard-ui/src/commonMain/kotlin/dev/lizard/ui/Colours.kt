package dev.lizard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin

// ai: Send's code colours (2026-11-16, both apps and the web): the code's ink and its paper, 0xRRGGBB, a frame's grey painted
// ai: along the line between them where it is shown, each channel apart (core/tx/sender.h setColours: expand.slang on the
// ai: GPU, Sender::take on the CPU; the web's lizard_tx_rgba and its WebGPU present), the margin round the code the
// ai: paper; black and white by default, the grey itself. The last row of Code: two tiles filled with their colours,
// ai: Primary (the ink: the dark modules and the border) and Secondary (the paper: the light, the quiet zone and the
// ai: area round the code), their hex on them; a tile tapped opens the picker under the tiles (a hue and saturation
// ai: wheel, a brightness bar, the hex), the code changing as it moves, Done or the tile again closing it. In the
// ai: column, not a dialog: Compose Desktop draws a dialog on the window's own canvas (compose.layers.type unset), under
// ai: the code's native canvas. No figure on the row (a contrast label with Show statistics on went the same day). No
// ai: floor (a palette a camera cannot separate fails at the receiver; an ink lighter than the paper paints an inverted
// ai: code, which no decoder reads); the measured margin is 60% (STATUS "Two brand colours instead of black and white").
const val INK_DEFAULT = 0x000000
const val PAPER_DEFAULT = 0xFFFFFF

class Colours(val ink: Int, val paper: Int, val choose: (ink: Int, paper: Int) -> Unit) : Setting

fun hexOf(c: Int): String = String.format(Locale.ROOT, "#%06X", c and 0xFFFFFF)
// ai: "#RRGGBB", "RRGGBB" or "#RGB" (any case) as 0xRRGGBB; null for anything else
fun colourOf(t: String): Int? {
    val h = t.trim().removePrefix("#")
    val six = if (h.length == 3) h.map { "$it$it" }.joinToString("") else h
    return if (six.length == 6 && six.all { it.isDigit() || it.lowercaseChar() in 'a'..'f' }) six.toInt(16) else null
}
fun lumaOf(c: Int): Double = (77 * ((c shr 16) and 255) + 150 * ((c shr 8) and 255) + 29 * (c and 255)) / 256.0
fun colourOfRgb(c: Int): Color = Color(0xFF000000.toInt() or (c and 0xFFFFFF))

// ai: hue (degrees), saturation and value (0..1) and back
fun hsvOf(c: Int): FloatArray {
    val r = ((c shr 16) and 255) / 255f; val g = ((c shr 8) and 255) / 255f; val b = (c and 255) / 255f
    val max = maxOf(r, g, b); val min = minOf(r, g, b); val d = max - min
    val h = when {
        d == 0f -> 0f
        max == r -> 60f * (((g - b) / d) % 6f)
        max == g -> 60f * ((b - r) / d + 2f)
        else -> 60f * ((r - g) / d + 4f)
    }
    return floatArrayOf((h + 360f) % 360f, if (max == 0f) 0f else d / max, max)
}
fun rgbOfHsv(hsv: FloatArray): Int {
    val (h, s, v) = Triple(hsv[0], hsv[1], hsv[2])
    val c = v * s; val x = c * (1 - kotlin.math.abs((h / 60f) % 2f - 1)); val m = v - c
    val (r, g, b) = when ((h / 60f).toInt().coerceIn(0, 5)) {
        0 -> Triple(c, x, 0f); 1 -> Triple(x, c, 0f); 2 -> Triple(0f, c, x)
        3 -> Triple(0f, x, c); 4 -> Triple(x, 0f, c); else -> Triple(c, 0f, x)
    }
    fun byte(f: Float) = ((f + m) * 255f).roundToInt().coerceIn(0, 255)
    return (byte(r) shl 16) or (byte(g) shl 8) or byte(b)
}

@Composable
internal fun ColoursView(s: Colours) {
    var open by remember { mutableStateOf(-1) }   // ai: the tile whose picker is open: 0 the ink, 1 the paper, -1 none
    Field("Colours") {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ColourTile("Primary", s.ink, open == 0, Modifier.weight(1f)) { open = if (open == 0) -1 else 0 }
                ColourTile("Secondary", s.paper, open == 1, Modifier.weight(1f)) { open = if (open == 1) -1 else 1 }
            }
            if (open >= 0) ColourPicker(if (open == 0) s.ink else s.paper, done = { open = -1 }) { c ->
                if (open == 0) s.choose(c, s.paper) else s.choose(s.ink, c)
            }
        }
    }
}

// ai: a colour's tile: filled with it, 64 dp, 12 dp corners, a hairline round it (2 dp of the foreground while its
// ai: picker is open), its name over its hex in white or the foreground, whichever reads on it
@Composable
private fun ColourTile(name: String, c: Int, open: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val on = if (lumaOf(c) < 140) Color.White else Fg
    val shape = RoundedCornerShape(12.dp)
    Column(modifier.height(64.dp).clip(shape).background(colourOfRgb(c))
        .border(if (open) 2.dp else 1.dp, if (open) Fg else Color.Black.copy(alpha = .14f), shape)
        .clickable(role = Role.Button, onClick = onClick).padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.SpaceBetween) {
        Text(name, style = MaterialTheme.typography.labelMedium, color = on.copy(alpha = .8f), maxLines = 1)
        Text(hexOf(c), style = MaterialTheme.typography.bodyLarge.copy(fontFeatureSettings = "tnum"), color = on, maxLines = 1)
    }
}

// ai: The picker: the wheel (hue round it, saturation out from its centre, drawn at the colour's brightness), the bar
// ai: (brightness, from black to the colour at its fullest) and the hex with Done. Its own hue and saturation kept while
// ai: the brightness is 0 or the saturation 0, which a colour alone does not hold; a hex typed sets all three.
@Composable
private fun ColourPicker(c: Int, done: () -> Unit, choose: (Int) -> Unit) {
    var hsv by remember { mutableStateOf(hsvOf(c)) }
    LaunchedEffect(c) { if (rgbOfHsv(hsv) != c) hsv = hsvOf(c) }
    val set: (FloatArray) -> Unit = { h -> hsv = h; choose(rgbOfHsv(h)) }
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Wheel(hsv, Modifier.size(220.dp)) { h, s -> set(floatArrayOf(h, s, hsv[2])) }
        ValueBar(hsv, Modifier.fillMaxWidth().height(28.dp)) { v -> set(floatArrayOf(hsv[0], hsv[1], v)) }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(Modifier.weight(1f)) { TextInput(hexOf(c), hexOf(INK_DEFAULT)) { t -> colourOf(t)?.let { hsv = hsvOf(it); choose(it) } } }
            Btn("Done", onClick = done)
        }
    }
}

private val HUES = listOf(Color.Red, Color.Yellow, Color.Green, Color.Cyan, Color.Blue, Color.Magenta, Color.Red)

// ai: a press or a drag picks: its angle from the centre the hue (clockwise from the right, as the sweep is drawn), its
// ai: distance the saturation (the rim 1)
@Composable
private fun Wheel(hsv: FloatArray, modifier: Modifier, pick: (Float, Float) -> Unit) {
    val onPick by rememberUpdatedState(pick)
    Canvas(modifier.pointerInput(Unit) {
        fun at(p: Offset) {
            val r = size.width / 2f
            val dx = p.x - r; val dy = p.y - r
            onPick(((Math.toDegrees(atan2(dy, dx).toDouble()) + 360.0) % 360.0).toFloat(), (hypot(dx, dy) / r).coerceIn(0f, 1f))
        }
        awaitEachGesture {
            val d = awaitFirstDown(); at(d.position); d.consume()
            drag(d.id) { ch -> at(ch.position); ch.consume() }
        }
    }) {
        val r = size.minDimension / 2
        drawCircle(Brush.sweepGradient(HUES, center), r)
        drawCircle(Brush.radialGradient(listOf(Color.White, Color.White.copy(alpha = 0f)), center, r), r)
        drawCircle(Color.Black.copy(alpha = 1f - hsv[2]), r)
        val a = Math.toRadians(hsv[0].toDouble())
        val k = Offset(center.x + cos(a).toFloat() * hsv[1] * r, center.y + sin(a).toFloat() * hsv[1] * r)
        drawCircle(Color.White, 9.dp.toPx(), k)
        drawCircle(colourOfRgb(rgbOfHsv(hsv)), 6.dp.toPx(), k)
        drawCircle(Color.Black.copy(alpha = .4f), 9.dp.toPx(), k, style = Stroke(1.dp.toPx()))
    }
}

@Composable
private fun ValueBar(hsv: FloatArray, modifier: Modifier, pick: (Float) -> Unit) {
    val onPick by rememberUpdatedState(pick)
    val full = colourOfRgb(rgbOfHsv(floatArrayOf(hsv[0], hsv[1], 1f)))
    Canvas(modifier.pointerInput(Unit) {
        fun at(x: Float) = onPick((x / size.width).coerceIn(0f, 1f))
        awaitEachGesture {
            val d = awaitFirstDown(); at(d.position.x); d.consume()
            drag(d.id) { ch -> at(ch.position.x); ch.consume() }
        }
    }) {
        val h = 16.dp.toPx(); val y = size.height / 2; val k = 10.dp.toPx()
        drawRoundRect(Brush.horizontalGradient(listOf(Color.Black, full)), Offset(0f, y - h / 2), Size(size.width, h), CornerRadius(h / 2))
        val x = (hsv[2] * size.width).coerceIn(k, size.width - k)
        drawCircle(Color.White, k, Offset(x, y))
        drawCircle(Color.Black.copy(alpha = .4f), k, Offset(x, y), style = Stroke(1.dp.toPx()))
    }
}
