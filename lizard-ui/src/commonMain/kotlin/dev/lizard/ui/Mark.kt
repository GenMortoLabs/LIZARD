package dev.lizard.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

// ai: LIZARD's mark (Icons.mark; the scales since 2026-11-16), the drawing of lizard-web/icon.svg, of the Android
// ai: launcher's drawables and of the macOS package's icon: the symbol's four corner marks and the dark ring between
// ai: them; inside them a lizard's scales in six greys, dark to light down the field, each scale its outline (a white
// ai: edge to the next); a white edge between the scales and the corner marks.
// ai: In the mark's 36-unit drawing, scaled 1.24444 about the 64-unit square's centre: the ink spans 70% of the square
// ai: (the standard of 2026-10-10). Written by the mark's generator with the other copies: change them together.
private const val CLIP = "M19.85 19.85H44.15V44.15H19.85Z"
private const val BARS = "M23.49 14h17.02v2.08h-17.02zM23.49 47.92h17.02v2.08h-17.02zM14 23.49h2.08v17.02h-2.08zM47.92 23.49h2.08v17.02h-2.08z"
private const val EDGES = "M13.155 13.155h10.14v10.14h-10.14zM40.705 13.155h10.14v10.14h-10.14zM13.155 40.705h10.14v10.14h-10.14zM40.705 40.705h10.14v10.14h-10.14z"
private const val MARKS = "M14 14h8.45v8.45h-8.45zM41.55 14h8.45v8.45h-8.45zM14 41.55h8.45v8.45h-8.45zM41.55 41.55h8.45v8.45h-8.45z"

// ai: the scales, back to front: the outline, its grey and its white edge's width
private val OUTLINES = arrayOf(
    "M12.56 40.991 C12.56 48.273 15.67 51.327 17.42 52.736 C19.17 51.327 22.28 48.273 22.28 40.991Z",
    "M22.28 40.991 C22.28 48.273 25.39 51.327 27.14 52.736 C28.89 51.327 32 48.273 32 40.991Z",
    "M32 40.991 C32 48.273 35.11 51.327 36.86 52.736 C38.61 51.327 41.72 48.273 41.72 40.991Z",
    "M41.72 40.991 C41.72 48.273 44.83 51.327 46.58 52.736 C48.33 51.327 51.44 48.273 51.44 40.991Z",
    "M17.42 34.884 C17.42 42.165 20.53 45.219 22.28 46.629 C24.03 45.219 27.14 42.165 27.14 34.884Z",
    "M27.14 34.884 C27.14 42.165 30.25 45.219 32 46.629 C33.75 45.219 36.86 42.165 36.86 34.884Z",
    "M36.86 34.884 C36.86 42.165 39.97 45.219 41.72 46.629 C43.47 45.219 46.58 42.165 46.58 34.884Z",
    "M12.56 28.776 C12.56 36.058 15.67 39.112 17.42 40.521 C19.17 39.112 22.28 36.058 22.28 28.776Z",
    "M22.28 28.776 C22.28 36.058 25.39 39.112 27.14 40.521 C28.89 39.112 32 36.058 32 28.776Z",
    "M32 28.776 C32 36.058 35.11 39.112 36.86 40.521 C38.61 39.112 41.72 36.058 41.72 28.776Z",
    "M41.72 28.776 C41.72 36.058 44.83 39.112 46.58 40.521 C48.33 39.112 51.44 36.058 51.44 28.776Z",
    "M17.42 22.669 C17.42 29.951 20.53 33.004 22.28 34.414 C24.03 33.004 27.14 29.951 27.14 22.669Z",
    "M27.14 22.669 C27.14 29.951 30.25 33.004 32 34.414 C33.75 33.004 36.86 29.951 36.86 22.669Z",
    "M36.86 22.669 C36.86 29.951 39.97 33.004 41.72 34.414 C43.47 33.004 46.58 29.951 46.58 22.669Z",
    "M12.56 16.561 C12.56 23.843 15.67 26.897 17.42 28.306 C19.17 26.897 22.28 23.843 22.28 16.561Z",
    "M22.28 16.561 C22.28 23.843 25.39 26.897 27.14 28.306 C28.89 26.897 32 23.843 32 16.561Z",
    "M32 16.561 C32 23.843 35.11 26.897 36.86 28.306 C38.61 26.897 41.72 23.843 41.72 16.561Z",
    "M41.72 16.561 C41.72 23.843 44.83 26.897 46.58 28.306 C48.33 26.897 51.44 23.843 51.44 16.561Z",
    "M17.42 10.454 C17.42 17.736 20.53 20.79 22.28 22.199 C24.03 20.79 27.14 17.736 27.14 10.454Z",
    "M27.14 10.454 C27.14 17.736 30.25 20.79 32 22.199 C33.75 20.79 36.86 17.736 36.86 10.454Z",
    "M36.86 10.454 C36.86 17.736 39.97 20.79 41.72 22.199 C43.47 20.79 46.58 17.736 46.58 10.454Z"
)
private val GREYS = longArrayOf(0xFFC3C3C3, 0xFFC3C3C3, 0xFFC3C3C3, 0xFFC3C3C3, 0xFFA5A5A5, 0xFFA5A5A5, 0xFFA5A5A5, 0xFF878787, 0xFF878787, 0xFF878787, 0xFF878787, 0xFF6E6E6E, 0xFF6E6E6E, 0xFF6E6E6E, 0xFF555555, 0xFF555555, 0xFF555555, 0xFF555555, 0xFF3C3C3C, 0xFF3C3C3C, 0xFF3C3C3C)
private val EDGE_WIDTHS = floatArrayOf(0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f, 0.68f)

private fun ImageVector.Builder.scales() {
    for (i in OUTLINES.indices)
        addPath(addPathNodes(OUTLINES[i]), fill = SolidColor(Color(GREYS[i])), stroke = SolidColor(Color.White), strokeLineWidth = EDGE_WIDTHS[i])
}

internal fun markVector(): ImageVector = ImageVector.Builder("mark", 64.dp, 64.dp, 64f, 64f).apply {
    addPath(addPathNodes("M0 0h64v64h-64z"), fill = SolidColor(Color.White))
    addGroup(pivotX = 32f, pivotY = 32f, scaleX = 1.24444f, scaleY = 1.24444f)
    addGroup(clipPathData = addPathNodes(CLIP))
    scales()
    clearGroup()
    addPath(addPathNodes(BARS), fill = SolidColor(Fg))
    addPath(addPathNodes(EDGES), fill = SolidColor(Color.White))
    addPath(addPathNodes(MARKS), fill = SolidColor(Fg))
    clearGroup()
}.build()

// ai: the desktop's window and taskbar icon (Icons.appIcon), as its Linux and Windows package icons: white inside the
// ai: ring alone (the field, its seams, the corner marks' white edge), clear outside it and in the gaps between the bars
// ai: and the corner marks; the scales clipped to the field less a notch at each corner mark (the white there the
// ai: inside's own); the ink 60 of the 64 square.
private const val INSIDE = "M16.08 16.08H47.92V47.92H16.08Z"
private const val PC_CLIP = "M23.295 19.85L40.705 19.85L40.705 23.295L44.15 23.295L44.15 40.705L40.705 40.705L40.705 44.15L23.295 44.15L23.295 40.705L19.85 40.705L19.85 23.295L23.295 23.295Z"

internal fun appIconVector(): ImageVector = ImageVector.Builder("appIcon", 64.dp, 64.dp, 64f, 64f).apply {
    addGroup(pivotX = 32f, pivotY = 32f, scaleX = 1.667f, scaleY = 1.667f)
    addPath(addPathNodes(INSIDE), fill = SolidColor(Color.White))
    addGroup(clipPathData = addPathNodes(PC_CLIP))
    scales()
    clearGroup()
    addPath(addPathNodes(BARS + MARKS), fill = SolidColor(Fg))
    clearGroup()
}.build()
