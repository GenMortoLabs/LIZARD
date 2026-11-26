package dev.lizard.desktop

import dev.lizard.lib.Ladder

// ai: The desktop sender's spacing and margin (liblizard/kotlin/Lib.kt Ladder's modules, the C's): two codes stand Spacing apart and
// ai: the area's edges Margin from them, both from the rings (lizard-web/send.mjs the same).
object Room {
    // ai: Spacing and Margin (2026-10-23; one Margin, two of them between codes, that morning; a gap between two codes
    // ai: before): the light between two codes' rings, two quiet zones (Ladder.QUIET, LIZARD_QUIET) at least, and between a
    // ai: ring and the area's edge, one at least, in modules. SPACING_MAX a painter's widest gap (64 modules) and its quiet
    // ai: zones; SPACING_DEFAULT the encoders' own gap (liblizard/gpu/encoder.mjs GAP_MODULES, 12) and its quiet zones
    val SPACING_MIN get() = 2 * Ladder.QUIET
    const val SPACING_MAX = 68
    const val SPACING_DEFAULT = 16
    const val MARGIN_MAX = 34
    const val MARGIN_DEFAULT = 8
    // ai: the gap a painter puts between two codes past their quiet zones, modules
    fun gapOf(spacing: Int) = spacing - 2 * Ladder.QUIET
    // ai: the light the presenter keeps at each edge past a frame's quiet zone, a share of the frame's height (a painted
    // ai: symbol's side: its modules and quiet zones) in the ring whose picture spans span modules
    fun edgeShare(span: Int, margin: Int) = (margin - Ladder.QUIET).toDouble() / (Ladder.modules(Ladder.RINGS.indexOf(span / 2)) + 2 * Ladder.QUIET)
}
