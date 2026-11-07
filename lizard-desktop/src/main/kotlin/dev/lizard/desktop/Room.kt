package dev.lizard.desktop

import dev.lizard.pick.Pick

// ai: The desktop sender's spacing and margin (liblizard/kotlin/Pick.kt's modules): two codes stand Spacing apart and
// ai: the area's edges Margin from them, both from the rings (lizard-web/send.mjs the same).
object Room {
    // ai: Spacing and Margin (2026-10-23; one Margin, two of them between codes, that morning; a gap between two codes
    // ai: before): the light between two codes' rings, two quiet zones (Pick.QUIET, LIZARD_QUIET) at least, and between a
    // ai: ring and the area's edge, one at least, in modules. SPACING_MAX a painter's widest gap (64 modules) and its quiet
    // ai: zones; SPACING_DEFAULT the encoders' own gap (liblizard/gpu/encoder.mjs GAP_MODULES, 12) and its quiet zones
    const val SPACING_MIN = 2 * Pick.QUIET
    const val SPACING_MAX = 68
    const val SPACING_DEFAULT = 16
    const val MARGIN_MAX = 34
    const val MARGIN_DEFAULT = 8
    // ai: the gap a painter puts between two codes past their quiet zones, modules
    fun gapOf(spacing: Int) = spacing - 2 * Pick.QUIET
    // ai: the light the presenter keeps at each edge past a frame's quiet zone, a share of the frame's height (a painted
    // ai: symbol's side) in the ring of span cells
    fun edgeShare(span: Int, margin: Int) = (margin - Pick.QUIET).toDouble() / (span + 2 * Pick.OB_MARGIN + 2 * Pick.QUIET)
}
