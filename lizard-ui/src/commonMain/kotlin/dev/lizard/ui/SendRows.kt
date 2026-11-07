package dev.lizard.ui

import dev.lizard.pick.Pick
import java.util.Locale

// ai: The sender's format rows as Show statistics names them (2026-11-07; the receiver's switch, one for both screens,
// ai: shown in Send's Developer Tools too). On: Blocks, its value the count and the bytes ("58, 27.4 KB"), and the ring
// ai: by its cells, 64, 128, 256. Off: Density, the bytes alone ("27.4 KB"), and the ring by its border's look, Thin (the
// ai: 256, its border 3% of the symbol's side), Medium (the 128) and Thick (the 64, 9%), in that order. Codes 1 and 2
// ai: either way. A choice stays a ring index, so a kept ring reads alike in both. The web sender's are send.mjs
// ai: showStatsLabels and blocksText.
object SendRows {
    fun blocksTitle(stats: Boolean) = if (stats) "Blocks" else "Density"

    // ai: the blocks a frame carries and what they hold, 1 KB = 1000 B (Readout's)
    fun blocksText(blocks: Int, stats: Boolean): String {
        val kb = "${String.format(Locale.ROOT, "%.1f", blocks * Pick.PAYLOAD / 1000.0)} KB"
        return if (stats) "$blocks, $kb" else kb
    }

    private val LOOK = mapOf(64 to "Thick", 128 to "Medium", 256 to "Thin")
    fun rings(stats: Boolean): List<Pair<Int, String>> =
        if (stats) Pick.RINGS.indices.map { it to "${Pick.RINGS[it]}" }
        else Pick.RINGS.indices.reversed().map { it to LOOK.getValue(Pick.RINGS[it]) }

    val CODES = listOf(1 to "1", 2 to "2")
}
