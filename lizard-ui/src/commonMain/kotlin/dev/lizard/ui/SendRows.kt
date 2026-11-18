package dev.lizard.ui

import dev.lizard.pick.Pick
import dev.lizard.readout.Readout
import java.util.Locale
import kotlin.math.floor

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

    // ai: What goes, as elements (2026-11-07; "Sending x, 2.4 GB (4.4 GB raw)" over "1.69 MB/s, pass 2, 7.6 s a pass"
    // ai: before): the file's name, a spinner while it is prepared, then Size (its own), Files (a bundle's), Speed (the
    // ai: rate the pictures shown carry, "" while paused or not yet measured) and with Show statistics on Compressed (the
    // ai: bytes that go), Pass and Pass time (a pass's data blocks, lap, over the blocks a second); the test stream its
    // ai: name and Speed. Every row from Start, a value "" until known, so the block does not move. Both apps'; the
    // ai: web's twin is send.mjs sendDetails.
    fun details(name: String, test: Boolean, preparing: Boolean, size: Long, sent: Long, files: Int, speedKBs: Double,
                pass: Double, lap: Double, stats: Boolean): Details {
        val speed = if (speedKBs > 0) Readout.rate(speedKBs) else ""
        if (test) return Details("Test stream", test = true, spinning = preparing, rows = listOf("Speed" to speed))
        val rows = buildList {
            add("Size" to Readout.bytes(size))
            if (stats) add("Compressed" to if (sent > 0) Readout.bytes(sent) else "")
            if (files > 1) add("Files" to "$files")
            add("Speed" to speed)
            if (stats) {
                add("Pass" to if (pass >= 1) "${floor(pass).toInt()}" else "")
                add("Pass time" to if (lap > 0 && speedKBs > 0) "${String.format(Locale.ROOT, "%.1f", lap / (speedKBs * 1000 / Pick.PAYLOAD))}\u00a0s" else "")
            }
        }
        return Details(name, spinning = preparing, rows = rows)
    }

    // ai: An export (2026-11-18, Send's Export): the stream file's name, its share written at the head and as the meter,
    // ai: Frames (written of all, "" until the lap is known), Size (the file's bytes so far), Folder (where it goes)
    fun exportDetails(name: String, written: Int, frames: Int, bytes: Long, folder: String, done: Boolean): Details {
        val frac = if (done) 1.0 else if (frames > 0) written.toDouble() / frames else 0.0
        return Details(name, end = "${(frac * 100).toInt()}%", spinning = !done && frames <= 0, frac = frac,
            rows = listOf("Frames" to if (frames > 0) "${String.format(Locale.ROOT, "%,d", written)} of ${String.format(Locale.ROOT, "%,d", frames)}" else "",
                "Size" to if (bytes > 0) Readout.bytes(bytes) else "", "Folder" to folder))
    }

    // ai: a stream file's format as its line says it, "LIZARD-480 · 2:1 · 15 fps" (its meta.json's format, layout, fps)
    fun streamLabel(blocks: Int, codes: Int, fps: Int): String = "LIZARD-${Pick.STEP * blocks} \u00b7 ${if (codes > 1) "2:1" else "1:1"} \u00b7 $fps fps"

    // ai: an export's file size before it is made (Laps' value): its frames (laps of the file's blocks over a picture's,
    // ai: odd) at a frame's raw bytes (frameBytes, the native side's: its squares) x 0.73, what MED and zstd at level 1
    // ai: leave of these frames (72 to 74%, 2026-11-18)
    fun exportEstimate(laps: Double, fileBytes: Long, pictureBlocks: Int, frameBytes: Long): Long {
        if (fileBytes <= 0 || pictureBlocks <= 0 || frameBytes <= 0) return 0
        val lap = (fileBytes + Pick.PAYLOAD - 1) / Pick.PAYLOAD
        val frames = kotlin.math.ceil(laps * lap / pictureBlocks).toLong() or 1L
        return (frames * frameBytes * 0.73).toLong()
    }
}
