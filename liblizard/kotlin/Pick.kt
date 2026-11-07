package dev.lizard.pick

import kotlin.math.abs

// ai: The ladder as the apps' senders read it (the C's constants copied: src/lizard_phy.h, src/fmt.h): the versions, the
// ai: rings, the rates offered, the border's modules. The Android app's and the desktop app's senders use this file
// ai: (compiled with their shared UI, lizard-ui); the picture size is the sender's own (lizard_n_for), never set here. No
// ai: pick since 2026-11-07 (the senders' Auto went: a version is always chosen); the library keeps its own, liz_pick
// ai: (liblizard/api/format.cpp, sim/lizard_pick.mjs pickVersion). Pure: no Android in here.
object Pick {
    // ai: the sub-channels a version (src/lizard_phy.h LIZARD_STEP): 16 x the version
    const val STEP = 16
    // ai: a block's payload, bytes (src/xfer.h XFER_PAYLOAD: the block's 476 less its 4-byte id): what a block carries
    const val PAYLOAD = 472
    // ai: every version from 4 to 64 (16 sub-channels each): LIZARD-64 to -1024, the web's VERSIONS (src/fmt.h
    // ai: OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX)
    const val VERSION_MIN = 4
    const val VERSION_MAX = 64
    val VERSIONS = (VERSION_MIN..VERSION_MAX).map { STEP * it }
    // ai: the senders' version unless one is chosen: 30, LIZARD-480 (the web sender's #subch default, 2026-10-01)
    const val VERSION_DEFAULT = 30
    // ai: liblizard/src/lizard_phy.h LIZARD_RING: 64, 128 and 256 band cells a side; LIZARD_RING_DEFAULT, the 128
    val RINGS = intArrayOf(64, 128, 256)
    const val RING_DEFAULT = 1
    // ai: the margin the codec paints round a symbol, modules (src/lizard_phy.h LIZARD_QUIET)
    const val QUIET = 2
    // ai: the pictures a second offered: 60 and its divisors to 10, every picture a whole number of refreshes at 60 Hz
    val FPS = listOf(60, 30, 20, 15, 10)
    // ai: a rate kept from a slider: the nearest offered, a tie to the higher
    fun fpsOf(v: Int) = FPS.minWith(compareBy<Int> { abs(it - v) }.thenByDescending { it })
    // ai: the border's depth past the ring's span, modules a side (src/layout.h OB_MARGIN)
    const val OB_MARGIN = 15

    // ai: the ring's span, 2 x its cells
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    // ai: a symbol's side in modules, its margin not counted
    fun modules(ring: Int = RING_DEFAULT) = span(ring) + 2 * OB_MARGIN
}
