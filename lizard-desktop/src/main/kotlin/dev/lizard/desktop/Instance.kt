package dev.lizard.desktop

import java.io.File
import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.concurrent.thread

// ai: One app takes what is handed in (2026-10-23): paths given to a later launch (a terminal's arguments, a file
// ai: manager's Open with or Broadcast with LIZARD, a double-clicked .lizard) go to the app already running, over a Unix socket in the session's
// ai: runtime folder, and that launch ends; the first app listens there. The socket is bound as a launch starts
// ai: (claim), before the native library and the window, so launches made together (Windows starts one a file
// ai: opened) find it bound: one binds, the rest hand over; its paths are taken once the app is made (serve), the
// ai: launches before that waiting in the socket's backlog. A socket left by an app that died is taken over. A launch
// ai: with no paths while one listens opens a window of its own, unheard, as before. What goes over: the paths one a
// ai: line, after a first line `--send` where they are to be broadcast (Main.kt), never a path (they are absolute).
object Instance {
    private val path: Path = Path.of(System.getenv("XDG_RUNTIME_DIR")?.takeIf { it.isNotEmpty() } ?: System.getProperty("java.io.tmpdir"), "lizard-sender.sock")
    private var server: ServerSocketChannel? = null

    private fun connect(): SocketChannel? = runCatching {
        SocketChannel.open(StandardProtocolFamily.UNIX).apply { connect(UnixDomainSocketAddress.of(path)) }
    }.getOrNull()

    private fun bind(): ServerSocketChannel? = runCatching {
        ServerSocketChannel.open(StandardProtocolFamily.UNIX).apply { bind(UnixDomainSocketAddress.of(path)) }
    }.getOrNull()

    // ai: true: the paths went to the app listening, and this launch ends. Otherwise this launch holds the socket where
    // ai: it can (a bind fails where another launch bound it a moment before, whose paths it then takes; a path nothing
    // ai: answers on is a dead app's, deleted and bound again)
    fun claim(files: List<File>, broadcast: Boolean): Boolean {
        repeat(2) { attempt ->
            connect()?.let { ch ->
                if (files.isEmpty()) { ch.close(); return false }
                return runCatching {
                    val lines = (if (broadcast) listOf(SEND) else emptyList()) + files.map { it.path }
                    ch.use { val b = ByteBuffer.wrap(lines.joinToString("\n").toByteArray()); while (b.hasRemaining()) it.write(b) }
                    true
                }.getOrDefault(false)
            }
            if (attempt == 1) runCatching { Files.deleteIfExists(path) }
            bind()?.let { s ->
                server = s
                Runtime.getRuntime().addShutdownHook(Thread { runCatching { s.close(); Files.deleteIfExists(path) } })
                return false
            }
        }
        System.err.println("instance: not listening at $path")
        return false
    }

    private const val SEND = "--send"

    // ai: the later launches' paths, those that exist, and whether to broadcast them, to take on this thread; nothing
    // ai: where claim did not bind
    fun serve(take: (List<File>, Boolean) -> Unit) {
        val s = server ?: return
        thread(name = "lizard-instance", isDaemon = true) {
            while (true) {
                val c = runCatching { s.accept() }.getOrNull() ?: break
                runCatching {
                    val text = c.use { Channels.newInputStream(it).readBytes().decodeToString() }
                    val lines = text.lines().filter { it.isNotEmpty() }
                    val broadcast = lines.firstOrNull() == SEND
                    val files = lines.drop(if (broadcast) 1 else 0).map(::File).filter { it.exists() }
                    if (files.isNotEmpty()) take(files, broadcast)
                }
            }
        }
    }
}
