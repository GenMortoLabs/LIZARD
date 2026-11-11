package dev.lizard.desktop

import androidx.compose.ui.window.application
import java.awt.EventQueue
import kotlin.system.exitProcess
import dev.lizard.ui.Screen

// ai: LIZARD on a PC (2026-10-04 the sender, the web sender's page as a window; since 2026-10-25 the app, App.kt: Home,
// ai: Send and Receive). The sender's column on the left and the code area on the right, the code drawn there by the
// ai: native presenter; the receiver's column and its camera's crop beside it. LIZ_SENDER_TEST (SenderTest.kt) and
// ai: LIZ_RECEIVER_TEST (ReceiverTest.kt) run an integration check instead of waiting for a hand, at most one of them.
// ai: args: paths handed in (App.handIn), opened: a .lizard (a double click: the packages register the type,
// ai: build.gradle.kts fileAssociation) to Receive's File camera, anything else (a terminal, a file manager's Open with
// ai: through the package's desktop entry, Exec %F) to Send; `--send` first, broadcast: all of them to Send, a .lizard
// ai: too (the file manager's Broadcast with LIZARD, packaging/lizard.nemo_action). An app already running takes them
// ai: instead (Instance.kt), and this launch ends. macOS hands a document opened to the running app as an event, not
// ai: as arguments (openFiles).
// ai: The first screen: Home, unless paths were given; Send under LIZ_SENDER_TEST; Receive under LIZ_RECEIVER_TEST.
fun main(args: Array<String>) {
    // ai: AWT erases nothing under the presenter's pictures (the code area's canvas is the swapchain's)
    System.setProperty("sun.awt.noerasebackground", "true")
    val sendTest = SenderTest.parse(System.getenv("LIZ_SENDER_TEST"))
    val recvTest = ReceiverTest.parse(System.getenv("LIZ_RECEIVER_TEST"))
    if (sendTest != null && recvTest != null) { println("FAIL LIZ_SENDER_TEST and LIZ_RECEIVER_TEST are both set"); exitProcess(1) }
    if (sendTest != null || recvTest != null) {
        Prefs.forget()
        Thread.setDefaultUncaughtExceptionHandler { _, e ->
            val why = "${e.javaClass.simpleName}: ${e.message}"
            if (recvTest != null) ReceiverTest.fail(why) else SenderTest.fail(why)
        }
    }
    val broadcast = args.firstOrNull() == "--send"
    val given = args.drop(if (broadcast) 1 else 0).map { java.io.File(it).absoluteFile }.filter { it.exists() }
    val testing = sendTest != null || recvTest != null
    if (!testing && Instance.claim(given, broadcast)) return
    val app = App()
    val s = app.send
    try { Native.load() } catch (e: Throwable) { app.loadError = "Native library missing: ${e.message}" }
    // ai: the launch's priority raised (native/priority.cpp) before the window's threads start, which inherit it;
    // ai: LIZ_PRIORITY=normal leaves it
    if (app.loadError.isEmpty() && System.getenv("LIZ_PRIORITY") != "normal")
        runCatching { Native.raisePriority() }.getOrNull()?.takeIf { it.isNotEmpty() }?.let { System.err.println("priority: $it") }
    sendTest?.apply(s)
    recvTest?.apply(app)
    app.screen = when {
        recvTest != null -> Screen.Receive
        sendTest != null -> Screen.Send
        else -> Screen.Home
    }
    if (!testing) {
        if (given.isNotEmpty()) app.handIn(given, broadcast)
        Instance.serve { files, b -> EventQueue.invokeLater { app.handIn(files, b) } }
        openFiles { files -> EventQueue.invokeLater { app.handIn(files, false) } }
    }
    application { LizardWindow(app, sendTest, recvTest) }
}

// ai: macOS: a document opened in Finder (a double click on a .lizard, the bundle's document type) reaches the app as an
// ai: open-file event, at its launch and while it runs; where the platform has no such event (Linux, Windows: the
// ai: path comes as an argument) nothing is set
private fun openFiles(take: (List<java.io.File>) -> Unit) {
    if (!java.awt.Desktop.isDesktopSupported()) return
    val d = java.awt.Desktop.getDesktop()
    if (!d.isSupported(java.awt.Desktop.Action.APP_OPEN_FILE)) return
    d.setOpenFileHandler { e -> e.files.map { it.absoluteFile }.filter { it.exists() }.takeIf { it.isNotEmpty() }?.let(take) }
}
