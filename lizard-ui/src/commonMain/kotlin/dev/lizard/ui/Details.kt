package dev.lizard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ai: What a send or a receive moves, as elements (2026-11-07; a sentence of state and figures before, "Sending x, 2.4 GB
// ai: (4.4 GB raw)" over "1.69 MB/s, pass 2, 7.6 s a pass"): a head, the item's icon and name (its stem cut, its ending
// ai: kept: NameLine) with at its end a percentage or, while it is being prepared, a spinner; the meter where there is
// ai: one; then a row a fact, its label muted at the left and its value at the right in tabular figures. No word for the
// ai: state: the buttons, the code and the meter say it. test: the test stream's head (its name not a file's). A value
// ai: "" keeps its row (a fact not yet known), so the block does not move as figures arrive. The web's twin is ui.mjs
// ai: details and ui.css .dt.
class Details(val name: String, val test: Boolean = false, val end: String = "", val spinning: Boolean = false,
              val frac: Double? = null, val rows: List<Pair<String, String>> = emptyList())

// ai: clear: drawn in no colour, its room kept (a run's block once shown keeps its place when it falls silent)
@Composable
fun DetailsBlock(d: Details, modifier: Modifier = Modifier, clear: Boolean = false) {
    val fg = if (clear) Color.Transparent else Fg
    val muted = if (clear) Color.Transparent else Muted
    val nameStyle = MaterialTheme.typography.titleSmall.copy(fontSize = 16.sp, lineHeight = 22.sp)
    val figures = MaterialTheme.typography.bodyMedium.copy(fontFeatureSettings = "tnum")
    Column(modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 30.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(Icons.file, contentDescription = null, Modifier.size(20.dp), tint = muted)
            if (d.test) Text(d.name, style = nameStyle, color = fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
            else NameLine("", d.name, "", nameStyle, fg, Modifier.weight(1f))
            if (d.spinning) CircularProgressIndicator(Modifier.size(16.dp), color = fg, strokeWidth = 2.dp, trackColor = if (clear) Color.Transparent else Line)
            else if (d.end.isNotEmpty()) Text(d.end, style = figures, color = muted, maxLines = 1)
        }
        if (d.frac != null) Meter(d.frac)
        d.rows.forEach { (label, value) ->
            Row(Modifier.fillMaxWidth().heightIn(min = 26.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(label, style = MaterialTheme.typography.bodyMedium, color = muted, maxLines = 1, modifier = Modifier.weight(1f))
                Text(value, style = figures, color = fg, maxLines = 1)
            }
        }
    }
}

// ai: A warning as a status label (2026-11-07; a sentence before, "Set the sender to at most 30 fps for this camera.";
// ai: sentences are for popups and covers): its label ("Sender too fast", "Throttling") and where a figure helps a value
// ai: ("max 30 fps", "camera 42 of 60 fps"). Drawn as a row under the block, the red triangle before its label (WarnRow;
// ai: rows, not chips); in a rail the triangle alone, its words the description (text).
class Warn(val label: String, val value: String = "") {
    val text get() = if (value.isEmpty()) label else "$label, $value"
}

@Composable
fun WarnRow(w: Warn, modifier: Modifier = Modifier, clear: Boolean = false) {
    Row(modifier.fillMaxWidth().heightIn(min = 26.dp).semantics(mergeDescendants = true) {}, verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(Icons.alert, contentDescription = "Warning", Modifier.size(16.dp), tint = if (clear) Color.Transparent else Bad)
        Text(w.label, style = MaterialTheme.typography.bodyMedium, color = if (clear) Color.Transparent else Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
        if (w.value.isNotEmpty()) Text(w.value, style = MaterialTheme.typography.bodyMedium.copy(fontFeatureSettings = "tnum"), color = if (clear) Color.Transparent else Muted, maxLines = 1)
    }
}

// ai: What went wrong, said over an area (the receiver's preview since 2026-11-02; Send's code area for a lasting fault,
// ai: its errors popups since 2026-11-07): the soft fill, the red triangle and the words under it, in the form what
// ai: happened, then what it touched ("Camera stopped: ..."), never in the column
@Composable
fun ErrorCover(text: String, modifier: Modifier = Modifier) = Column(modifier.background(Soft).padding(horizontal = 24.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
    Icon(Icons.alert, contentDescription = null, Modifier.size(24.dp), tint = Bad)
    Text(text, style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
}

// ai: A note as a popup (2026-11-07; a red line on Home before; Send's errors too, a cover before): the part before its
// ai: first ": " the title, the rest the body ("Not saved to Downloads" over the reason, "File changed" over "Choose it
// ai: again"), a body opening on a plain word taking a capital (a name such as dev.fosslabs.lizard kept), OK closing it
@Composable
fun NoteDialog(note: String, onDismiss: () -> Unit) {
    val at = note.indexOf(": ")
    val title = if (at > 0) note.substring(0, at) else note
    val body = (if (at > 0) note.substring(at + 2) else "").let { if (Regex("^[a-z]+(\\s|$)").containsMatchIn(it)) it.replaceFirstChar { c -> c.uppercaseChar() } else it }
    AlertDialog(onDismissRequest = onDismiss,
        confirmButton = { Btn("OK", Kind.Text, onClick = onDismiss) },
        title = { Text(title, style = MaterialTheme.typography.titleMedium, color = Fg) },
        text = if (body.isNotEmpty()) ({ Text(body, style = MaterialTheme.typography.bodyMedium, color = Muted) }) else null,
        shape = RoundedCornerShape(12.dp), containerColor = Bg)
}
