package dev.lizard.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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
