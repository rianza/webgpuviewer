package ca.mpreg.webgpuviewer.log

import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Central logging facade for the WebGPUViewer library.
 *
 * Every log line the library emits goes through here with a stable `WGV.<module>` tag
 * ("WGV.Render", "WGV.Page", "WGV.Tiles", ...), so the whole library can be filtered with a
 * single `adb logcat` command - see LOGGING.md at the repository root.
 *
 * Levels map 1:1 to android.util.Log. [minLevel] defaults to [Log.VERBOSE] so the full
 * instrumentation is visible out of the box; quiet deployments can raise [minLevel] or flip
 * [enabled] off entirely (the hot path pays a single volatile read in that case).
 */
object WgvLog {

    /** Tag prefix shared by every module tag ("WGV.Render", "WGV.Page", ...). */
    const val PREFIX = "WGV"

    /** Master switch. When false nothing is emitted at all. */
    @Volatile
    var enabled: Boolean = true

    /** Lowest level that gets emitted; uses the [Log.VERBOSE]..[Log.ASSERT] constants. */
    @Volatile
    var minLevel: Int = Log.VERBOSE

    fun v(tag: String, msg: String) = log(Log.VERBOSE, tag, msg, null)

    fun d(tag: String, msg: String) = log(Log.DEBUG, tag, msg, null)

    fun i(tag: String, msg: String) = log(Log.INFO, tag, msg, null)

    fun w(tag: String, msg: String, tr: Throwable? = null) = log(Log.WARN, tag, msg, tr)

    fun e(tag: String, msg: String, tr: Throwable? = null) = log(Log.ERROR, tag, msg, tr)

    /**
     * Rate-limited debug log for per-frame / per-gesture hot paths: at most one line per
     * [intervalMs] for each (tag, key) pair. [msg] is only evaluated when a line is actually
     * emitted, so a throttled hot path pays no string formatting at all.
     */
    fun throttled(
        tag: String,
        key: String = "default",
        intervalMs: Long = 250L,
        msg: () -> String,
    ) {
        if (!enabled || minLevel > Log.DEBUG) return
        val now = System.nanoTime()
        val slot = slots.computeIfAbsent("$tag#$key") { Slot() }
        synchronized(slot) {
            if (now - slot.lastNanos < intervalMs * 1_000_000L) return
            slot.lastNanos = now
        }
        d(tag, msg())
    }

    private class Slot {
        var lastNanos = 0L
    }

    private val slots = ConcurrentHashMap<String, Slot>()

    private fun log(level: Int, tag: String, msg: String, tr: Throwable?) {
        if (!enabled || level < minLevel) return
        val full = if (tr != null) "$msg\n${Log.getStackTraceString(tr)}" else msg
        Log.println(level, tag, full)
    }
}
