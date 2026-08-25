package ca.mpreg.webgpuviewer

import ca.mpreg.webgpuviewer.log.WgvLog
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.ui.MotionDurationScale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventTimeoutCancellationException
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.changedToUp

internal object NormalMotionDurationScale : MotionDurationScale {
    override val scaleFactor: Float = 1f
}

suspend fun AwaitPointerEventScope.waitForCleanUp(
    pointerId: PointerId, timeout: Long, touchSlop: Float
): PointerEvent? = try {
    withTimeout(timeout) {
        var acc = Offset.Zero

        while (true) {
            val event = awaitPointerEvent()

            if (event.changes.any { it.isConsumed }) {
                WgvLog.d(TAG, "waitForCleanUp: event consumed elsewhere - gesture may start")
                return@withTimeout null
            }

            val change = event.changes.firstOrNull { it.id == pointerId } ?: return@withTimeout null

            if (event.changes.any { it.id != pointerId && it.pressed }) {
                WgvLog.d(TAG, "waitForCleanUp: second finger down - not a clean up")
                return@withTimeout null
            }

            acc += event.calculatePan()
            if (acc.getDistance() > touchSlop) {
                WgvLog.d(TAG, "waitForCleanUp: exceeded touchSlop ($touchSlop) - drag")
                return@withTimeout null
            }
            if (change.changedToUp()) {
                return@withTimeout event
            }
        }
    }
} catch (e: PointerEventTimeoutCancellationException) {
    WgvLog.v(TAG, "waitForCleanUp: timed out after ${timeout}ms")
    null
} as PointerEvent?

suspend fun AwaitPointerEventScope.waitForDown(timeout: Long) = try {
    withTimeout(timeout) {
        var down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        while (down == null) {
            down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        }
        down
    }
} catch (e: PointerEventTimeoutCancellationException) {
    WgvLog.v(TAG, "waitForDown: timed out after ${timeout}ms")
    null
}

private const val TAG = "WGV.Gesture"

fun Float.orZero(): Float = if (this.isNaN()) 0f else this
