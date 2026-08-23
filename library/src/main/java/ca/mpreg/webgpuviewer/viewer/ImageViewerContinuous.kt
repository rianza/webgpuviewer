package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.exponentialDecay
import androidx.compose.animation.core.spring
import androidx.compose.animation.splineBasedDecay
import androidx.compose.foundation.AndroidEmbeddedExternalSurface
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.input.pointer.util.addPointerInputChange
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.util.fastCoerceIn
import ca.mpreg.webgpuviewer.NormalMotionDurationScale
import ca.mpreg.webgpuviewer.waitForCleanUp
import ca.mpreg.webgpuviewer.waitForDown
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.time.Duration.Companion.milliseconds

@Composable
fun ImageViewerContinuous(
    modifier: Modifier = Modifier,
    state: ImageViewerContinuousState,
) {
    val scope = rememberCoroutineScope()
    val view = LocalView.current
    val density = LocalDensity.current
    val flingX = remember { Animatable(0f) }
    val decay = remember(density) { splineBasedDecay<Float>(density) }

    val minScale = 1f
    val doubleTapScale = minScale * 2f
    val maxScale = max(doubleTapScale * 2f, 4f)

    // See ImageViewer.kt: AndroidEmbeddedExternalSurface over AndroidExternalSurface.
    AndroidEmbeddedExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    state.animationJob?.cancel()
                    view.parent?.requestDisallowInterceptTouchEvent(true)

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        // Tap - wait for possible double tap
                        val secondDown = waitForDown(doubleTapTimeout)
                        if (secondDown == null) {
                            // Single tap
                            state.onTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )

                            if (state.scale < minScale) {
                                state.animationJob = scope.launch {
                                    state.isScaleAnimating = true
                                    try {
                                        val startScale = state.scale
                                        val startOffsetX = state.offsetX
                                        animate(
                                            0f, 1f, animationSpec = spring(
                                                stiffness = Spring.StiffnessMediumLow,
                                                visibilityThreshold = 0.002f
                                            )
                                        ) { t, _ ->
                                            state.scale = startScale + (minScale - startScale) * t
                                            state.offsetX = startOffsetX * (1f - t)
                                            state.invalidate()
                                        }
                                    } finally {
                                        state.isScaleAnimating = false
                                    }
                                }
                            } else if (state.scale > maxScale) {
                                state.animationJob = scope.launch {
                                    state.isScaleAnimating = true
                                    try {
                                        val startScale = state.scale
                                        val startOffsetX = state.offsetX
                                        val targetMaxOffsetX =
                                            max(0f, (maxScale - 1f) / (2f * maxScale))
                                        val targetOffsetX = startOffsetX.fastCoerceIn(
                                            -targetMaxOffsetX, targetMaxOffsetX
                                        )
                                        animate(
                                            0f, 1f, animationSpec = spring(
                                                stiffness = Spring.StiffnessMediumLow,
                                                visibilityThreshold = 0.002f
                                            )
                                        ) { t, _ ->
                                            state.scale = startScale + (maxScale - startScale) * t
                                            state.offsetX =
                                                startOffsetX + (targetOffsetX - startOffsetX) * t
                                            state.invalidate()
                                        }
                                    } finally {
                                        state.isScaleAnimating = false
                                    }
                                }
                            }
                            return@awaitEachGesture
                        }

                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // Double tap: toggle zoom
                            if (state.scale > minScale + 0.1f) {
                                // Zoom out: animate offsetX to 0, anchor Y to tap point
                                val py = secondDown.position.y / state.height - 0.5f
                                state.animationJob = scope.launch {
                                    state.isScaleAnimating = true
                                    try {
                                        val startScale = state.scale
                                        val startOffsetX = state.offsetX
                                        val totalDiff = 1f / minScale - 1f / startScale
                                        val px =
                                            if (totalDiff != 0f) -startOffsetX / totalDiff else 0f
                                        animate(
                                            0f, 1f, animationSpec = spring(
                                                stiffness = Spring.StiffnessMediumLow,
                                                visibilityThreshold = 0.002f
                                            )
                                        ) { t, _ ->
                                            val newScale = startScale + (minScale - startScale) * t
                                            val diff = 1f / newScale - 1f / state.scale
                                            state.offsetX += px * diff
                                            state.scrollY -= py * diff * state.height
                                            state.scale = newScale
                                            state.invalidate()
                                        }
                                    } finally {
                                        state.isScaleAnimating = false
                                    }
                                }
                            } else {
                                // Zoom in at tap point
                                val px = secondDown.position.x / state.width - 0.5f
                                val py = secondDown.position.y / state.height - 0.5f
                                state.animationJob = scope.launch {
                                    state.isScaleAnimating = true
                                    try {
                                        val startScale = state.scale
                                        val startOffsetX = state.offsetX
                                        val startScrollY = state.scrollY
                                        animate(
                                            0f, 1f, animationSpec = spring(
                                                stiffness = Spring.StiffnessMediumLow,
                                                visibilityThreshold = 0.002f
                                            )
                                        ) { t, _ ->
                                            val newScale =
                                                startScale + (doubleTapScale - startScale) * t
                                            val diff = 1f / newScale - 1f / state.scale
                                            state.offsetX += px * diff
                                            state.scrollY -= py * diff * state.height
                                            state.scale = newScale
                                            state.invalidate()
                                        }
                                    } finally {
                                        state.isScaleAnimating = false
                                    }
                                }
                            }
                        } else {
                            // Double tap drag: zoom by dragging
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)
                            val dragPointerId = secondDown.id
                            val originalScale = state.scale
                            val originalOffsetX = state.offsetX
                            val originalScrollY = state.scrollY
                            val px = secondDown.position.x / state.width - 0.5f
                            val py = secondDown.position.y / state.height - 0.5f
                            var totalDeltaY = 0f

                            state.isScaleAnimating = true
                            var willFlingZoom = false
                            try {
                                while (true) {
                                    val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                    val change =
                                        event.changes.firstOrNull { it.id == dragPointerId }
                                    if (change == null || change.changedToUp() || change.isConsumed) break

                                    velocityTracker.addPointerInputChange(change)

                                    if (change.positionChanged()) {
                                        val pan = event.calculatePan()
                                        totalDeltaY += pan.y

                                        if (totalDeltaY != 0f) {
                                            val newScale =
                                                originalScale * 10f.pow(2 * totalDeltaY / state.height)
                                            val diff = 1f / newScale - 1f / originalScale
                                            state.scale = newScale
                                            state.offsetX = originalOffsetX + px * diff
                                            state.scrollY =
                                                originalScrollY - py * diff * state.height
                                            state.invalidate()
                                            change.consume()
                                        }
                                    }
                                }
                                val dragVelocity = velocityTracker.calculateVelocity()
                                // Decided before the finally below, so isScaleAnimating has no
                                // gap between this drag ending and its fling starting.
                                willFlingZoom =
                                    abs(dragVelocity.y) > 200 && state.scale in minScale..maxScale
                            } finally {
                                if (!willFlingZoom) state.isScaleAnimating = false
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (willFlingZoom) {
                                // Fling zoom
                                state.animationJob = scope.launch(NormalMotionDurationScale) {
                                    try {
                                        Animatable(0f).animateDecay(
                                            velocity.y, exponentialDecay(frictionMultiplier = 0.5f)
                                        ) {
                                            val newScale =
                                                (originalScale * 10f.pow(2 * (totalDeltaY + value) / state.height)).fastCoerceIn(
                                                    minScale, maxScale
                                                )
                                            val diff = 1f / newScale - 1f / originalScale
                                            val maxOffsetX =
                                                max(0f, (newScale - 1f) / (2f * newScale))
                                            state.scale = newScale
                                            state.offsetX =
                                                (originalOffsetX + px * diff).fastCoerceIn(
                                                    -maxOffsetX, maxOffsetX
                                                )
                                            state.scrollY =
                                                originalScrollY - py * diff * state.height
                                            state.invalidate()
                                        }
                                    } finally {
                                        state.isScaleAnimating = false
                                    }
                                }
                            } else {
                                // Snap scale and offsetX back if overshot
                                val targetScale = state.scale.fastCoerceIn(minScale, maxScale)
                                val targetMaxOffsetX =
                                    max(0f, (targetScale - 1f) / (2f * targetScale))
                                val targetOffsetX =
                                    state.offsetX.fastCoerceIn(-targetMaxOffsetX, targetMaxOffsetX)
                                if (targetScale != state.scale || targetOffsetX != state.offsetX) {
                                    state.animationJob = scope.launch {
                                        state.isScaleAnimating = true
                                        try {
                                            val startScale = state.scale
                                            val startOffsetX = state.offsetX
                                            animate(
                                                0f, 1f, animationSpec = spring(
                                                    stiffness = Spring.StiffnessMediumLow,
                                                    visibilityThreshold = 0.002f
                                                )
                                            ) { t, _ ->
                                                state.scale =
                                                    startScale + (targetScale - startScale) * t
                                                state.offsetX =
                                                    startOffsetX + (targetOffsetX - startOffsetX) * t
                                                state.invalidate()
                                            }
                                        } finally {
                                            state.isScaleAnimating = false
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Drag gesture
                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

                        var single = true
                        var lastZoomTime = firstDown.uptimeMillis
                        var zoomVelocity = 0f
                        var lastCentroid = Offset(0.5f, 0.5f)

                        var longPressed = false
                        val longPressJob = scope.launch {
                            delay(viewConfiguration.longPressTimeoutMillis.milliseconds)
                            longPressed = true
                            state.onLongTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )
                        }

                        var willFlingZoom = false
                        try {
                            do {
                                val event = awaitPointerEvent()
                                val canceled = event.changes.any { it.isConsumed }
                                if (!canceled) {
                                    val change = event.changes[0]

                                    if (event.changes.size > 1 && event.changes.all { it.pressed }) {
                                        if (single) {
                                            longPressJob.cancel()
                                            velocityTracker.resetTracking()
                                        }
                                        single = false
                                    }

                                    velocityTracker.addPointerInputChange(change)

                                    val pan = event.calculatePan()
                                    val zoom = event.calculateZoom()
                                    state.isScaleAnimating = zoom != 1f

                                    if (pan != Offset.Zero || zoom != 1f) {
                                        longPressJob.cancel()

                                        if (zoom != 1f) {
                                            velocityTracker.resetTracking()
                                            val centroid =
                                                event.calculateCentroid(useCurrent = true)
                                            lastCentroid = Offset(
                                                centroid.x / state.width, centroid.y / state.height
                                            )
                                            val newScale = state.scale * zoom
                                            val diff = 1f / newScale - 1f / state.scale
                                            val cx = lastCentroid.x - 0.5f
                                            val cy = lastCentroid.y - 0.5f
                                            state.offsetX += cx * diff
                                            state.scrollBy(-cy * diff * state.height)
                                            state.scale = newScale

                                            // Track zoom velocity in log-scale space
                                            val now = change.uptimeMillis
                                            val dt = (now - lastZoomTime).coerceAtLeast(1L)
                                            val logZoom = ln(zoom) / (dt / 1000f)
                                            zoomVelocity = zoomVelocity * 0.5f + logZoom * 0.5f
                                            lastZoomTime = now
                                        } else {
                                            zoomVelocity *= 0.8f
                                        }

                                        if (single) {
                                            val maxOffsetX =
                                                max(0f, (state.scale - 1f) / (2f * state.scale))
                                            state.offsetX =
                                                (state.offsetX + pan.x / state.width / state.scale).fastCoerceIn(
                                                    -maxOffsetX, maxOffsetX
                                                )
                                        } else {
                                            state.offsetX += pan.x / state.width / state.scale
                                        }
                                        state.scrollBy(-pan.y / state.scale)
                                        state.invalidate()
                                        event.changes.forEach { if (it.positionChanged()) it.consume() }
                                    }
                                }
                                // Decided before the finally below, so isScaleAnimating has no gap
                                // between this pinch ending and its fling starting. Forced true
                                // rather than left as the last iteration's own instantaneous zoom
                                // left it, which could already be false during a quiet panning tail.
                                willFlingZoom =
                                    !longPressed && !single && abs(zoomVelocity) > 0.5f &&
                                            state.scale in minScale..maxScale
                                if (willFlingZoom) state.isScaleAnimating = true
                            } while (!canceled && event.changes.any { it.pressed })
                        } finally {
                            if (!willFlingZoom) state.isScaleAnimating = false
                        }

                        longPressJob.cancel()
                        if (longPressed) return@awaitEachGesture

                        if (willFlingZoom) {
                            // Fling zoom
                            val cx = lastCentroid.x - 0.5f
                            val cy = lastCentroid.y - 0.5f
                            val startScale = state.scale
                            val startOffsetX = state.offsetX
                            val startScrollY = state.scrollY
                            state.animationJob = scope.launch(NormalMotionDurationScale) {
                                try {
                                    Animatable(0f).animateDecay(
                                        zoomVelocity, exponentialDecay(frictionMultiplier = 0.5f)
                                    ) {
                                        val newScale =
                                            (startScale * exp(value)).fastCoerceIn(
                                                minScale,
                                                maxScale
                                            )
                                        val diff = 1f / newScale - 1f / startScale
                                        val maxOffsetX = max(0f, (newScale - 1f) / (2f * newScale))
                                        state.scale = newScale
                                        state.offsetX = (startOffsetX + cx * diff).fastCoerceIn(
                                            -maxOffsetX, maxOffsetX
                                        )
                                        state.scrollY = startScrollY - cy * diff * state.height
                                        state.invalidate()
                                    }
                                } finally {
                                    state.isScaleAnimating = false
                                }
                            }
                        } else if (state.scale < minScale) {
                            // Snap scale back up
                            state.animationJob = scope.launch {
                                state.isScaleAnimating = true
                                try {
                                    val startScale = state.scale
                                    val startOffsetX = state.offsetX
                                    animate(
                                        0f, 1f, animationSpec = spring(
                                            stiffness = Spring.StiffnessMediumLow,
                                            visibilityThreshold = 0.002f
                                        )
                                    ) { t, _ ->
                                        state.scale = startScale + (minScale - startScale) * t
                                        state.offsetX = startOffsetX * (1f - t)
                                        state.invalidate()
                                    }
                                } finally {
                                    state.isScaleAnimating = false
                                }
                            }
                        } else if (state.scale > maxScale) {
                            // Snap scale back down
                            state.animationJob = scope.launch {
                                state.isScaleAnimating = true
                                try {
                                    val startScale = state.scale
                                    val startOffsetX = state.offsetX
                                    val targetMaxOffsetX =
                                        max(0f, (maxScale - 1f) / (2f * maxScale))
                                    val targetOffsetX =
                                        startOffsetX.fastCoerceIn(
                                            -targetMaxOffsetX,
                                            targetMaxOffsetX
                                        )
                                    animate(
                                        0f, 1f, animationSpec = spring(
                                            stiffness = Spring.StiffnessMediumLow,
                                            visibilityThreshold = 0.002f
                                        )
                                    ) { t, _ ->
                                        state.scale = startScale + (maxScale - startScale) * t
                                        state.offsetX =
                                            startOffsetX + (targetOffsetX - startOffsetX) * t
                                        state.invalidate()
                                    }
                                } finally {
                                    state.isScaleAnimating = false
                                }
                            }
                        } else {
                            // Scale in bounds: fling pan or snap offsetX
                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 400 || abs(velocity.x) > 400) {
                                state.animationJob = scope.launch(NormalMotionDurationScale) {
                                    state.isFlinging = true
                                    try {
                                        val speed = hypot(velocity.x, velocity.y)
                                        val dirX = velocity.x / speed
                                        val dirY = velocity.y / speed
                                        flingX.snapTo(0f)
                                        var last = 0f
                                        flingX.animateDecay(speed, decay) {
                                            val delta = value - last
                                            last = value
                                            val maxOffsetX =
                                                max(0f, (state.scale - 1f) / (2f * state.scale))
                                            state.offsetX =
                                                (state.offsetX + dirX * delta / state.width / state.scale).fastCoerceIn(
                                                    -maxOffsetX, maxOffsetX
                                                )
                                            state.scrollBy(-dirY * delta / state.scale)
                                            state.invalidate()
                                        }
                                    } finally {
                                        // A final invalidate so generation resumes promptly rather
                                        // than waiting on whatever gesture happens to invalidate next.
                                        state.isFlinging = false
                                        state.invalidate()
                                    }
                                }
                            } else {
                                val maxOffsetX = max(0f, (state.scale - 1f) / (2f * state.scale))
                                val clampedX = state.offsetX.fastCoerceIn(-maxOffsetX, maxOffsetX)
                                if (clampedX != state.offsetX) {
                                    state.animationJob = scope.launch {
                                        val startX = state.offsetX
                                        animate(
                                            0f, 1f, animationSpec = spring(
                                                stiffness = Spring.StiffnessMediumLow,
                                                visibilityThreshold = 0.002f
                                            )
                                        ) { t, _ ->
                                            state.offsetX = startX + (clampedX - startX) * t
                                            state.invalidate()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }, isOpaque = false
    ) {
        onSurface { surface, width, height ->
            try {
                state.init(scope, surface, width, height)
                state.invalidate()
                state.collect()
            } finally {
                state.cleanup()
            }
        }
    }
}
