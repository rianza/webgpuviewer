package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.exponentialDecay
import androidx.compose.animation.core.spring
import androidx.compose.foundation.AndroidEmbeddedExternalSurface
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.util.fastCoerceIn
import androidx.compose.ui.util.fastForEach
import ca.mpreg.webgpuviewer.NormalMotionDurationScale
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.waitForCleanUp
import ca.mpreg.webgpuviewer.waitForDown
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.pow
import kotlin.time.Duration.Companion.milliseconds

@Composable
fun ImageViewer(
    modifier: Modifier = Modifier,
    state: ImageViewerState,
) {
    val scope = rememberCoroutineScope()

    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    val view = LocalView.current
    val density = LocalDensity.current
    // Get cutout top directly in px
    val cutoutPx = WindowInsets.displayCutout.getTop(density).let { px ->
        if (px == 0) {
            val context = LocalContext.current
            val resourceId =
                context.resources.getIdentifier("status_bar_height", "dimen", "android")
            context.resources.getDimensionPixelSize(resourceId).toFloat()
        } else {
            px.toFloat()
        }
    }

    LaunchedEffect(state.avoidCutout, cutoutPx) {
        state.cutoutTopPx = if (state.avoidCutout) cutoutPx else 0f
    }

    // AndroidExternalSurface places the SurfaceView on a window layer behind the activity's
    // window (AndroidExternalSurfaceZOrder.Behind) and relies on the HWUI hole-punch for
    // visibility. On some Android 11 OEM builds that punch silently fails - every frame
    // renders into the surface with no errors, but the screen stays black and the content
    // only flashes when the reader closes (mihonapp/mihon#3773). The embedded variant
    // composites the surface like any regular view and has the same callback contract.
    AndroidEmbeddedExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    val wasScrolling = state.pageOffset != 0f
                    val pageTurnJob = state.animationJob
                    val page = state.getPage(0) ?: return@awaitEachGesture
                    page.animationJob?.cancel()

                    view.parent?.requestDisallowInterceptTouchEvent(true)

                    var longPressed = false

                    val edgeThreshold = 50f
                    val nearEdge =
                        firstDown.position.x < edgeThreshold || firstDown.position.x > state.width - edgeThreshold
                    val longPressJob = if (!nearEdge) {
                        scope.launch {
                            delay(viewConfiguration.longPressTimeoutMillis.milliseconds)
                            longPressed = true
                            state.onLongTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )
                        }
                    } else null

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        longPressJob?.cancel()
                        val secondDown = waitForDown(doubleTapTimeout)
                        if (secondDown == null) {
                            pageTurnJob?.cancel()
                            if (state.pageOffset != 0f) {
                                state.animationJob = scope.launch {
                                    Animatable(state.pageOffset).animateTo(
                                        0f, animationSpec = spring(
                                            stiffness = Spring.StiffnessMediumLow,
                                            visibilityThreshold = 0.002f
                                        )
                                    ) {
                                        state.pageOffset = value
                                        state.invalidate()
                                    }
                                }
                            }
                            page.animateTo(Offset(0.5f, 0.5f))
                            state.onTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )
                            return@awaitEachGesture
                        }

                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // double tap — let any in-progress page turn finish committing first
                            val tapX = secondDown.position.x / state.width
                            val tapY = secondDown.position.y / state.height
                            scope.launch {
                                pageTurnJob?.join()
                                val zoomPage = state.getPage(0) ?: return@launch
                                if (zoomPage.atHomeScale) {
                                    zoomPage.animateTo(
                                        Offset(tapX, tapY),
                                        targetScale = zoomPage.doubleTapScale
                                    )
                                } else {
                                    zoomPage.animateTo(
                                        Offset(tapX, tapY),
                                        targetScale = zoomPage.homeScale
                                    )
                                }
                            }
                        } else {
                            // double tap drag
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)

                            val dragPointerId = secondDown.id

                            val originalScale = page.scale
                            val originalX = page.x
                            val originalY = page.y
                            var totalDeltaY = 0f

                            state.animationJob?.cancel()

                            page.isScaleAnimating = true
                            var willFlingZoom = false
                            try {
                                while (true) {
                                    val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                    val change =
                                        event.changes.firstOrNull { it.id == dragPointerId }

                                    if (change == null || change.changedToUp() || change.isConsumed) {
                                        break
                                    }

                                    velocityTracker.addPointerInputChange(change)

                                    if (change.positionChanged()) {
                                        val pan = event.calculatePan()
                                        totalDeltaY += pan.y
                                        if (totalDeltaY != 0f) {

                                            val px = secondDown.position.x / state.width - 0.5f
                                            val py = secondDown.position.y / state.height - 0.5f

                                            val newScale =
                                                originalScale * 10f.pow(2 * totalDeltaY / state.height)

                                            page.scale = newScale
                                            val diff = 1 / page.scale - 1 / originalScale

                                            page.setPos(
                                                (originalX + px * diff).orZero(),
                                                (originalY + py * diff).orZero()
                                            )

                                            change.consume()
                                        }
                                    }
                                }
                                val dragVelocity = velocityTracker.calculateVelocity()
                                // Decided before the finally below, so isScaleAnimating has no gap
                                // between this drag ending and its fling starting.
                                willFlingZoom = abs(dragVelocity.y) > 200 &&
                                        page.scale > page.homeScale && page.scale < page.maxScale
                            } finally {
                                if (!willFlingZoom) page.isScaleAnimating = false
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (willFlingZoom) {
                                // fling zoom
                                page.animationJob = scope.launch(NormalMotionDurationScale) {
                                    try {
                                        Animatable(0f).animateDecay(
                                            velocity.y,
                                            exponentialDecay()
                                        ) {
                                            val px = secondDown.position.x / state.width - 0.5f
                                            val py = secondDown.position.y / state.height - 0.5f

                                            val newScale =
                                                originalScale * 10f.pow(2 * (totalDeltaY + value) / state.height)

                                            page.scale =
                                                newScale.fastCoerceIn(page.homeScale, page.maxScale)
                                            val diff = 1 / page.scale - 1 / originalScale

                                            val x = (originalX + px * diff).orZero()
                                            val y = (originalY + py * diff).orZero()

                                            val maxX = page.maxX(page.scale)
                                            val minY = page.minY(page.scale)
                                            val maxY = page.maxY(page.scale)

                                            page.setPos(
                                                x.fastCoerceIn(-maxX, maxX),
                                                y.fastCoerceIn(minY, maxY)
                                            )
                                        }
                                    } finally {
                                        page.isScaleAnimating = false
                                    }
                                }
                            } else {
                                page.animateTo(
                                    Offset(
                                        secondDown.position.x / state.width,
                                        secondDown.position.y / state.height
                                    )
                                )
                            }
                        }
                    } else {
                        if (!wasScrolling) page.animateTo(Offset(0.5f, 0.5f))
                        pageTurnJob?.cancel()

                        var lastMoveTime = firstDown.uptimeMillis
                        var lastEventTime: Long = firstDown.uptimeMillis
                        var acc = Offset.Zero

                        var scaleOrigin = Offset(0.5f, 0.5f)

                        var single = true
                        var pageTurning = wasScrolling

                        // If grabbing mid-animation, update firstPos so panning continues smoothly
                        if (wasScrolling) {
                            state.firstPos = firstDown.position
                        }

                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

                        page.animationJob?.cancel()

                        var canceled = false
                        try {
                            do {
                                val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                canceled = event.changes.any { it.isConsumed }
                                if (canceled) {
                                    longPressJob?.cancel()
                                } else {
                                    val change = event.changes[0]
                                    lastEventTime = change.uptimeMillis

                                    if (change.positionChanged()) {
                                        lastMoveTime = change.uptimeMillis
                                    }

                                    val centroid = event.calculateCentroid(useCurrent = true)

                                    val twoFingers =
                                        event.changes.size > 1 && event.changes.all { it.pressed }
                                    page.isScaleAnimating = twoFingers

                                    var pointerCountChanged = false
                                    if (twoFingers) {
                                        if (single && !pageTurning) {
                                            longPressJob?.cancel()
                                            velocityTracker.resetTracking()
                                            acc = Offset.Zero
                                            pointerCountChanged = true
                                        }
                                        if (!pageTurning) {
                                            velocityTracker.addPointerInputChange(change)
                                            single = false
                                            scaleOrigin = Offset(
                                                centroid.x / state.width, centroid.y / state.height
                                            )
                                        }
                                    } else if (single) {
                                        velocityTracker.addPointerInputChange(change)
                                    }

                                    if (pointerCountChanged) continue

                                    val pan = event.calculatePan()
                                    if (acc.getDistance() > touchSlop) longPressJob?.cancel()
                                    acc += pan

                                    if (pageTurning) {
                                        val prev = state.pageOffset
                                        // Always the raw drag direction - pageOffset's sign is
                                        // also the live visual pan amount (see renderSnapshot's use
                                        // of it as a transition's frac), so it has to keep tracking
                                        // the finger 1:1 regardless of reading direction. isReversed
                                        // is resolved separately, in getPage/onPageChange, where it
                                        // only affects which page a crossing reveals - not how far
                                        // the finger has to move to cause one.
                                        val panAmount =
                                            if (state.isVertical) -pan.y / state.height else -pan.x / state.width
                                        state.pageOffset += panAmount
                                        if ((prev > 0f && state.pageOffset <= 0f) || (prev < 0f && state.pageOffset >= 0f)) {
                                            state.pageOffset = 0f
                                            pageTurning = false
                                            acc = Offset.Zero
                                        }
                                        state.currentPos = event.changes[0].position
                                        state.invalidate()
                                        event.changes.fastForEach { if (it.positionChanged()) it.consume() }
                                    } else {
                                        val zoom = event.calculateZoom()

                                        if (zoom != 1f || pan != Offset.Zero) {
                                            val newScale = page.scale * zoom
                                            val diff = 1 / newScale - 1 / page.scale

                                            var x = page.x + (pan.x / state.width) / page.scale
                                            var y = page.y + (pan.y / state.height) / page.scale

                                            x += (centroid.x / state.width - 0.5f) * diff
                                            y += (centroid.y / state.height - 0.5f) * diff

                                            val maxX = page.maxX(newScale)
                                            val minY = page.minY(newScale)
                                            val maxY = page.maxY(newScale)

                                            if (single) {
                                                val clampedX = x.fastCoerceIn(-maxX, maxX)
                                                val clampedY = y.fastCoerceIn(minY, maxY)
                                                val overflow = if (state.isVertical) {
                                                    y - clampedY
                                                } else {
                                                    x - clampedX
                                                }
                                                val isBiased = if (state.isVertical) {
                                                    abs(acc.y) > abs(acc.x)
                                                } else {
                                                    abs(acc.x) > abs(acc.y)
                                                }
                                                if (overflow != 0f && isBiased) {
                                                    page.animateTo(Offset(0.5f, 0.5f))
                                                    pageTurning = true
                                                    state.firstPos = firstDown.position
                                                    state.pageOffset += -overflow * page.scale
                                                    state.invalidate()
                                                }
                                                x = clampedX
                                                y = clampedY
                                            }

                                            page.scale = newScale
                                            page.setPos(x.orZero(), y.orZero())

                                            event.changes.fastForEach { if (it.positionChanged()) it.consume() }
                                        }
                                    }
                                }
                            } while (!canceled && event.changes.any { it.pressed })
                        } finally {
                            page.isScaleAnimating = false
                        }

                        longPressJob?.cancel()
                        if (longPressed || canceled) return@awaitEachGesture

                        if (pageTurning) {
                            val velocity = velocityTracker.calculateVelocity()
                            // Always raw, matching panAmount above - see that comment.
                            val initialVelocity =
                                if (state.isVertical) -velocity.y / state.height else -velocity.x / state.width

                            // Flicking opposite to current direction = go back to 0
                            val flickingOpposite =
                                (state.pageOffset > 0f && initialVelocity < -0.5f) || (state.pageOffset < 0f && initialVelocity > 0.5f)

                            val target = when {
                                flickingOpposite -> 0f
                                initialVelocity > 1f && state.haveNext -> 1f
                                initialVelocity < -1f && state.havePrev -> -1f
                                state.pageOffset > 0.5f && state.haveNext -> 1f
                                state.pageOffset < -0.5f && state.havePrev -> -1f
                                else -> 0f
                            }

                            state.animationJob = scope.launch {
                                val anim = Animatable(state.pageOffset)
                                anim.updateBounds(lowerBound = -1f, upperBound = 1f)
                                anim.animateTo(
                                    target,
                                    initialVelocity = initialVelocity,
                                    animationSpec = spring(
                                        stiffness = Spring.StiffnessMediumLow,
                                        visibilityThreshold = 0.002f
                                    )
                                ) {
                                    state.pageOffset = value
                                    state.invalidate()
                                }
                            }
                        } else {
                            val maxX = page.maxX(page.scale)
                            val minY = page.minY(page.scale)
                            val maxY = page.maxY(page.scale)

                            val velocity = velocityTracker.calculateVelocity()
                            if ((page.scale >= page.homeScale) && (page.scale <= page.maxScale) && (lastEventTime - lastMoveTime) < 100 && (abs(
                                    velocity.x
                                ) > 400 || abs(velocity.y) > 400) && (page.x.fastCoerceIn(
                                    -maxX, maxX
                                ) == page.x || page.y.fastCoerceIn(minY, maxY) == page.y)
                            ) {
                                // fling pan
                                page.animationJob = scope.launch(NormalMotionDurationScale) {
                                    fling.snapTo(Offset.Zero)
                                    var lastOffset = Offset.Zero
                                    fling.animateDecay(
                                        Offset(velocity.x, velocity.y), exponentialDecay()
                                    ) {
                                        val delta = value - lastOffset
                                        lastOffset = value
                                        val dx = (delta.x / state.width) / page.scale
                                        val dy = (delta.y / state.height) / page.scale
                                        page.setPos(
                                            (page.x + dx).fastCoerceIn(-maxX, maxX).orZero(),
                                            (page.y + dy).fastCoerceIn(minY, maxY).orZero()
                                        )
                                    }
                                }
                            } else {
                                page.animateTo(scaleOrigin)
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
