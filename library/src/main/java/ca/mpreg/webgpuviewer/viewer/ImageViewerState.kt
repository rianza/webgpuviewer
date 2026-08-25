package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.view.Surface
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.Density
import androidx.compose.ui.util.fastCoerceAtLeast
import androidx.compose.ui.util.fastCoerceAtMost
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDepthStencilAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.dispatcher
import ca.mpreg.webgpuviewer.transition.Transition
import ca.mpreg.webgpuviewer.transition.TransitionBasic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "WGV.Viewer"

open class ImageViewerState(var isVertical: Boolean = false, var isReversed: Boolean = false) {
    val renderer = WebGpuRenderer()

    internal val tiles = TileRenderer { invalidate() }

    var animationJob: Job? = null

    val width get() = renderer.width
    val height get() = renderer.height

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f
    var density: Density =
        Density(density = Resources.getSystem().displayMetrics.density, fontScale = 1f)

    var scope: CoroutineScope? = null

    /** Top padding in pixels to avoid display cutout. Set automatically by ImageViewer when avoidCutout is true. */
    var cutoutTopPx: Float = 0f
        set(value) {
            if (field != value) {
                WgvLog.d(TAG, "cutoutTopPx: $field -> $value (re-homing current page)")
                field = value
                // Move current page to new home position when cutout changes
                getPage(0)?.home()
            }
        }

    /** When true, images will be positioned/scaled to avoid the display cutout. */
    var avoidCutout: Boolean by mutableStateOf(false)

    /** When true, always shift images below cutout. When false, only shift if image would overlap cutout. */
    var alwaysAvoidCutout: Boolean by mutableStateOf(false)

    private var suppressPageChange = false

    var pageOffset = 0f
        set(value) {
            var v = value
            var pageDelta = 0

            if (!suppressPageChange) {
                while (v >= 1f && haveNext) {
                    pageDelta += 1
                    v -= 1f
                }
                while (v <= -1f && havePrev) {
                    pageDelta -= 1
                    v += 1f
                }
            }

            if (!haveNext) v = v.fastCoerceAtMost(1f)
            if (!havePrev) v = v.fastCoerceAtLeast(-1f)

            val settling = field != 0f && v == 0f

            field = v

            if (pageDelta != 0) {
                WgvLog.i(TAG, "Page turn: delta=$pageDelta (pageOffset=$value, reversed=$isReversed)")
                onPageChange?.invoke(if (isReversed) -pageDelta else pageDelta)
            }

            // Rotate rather than invalidate: onPageChange has already updated whatever backs
            // getPage, so slot 2 often already holds a valid render of this new current page.
            if (settling) {
                val current = getPage(0)
                if (current != null) Transition.rotateCacheOnPageChange(current) else Transition.invalidateCache()
            }
        }

    private fun setPageOffsetDirect(value: Float) {
        suppressPageChange = true
        pageOffset = value
        suppressPageChange = false
    }

    fun animatePageTurn(direction: Int) {
        WgvLog.d(TAG, "animatePageTurn(direction=$direction)")
        animationJob?.cancel()
        animationJob = scope?.launch {
            setPageOffsetDirect(direction.toFloat())
            invalidate()
            try {
                Animatable(direction.toFloat()).animateTo(
                    0f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
                ) {
                    setPageOffsetDirect(value)
                    invalidate()
                }
            } finally {
                // Always clear transitionFromPage - if cancelled, getPage will provide the right page
                transitionFromPage = null
            }
            // Only snap to 0 when animation completes normally
            setPageOffsetDirect(0f)
            invalidate()
        }
    }

    val havePrev get() = getPage(if (isReversed) 1 else -1) != null
    val haveNext get() = getPage(if (isReversed) -1 else 1) != null

    var fetchPage: ((Int) -> ImagePage?)? = null

    var onPageChange: ((Int) -> Unit)? = null
    var onTap: ((Offset) -> Unit)? = null
    var onLongTap: ((Offset) -> Unit)? = null

    /** Override for the "from" page during far navigation animation */
    var transitionFromPage: ImagePage? = null

    // Pre-allocated invalidate lambda - same for the lifetime of this state
    private val invalidateCallback: () -> Unit = { invalidate() }

    /**
     * The page [index] steps from current. [isReversed] plays no part here - [fetchPage] and
     * [onPageChange] are what decide what a step actually means.
     */
    fun getPage(index: Int): ImagePage? {
        val page = fetchPage?.invoke(index)
        if (page == null) {
            WgvLog.v(TAG, "getPage($index): null (no fetchPage or no page)")
        }
        return page?.also { p ->
            if (p.parent !== this) p.parent = this
            if (p.scope !== this.scope) p.scope = this.scope
            if (p.onInvalidate !== invalidateCallback) p.onInvalidate = invalidateCallback
        }
    }

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        WgvLog.i(TAG, "ImageViewerState.init: ${width}x$height, vertical=$isVertical, reversed=$isReversed")
        this.renderer.init(scope, surface, width, height)
        this.scope = scope

        scope.launch {
            _postInit.forEach { it() }
            _postInit.clear()
        }
    }

    var firstPos = Offset.Zero
    var currentPos = Offset.Zero

    var transition: Transition = if (isVertical) TransitionBasic.Vertical else TransitionBasic

    var renderFlow = MutableSharedFlow<Int>(
        replay = 1, extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    fun invalidate() {
        renderFlow.tryEmit(0)
    }

    suspend fun collect() {
        renderFlow.collectLatest {
            // Capture render state on main thread before any thread switching
            val snapshot = captureRenderState() ?: run {
                WgvLog.v(TAG, "collect: no render state captured, skipping frame")
                return@collectLatest
            }
            // Now render on GPU thread with captured state
            withContext(dispatcher) {
                renderer.render { encoder, texture ->
                    renderSnapshot(encoder, texture, snapshot)
                }
            }
        }
    }

    protected open fun captureRenderState(): Any? {
        val currentPage = getPage(0) ?: return null
        val offset = pageOffset
        val adjacentPage = when {
            offset == 0f -> null
            // Use override if set (for far navigation)
            transitionFromPage != null -> transitionFromPage
            offset > 0f -> getPage(if (isReversed) -1 else 1)
            else -> getPage(if (isReversed) 1 else -1)
        }
        // Only used to pre-warm the transition cache while at rest (see renderSnapshot), so
        // there's no need to look it up while a turn is already in progress.
        val nextPage = if (offset == 0f) getPage(1) else null
        WgvLog.throttled(TAG, key = "capture", intervalMs = 500L) {
            "captureRenderState: page=${currentPage::class.simpleName} offset=$offset " +
                "adjacent=${adjacentPage != null} transition=${transition::class.simpleName}"
        }
        return RenderSnapshot(
            currentPage, adjacentPage, nextPage, offset, transition, firstPos, currentPos
        )
    }

    private class RenderSnapshot(
        val currentPage: ImagePage,
        val adjacentPage: ImagePage?,
        val nextPage: ImagePage?,
        val offset: Float,
        val transition: Transition,
        val firstPos: Offset,
        val currentPos: Offset,
    )

    /**
     * Run [block] against a render pass over [texture], ending the pass afterwards either way.
     *
     * Pass ownership sits here rather than inside the transitions, since only the code that knows
     * the whole frame's contents can decide where the pass starts and ends. A draw that throws
     * still leaves the pass closed, and [WebGpuRenderer.render] turns it into a dropped frame.
     *
     * Always clears: `getCurrentTexture` rotates buffers, so loading would show stale content
     * from several frames ago around the page.
     */
    protected fun renderPass(
        encoder: GPUCommandEncoder, texture: GPUTexture, block: (GPURenderPassEncoder) -> Unit
    ) {
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = texture.createView(),
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                ),
                // Cleared fresh every frame so TileRenderer's blit can mark which pixels it just
                // covered and RenderPage's masked draws can skip re-shading them - see
                // TileRenderer.stencilViewFor. Discarded afterward: nothing reads it across frames.
                depthStencilAttachment = GPURenderPassDepthStencilAttachment(
                    view = tiles.stencilViewFor(texture),
                    stencilLoadOp = LoadOp.Clear,
                    stencilStoreOp = StoreOp.Discard,
                    stencilClearValue = 0,
                )
            )
        )
        try {
            block(pass)
        } finally {
            pass.end()
        }
    }

    protected open suspend fun renderSnapshot(
        encoder: GPUCommandEncoder, texture: GPUTexture, snapshot: Any
    ) {
        val s = snapshot as RenderSnapshot
        tiles.newFrame()
        val page = s.currentPage

        if (s.adjacentPage != null && s.offset != 0f) {
            WgvLog.throttled(TAG, key = "transition", intervalMs = 500L) {
                "renderSnapshot: transition=${s.transition::class.simpleName} offset=${s.offset}"
            }
            s.transition.render(
                page, s.adjacentPage, encoder, texture, s.offset, s.firstPos, s.currentPos, tiles
            )
            return
        }

        val covered = page.drawLive(encoder, texture, tiles)

        // Opportunistic: once the current page's tiles settle, prewarm the next page's too,
        // so a transition into it starts already mostly sharp (Transition.getCachedTexture
        // seeds itself and layers tiles in live, so this no longer needs to be complete
        // first). Gated on atHome since the cache is keyed by (x, y, scale).
        if (covered && page is ImagePage.Images && page.atHome) {
            val next = s.nextPage as? ImagePage.Images
            if (next != null && next.highQuality && !next.isAnimated && next.atHome) {
                tiles.prewarm(next, texture)
            }
        }
    }

    private val _postInit = mutableListOf<(suspend () -> Unit)>()

    @Synchronized
    fun post(fn: suspend () -> Unit) {
        WgvLog.v(TAG, "post(fn) - scope active=${scope?.isActive == true}")
        val activeScope = scope
        if (activeScope?.isActive == true) {
            activeScope.launch(dispatcher) {
                fn()
            }
        } else {
            _postInit.add(fn)
        }
    }

    fun cleanup() {
        WgvLog.i(TAG, "ImageViewerState.cleanup")
        animationJob?.cancel()
        tiles.cleanup()
        renderer.cleanup()
    }
}