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
import ca.mpreg.webgpuviewer.renderer.RenderPage
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

open class ImageViewerState(var isVertical: Boolean = false, var isReversed: Boolean = false) {
    val renderer = WebGpuRenderer()

    internal val tiles = TileRenderer { invalidate() }

    var animationJob: Job? = null

    val width get() = renderer.width
    val height get() = renderer.height

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    var scope: CoroutineScope? = null

    /** Top padding in pixels to avoid display cutout. Set automatically by ImageViewer when avoidCutout is true. */
    var cutoutTopPx: Float = 0f
        set(value) {
            if (field != value) {
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
        return fetchPage?.invoke(index)?.also { page ->
            if (page.parent !== this) page.parent = this
            if (page.scope !== this.scope) page.scope = this.scope
            if (page.onInvalidate !== invalidateCallback) page.onInvalidate = invalidateCallback
        }
    }

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
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
            val snapshot = captureRenderState() ?: return@collectLatest
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
                        // TEMPORARY diagnostic (fix/blank/old-driver): opaque red proves whether
                        // presented frames reach the screen at all. Revert to GPUColor(0,0,0,0)
                        // once the blank-screen root cause is confirmed.
                        clearValue = GPUColor(1.0, 0.0, 0.0, 1.0)
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
        if (s.adjacentPage != null && s.offset != 0f) {
            s.transition.render(
                s.currentPage,
                s.adjacentPage,
                encoder,
                texture,
                s.offset,
                s.firstPos,
                s.currentPos,
                tiles
            )
        } else {
            // Computed once and reused below - isFullyCoveredCore already treats a false
            // highQuality as "not covered", so the plain-sampler branch just ends up unused.
            val covered = tiles.isFullyCovered(s.currentPage, texture, 0f, 0f, 1f)

            renderPass(encoder, texture) { pass ->
                // Content not worth the tile cache's sharpness or the fast path's linear-light
                // correctness (see ImagePage.highQuality) skips both entirely - just the plain
                // sampler, every frame.
                if (!s.currentPage.highQuality) {
                    RenderPage.renderPlainMasked(pass, s.currentPage, texture, 0f, 0f, 1f)
                    return@renderPass
                }
                // Background always drawn live first (its fades are position-dependent, never
                // from a stale tile) so it stays underneath everything else. Tiles draw next,
                // marking every pixel they cover in the stencil buffer tiles.draw() writes to;
                // renderFastImageOnly then only shades what's left uncovered instead of the whole
                // viewport, since tiles.draw() already produced the right pixel wherever it drew.
                RenderPage.renderBackground(pass, s.currentPage, texture, 0f, 0f, 1f)
                tiles.draw(pass, s.currentPage, texture, 0f, 0f, 1f)
                if (!covered) {
                    RenderPage.renderFastImageOnly(pass, s.currentPage, texture, 0f, 0f, 1f)
                }
            }

            // Opportunistic: once the current page's tiles settle, prewarm the next page's too,
            // so a transition into it starts already mostly sharp (Transition.getCachedTexture
            // seeds itself and layers tiles in live, so this no longer needs to be complete
            // first). Gated on atHome since the cache is keyed by (x, y, scale).
            if (s.currentPage.highQuality && !s.currentPage.isAnimated && s.currentPage.atHome && covered) {
                val next = s.nextPage
                if (next != null && next.highQuality && !next.isAnimated && next.atHome) {
                    tiles.prewarm(next, texture)
                }
            }
        }
    }

    private val _postInit = mutableListOf<(suspend () -> Unit)>()

    @Synchronized
    fun post(fn: suspend () -> Unit) {
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
        animationJob?.cancel()
        tiles.cleanup()
        renderer.cleanup()
    }
}