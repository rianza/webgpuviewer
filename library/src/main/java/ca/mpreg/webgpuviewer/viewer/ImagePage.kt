package ca.mpreg.webgpuviewer.viewer

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDepthStencilAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.Font
import ca.mpreg.webgpuviewer.draw.TextAlign
import ca.mpreg.webgpuviewer.draw.circle
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.draw.text
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.time.Duration.Companion.milliseconds

private const val TAG = "WGV.Page"

/**
 * A page in the viewer, with shared transform (x, y, scale), pan/zoom-to-fit bounds, and
 * animation.
 *
 * [Images] is the ordinary case - one or two decoded [Image]s. [Dummy] is a placeholder with
 * known dimensions but no content. [Render]/[Progress] draw their own content via [renderWith]
 * instead of blitting an image.
 *
 * Handles:
 * - Transform state (x, y, scale) and home position calculations
 * - Pan/zoom animation
 * - Lifecycle (cleanup)
 */
open class ImagePage {

    /** Placeholder page with dimensions but no image data */
    class Dummy(override val width: Int, override val height: Int) : ImagePage()

    /**
     * ImagePage whose content is supplied by an app-overridden [render] instead of an [Image] - for
     * rendering progress indicators or other app-drawn content with its own shader. Called
     * instead of blitting a texture wherever this page would otherwise be drawn into the regular
     * view or a [ca.mpreg.webgpuviewer.transition.Transition]'s cache texture - unlike [Images],
     * it has no [Images.highQuality] to opt into either path, so it always draws this way.
     *
     * Opens one pass per [render] call (no stencil attachment - unlike [Images], nothing here
     * needs [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s masking), shared by every [rect]/
     * [circle] call inside it rather than each opening its own.
     *
     * [renderWith] clears that pass's destination first - right whenever this page owns the whole
     * thing (a rotated screen buffer, or a [ca.mpreg.webgpuviewer.transition.Transition]'s
     * per-page cache slot). [renderLoaded] is the one exception:
     * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] draws several pages into one
     * shared screen texture, so a Render page's destination there is that whole shared texture,
     * not a page-sized one of its own - clearing it would blank every other visible page too.
     *
     * Override [backgroundColor] to fill a background before [render] runs: when
     * [renderWith] owns the whole destination, that fill is just its clear color, so it covers
     * the full screen (letterboxing beyond this page's own footprint, same as a transition
     * blending toward it); [renderLoaded] instead fills just [fillPage]'s scoped rect, since it
     * can't clear the shared texture. Leave it null to paint nothing and rely entirely on [render].
     */
    open class Render(override val width: Int, override val height: Int) : ImagePage() {

        // Always has drawable content via render(), unlike Dummy - needed so
        // Transition.getCachedTexture doesn't skip this page as if it were undecoded.
        override val isDecoded: Boolean get() = true

        // Unlike the base ImagePage default (fixed "home" position, right for Dummy - it has
        // nothing pan-worthy anyway), a Render page's own render() gets its *live* pan/zoom
        // transform - the same x/y/scale gestures already drive on any ImagePage - so e.g. a
        // custom Render page's own drawn content can track a drag/pinch the same way an Images
        // page's would, rather than being stuck rendering at (0, 0, 1) forever.
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            renderWith(encoder, x, y, scale, dst)
            return false
        }

        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            renderWith(encoder, x, y, scale, tex)
        }

        // render() treats dst as entirely its own canvas (see the class doc) - so unlike Images,
        // whose real content only ever occupies part of dst, this page's rect within a flat
        // render of it IS the whole thing, just panned/zoomed by its own live x/y/scale the same
        // way renderWith/renderCacheSeed already thread through. Without this, TransitionFlipLeft/
        // TransitionFlipRight/TransitionSphere - which all bail out on a null pageRect rather than
        // treating it as screen-shaped - would just never draw a Render page at all.
        override fun pageRect(dst: GPUTexture): FloatArray = floatArrayOf(
            0.5f + scale * (x - 0.5f),
            0.5f + scale * (y - 0.5f),
            0.5f + scale * (x + 0.5f),
            0.5f + scale * (y + 0.5f),
        )

        // Set by renderWith right before calling render(), and only valid for the duration of
        // that call - rect/circle/text read it instead of taking a pass parameter, since there's
        // only ever one pass open per render() call (see the class doc).
        private lateinit var pass: GPURenderPassEncoder

        /** Draws this page's content. Use [rect]/[circle]/[text] to draw into the open pass. */
        open fun render(
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
        ) {
        }

        protected fun rect(
            x1: Float, y1: Float, x2: Float, y2: Float, color: Int
        ) = Draw.rect(pass, x1, y1, x2, y2, color)

        /**
         * Fills this page's own [width] x [height] footprint with [color] - unlike [rect]'s raw
         * [dst]-relative `[0,1]` coordinates (which always cover the *entire* [dst], full stop),
         * this is sized and positioned from this page's own declared [width]/[height] plus the
         * same [x]/[y]/[scale] [render] itself received, rather than assuming this page fills the
         * whole of [dst] the way [renderWith]'s full-[dst] clear does. Matters once [dst] is
         * shared with other pages ([ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState])
         * instead of being this page's own - filling all of `[0,1]` there would blank every other
         * visible page too, and even a same-sized fill would come out the wrong aspect ratio
         * whenever [dst] (the real screen) isn't [width]x[height]'s own aspect.
         */
        protected fun fillPage(dst: GPUTexture, x: Float, y: Float, scale: Float, color: Int) {
            val halfWidthFrac = scale * width / (2f * dst.width)
            val halfHeightFrac = scale * height / (2f * dst.height)
            val cx = 0.5f + scale * x
            val cy = 0.5f + scale * y
            rect(
                cx - halfWidthFrac,
                cy - halfHeightFrac,
                cx + halfWidthFrac,
                cy + halfHeightFrac,
                color
            )
        }

        protected fun circle(
            cx: Float, cy: Float, radius: Float, color: Int
        ) = Draw.circle(pass, cx, cy, radius, color)

        protected fun text(
            dst: GPUTexture,
            font: Font,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(pass, dst, font, text, x, y, size, color, align, maxWidth)

        protected fun text(
            dst: GPUTexture,
            context: Context,
            fontFamily: FontFamily,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            weight: FontWeight = FontWeight.Normal,
            style: FontStyle = FontStyle.Normal,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(
            pass, dst, context, fontFamily, text, x, y, size, color, weight, style, align, maxWidth
        )

        final override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first - see
            // ImagePage.renderWith's default for why that's still needed for a page (like Dummy)
            // that doesn't override this at all. [dst] is this call's own - a rotated screen
            // buffer or a Transition's per-page cache slot - so nothing else on screen depends on
            // whatever was already there.
            openPassAndRender(encoder, x, y, scale, dst, clear = true)
        }

        /**
         * As [renderWith], but loads [dst] instead of clearing it -
         * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] uses this instead, since there
         * [dst] is one screen texture shared by several visible pages at once, and clearing it
         * would blank every other one. [render] is responsible for painting over every pixel of
         * its own footprint here - see [Progress] painting a full background rect before its
         * circle - since anything it doesn't touch keeps whatever [dst] already had (a decoded
         * neighbour's pixels, or simply undefined content the first time a fresh texture is used
         * - [ImageViewerContinuousState] clears once up front to guard against that).
         */
        internal fun renderLoaded(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            openPassAndRender(encoder, x, y, scale, dst, clear = false)
        }

        private fun argbToGPUColor(color: Int): GPUColor {
            val r = ((color shr 16) and 0xFF) / 255.0
            val g = ((color shr 8) and 0xFF) / 255.0
            val b = (color and 0xFF) / 255.0
            val a = ((color ushr 24) and 0xFF) / 255.0
            return GPUColor(r, g, b, a)
        }

        // Background fill driven by backgroundColor() (null by default; override it to
        // opt in - see Progress/an app's own transition placeholder) instead of a dedicated
        // property, since that's the exact same hook every Transition already reads for this
        // page's letterbox colour - one override covers both without the two ever disagreeing.
        private fun openPassAndRender(
            encoder: GPUCommandEncoder,
            x: Float,
            y: Float,
            scale: Float,
            dst: GPUTexture,
            clear: Boolean
        ) {
            val clearValue =
                backgroundColor?.let { argbToGPUColor(it) } ?: GPUColor(0.0, 0.0, 0.0, 0.0)

            val openedPass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = dst.createView(),
                            loadOp = if (clear) LoadOp.Clear else LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = clearValue
                        )
                    )
                )
            )
            pass = openedPass
            try {
                // clear=true already painted the whole dst this colour via clearValue above - a
                // page-scoped fillPage on top would be redundant. clear=false (the shared-texture
                // continuous-mode pass) has no clear to fall back on, so paint the footprint here.
                if (!clear) {
                    backgroundColor?.let { fillPage(dst, x, y, scale, it) }
                }
                render(dst, x, y, scale)
            } finally {
                openedPass.end()
            }
        }
    }

    /**
     * A page backed by one or two decoded images.
     *
     * For single page: images = [image]
     * For dual page spread: images = [leftImage, rightImage]
     */
    class Images(val images: List<Image?>) : ImagePage() {

        constructor(image: Image?) : this(listOf(image))
        constructor(left: Image?, right: Image?) : this(listOf(left, right))

        companion object {
            suspend operator fun invoke(
                pixels: ByteBuffer, width: Int, height: Int, createMipMaps: Boolean = true
            ): Images {
                WgvLog.i(TAG, "Images.fromPixels ${width}x$height (mipmaps=$createMipMaps)")
                return Images(Image(pixels, width, height, createMipMaps))
            }

            suspend operator fun invoke(bitmap: Bitmap, createMipMaps: Boolean = true): Images {
                WgvLog.d(
                    TAG, "Images.fromBitmap ${bitmap.width}x${bitmap.height} (${bitmap.byteCount} bytes)"
                )
                val buf = ByteBuffer.allocateDirect(bitmap.byteCount)
                bitmap.copyPixelsToBuffer(buf)
                return Images(buf, bitmap.width, bitmap.height, createMipMaps)
            }
        }

        override val isDecoded: Boolean
            get() = images.any { it != null }

        /** If true, this page owns its images and will clean them up. If false, images are borrowed. */
        var ownsImages: Boolean = true

        /**
         * When false, this page skips [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s tile cache
         * entirely and its fast path renders through [renderPage] with `linear = false` instead of
         * the default `linear = true` - for content not worth either path's extra correctness or
         * sharpness, such as an app-drawn transition/error bitmap.
         */
        var highQuality: Boolean = true

        /** Total width (sum of image widths) */
        override val width: Int
            get() = images.filterNotNull().sumOf { it.width }

        /** Total height (max of image heights) */
        override val height: Int
            get() = images.filterNotNull().maxOfOrNull { it.height } ?: 0

        /**
         * Visible width after trim. For dual pages, inner edges are ignored.
         */
        override val trimWidth: Int
            get() {
                if (images.size == 2) {
                    val left = images[0]
                    val right = images[1]
                    // Left: from left trim to full width (ignore right trim)
                    val leftW = left?.let { it.width - (it.trim?.left ?: 0) } ?: 0
                    // Right: from 0 to right trim (ignore left trim)
                    val rightW = right?.let { it.trim?.right ?: it.width } ?: 0
                    return leftW + rightW
                }
                return images.sumOf { it?.trim?.width() ?: it?.width ?: 0 }
            }

        /** Visible height after trim (max of trim heights) */
        override val trimHeight: Int
            get() = images.maxOfOrNull { it?.trim?.height() ?: it?.height ?: 0 } ?: 0

        /** True if this page uses half-screen layout (dual page or single LEFT/RIGHT) */
        override val isHalfWidth: Boolean
            get() = images.size == 2 || images.firstOrNull()?.position.let {
                it == Image.Position.LEFT || it == Image.Position.RIGHT
            }

        override fun xEdges(trimmed: Boolean): Pair<Float, Float> {
            if (isHalfWidth || !trimmed) return 0f to width.toFloat()
            val trim = images.firstOrNull()?.trim ?: return 0f to width.toFloat()
            return trim.left.toFloat() to trim.right.toFloat()
        }

        override fun yEdges(trimmed: Boolean): Pair<Int, Int>? {
            if (isHalfWidth || !trimmed || images.all { it?.trim == null }) return null
            val trimTop =
                images.mapNotNull { img -> img?.let { it.trim?.top ?: 0 } }.minOrNull() ?: 0
            val trimBottom =
                images.mapNotNull { it?.trim?.bottom ?: it?.height }.maxOrNull() ?: height
            return trimTop to trimBottom
        }

        /**
         * Each image independently fit to its own half, since a dual spread's two images can
         * differ in size - unlike [width]/[height]'s combined span, which [ImagePage]'s default (for a
         * single non-spread image) uses instead.
         */
        override fun halfWidthScale(halfWidth: Float, contentHeight: Float): Float =
            images.filterNotNull().minOfOrNull { img ->
                minOf(halfWidth / img.width, contentHeight / img.height)
            }?.coerceAtLeast(0.01f) ?: 0.01f

        private var animationLoop: Job? = null
        private var frames: List<Pair<Image, Int>>? = null
        private var currentFrameImage: Image? = null

        /** True while an animation frame loop owns [image]. The tile cache skips animated pages. */
        override val isAnimated: Boolean
            get() = frames != null

        @Volatile
        private var _frameVersion: Int = 0

        /** Incremented each time the animation frame changes. Used by the render cache to detect stale frames. */
        override val frameVersion: Int
            get() = _frameVersion

        /** Current image for rendering (may change during animation) */
        val image: Image?
            get() = currentFrameImage ?: images.firstOrNull()

        fun startAnimationLoop(frames: List<Pair<Image, Int>>, invalidate: () -> Unit) {
            WgvLog.d(TAG, "startAnimationLoop: ${frames.size} frame(s), frameDurations=${frames.map { it.second }}")
            animationLoop?.cancel()
            this.frames = frames
            currentFrameImage = frames.firstOrNull()?.first

            // Use the page's scope if available, otherwise use the shared background scope
            val loopScope = scope ?: cleanupScope
            animationLoop = loopScope.launch {
                var frameIndex = 0
                while (true) {
                    this@Images.frames?.getOrNull(frameIndex)?.let { (img, duration) ->
                        currentFrameImage = img
                        _frameVersion++
                        invalidate()
                        delay(duration.coerceAtLeast(0).milliseconds)
                    } ?: break
                    frameIndex = (frameIndex + 1) % (this@Images.frames?.size ?: 1)
                }
            }
        }

        override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first. No
            // stencil attachment - this fallback never needs TileRenderer's masking, unlike the
            // overrides below that bypass it entirely.
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = dst.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                renderPage(pass, dst, x, y, scale, linear = false, masked = false)
            } finally {
                pass.end()
            }
        }

        /** Opens a `LoadOp.Clear` pass on [dst], with a stencil attachment for [TileRenderer]'s masking. */
        private fun beginLivePass(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ) = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                ), depthStencilAttachment = GPURenderPassDepthStencilAttachment(
                    view = tiles.stencilViewFor(dst),
                    stencilLoadOp = LoadOp.Clear,
                    stencilStoreOp = StoreOp.Discard,
                    stencilClearValue = 0,
                )
            )
        )

        /** Opens a `LoadOp.Clear` pass on [tex], no stencil attachment - a transition's cache is never masked. */
        private fun beginCachePass(encoder: GPUCommandEncoder, tex: GPUTexture) =
            encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = tex.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )

        /**
         * As [ImagePage.drawLive]. Animated frames always want the fast path regardless of
         * [highQuality] (never worth a tile cache that would just churn every frame); a
         * non-[highQuality], non-animated page falls back to the plain [renderWith] (via `super`);
         * everything else goes through the tile cache, backfilling with [renderPage] wherever it
         * isn't covered yet. Opens its own pass (with a stencil attachment, for [TileRenderer]'s
         * masking) rather than sharing one from the caller - see [Render] for why that split exists.
         */
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            if (isAnimated) {
                val pass = beginLivePass(encoder, dst, tiles)
                try {
                    renderBackground(pass, dst, 0f, 0f, 1f)
                    renderPage(pass, dst, 0f, 0f, 1f)
                } finally {
                    pass.end()
                }
                return false
            }

            if (!highQuality) return super.drawLive(encoder, dst, tiles)

            val pass = beginLivePass(encoder, dst, tiles)
            try {
                // Background always drawn live first (its fades are position-dependent, never
                // from a stale tile) so it stays underneath everything else. Tiles draw next,
                // marking every pixel they cover in the stencil buffer tiles.draw() writes to;
                // renderPage then only shades what's left uncovered instead of the whole
                // viewport, since tiles.draw() already produced the right pixel wherever it drew.
                val covered = tiles.isFullyCovered(this, dst, 0f, 0f, 1f)
                WgvLog.throttled(TAG, key = "drawLive", intervalMs = 500L) {
                    "Images.drawLive: page=${Integer.toHexString(System.identityHashCode(this))} " +
                        "covered=$covered"
                }
                renderBackground(pass, dst, 0f, 0f, 1f)
                tiles.draw(pass, this, dst, 0f, 0f, 1f)
                if (!covered) {
                    renderPage(pass, dst, 0f, 0f, 1f)
                }
                return covered
            } finally {
                pass.end()
            }
        }

        /**
         * As [drawLive], seeding a transition's cache slot instead of the screen - same
         * isAnimated/highQuality precedence, but never with a stencil attachment (a transition's
         * cache is never stencil-masked either way).
         */
        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            WgvLog.v(TAG, "Images.renderCacheSeed: tex ${tex.width}x${tex.height}, animated=$isAnimated, highQuality=$highQuality")
            if (isAnimated) {
                val pass = beginCachePass(encoder, tex)
                try {
                    renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                } finally {
                    pass.end()
                }
                return
            }

            if (!highQuality) {
                super.renderCacheSeed(encoder, tex, tiles)
                return
            }

            val pass = beginCachePass(encoder, tex)
            try {
                renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                tiles.blitAvailableTiles(pass, this, tex)
            } finally {
                pass.end()
            }
        }

        override fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
            if (!highQuality || isAnimated) null else tiles.availableTileKeys(this, tex)

        override fun renderIntoCache(
            encoder: GPUCommandEncoder,
            tex: GPUTexture,
            tiles: TileRenderer,
            identityMatches: Boolean
        ) {
            if (!identityMatches) {
                renderCacheSeed(encoder, tex, tiles)
                return
            }
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = tex.createView(),
                            loadOp = LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                tiles.blitAvailableTiles(pass, this, tex)
            } finally {
                pass.end()
            }
        }

        override fun pageRect(dst: GPUTexture): FloatArray? {
            val image = images.firstOrNull() ?: return null
            if (image.mipmaps.isEmpty()) return null
            return image.placement(dst, x, y, scale)
        }

        override val backgroundColor: Int? = images.firstOrNull()?.backgroundColor

        override fun drawBackgroundColumns(
            pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
        ) {
            val renderImages = if (images.size == 1) listOf(image) else images
            renderImages.forEach { img ->
                img ?: return@forEach
                if (img.mipmaps.isEmpty()) return@forEach
                val imgOffsetX = img.spreadOffsetX / dst.width
                val placedRect = img.placement(dst, x + imgOffsetX, y, scale)
                val x1 = if (img.position == Image.Position.SINGLE) 0f else placedRect[0]
                val x2 = if (img.position == Image.Position.SINGLE) 1f else placedRect[2]
                Draw.rect(
                    pass, offsetX + x1, offsetY, offsetX + x2, offsetY + 1f, img.backgroundColor
                )
            }
        }

        /**
         * Draw this page into [pass], one bilinear tap per pixel. [linear]/[masked] pick the same
         * 4 pipelines as the image-level [RenderPage.renderFast]. Background handling is derived
         * from the pair rather than its own parameter: masked+linear skips it (that caller already
         * drew it via [renderBackground]), masked-only folds it into the masked draw, unmasked
         * always draws a plain one alongside.
         */
        fun renderPage(
            pass: GPURenderPassEncoder,
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            linear: Boolean = true,
            masked: Boolean = true
        ) {
            WgvLog.throttled(TAG, key = "renderPage", intervalMs = 500L) {
                "Images.renderPage: ${images.size} image(s), scale=$scale, linear=$linear, masked=$masked"
            }
            val variant = RenderPage.variantFor(linear, masked)
            forEachPlacedImage(dst, x, y, scale) { image, rect, placeX, placeY, placeScale ->
                if (!linear || !masked) {
                    drawImageBackground(pass, image, rect, scale, maskedBackground = masked)
                }
                for (tile in image.prepareTilesForRender(dst, placeX, placeY, placeScale)) {
                    RenderPage.drawTile(pass, dst, tile, variant)
                }
            }
        }

        /**
         * Draw just this page's per-image background colour, skipping the image itself - for
         * ImageViewerState's masked path, which draws this first (so it stays underneath
         * [ca.mpreg.webgpuviewer.renderer.TileRenderer.draw] and [renderPage]), or as the whole
         * draw once [ca.mpreg.webgpuviewer.renderer.TileRenderer] already covers the image itself.
         * The background's alpha depends on live pan/scale, so it's drawn every frame regardless
         * of tile coverage. Uses the stencil-pass-compatible rect pipeline - see
         * [RenderPage.drawMaskedRect] - since it always runs inside that pass.
         */
        fun renderBackground(
            pass: GPURenderPassEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float
        ) = forEachPlacedImage(dst, x, y, scale) { image, rect, _, _, _ ->
            drawImageBackground(pass, image, rect, scale, maskedBackground = true)
        }

        /**
         * Draws [image]'s fading background rect, for [renderPage]'s `drawBackground` branch.
         * Alpha fades with distance from home/min scale or the page's pan bounds, so it only shows
         * near the edges of the zoom/pan range where the image itself doesn't fill the viewport.
         */
        private fun drawImageBackground(
            pass: GPURenderPassEncoder,
            image: Image,
            rect: FloatArray,
            scale: Float,
            maskedBackground: Boolean,
        ) {
            val parent = parent
            val minScale = minScale
            val homeScale = homeScale
            val currentScale = this.scale * scale

            val fadeDistancePixels = 200f
            val imageSize = (width.coerceAtLeast(height)).toFloat()

            fun proximity(anchorScale: Float): Float {
                if (anchorScale <= 0f) return 0f
                val deltaPixels = abs(imageSize * (currentScale - anchorScale))
                return (1f - deltaPixels / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundProximity(value: Float, lo: Float, hi: Float, pixelsPerUnit: Float): Float {
                val overflow = when {
                    value < lo -> lo - value
                    value > hi -> value - hi
                    else -> return 1f
                }
                return (1f - overflow * pixelsPerUnit / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundsProximityAt(anchorScale: Float): Float {
                if (parent == null || anchorScale <= 0f) return 0f
                val minX = minX(anchorScale)
                val maxX = maxX(anchorScale)
                val minY = minY(anchorScale)
                val maxY = maxY(anchorScale)
                val pixelsPerUnitX = parent.width.toFloat() * anchorScale
                val pixelsPerUnitY = parent.height.toFloat() * anchorScale
                return min(
                    boundProximity(this@Images.x, minX, maxX, pixelsPerUnitX),
                    boundProximity(this@Images.y, minY, maxY, pixelsPerUnitY)
                )
            }

            val bgAlpha = if (parent != null) {
                if (currentScale > minScale) {
                    boundsProximityAt(currentScale)
                } else {
                    val homeProximity = min(proximity(homeScale), boundsProximityAt(homeScale))
                    val minProximity = min(proximity(minScale), boundsProximityAt(minScale))
                    max(homeProximity, minProximity)
                }
            } else {
                1f
            }

            val origA = (image.backgroundColor ushr 24) and 0xFF
            val a = (origA * bgAlpha).toInt()
            if (a <= 0) return

            val x1 = if (image.position == Image.Position.SINGLE) 0f else rect[0]
            val x2 = if (image.position == Image.Position.SINGLE) 1f else rect[2]
            val origR = (image.backgroundColor shr 16) and 0xFF
            val origG = (image.backgroundColor shr 8) and 0xFF
            val origB = image.backgroundColor and 0xFF
            val r = (origR * bgAlpha).toInt()
            val g = (origG * bgAlpha).toInt()
            val b = (origB * bgAlpha).toInt()
            val bgColor = (a shl 24) or (r shl 16) or (g shl 8) or b
            if (maskedBackground) {
                RenderPage.drawMaskedRect(pass, x1, 0f, x2, 1f, bgColor)
            } else {
                Draw.rect(pass, x1, 0f, x2, 1f, bgColor)
            }
        }

        /** Walks this page's image(s), placing each one for [action] to draw against. */
        private inline fun forEachPlacedImage(
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            action: (image: Image, rect: FloatArray, placeX: Float, placeY: Float, placeScale: Float) -> Unit
        ) {
            // A snapshot is captured on the main thread and drawn later, so the page may have been
            // evicted in between. Its images' buffers are already destroyed, and touching one throws.
            if (destroyed || images.all { it == null }) return

            val renderImages = if (images.size == 1) listOf(image) else images
            renderImages.forEach { img ->
                img ?: return@forEach
                if (img.mipmaps.isEmpty()) return@forEach
                val offsetX = img.spreadOffsetX / dst.width
                val placeX = this.x + x + offsetX
                val placeY = this.y + y
                val placeScale = this.scale * scale
                val rect = img.placement(dst, placeX, placeY, placeScale)
                action(img, rect, placeX, placeY, placeScale)
            }
        }

        @Synchronized
        override fun cleanup() {
            if (destroyed) return
            WgvLog.d(TAG, "Images.cleanup: page=${Integer.toHexString(System.identityHashCode(this))}, ${images.size} image(s), ownsImages=$ownsImages")
            super.cleanup()

            animationLoop?.cancel()
            animationLoop = null

            // Only clean images if we own them
            if (!ownsImages) {
                frames = null
                currentFrameImage = null
                return
            }

            val framesToClean = frames
            frames = null
            currentFrameImage = null

            // Frames include all images; otherwise clean individual images
            val imagesToClean = framesToClean?.map { it.first } ?: images.filterNotNull()

            if (imagesToClean.isNotEmpty()) {
                cleanupScope.launch {
                    try {
                        // Eviction fires exactly when the viewer reaches a new page, so freeing a
                        // page's textures competes with the frames that are drawing the new one.
                        // Yield between images and stay off the render mutex, for the same reason
                        // uploads do. Dawn keeps a destroyed texture alive until the command buffers
                        // referencing it retire, so a frame already in flight is unaffected.
                        WebGpuRenderer.onDispatcher {
                            imagesToClean.forEach { image ->
                                image.cleanup()
                                yield()
                            }
                        }
                    } catch (e: Exception) {
                        WgvLog.e(TAG, "Cleanup error while freeing ${imagesToClean.size} image(s)", e)
                    }
                }
            }
        }
    }

    companion object {
        /**
         * Shared scope for fire-and-forget GPU cleanup work.
         * Lives for the application lifetime; individual cleanups are tiny and non-cancellable anyway.
         */
        internal val cleanupScope = CoroutineScope(Dispatchers.Default)
    }

    /** True once page content has been decoded/is otherwise ready to draw. */
    open val isDecoded: Boolean
        get() = false

    /**
     * True once [cleanup] has run and the page's resources are gone or going.
     *
     * Volatile because it is set on whatever thread evicts the page but read on the GPU thread,
     * which uses it to skip drawing a page whose textures are being freed. A render snapshot is
     * captured on the main thread and drawn later, so it can outlive the page it names.
     */
    @Volatile
    var destroyed = false
        private set

    /** True while an animation frame loop owns the current frame. Only ever true for [Images]. */
    open val isAnimated: Boolean get() = false

    /** Incremented each time an animated page's frame changes. Only ever nonzero for [Images]. */
    open val frameVersion: Int get() = 0

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (this.x == x && this.y == y && this.scale == scale) return
        this.x = x
        this.y = y
        this.scale = scale
        onInvalidate?.invoke()
    }

    /**
     * Draws this page's content instead of blitting an image - see [Render] and [Images]. Given
     * the raw [encoder] rather than an already-open pass, so a page like [Render] that doesn't
     * need [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s stencil masking can open its own
     * pass(es) instead of being forced to match one it has no use for. Must clear [dst] itself -
     * this default does, in its own pass, since [Dummy] (which never overrides this) still needs
     * one: `getCurrentTexture` rotates buffers, so leaving it alone would show stale content from
     * several frames ago. [Render]/[Images] override this to clear as part of their own drawing
     * pass instead of paying for a separate one.
     */
    open fun renderWith(
        encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
    ) {
        WgvLog.v(TAG, "ImagePage.renderWith (base): clear ${dst.width}x${dst.height}")
        Draw.clear(encoder, dst, 0)
    }

    /**
     * Draws this page's current live content into [dst] - the paged viewer's per-frame
     * (non-transition) path. Just [renderWith] at home position by default, which already does
     * the right thing for every non-[Images] page (including a no-op [Dummy]); [Images] overrides
     * this to add its [Images.highQuality] tile cache and animated-frame handling.
     *
     * Returns true if the page is now fully covered by sharp tiles, so [ImageViewerState] knows
     * it's safe to prewarm the next page - always false here, since only [Images] has a tile cache.
     */
    internal open fun drawLive(
        encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
    ): Boolean {
        renderWith(encoder, 0f, 0f, 1f, dst)
        return false
    }

    /**
     * As [drawLive], but seeding a [ca.mpreg.webgpuviewer.transition.Transition]'s cache slot
     * instead of the screen - see [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture].
     * Just [renderWith] by default; [Images] overrides this the same way it overrides [drawLive].
     */
    internal open fun renderCacheSeed(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
    ) {
        renderWith(encoder, 0f, 0f, 1f, tex)
    }

    /**
     * Tile keys newly available to blit since the last call, or null if this page is never tiled
     * (every non-[Images] page, or an [Images] one that isn't [Images.highQuality] or is animated)
     * - see [ca.mpreg.webgpuviewer.renderer.TileRenderer.availableTileKeys].
     */
    internal open fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
        null

    /**
     * Renders this page into a transition's cache slot - see
     * [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture]. [identityMatches] false
     * means a fresh slot ([renderCacheSeed] from scratch, `LoadOp.Clear`); true means an unchanged
     * one that just needs whatever's newly available layered on (`LoadOp.Load`). Just
     * [renderCacheSeed] by default, since a non-[Images] page never has anything incremental to
     * layer on top of - [Images] overrides this to blit instead when [identityMatches].
     */
    internal open fun renderIntoCache(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer, identityMatches: Boolean
    ) {
        renderCacheSeed(encoder, tex, tiles)
    }

    /**
     * This page's own rect within a flat render of it into [dst], as normalised (x1, y1, x2, y2)
     * surface coordinates - for a warp transition ([ca.mpreg.webgpuviewer.transition.TransitionSphere]/
     * flip) to map the page's actual rect rather than treating it as screen-shaped. Null if this
     * page has nothing to draw, which is always true for a non-[Images] page.
     */
    open fun pageRect(dst: GPUTexture): FloatArray? = null

    /**
     * The color a transition should blend/fade toward for this page as a whole - [Images] reads
     * it from its first image's own [ca.mpreg.webgpuviewer.renderer.Image.backgroundColor];
     * [Render] also uses it (when overridden non-null) as its own background fill, both during a
     * transition and for its regular [Render.renderWith]/[Render.renderLoaded] draws - see
     * [Render]'s class doc. Null (nothing to fill/blend toward) by default.
     */
    open val backgroundColor: Int? = null

    /**
     * Draws this page's per-image background colour as separate columns at [offsetX]/[offsetY]
     * within its own cached-surface slide - see
     * [ca.mpreg.webgpuviewer.transition.TransitionBasic] and the
     * [ca.mpreg.webgpuviewer.transition.TransitionStackUp] family. A no-op for a non-[Images] page.
     */
    open fun drawBackgroundColumns(
        pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
    ) {
    }

    open val width: Int get() = 0
    open val height: Int get() = 0

    /** As [width]/[height], but after trim - defaults to the untrimmed size. */
    open val trimWidth: Int get() = width
    open val trimHeight: Int get() = height

    /** True if this page uses half-screen layout (dual page or single LEFT/RIGHT). */
    open val isHalfWidth: Boolean get() = false

    /** Left/right edges [minX]/[maxX] pan between: trim's edges once [trimmed], else raw span. */
    protected open fun xEdges(trimmed: Boolean): Pair<Float, Float> = 0f to width.toFloat()

    /**
     * As [xEdges], for top/bottom - null (not the raw fallback) so [nudgedYBounds] can tell real
     * trim edges (margin worth protecting) from the raw image's own (safe to pan past).
     */
    protected open fun yEdges(trimmed: Boolean): Pair<Int, Int>? = null

    var animationJob: Job? = null
    var animationTargetX: Float? = null
    var animationTargetY: Float? = null
    var animationTargetScale: Float? = null

    /**
     * True while [scale] is being animated - by [animateTo] or externally (e.g. fling-zoom decay,
     * which sets this directly). Gates the tile cache, which otherwise can't tell a settled scale
     * from a spring that's merely repeating a value for a frame mid-flight.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    /**
     * True while a plain (non-zoom) pan fling is actively decaying - set/cleared directly around
     * that decay in [ca.mpreg.webgpuviewer.viewer.ImageViewer]'s gesture handling. Checked there
     * so a tap landing while a fling from the previous gesture is still gliding doesn't also fire
     * [ImageViewerState.onTap] - the tap itself has no motion (that's what makes it a tap, not a
     * drag), so nothing else would otherwise tell the two apart.
     */
    @Volatile
    var isFlinging: Boolean = false

    var parent: ImageViewerState? = null
        set(value) {
            val wasNull = field == null
            field = value
            // Initialize to home when parent first set
            if (wasNull && value != null && x == 0f && y == 0f && scale == 1f) {
                x = homeX
                y = homeY
                scale = homeScale
            }
        }

    var scope: CoroutineScope? = null
    var onInvalidate: (() -> Unit)? = null

    private val contentWidth: Float
        get() = parent?.width?.toFloat() ?: 0f

    private val contentHeight: Float
        get() {
            val parent = parent ?: return 0f
            return if (parent.avoidCutout && parent.cutoutTopPx > 0f) parent.height - parent.cutoutTopPx else parent.height.toFloat()
        }

    /**
     * [isHalfWidth]'s fit scale: each side of a spread can be a differently sized image, so
     * [Images] overrides this to fit each one independently rather than [width]/[height]'s
     * combined span - the default here is only ever exercised by a non-spread page.
     */
    protected open fun halfWidthScale(halfWidth: Float, contentHeight: Float): Float =
        minOf(halfWidth / width, contentHeight / height).coerceAtLeast(0.01f)

    val atHome: Boolean
        get() {
            val eps = 0.0001f
            return abs(x - homeX) < eps && abs(y - homeY) < eps && atHomeScale
        }

    val atHomeScale: Boolean
        get() {
            val eps = 0.0001f
            return abs(scale - homeScale) < eps
        }

    var homeScale: Float = -1f
        get() {
            if (field > 0) return field

            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen, no trim
                return halfWidthScale(contentWidth / 2f, contentHeight)
            }

            // Single SINGLE page: fit trim to full screen
            val w = trimWidth.toFloat().takeIf { it > 0f } ?: return 0.01f
            val h = trimHeight.toFloat().takeIf { it > 0f } ?: return 0.01f
            return minOf(contentWidth / w, contentHeight / h).coerceAtLeast(0.01f)
        }

    var homeX: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxX(scale).fastCoerceIn(minX(scale), maxX(scale))
        }

    var homeY: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxY(scale).fastCoerceIn(minY(scale), maxY(scale))
        }

    var minScale = 0f
        get() {
            if (field > 0) return field
            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen
                return halfWidthScale(contentWidth / 2f, contentHeight)
            }

            // Single SINGLE page
            return minOf(contentWidth / width, contentHeight / height).coerceAtLeast(0.01f)
        }

    var maxScale = 0f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

    var doubleTapScale: Float = 0f
        get() = if (field != 0f) field else max(minScale, homeScale) * 2

    // BOUNDS:
    // cutout ignore:
    //  with trim:
    //      >= homeScale: viewport pan over trimmed content
    //      < homeScale: viewport pan over untrimmed content
    //  without trim: viewport pan over content
    //
    // cutout avoid:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: nudge below cutout
    //
    // cutout shift:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: center in cut viewport

    /**
     * Natural pan range at [scale] (no cutout nudge): [nearEdge]/[farEdge] bound where they'd
     * land exactly on the viewport's near/far edge. Collapses to center when content fits.
     */
    private fun edgeRange(
        size: Int, nearEdge: Float, farEdge: Float, parentSize: Int, scale: Float
    ): Pair<Float, Float> {
        val (minV, maxV) = rawBounds(size, nearEdge, farEdge, parentSize, scale)
        if (minV > maxV) {
            val center = (minV + maxV) / 2f
            return center to center
        }
        return minV to maxV
    }

    /**
     * As [edgeRange] but never collapsed (min > max when there's slack). [nudgedYBounds] needs
     * the true floor - the collapsed center sits below it, letting panning reveal past it.
     */
    private fun rawBounds(
        size: Int, nearEdge: Float, farEdge: Float, parentSize: Int, scale: Float
    ): Pair<Float, Float> {
        val maxV = (0.5f * size - nearEdge) / parentSize - 0.5f / scale
        val minV = (0.5f * size - farEdge) / parentSize + 0.5f / scale
        return minV to maxV
    }

    /** [minX]/[maxX] together, computing [homeScale] and [edgeRange] only once per call. */
    private fun xBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val (left, right) = xEdges(scale >= homeScale)
        return edgeRange(width, left, right, parent.width, scale)
    }

    fun minX(scale: Float): Float = xBounds(scale).first

    fun maxX(scale: Float): Float = xBounds(scale).second

    /**
     * "Ignore": [edgeRange]'s plain collapse-when-it-fits.
     *
     * "Avoid"/"shift": near/top bound pushed further by the *full* [ImageViewerState.cutoutTopPx]
     * so the cut viewport (real viewport minus the cutout) can pan over all the content - unless
     * still fully visible there even after the push, which collapses to one rest point instead of
     * a pointless range: "shift" always rests centered in the cut viewport (half the push, since
     * shrinking the viewport only moves its center by half); "avoid" only nudges - just far enough
     * to clear the cutout, not all the way to centered-in-cut-viewport - and only when the plain
     * whole-screen-centered rest would actually overlap the cutout.
     */
    private fun nudgedYBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val trimmed = scale >= homeScale
        val (top, bottom) = yEdges(trimmed) ?: (0 to height)
        val (floor, natMax) = rawBounds(
            height, top.toFloat(), bottom.toFloat(), parent.height, scale
        )
        val slack = floor > natMax
        val center = (floor + natMax) / 2f

        if (!parent.avoidCutout || parent.cutoutTopPx <= 0f || parent.height <= 0) {
            return if (slack) center to center else floor to natMax
        }

        val fullPush =
            if (isHalfWidth && !trimmed) 0f else parent.cutoutTopPx / (scale * parent.height)
        val pushed = natMax + fullPush

        if (slack) {
            // "pushed" (natMax + fullPush) is exactly the rest position where the near edge sits
            // flush against the cutout - so center < pushed means the plain centered rest would
            // sit above that line, i.e. under the cutout.
            val overlapsCutout = center < pushed
            if (!parent.alwaysAvoidCutout && !overlapsCutout) return center to center
            val rest = if (parent.alwaysAvoidCutout) center + fullPush / 2f else pushed
            if (rest < floor) return rest to rest
        }
        return floor to pushed
    }

    fun minY(scale: Float): Float = nudgedYBounds(scale).first

    fun maxY(scale: Float): Float = nudgedYBounds(scale).second

    fun home() {
        animateTo(targetScale = homeScale)
    }

    fun animateTo(
        origin: Offset? = null,
        targetX: Float = homeX,
        targetY: Float = homeY,
        targetScale: Float = scale,
    ) {
        WgvLog.d(
            TAG,
            "animateTo: (${x},$y@$scale) -> ($targetX,$targetY@$targetScale) origin=$origin"
        )
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val targetScale = targetScale.fastCoerceIn(minScale, maxScale)

        val minX = minX(targetScale)
        val maxX = maxX(targetScale)
        val minY = minY(targetScale)
        val maxY = maxY(targetScale)

        val scaleChanging = targetScale != startScale
        val diffEnd = if (scaleChanging) 1 / targetScale - 1 / startScale else 1f

        val endX = when {
            origin != null && scaleChanging -> (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(
                minX, maxX
            )

            origin != null -> x.fastCoerceIn(minX, maxX)
            else -> targetX
        }
        val endY = when {
            origin != null && scaleChanging -> (startY + (origin.y - 0.5f) * diffEnd).fastCoerceIn(
                minY, maxY
            )

            origin != null -> y.fastCoerceIn(minY, maxY)
            else -> targetY
        }

        animationJob = scope?.launch {
            animationTargetX = endX
            animationTargetY = endY
            animationTargetScale = targetScale
            if (scaleChanging) isScaleAnimating = true
            try {
                animate(
                    0f, 1f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
                ) { value, _ ->
                    val currentScale = startScale + (targetScale - startScale) * value
                    val c = if (scaleChanging) {
                        ((1 / currentScale - 1 / startScale) / diffEnd).fastCoerceIn(0f, 1f)
                    } else {
                        value
                    }

                    setPos(
                        (startX + (endX - startX) * c).orZero(),
                        (startY + (endY - startY) * c).orZero(),
                        currentScale
                    )
                }
            } finally {
                animationTargetX = null
                animationTargetY = null
                animationTargetScale = null
                isScaleAnimating = false
            }
        }
    }

    @Synchronized
    open fun cleanup() {
        if (destroyed) return
        WgvLog.d(TAG, "ImagePage.cleanup: ${this::class.simpleName}")
        destroyed = true

        animationJob?.cancel()
        animationJob = null
    }

    /** Alias for cleanup() */
    fun destroy() = cleanup()
}
