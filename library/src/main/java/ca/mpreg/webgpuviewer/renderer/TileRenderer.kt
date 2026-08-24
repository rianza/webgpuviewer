package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferBindingType
import androidx.webgpu.BufferUsage
import androidx.webgpu.CompareFunction
import androidx.webgpu.FeatureName
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBindGroupLayoutDescriptor
import androidx.webgpu.GPUBindGroupLayoutEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferBindingLayout
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUDepthStencilState
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPUPipelineLayoutDescriptor
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPUQuerySet
import androidx.webgpu.GPUQuerySetDescriptor
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUSamplerBindingLayout
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUStencilFaceState
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureBindingLayout
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.MapMode
import androidx.webgpu.OptionalBool
import androidx.webgpu.PrimitiveTopology
import androidx.webgpu.QueryType
import androidx.webgpu.SamplerBindingType
import androidx.webgpu.ShaderStage
import androidx.webgpu.StencilOperation
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureSampleType
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.OFF_SCREEN_SCORE
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.STENCIL_BUFFER_COUNT
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILES_PER_BATCH_FALLBACK
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE
import ca.mpreg.webgpuviewer.viewer.ImagePage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.time.Duration.Companion.milliseconds

/**
 * Solve the (x, y) [RenderPage.render]/[RenderPage.renderFast] need so [image] lands centred at
 * screen position ([targetX], [targetY]) - inverts [Image.prepareForRender]'s placement math for
 * an arbitrary target.
 */
internal fun solveImagePlacement(
    targetX: Float,
    targetY: Float,
    imageScale: Float,
    image: Image,
    dstWidth: Float,
    dstHeight: Float
): Pair<Float, Float> {
    val x =
        (targetX - dstWidth / 2f) / (imageScale * dstWidth) - image.x / dstWidth - WebGpuRenderer.offsetX
    val y =
        (targetY - dstHeight / 2f) / (imageScale * dstHeight) - image.y / dstHeight - WebGpuRenderer.offsetY
    return x to y
}

/**
 * A cache of the filtered render, cut into [TILE_SIZE]-square screen-resolution tiles.
 *
 * [RenderPage.render] is too expensive every frame; [RenderPage.renderFast] is cheap but
 * unfiltered. Each frame draws the fast path, then blits whatever filtered tiles already exist
 * on top, while a background worker fills in the rest a few at a time.
 *
 * One grid per whole [ImagePage], not per image, so a spread's seam bakes into whichever tile
 * straddles it rather than meeting two independently-snapped layers.
 *
 * The same grid serves both viewers. The paged viewer has one page on screen at a time and
 * rounds its own anchor; the continuous viewer can have several pages' grids live at once, so it
 * rounds only the shared *camera* position and leaves each page's offset from it exact - keeps
 * adjacent pages' tiles pixel-aligned at their shared boundary (see that [draw] overload).
 *
 * Tiles live in content space: tile (tx, ty) holds the square [TILE_SIZE] pixels right/down of
 * the grid's anchor, so panning survives untouched. Changing scale (or a document position shift
 * in the continuous viewer) invalidates the whole grid, which regenerates once scale has held
 * stable for two frames.
 *
 * The whole grid snaps to the nearest screen pixel so every blit is an exact 1:1 texel copy. Each
 * tile's bind group and uniform are created once at generation, so a blit is just setBindGroup +
 * draw; only the per-grid anchor/clip uniform is rewritten, and only when it changes.
 *
 * Generation runs on the render thread but outside the render mutex, a few tiles at a time with a
 * suspend between batches so a queued frame always gets the thread back. Pending tiles are pulled
 * on-screen-first, centre-out.
 *
 * Everything here is touched only on the render thread. [cleanup] may be called from any thread
 * and posts its work there.
 */
internal class TileRenderer(private val invalidate: () -> Unit) {
    companion object {
        const val TILE_SIZE = 256

        private const val TILES_PER_BATCH_FALLBACK = 1
        private const val BATCH_TARGET_NS = 4_000_000.0
        private const val MAX_TILES_PER_BATCH = 8

        /**
         * Grace window of extra pages (past "whichever is current") [draw] keeps a grid for, so
         * leaving a page doesn't force full regeneration on turning right back. Paged overload
         * only - the continuous viewer relies on [evict]'s shared LRU cap instead.
         */
        private const val RETAIN_MARGIN = 2

        private const val TAG = "TileRenderer"

        /**
         * Score threshold [nextRequest] uses to tell a genuinely on-screen tile request from one
         * outside a grid's wanted range (e.g. [prewarm]'s tiles, which leave that range empty).
         */
        private const val OFF_SCREEN_SCORE = 1e6f

        /**
         * Ring buffer size for [stencilViewFor] - matches a typical Android/Vulkan surface's
         * buffer count so a rotated stencil texture is never still in flight from a prior frame.
         */
        private const val STENCIL_BUFFER_COUNT = 3
    }

    /** Upper bound on cached tiles, ~[maxTiles] * 256KB of texture memory. */
    var maxTiles = 192

    private val device get() = WebGpuRenderer.device

    private var frame = 0L
    private var workerActive = false
    private val workerScope = CoroutineScope(WebGpuRenderer.dispatcher + SupervisorJob())

    // Timestamp-query based GPU cost measurement for [generateTile]'s batches - null wherever the
    // adapter didn't have the feature (see WebGpuRenderer's requiredFeatures), in which case
    // batch sizing just falls back to [TILES_PER_BATCH_FALLBACK] forever.
    private val timestampQuerySet: GPUQuerySet? by lazy {
        if (!device.hasFeature(FeatureName.TimestampQuery)) return@lazy null
        device.createQuerySet(GPUQuerySetDescriptor(type = QueryType.Timestamp, count = 16))
    }

    private fun createTimestampBuffers(): Pair<GPUBuffer, GPUBuffer> {
        val resolve = device.createBuffer(
            GPUBufferDescriptor(size = 16, usage = BufferUsage.QueryResolve or BufferUsage.CopySrc)
        )
        val result = device.createBuffer(
            GPUBufferDescriptor(size = 16, usage = BufferUsage.MapRead or BufferUsage.CopyDst)
        )
        return resolve to result
    }

    // Exponential moving average of one tile's GPU render-pass duration, in nanoseconds - 0
    // until the first measurement lands, which is when [nextBatchSize] starts trusting it over
    // [TILES_PER_BATCH_FALLBACK].
    private var avgTileGpuNs = 0.0

    /** How many tiles [schedule] should generate before its next yield - see [avgTileGpuNs]. */
    private fun nextBatchSize(): Int {
        if (avgTileGpuNs <= 0.0) return TILES_PER_BATCH_FALLBACK
        return (BATCH_TARGET_NS / avgTileGpuNs).toInt().coerceIn(1, MAX_TILES_PER_BATCH)
    }

    private class Tile(
        val texture: GPUTexture, val uniform: GPUBuffer, val bindGroup: GPUBindGroup
    ) {
        var lastUsed = 0L

        fun destroy() {
            bindGroup.close()
            uniform.destroy()
            texture.destroy()
        }
    }

    /** One grid per whole [ImagePage] - both images of a spread share it, seam baked in. */
    private class PageTiles(var scale: Float, val page: ImagePage, val frameUniform: GPUBuffer) {
        val tiles = HashMap<Long, Tile>()
        val pending = HashSet<Long>()

        // Values the current frameUniform contents were derived from, so a frame where the grid
        // didn't move skips the write entirely and encodes nothing but the blit draws - see
        // writeFrameUniformIfChanged.
        var writtenSnapX = Float.NaN
        var writtenSnapY = Float.NaN
        var writtenDstW = Float.NaN
        var writtenDstH = Float.NaN
        var writtenClipL = Float.NaN
        var writtenClipT = Float.NaN
        var writtenClipR = Float.NaN
        var writtenClipB = Float.NaN

        /** True once the scale has held for two consecutive frames; gates generation. */
        var stable = false

        /**
         * This page's exact, unrounded vertical offset from the grid's shared anchor - see
         * [draw]'s continuous overload. Always 0 for the paged overload.
         *
         * Also doubles as a staleness key, compared each call like [scale]: a changed offset at
         * fixed scale means the page's document position shifted, so existing tiles no longer
         * agree with where it sits.
         */
        var centerYOffset = 0f

        // The strictly visible tile range as of the last draw, in tile coordinates. The worker
        // prioritises against it at pull time, so a pan mid-fill redirects generation without
        // touching the queue.
        var txMin = 0
        var txMax = -1
        var tyMin = 0
        var tyMax = -1

        val destroyed get() = page.destroyed

        fun destroyAll() {
            tiles.values.forEach { it.destroy() }
            tiles.clear()
            pending.clear()
            frameUniform.destroy()
        }
    }

    /** One tile of work for the shared worker - see [schedule]/[nextRequest]. */
    private class Request(val state: PageTiles, val tx: Int, val ty: Int)

    // Access-ordered so getOrPut's read-then-maybe-write always moves the touched page to the
    // end (most recently drawn), whether or not it was already present - see RETAIN_MARGIN.
    private val pages = LinkedHashMap<ImagePage, PageTiles>(16, 0.75f, true)

    private fun key(tx: Int, ty: Int) = (tx.toLong() shl 32) or (ty.toLong() and 0xFFFFFFFFL)

    // Thread-local ByteBuffer to avoid per-blit allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    // Nearest: tiles are blitted 1:1 at integer pixel positions, so this is an exact copy.
    private val blitSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Nearest,
                minFilter = FilterMode.Nearest,
            )
        )
    }

    // Explicit (not auto-inferred) so it can be shared across blitPipeline/blitPipelineStencilWrite -
    // an implicit/auto pipeline layout is unique to the pipeline it was inferred for, so a bind
    // group made against one pipeline's auto layout is invalid on the other, even though both
    // pipelines share the exact same shader and bindings. Tile.bindGroup is created once and
    // reused across both pipelines (see tileBindGroup), so it must be built against this instead.
    private val blitBindGroupLayout by lazy {
        device.createBindGroupLayout(
            GPUBindGroupLayoutDescriptor(
                entries = arrayOf(
                    GPUBindGroupLayoutEntry(
                        binding = 0,
                        visibility = ShaderStage.Vertex or ShaderStage.Fragment,
                        buffer = GPUBufferBindingLayout(type = BufferBindingType.Uniform)
                    ),
                    GPUBindGroupLayoutEntry(
                        binding = 1,
                        visibility = ShaderStage.Vertex or ShaderStage.Fragment,
                        buffer = GPUBufferBindingLayout(type = BufferBindingType.Uniform)
                    ),
                    GPUBindGroupLayoutEntry(
                        binding = 2,
                        visibility = ShaderStage.Fragment,
                        texture = GPUTextureBindingLayout(sampleType = TextureSampleType.Float)
                    ),
                    GPUBindGroupLayoutEntry(
                        binding = 3,
                        visibility = ShaderStage.Fragment,
                        sampler = GPUSamplerBindingLayout(type = SamplerBindingType.Filtering)
                    ),
                )
            )
        )
    }

    private val blitPipelineLayout by lazy {
        device.createPipelineLayout(
            GPUPipelineLayoutDescriptor(bindGroupLayouts = arrayOf(blitBindGroupLayout))
        )
    }

    private fun buildBlitPipeline(depthStencil: GPUDepthStencilState?): GPURenderPipeline {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(BLIT_SHADER))
        )
        return device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(module = shaderModule, entryPoint = "vs_main"),
                layout = blitPipelineLayout,
                fragment = GPUFragmentState(
                    module = shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            // Tiles hold RenderPage's output, which is premultiplied, so One
                            // rather than SrcAlpha.
                            format = TextureFormat.RGBA8Unorm, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = PrimitiveTopology.TriangleList),
                depthStencil = depthStencil,
            )
        )
    }

    /** Plain blit, no stencil attachment - [blitAvailableTiles]/[renderFullyTiled]'s own pass. */
    private val blitPipeline: GPURenderPipeline by lazy { buildBlitPipeline(null) }

    /**
     * As [blitPipeline], but always writes 1 into the stencil attachment wherever it draws - for
     * [drawCore]'s live-render callers ([draw]), which mask [RenderPage.renderFast]'s per-pixel
     * work against exactly this: a tile pixel drawn here needs no further shading underneath it,
     * so a pixel [RenderPage] would otherwise redraw stays skipped once this stencil value marks
     * it done. See [stencilViewFor].
     */
    private val blitPipelineStencilWrite: GPURenderPipeline by lazy {
        buildBlitPipeline(
            GPUDepthStencilState(
                format = TextureFormat.Stencil8,
                depthWriteEnabled = OptionalBool.False,
                depthCompare = CompareFunction.Always,
                stencilFront = GPUStencilFaceState(
                    compare = CompareFunction.Always, passOp = StencilOperation.Replace
                ),
                stencilBack = GPUStencilFaceState(
                    compare = CompareFunction.Always, passOp = StencilOperation.Replace
                ),
                stencilWriteMask = 0xFF,
            )
        )
    }

    // Ring-buffered, not one shared texture: reusing a single stencil texture every frame would
    // force the GPU to serialize each frame's clear/write against the previous frame's, still
    // in flight since render() only serializes encoding - silently undoing the swapchain's own
    // buffering and showing up as tiles trailing the current pan/scroll position.
    private val stencilTextures = arrayOfNulls<GPUTexture>(STENCIL_BUFFER_COUNT)
    private val stencilViews = arrayOfNulls<GPUTextureView>(STENCIL_BUFFER_COUNT)
    private var stencilWidth = 0
    private var stencilHeight = 0

    /**
     * Stencil-only attachment matching [dst]'s size, shared by every live-render pass this frame
     * so [blitPipelineStencilWrite] can mark tile-covered pixels and [RenderPage]'s masked
     * variants can skip re-shading them. Rotates across [STENCIL_BUFFER_COUNT] textures by
     * [frame] so a still-in-flight previous frame never shares one with this one.
     */
    fun stencilViewFor(dst: GPUTexture): GPUTextureView {
        if (stencilWidth != dst.width || stencilHeight != dst.height) {
            stencilWidth = dst.width
            stencilHeight = dst.height
            for (i in 0 until STENCIL_BUFFER_COUNT) {
                stencilTextures[i]?.destroy()
                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        usage = TextureUsage.RenderAttachment,
                        size = GPUExtent3D(dst.width, dst.height),
                        format = TextureFormat.Stencil8,
                    )
                )
                stencilTextures[i] = texture
                stencilViews[i] = texture.createView()
            }
        }
        return stencilViews[(frame % STENCIL_BUFFER_COUNT).toInt()]!!
    }

    /**
     * Advance the frame counter and drop tiles for any page the app has since evicted. Called
     * once at the top of every rendered frame, including ones that don't draw tiles at all (e.g.
     * a page transition), so a destroyed page's textures are freed right away.
     */
    fun newFrame() {
        frame++
        if (pages.isEmpty()) return
        val it = pages.iterator()
        while (it.hasNext()) {
            val st = it.next().value
            if (st.destroyed) {
                st.destroyAll()
                it.remove()
            }
        }
    }

    /**
     * Left/right screen-pixel extent of [page] from its own anchor (x=0), in [pageScale] units.
     *
     * Not symmetric halves of [ImagePage.width]: a LEFT/RIGHT image extends outward from the
     * anchor by its *own* width, so the anchor sits at the spread's seam, not the centre of its
     * combined footprint - they can differ, e.g. a cover with no partner. No LEFT/RIGHT image at
     * all is the ordinary centred case, so its extent stays symmetric.
     */
    private fun pageHorizontalExtent(page: ImagePage, pageScale: Float): Pair<Float, Float> {
        val leftWidth = page.images.firstOrNull { it?.position == Image.Position.LEFT }?.width
        val rightWidth = page.images.firstOrNull { it?.position == Image.Position.RIGHT }?.width
        if (leftWidth == null && rightWidth == null) {
            val half = pageScale * page.width / 2f
            return half to half
        }
        return pageScale * (leftWidth ?: 0) to pageScale * (rightWidth ?: 0)
    }

    /** [pageScale]/[anchorX]/[anchorY] a paged overload resolves its (x, y, scale) placement to. */
    private class PagedAnchor(val pageScale: Float, val anchorX: Float, val anchorY: Float)

    private fun pagedAnchor(
        page: ImagePage, dst: GPUTexture, x: Float, y: Float, scale: Float
    ): PagedAnchor {
        // Pin the grid to the animation's target while actually scale-animating home, so
        // drawCore wipes it once instead of every interpolated frame. Gated on isScaleAnimating,
        // not just the target being homeScale, since a pure-position animateTo at a constant
        // homeScale never sets that flag - pinning then would just freeze the grid for nothing.
        val goingHome = page.isScaleAnimating && page.animationTargetScale == page.homeScale
        val effectivePageX = if (goingHome) page.animationTargetX ?: page.x else page.x
        val effectivePageY = if (goingHome) page.animationTargetY ?: page.y else page.y
        val effectivePageScale = if (goingHome) page.homeScale else page.scale
        val pageScale = effectivePageScale * scale
        val anchorX =
            dst.width / 2f + pageScale * ((effectivePageX + x + WebGpuRenderer.offsetX) * dst.width)
        val anchorY =
            dst.height / 2f + pageScale * ((effectivePageY + y + WebGpuRenderer.offsetY) * dst.height)
        return PagedAnchor(pageScale, anchorX, anchorY)
    }

    /**
     * Shared placement math for a page's tile grid at [anchorX]/[anchorY], scaled by [pageScale]:
     * the tile region the viewport (plus a one-tile margin) wants, clipped to the page's own
     * extent, and the snapped clip rect the shader clamps blits to.
     *
     * One definition shared by [isFullyCoveredCore], [drawCore], and [prewarm] - each used to
     * carry its own copy, which is how [prewarm] ended up silently missing a step the others had.
     */
    private class GridPlacement(
        val snapX: Float,
        val snapY: Float,
        val clipL: Float,
        val clipT: Float,
        val clipR: Float,
        val clipB: Float,
        val wantL: Float,
        val wantR: Float,
        val wantT: Float,
        val wantB: Float
    )

    private fun gridPlacement(
        page: ImagePage,
        dst: GPUTexture,
        anchorX: Float,
        anchorY: Float,
        centerYOffset: Float,
        pageScale: Float
    ): GridPlacement? {
        val ts = TILE_SIZE.toFloat()
        val (leftHalf, rightHalf) = pageHorizontalExtent(page, pageScale)
        val halfH = pageScale * page.height / 2f
        if (leftHalf + rightHalf <= 0f || halfH <= 0f) return null

        val wantL = max(-anchorX - ts, -leftHalf)
        val wantR = min(dst.width - anchorX + ts, rightHalf)
        val wantT = max(-anchorY - ts, centerYOffset - halfH)
        val wantB = min(dst.height - anchorY + ts, centerYOffset + halfH)

        val snapX = round(anchorX)
        val snapY = round(anchorY)
        val clipL = snapX - leftHalf
        val clipT = snapY + centerYOffset - halfH
        val clipR = snapX + rightHalf
        val clipB = snapY + centerYOffset + halfH

        return GridPlacement(snapX, snapY, clipL, clipT, clipR, clipB, wantL, wantR, wantT, wantB)
    }

    /**
     * Every (tx, ty) [gp] wants, visible or not - the one definition [drawCore],
     * [isFullyCoveredCore], and [availableTileKeys] all share, so they can't drift apart the way
     * [gridPlacement]'s own doc describes [prewarm] once doing. Visibility is a separate per-tile
     * question - see [tileVisible].
     */
    private inline fun forEachWantedGridTile(
        gp: GridPlacement,
        action: (txi: Int, tyi: Int) -> Unit
    ) {
        val ts = TILE_SIZE.toFloat()
        val tx0 = floor(gp.wantL / ts).toInt()
        val tx1 = ceil(gp.wantR / ts).toInt() - 1
        val ty0 = floor(gp.wantT / ts).toInt()
        val ty1 = ceil(gp.wantB / ts).toInt() - 1
        for (tyi in ty0..ty1) {
            for (txi in tx0..tx1) {
                action(txi, tyi)
            }
        }
    }

    /** True if tile ([txi], [tyi]) of [gp]'s grid actually overlaps [dst]'s visible bounds. */
    private fun tileVisible(gp: GridPlacement, dst: GPUTexture, txi: Int, tyi: Int): Boolean {
        val ts = TILE_SIZE.toFloat()
        val px = gp.snapX + txi * ts
        val py = gp.snapY + tyi * ts
        val visL = max(gp.clipL, 0f)
        val visT = max(gp.clipT, 0f)
        val visR = min(gp.clipR, dst.width.toFloat())
        val visB = min(gp.clipB, dst.height.toFloat())
        return px < visR && px + ts > visL && py < visB && py + ts > visT
    }

    /** As [PagedAnchor], for the continuous overloads; also carries [centerYOffset]. */
    private class ContinuousAnchor(
        val pageScale: Float, val anchorX: Float, val anchorY: Float, val centerYOffset: Float
    )

    private fun continuousAnchor(
        page: ImagePage,
        dst: GPUTexture,
        cameraDocY: Float,
        docTop: Float,
        viewerOffsetX: Float,
        scale: Float
    ): ContinuousAnchor? {
        if (page.width <= 0) return null
        val pageScaleAtZoom1 = dst.width / page.width.toFloat()
        val pageScale = pageScaleAtZoom1 * scale
        val anchorX =
            dst.width / 2f + scale * (viewerOffsetX * dst.width + WebGpuRenderer.offsetX * dst.width)
        val anchorY =
            dst.height / 2f - scale * cameraDocY + scale * WebGpuRenderer.offsetY * dst.height
        val pageHeightDoc = page.height * pageScaleAtZoom1
        val centerYOffset = scale * (docTop + pageHeightDoc / 2f)
        return ContinuousAnchor(pageScale, anchorX, anchorY, centerYOffset)
    }

    /**
     * True if [page]'s grid already has full tile coverage, so the caller can skip
     * [RenderPage.renderFast] entirely this frame - the paged viewer's placement, via its own
     * [page]-relative (x, y).
     */
    fun isFullyCovered(
        page: ImagePage, dst: GPUTexture, x: Float, y: Float, scale: Float
    ): Boolean {
        val a = pagedAnchor(page, dst, x, y, scale)
        return isFullyCoveredCore(
            page, dst, a.anchorX, a.anchorY, 0f, a.pageScale, page.isScaleAnimating
        )
    }

    /**
     * True if [page]'s grid already has full tile coverage, so the caller can skip
     * [RenderPage.renderFast] entirely this frame - the continuous viewer's placement, via
     * [cameraDocY] (the camera's document position) and [docTop] (this page's own, both in screen
     * pixels at zoom 1).
     */
    fun isFullyCovered(
        page: ImagePage,
        dst: GPUTexture,
        cameraDocY: Float,
        docTop: Float,
        viewerOffsetX: Float,
        scale: Float,
        suppressGeneration: Boolean
    ): Boolean {
        val a =
            continuousAnchor(page, dst, cameraDocY, docTop, viewerOffsetX, scale) ?: return false
        return isFullyCoveredCore(
            page, dst, a.anchorX, a.anchorY, a.centerYOffset, a.pageScale, suppressGeneration
        )
    }

    /**
     * Shared read-only coverage check for both [isFullyCovered] overloads. [anchorX]/[anchorY]
     * are the (unrounded) anchor either overload computed; [centerYOffset] is this page's own
     * exact, unrounded vertical offset from that anchor - zero for the paged overload. Compared
     * against [PageTiles.centerYOffset] (like [pageScale] against [PageTiles.scale]) since this
     * call may race ahead of [draw] invalidating a grid whose page just shifted.
     */
    private fun isFullyCoveredCore(
        page: ImagePage,
        dst: GPUTexture,
        anchorX: Float,
        anchorY: Float,
        centerYOffset: Float,
        pageScale: Float,
        suppressGeneration: Boolean
    ): Boolean {
        if (page.destroyed || !page.highQuality || page.isAnimated || suppressGeneration) return false
        if (page.images.isEmpty() || page.images.all { it == null || it.mipmaps.isEmpty() }) return false

        val st = pages[page] ?: return false
        if (st.scale != pageScale || st.centerYOffset != centerYOffset || !st.stable) return false

        val gp =
            gridPlacement(page, dst, anchorX, anchorY, centerYOffset, pageScale) ?: return false
        // Genuinely off-screen: the fast path would draw nothing either, so there is nothing for
        // tiles to be "covering".
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) return true

        var covered = true
        forEachWantedGridTile(gp) { txi, tyi ->
            if (covered && tileVisible(gp, dst, txi, tyi) && !st.tiles.containsKey(key(txi, tyi))) {
                covered = false
            }
        }
        return covered
    }

    /**
     * The set of (tx, ty) grid keys [page]'s tile cache currently has cached and visible within
     * [dst] - lets a caller like [Transition] track exactly what it's already blitted and detect
     * when something new lands, instead of re-deriving a "done yet" boolean that has to agree
     * with [drawCore] (see [forEachWantedGridTile]'s doc). Null if the page isn't drawable.
     */
    fun availableTileKeys(page: ImagePage, dst: GPUTexture): Set<Long>? {
        if (page.destroyed || !page.highQuality || page.isAnimated) return null
        if (page.images.isEmpty() || page.images.all { it == null || it.mipmaps.isEmpty() }) return null

        val st = pages[page] ?: return null
        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        val gp = gridPlacement(page, dst, a.anchorX, a.anchorY, 0f, a.pageScale) ?: return null
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) return emptySet()

        val keys = HashSet<Long>()
        forEachWantedGridTile(gp) { txi, tyi ->
            if (tileVisible(gp, dst, txi, tyi)) {
                val tkey = key(txi, tyi)
                if (st.tiles.containsKey(tkey)) keys.add(tkey)
            }
        }
        return keys
    }

    /**
     * Ensure [page]'s tile grid exists and enqueue its missing tiles, without blitting anything -
     * for getting a page sharp before it's on screen (the paged viewer's next page). Always at
     * the page's own home position, since it has no live pan/zoom yet.
     *
     * Deliberately leaves [PageTiles.txMin]/etc at their empty default, so [nextRequest] always
     * ranks this grid's tiles as "off-screen", behind whichever page is genuinely being drawn.
     * Doesn't run [draw]'s retain-window trim either; that catches this grid once the page it
     * displaced becomes current and calls [draw] again.
     */
    fun prewarm(page: ImagePage, dst: GPUTexture) {
        if (page.destroyed || !page.highQuality || page.isAnimated) return
        if (page.images.isEmpty() || page.images.all { it == null || it.mipmaps.isEmpty() }) return

        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        val st = pages.getOrPut(page) {
            PageTiles(
                a.pageScale, page, device.createBuffer(
                    GPUBufferDescriptor(
                        size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                    )
                )
            )
        }

        if (st.scale != a.pageScale) {
            st.tiles.values.forEach { it.destroy() }
            st.tiles.clear()
            st.pending.clear()
            st.scale = a.pageScale
            st.stable = false
            invalidate()
        } else {
            st.stable = true
        }
        if (!st.stable) return

        val gp = gridPlacement(page, dst, a.anchorX, a.anchorY, 0f, a.pageScale) ?: return
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) return

        val ts = TILE_SIZE.toFloat()
        val tx0 = floor(gp.wantL / ts).toInt()
        val tx1 = ceil(gp.wantR / ts).toInt() - 1
        val ty0 = floor(gp.wantT / ts).toInt()
        val ty1 = ceil(gp.wantB / ts).toInt() - 1

        // A prewarmed tile's bind group references this same frame uniform, but unlike [drawCore]
        // this grid is never drawn on screen to write it - without this, a page that's only ever
        // been prewarmed (never on screen) blits with a stale/never-written clip rect once
        // [blitAvailableTiles] uses its tiles, which reads as solid black. Shares [gridPlacement]
        // with [drawCore] now, so this can't happen again without both call sites noticing.
        writeFrameUniformIfChanged(
            st, dst, gp.snapX, gp.snapY, gp.clipL, gp.clipT, gp.clipR, gp.clipB
        )

        // Captured before the loop below can add to it, so this only fires the one time pending
        // work actually starts for this grid - every later call while it's still filling in finds
        // pending already non-empty and stays quiet.
        val alreadyPrewarming = st.pending.isNotEmpty()

        var added = false
        for (tyi in ty0..ty1) {
            for (txi in tx0..tx1) {
                val tkey = key(txi, tyi)
                if (!st.tiles.containsKey(tkey)) {
                    st.pending.add(tkey)
                    added = true
                }
            }
        }
        if (added) {
            if (!alreadyPrewarming) Log.d(TAG, "Pre-warming next page tiles ${pageId(page)}")
            schedule()
        }
    }

    /**
     * Blit [page]'s cached tiles and enqueue the missing ones - the paged viewer's placement, via
     * its own [page]-relative (x, y).
     *
     * [ImagePage.Draw] pages are drawn into externally and animated pages swap images per frame -
     * neither worth the cache's sharpness, so both stay on the fast/plain path instead via
     * [ImagePage.highQuality] being false.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) {
        val a = pagedAnchor(page, dst, x, y, scale)
        drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            0f,
            a.pageScale,
            page.isScaleAnimating,
            applyRetainWindow = true,
            useStencilMask = true
        )
    }

    /**
     * Blit [page]'s cached tiles and enqueue the missing ones - the continuous viewer's
     * placement, via [cameraDocY] (the camera's document position) and [docTop] (this page's
     * own, both in screen pixels at zoom 1).
     *
     * Rounds only the shared *camera* anchor, leaving each page's own offset from it exact -
     * unlike the paged overload, several pages can draw through here in the same frame, and
     * independently rounding each one's own anchor could leave adjacent grids disagreeing by a
     * pixel at their shared seam (`round(a) + b` isn't generally `round(a + b)`). Recomputing the
     * same camera anchor from the same inputs every caller passes keeps it identical regardless
     * of which page is drawing, so two adjacent pages' clip boundaries always agree exactly.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        cameraDocY: Float,
        docTop: Float,
        viewerOffsetX: Float,
        scale: Float,
        suppressGeneration: Boolean
    ) {
        val a = continuousAnchor(page, dst, cameraDocY, docTop, viewerOffsetX, scale) ?: return
        drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            a.centerYOffset,
            a.pageScale,
            suppressGeneration,
            applyRetainWindow = false,
            useStencilMask = true
        )
    }

    /**
     * Shared blit-and-enqueue core for both [draw] overloads. [anchorX]/[anchorY] are the
     * (unrounded) anchor either overload computed; [centerYOffset] is this page's own exact,
     * unrounded vertical offset from that anchor - zero for the paged overload. See
     * [PageTiles.centerYOffset] for why comparing it against the stored value also catches a
     * changed document position.
     */
    private fun drawCore(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        anchorX: Float,
        anchorY: Float,
        centerYOffset: Float,
        pageScale: Float,
        suppressGeneration: Boolean,
        applyRetainWindow: Boolean,
        useStencilMask: Boolean = false
    ) {
        if (page.destroyed || !page.highQuality || page.isAnimated) return
        if (page.images.isEmpty() || page.images.all { it == null || it.mipmaps.isEmpty() }) return

        val st = pages.getOrPut(page) {
            PageTiles(
                pageScale, page, device.createBuffer(
                    GPUBufferDescriptor(
                        size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                    )
                )
            )
        }

        if (applyRetainWindow) {
            // getOrPut just moved page to the end of this access-ordered map - trim the front
            // (least recently drawn) down to the grace window. A page turn animates via
            // Transition's own cache, never this one, so anything evicted here isn't on screen.
            while (pages.size > RETAIN_MARGIN) {
                val eldest = pages.entries.iterator()
                val entry = eldest.next()
                entry.value.destroyAll()
                eldest.remove()
            }
        }

        if (st.scale != pageScale || st.centerYOffset != centerYOffset) {
            val reason = if (st.scale != pageScale && st.centerYOffset != centerYOffset) "scale and centerYOffset" else if (st.scale != pageScale) "scale" else "centerYOffset"
            Log.d(TAG, "Wipe tile grid page=${pageId(page)} reason=$reason oldScale=${st.scale} newScale=$pageScale oldCenterYOffset=${st.centerYOffset} newCenterYOffset=$centerYOffset")
            // Dawn keeps a destroyed texture alive until its command buffers retire, so
            // destroying now is safe. A changed centerYOffset at fixed scale means a placeholder
            // corrected its guessed height - invalidate the same way a scale change does.
            st.tiles.values.forEach { it.destroy() }
            st.tiles.clear()
            st.pending.clear()
            st.scale = pageScale
            st.stable = false
            invalidate()
        } else {
            // Two frames landing on the same scale isn't enough proof of settling while a
            // gesture/animation is still actively driving it.
            st.stable = !suppressGeneration
        }
        // Recomputed every call regardless of whether it actually changed - see the field's own
        // doc for why that's safe. generate() has no other way to reach this value.
        st.centerYOffset = centerYOffset

        val gp = gridPlacement(page, dst, anchorX, anchorY, centerYOffset, pageScale)
        if (gp == null) {
            st.pending.clear()
            return
        }
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) {
            st.pending.clear()
            return
        }

        val ts = TILE_SIZE.toFloat()

        // In tile coordinates, unlike wantT/wantB - not offset by centerYOffset, since a tile's
        // blit position is snapY + ty*ts regardless of which page it belongs to.
        st.txMin = floor(-anchorX / ts).toInt()
        st.txMax = ceil((dst.width - anchorX) / ts).toInt() - 1
        st.tyMin = floor(-anchorY / ts).toInt()
        st.tyMax = ceil((dst.height - anchorY) / ts).toInt() - 1

        writeFrameUniformIfChanged(
            st, dst, gp.snapX, gp.snapY, gp.clipL, gp.clipT, gp.clipR, gp.clipB
        )

        var pipelineSet = false
        val desired = HashSet<Long>()
        forEachWantedGridTile(gp) { txi, tyi ->
            val tkey = key(txi, tyi)
            desired.add(tkey)
            val tile = st.tiles[tkey]
            if (tile != null) {
                tile.lastUsed = frame
                if (tileVisible(gp, dst, txi, tyi)) {
                    if (!pipelineSet) {
                        if (useStencilMask) {
                            pass.setPipeline(blitPipelineStencilWrite)
                            pass.setStencilReference(1)
                        } else {
                            pass.setPipeline(blitPipeline)
                        }
                        pipelineSet = true
                    }
                    pass.setBindGroup(0, tile.bindGroup)
                    pass.draw(6)
                }
            } else if (st.stable) {
                st.pending.add(tkey)
            }
        }
        st.pending.retainAll(desired)

        // Drop tiles outside this frame's wanted range - without this, a continuous-mode page
        // scrolling past keeps accumulating tiles that may never hit evict()'s global cap on
        // their own. Same staleness guard evict() uses; in-place removal costs nothing when
        // nothing is stale.
        val staleIt = st.tiles.entries.iterator()
        while (staleIt.hasNext()) {
            val (k, t) = staleIt.next()
            if (k !in desired && t.lastUsed < frame - 1) {
                t.destroy()
                staleIt.remove()
            }
        }

        if (st.pending.isNotEmpty()) schedule()
    }

    /**
     * Start the generation worker if it isn't running.
     *
     * The worker shares the render thread but not the render mutex, so suspending between
     * batches actually lets a queued frame through - the same reasoning as
     * [WebGpuRenderer.onDispatcher]. Batch size comes from [nextBatchSize], re-read every batch
     * as [avgTileGpuNs] accumulates measurements. Each tile's [generate] job is collected and
     * joined once per batch (not per tile) on `this` coroutine directly - no nested
     * [coroutineScope], and that join is itself what lets a queued frame through, so no separate
     * [yield] is needed. Without [FeatureName.TimestampQuery], [generate] never returns a [Job],
     * so joining an always-empty [measurements] does nothing - pacing falls back to a flat
     * [delay] between [TILES_PER_BATCH_FALLBACK]-sized batches instead.
     */
    private fun schedule() {
        if (WebGpuRenderer.deviceLost || workerActive) return
        workerActive = true
        val timestampsSupported = device.hasFeature(FeatureName.TimestampQuery)
        workerScope.launch {
            try {
                while (true) {
                    val batchSize = nextBatchSize()
                    var generated = 0
                    val measurements = ArrayList<Job>(batchSize)
                    while (generated < batchSize) {
                        val req = nextRequest() ?: break
                        try {
                            generate(req, this)?.let { measurements.add(it) }
                            generated++
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            Log.e(TAG, "Tile render failed", e)
                        }
                    }
                    if (generated == 0) break
                    invalidate()
                    if (timestampsSupported) measurements.joinAll() else delay(5.milliseconds)
                }
            } finally {
                workerActive = false
            }
        }
    }

    /**
     * Pull the highest-priority pending tile: on-screen first, then everything else ([prewarm]'d
     * tiles or a stability-losing grid's leftovers).
     *
     * The on-screen/other split reuses the per-tile distance score: [prewarm] leaves a grid's
     * [PageTiles.txMin]/etc at their empty default, so its tiles always score at or above
     * [OFF_SCREEN_SCORE].
     */
    private fun nextRequest(): Request? {
        var bestPriority = Float.MAX_VALUE
        var bestState: PageTiles? = null
        var bestKey = 0L

        fun priorityOf(st: PageTiles, key: Long): Float {
            val tx = (key shr 32).toInt()
            val ty = key.toInt()
            val centerTx = (st.txMin + st.txMax) * 0.5f
            val centerTy = (st.tyMin + st.tyMax) * 0.5f
            val outX = max(max(st.txMin - tx, tx - st.txMax), 0)
            val outY = max(max(st.tyMin - ty, ty - st.tyMax), 0)
            val cx = tx - centerTx
            val cy = ty - centerTy
            return max(outX, outY) * OFF_SCREEN_SCORE + cx * cx + cy * cy
        }

        val pageIt = pages.iterator()
        while (pageIt.hasNext()) {
            val (_, st) = pageIt.next()
            if (st.pending.isEmpty()) continue
            if (st.destroyed || !st.stable) {
                st.pending.clear()
                continue
            }
            for (pkey in st.pending) {
                val priority = priorityOf(st, pkey)
                if (priority < bestPriority) {
                    bestPriority = priority
                    bestState = st
                    bestKey = pkey
                }
            }
        }

        if (bestState == null) return null
        bestState.pending.remove(bestKey)
        return Request(bestState, (bestKey shr 32).toInt(), bestKey.toInt())
    }

    /**
     * Generates [req], returning the GPU timing measurement's [Job] if this tile started one -
     * launched onto [measurementScope], which [schedule] sets to its own coroutine so a whole
     * batch's worth can be joined together. [blitAvailableTiles]'s direct [generateTileNow] call
     * has no such scope, so it just falls back to [workerScope], fire-and-forget.
     */
    private fun generate(req: Request, measurementScope: CoroutineScope): Job? {
        val st = req.state
        if (st.destroyed || !st.stable) return null
        return generateTileNow(st, req.tx, req.ty, measurementScope)
    }

    /** Generate [st]'s tile at ([tx], [ty]) right now if it isn't already cached. */
    private fun generateTileNow(
        st: PageTiles, tx: Int, ty: Int, measurementScope: CoroutineScope = workerScope
    ): Job? {
        return generateTile(st, tx, ty, measurementScope) { pass, texture ->
            renderTileContent(
                st, tx, ty, pass, texture
            )
        }
    }

    /**
     * Render a page's tile, drawing every one of its images into the same pass so a tile
     * straddling their seam comes out with both already in place.
     *
     * Positions each image via [solveImagePlacement], the same inversion of
     * [Image.prepareForRender]'s placement the fast path already uses. The target passed in is
     * this image's ordinary full-frame placement minus the tile's own origin, so solving against
     * a [TILE_SIZE]-square destination places exactly the crop this tile is responsible for.
     */
    private fun renderTileContent(
        st: PageTiles, tx: Int, ty: Int, pass: GPURenderPassEncoder, texture: GPUTexture
    ) {
        val ts = TILE_SIZE.toFloat()
        val s = st.scale
        st.page.images.forEach { image ->
            image ?: return@forEach
            if (image.mipmaps.isEmpty()) return@forEach
            // In raw (unscaled) pixels since solveImagePlacement scales by s itself.
            val targetX = -tx * ts + s * (image.spreadOffsetX + image.x)
            val targetY = st.centerYOffset - ty * ts + s * image.y
            val (x, y) = solveImagePlacement(targetX, targetY, s, image, ts, ts)
            RenderPage.render(pass, image, texture, x, y, s)
        }
    }

    /**
     * Compact, stable-per-object identifier for [page] in log messages - "current"/"next" are
     * relative labels that get reassigned to a different actual page on every turn, so this is
     * what lets a log reader tell whether two log lines are really about the same page.
     */
    private fun pageId(page: ImagePage) = Integer.toHexString(System.identityHashCode(page))

    /**
     * Shared setup for [renderFullyTiled]/[blitAvailableTiles]: draws whatever's cached into
     * [dst] and queues anything missing (via [drawCore]'s own `schedule()` call), returning the
     * grid's state - or null if [page] has no drawable images.
     */
    private fun drawGridForFullPage(
        pass: GPURenderPassEncoder, page: ImagePage, dst: GPUTexture
    ): PageTiles? {
        if (page.destroyed || !page.highQuality || page.isAnimated) return null
        if (page.images.isEmpty() || page.images.all { it == null || it.mipmaps.isEmpty() }) return null

        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            0f,
            a.pageScale,
            suppressGeneration = false,
            applyRetainWindow = false
        )

        val st = pages[page] ?: return null

        // A scale (or centerYOffset) change wipes the grid and marks it unstable within that same
        // drawCore call - which also means its want/pending pass ran before stability was granted,
        // so nothing got queued. drawCore's two-call gate exists to avoid re-wiping every frame
        // while a gesture actively drives scale, but this caller has no next call to benefit from
        // that - it must finish now regardless, e.g. a page turn triggered while the page is still
        // mid-animation back to its home scale. Re-running once more (same anchor/scale, so no
        // further wipe) grants stability immediately and lets the want/pending pass actually queue
        // this frame's tiles.
        if (!st.stable) {
            drawCore(
                pass,
                page,
                dst,
                a.anchorX,
                a.anchorY,
                0f,
                a.pageScale,
                suppressGeneration = false,
                applyRetainWindow = false
            )
        }

        return st
    }

    /**
     * Render [page]'s full tile grid into [dst], generating any tile the worker hasn't reached
     * yet right here rather than leaving it queued - for callers that can't show less than fully
     * complete. [blitAvailableTiles] is the partial/progressive counterpart. Returns false,
     * drawing nothing, if the page has no drawable images.
     */
    fun renderFullyTiled(pass: GPURenderPassEncoder, page: ImagePage, dst: GPUTexture): Boolean {
        val st = drawGridForFullPage(pass, page, dst) ?: return false
        if (st.pending.isEmpty()) return true

        val toGenerate = st.pending.toList()
        st.pending.clear()
        toGenerate.forEach { tkey -> generateTileNow(st, (tkey shr 32).toInt(), tkey.toInt()) }

        pass.setPipeline(blitPipeline)
        toGenerate.forEach { tkey ->
            val tile = st.tiles[tkey] ?: return@forEach
            tile.lastUsed = frame
            pass.setBindGroup(0, tile.bindGroup)
            pass.draw(6)
        }
        return true
    }

    /**
     * Blit whatever's already cached into [dst], queuing anything missing for the background
     * worker rather than force-generating it - see [renderFullyTiled] for the force-complete
     * counterpart. Used by [Transition]'s cache to layer tiles over an immediate fast-rendered
     * seed without blocking on full coverage. Returns false, drawing nothing, if [page] has no
     * drawable images.
     */
    fun blitAvailableTiles(pass: GPURenderPassEncoder, page: ImagePage, dst: GPUTexture): Boolean =
        drawGridForFullPage(pass, page, dst) != null

    /**
     * Render one [TILE_SIZE] tile at ([tx], [ty]) into [st] via [render], then store it. The GPU
     * timing measurement this starts is launched onto [measurementScope] - see [generate]'s doc
     * for why that's a per-batch scope from [schedule] rather than [workerScope] directly.
     */
    private inline fun generateTile(
        st: PageTiles,
        tx: Int,
        ty: Int,
        measurementScope: CoroutineScope,
        render: (GPURenderPassEncoder, GPUTexture) -> Unit
    ): Job? {
        val key = key(tx, ty)
        if (st.tiles.containsKey(key)) return null
        evict()

        val queries = timestampQuerySet
        if (queries == null) {
            withTileTexture { texture, uniform ->
                val encoder = device.createCommandEncoder()
                val pass = encoder.beginRenderPass(clearedColorPass(texture))
                try {
                    render(pass, texture)
                } finally {
                    pass.end()
                }
                device.queue.submit(arrayOf(encoder.finish()))

                writeTileUniform(uniform, tx, ty)
                val bindGroup = tileBindGroup(st.frameUniform, uniform, texture)
                st.tiles[key] = Tile(texture, uniform, bindGroup).also { it.lastUsed = frame }
            }
            return null
        }
        val (resolve, result) = createTimestampBuffers()

        withTileTexture { texture, uniform ->
            val encoder = device.createCommandEncoder()
            val pass = encoder.beginRenderPass(
                clearedColorPass(
                    texture, timestampWrites = GPUPassTimestampWrites(
                        queries, beginningOfPassWriteIndex = 0, endOfPassWriteIndex = 1
                    )
                )
            )

            try {
                render(pass, texture)
            } finally {
                pass.end()
            }

            encoder.resolveQuerySet(queries, 0, 2, resolve, 0)
            encoder.copyBufferToBuffer(resolve, 0, result, 0, 16)

            device.queue.submit(arrayOf(encoder.finish()))
            resolve.destroy()

            writeTileUniform(uniform, tx, ty)
            val bindGroup = tileBindGroup(st.frameUniform, uniform, texture)
            st.tiles[key] = Tile(texture, uniform, bindGroup).also { it.lastUsed = frame }
        }

        return measurementScope.launch { measureTileGpuTime(result) }
    }

    private suspend fun measureTileGpuTime(result: GPUBuffer) {
        try {
            if (WebGpuRenderer.deviceLost) return
            awaitPumped { result.mapAndAwait(MapMode.Read, 0, result.size) }
            if (WebGpuRenderer.deviceLost) return
            val timestamps = result.getConstMappedRange(0, 16)
            timestamps.order(ByteOrder.nativeOrder())
            val start = timestamps.getLong(0)
            val end = timestamps.getLong(8)
            result.unmap()
            if (end > start) {
            val sampleNs = (end - start).toDouble()
            avgTileGpuNs =
                if (avgTileGpuNs <= 0.0) sampleNs else avgTileGpuNs * 0.8 + sampleNs * 0.2
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Skipping GPU tile timing after device/surface loss", e)
        } finally {
            try { result.destroy() } catch (_: Exception) { }
        }
    }

    private suspend fun awaitPumped(block: suspend () -> Unit) = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1.milliseconds)
            }
        }
        try {
            block()
        } finally {
            pump.cancel()
        }
    }

    private inline fun withTileTexture(block: (GPUTexture, GPUBuffer) -> Unit) {
        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(TILE_SIZE, TILE_SIZE),
                usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding,
                format = TextureFormat.RGBA8Unorm
            )
        )
        // Holds just the tile's grid coordinate, which never changes, so it and the bind group
        // are created once here. 32 bytes rather than the 8 the shader reads, since writeBuffer
        // writes the whole scratch ByteBuffer and a write must fit in the destination.
        val uniform = device.createBuffer(
            GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        )
        try {
            block(texture, uniform)
        } catch (e: Exception) {
            texture.destroy()
            uniform.destroy()
            throw e
        }
    }

    private fun clearedColorPass(
        texture: GPUTexture, timestampWrites: GPUPassTimestampWrites? = null
    ) = GPURenderPassDescriptor(
        colorAttachments = arrayOf(
            GPURenderPassColorAttachment(
                view = texture.createView(),
                loadOp = LoadOp.Clear,
                storeOp = StoreOp.Store,
                clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
            )
        ), timestampWrites = timestampWrites
    )

    /**
     * Write [st]'s frame uniform - snapped anchor, [dst]'s size, and a clip rect - if any of them
     * actually changed since the last write. Shared by both [draw] overloads and [prewarm], which
     * each derive the same 32-byte layout from their own notion of "this grid's clip".
     */
    private fun writeFrameUniformIfChanged(
        st: PageTiles,
        dst: GPUTexture,
        snapX: Float,
        snapY: Float,
        clipL: Float,
        clipT: Float,
        clipR: Float,
        clipB: Float
    ) {
        val dstW = dst.width.toFloat()
        val dstH = dst.height.toFloat()
        if (st.writtenSnapX == snapX && st.writtenSnapY == snapY && st.writtenDstW == dstW && st.writtenDstH == dstH && st.writtenClipL == clipL && st.writtenClipT == clipT && st.writtenClipR == clipR && st.writtenClipB == clipB) return

        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(snapX)
        byteBuffer.putFloat(snapY)
        byteBuffer.putFloat(dstW)
        byteBuffer.putFloat(dstH)
        byteBuffer.putFloat(clipL)
        byteBuffer.putFloat(clipT)
        byteBuffer.putFloat(clipR)
        byteBuffer.putFloat(clipB)
        byteBuffer.flip()
        device.queue.writeBuffer(st.frameUniform, 0, byteBuffer)
        st.writtenSnapX = snapX
        st.writtenSnapY = snapY
        st.writtenDstW = dstW
        st.writtenDstH = dstH
        st.writtenClipL = clipL
        st.writtenClipT = clipT
        st.writtenClipR = clipR
        st.writtenClipB = clipB
    }

    private fun writeTileUniform(uniform: GPUBuffer, tx: Int, ty: Int) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(tx.toFloat())
        byteBuffer.putFloat(ty.toFloat())
        byteBuffer.flip()
        device.queue.writeBuffer(uniform, 0, byteBuffer)
    }

    private fun tileBindGroup(
        frameUniform: GPUBuffer, tileUniform: GPUBuffer, texture: GPUTexture
    ): GPUBindGroup = device.createBindGroup(
        GPUBindGroupDescriptor(
            layout = blitBindGroupLayout, entries = arrayOf(
                GPUBindGroupEntry(0, buffer = frameUniform),
                GPUBindGroupEntry(1, buffer = tileUniform),
                GPUBindGroupEntry(2, textureView = texture.createView()),
                GPUBindGroupEntry(3, sampler = blitSampler),
            )
        )
    )

    /**
     * Evict least-recently-used tiles down to [maxTiles], best-effort.
     *
     * Tiles used this frame or last are never touched: the pass that blitted them may not have
     * been submitted yet, and destroying a texture a recording encoder references is a
     * validation error. If everything is that fresh the cap is allowed to overshoot - the
     * wanted set is bounded by the viewport, so the overshoot is too.
     */
    private fun evict() {
        var total = 0
        for (st in pages.values) total += st.tiles.size
        if (total < maxTiles) return

        val candidates = ArrayList<Triple<PageTiles, Long, Tile>>()
        for (st in pages.values) {
            for ((k, t) in st.tiles) if (t.lastUsed < frame - 1) candidates.add(Triple(st, k, t))
        }
        candidates.sortBy { it.third.lastUsed }
        var i = 0
        while (total >= maxTiles && i < candidates.size) {
            val (st, k, t) = candidates[i]
            st.tiles.remove(k)
            t.destroy()
            total--
            i++
        }
    }

    /**
     * Free every tile. Safe from any thread; destruction runs on the render thread. Not a
     * permanent shutdown - the surface can be recreated with the same viewer state afterward, and
     * rendering simply refills the cache.
     */
    fun cleanup() {
        workerScope.coroutineContext[Job]?.children?.forEach { it.cancel() }
        workerScope.launch {
            pages.values.forEach { it.destroyAll() }
            pages.clear()
        }
    }
}

private const val BLIT_SHADER = """
const TS: f32 = $TILE_SIZE.0;

struct FrameParams {
    // Snapped screen-pixel position of the grid's centre; the tile grid hangs off it.
    snap: vec2<f32>,
    dst_size: vec2<f32>,
    // The grid's rect in screen pixels that blits are clipped to - see the class doc.
    clip: vec4<f32>,
}

struct TileParams {
    t: vec2<f32>,  // this tile's grid coordinate, fixed for the tile's lifetime
}

@group(0) @binding(0) var<uniform> frame_params: FrameParams;
@group(0) @binding(1) var<uniform> tile_params: TileParams;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );

    let pos = corners[vertex_index];

    // Clamping the corners to the clip rect shrinks the quad and its uv window in step, so
    // whatever survives is still a 1:1 texel copy. Offscreen overhang is left to NDC clipping.
    let origin = frame_params.snap + tile_params.t * TS;
    let p = clamp(origin + pos * TS, frame_params.clip.xy, frame_params.clip.zw);

    var out: VertexOutput;
    out.position = vec4<f32>(
        p.x / frame_params.dst_size.x * 2.0 - 1.0,
        1.0 - p.y / frame_params.dst_size.y * 2.0,
        0.0, 1.0
    );
    out.uv = (p - origin) / TS;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1:1 at integer positions with a nearest sampler: an exact copy of the tile's texels,
    // already premultiplied by RenderPage.
     return textureSample(src_tex, src_sampler, in.uv);
}
"""
