package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.solveImagePlacement
import kotlinx.coroutines.launch

private const val TAG = "WGV.Continuous"

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    companion object {
        private const val MAX_VISIBLE_PAGES = 2
    }

    var scale = 1f

    var offsetX = 0f

    /**
     * True while [ImageViewerContinuous]'s gestures are actively driving zoom (pinch, drag, fling,
     * snap-back). Gates every visible page's tile grid here the same way [ImagePage.isScaleAnimating]
     * gates the paged viewer's.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    /**
     * True while a plain (non-zoom) fling is actively scrolling. Generating a filtered tile is
     * real GPU work sharing the render thread with the frame itself, so doing it while the camera
     * is moving fast under its own momentum both wastes the work (the content is about to scroll
     * back out of view) and is a real source of visible frame lag. Combined with
     * [isScaleAnimating] wherever a caller needs "don't generate right now" - kept separate here
     * since the two are driven by different gestures and one may be true without the other.
     */
    @Volatile
    var isFlinging: Boolean = false

    private val scrollLock = Any()
    var scrollY = 0f

    /**
     * Layout height of [page] in screen pixels.
     *
     * Measured the same way whether or not the page has decoded yet: a placeholder that carries
     * the real aspect ratio has to occupy exactly the space its decoded self will, or the pages
     * below it jump the moment it decodes. The guard is only for pages with no width to fit
     * against (all images null), which have no aspect ratio to scale by.
     *
     * Only an [ImagePage.Images] page is fit to the viewer's full width this way - that's this
     * mode's whole reading convention for raster content. A [ImagePage.Render] page's width/height
     * are an author's deliberate choice (see e.g. a square chapter-transition placeholder sized to
     * whichever screen dimension is smaller), not something to stretch to fill the width, so it's
     * reserved/drawn at its own native size instead - see the matching pageScale in
     * [renderSnapshot].
     */
    fun getPageHeight(page: ImagePage): Float {
        if (page !is ImagePage.Images) return page.height.toFloat()
        val pageWidth = page.width
        if (pageWidth <= 0) return page.height.toFloat()
        return page.height * (width.toFloat() / pageWidth)
    }

    var currentPageHeight: Float? = null

    /**
     * Document-space top of whatever page currently sits at [scrollY] == 0, in screen pixels at
     * zoom 1. The only state the continuous coordinate space needs to persist across
     * frames: every other visible page's position is re-derived fresh each frame from this one
     * value (see [captureRenderState]'s walk), rather than stored per page - a page's identity
     * isn't stable across a decode (the app hands over a new object), so anything kept on the
     * page itself would be silently lost exactly when a placeholder corrects to its real height.
     * Updated only here, in [scrollBy], using the height of whichever page is actually being
     * crossed.
     */
    private var anchorDocY = 0f

    /**
     * Scroll by [deltaPixels], moving the current page as many times as the delta covers.
     *
     * A single fling frame can cross more than one page when pages are short, so both walks
     * loop. Each also stops on a zero-height page, which would otherwise never advance the
     * position and spin here forever.
     */
    fun scrollBy(deltaPixels: Float) {
        synchronized(scrollLock) {
            getPage(0) ?: return

            val before = scrollY
            scrollY += deltaPixels

            // Backwards, while the position sits above the top of the current page.
            while (scrollY < 0) {
                if (getPage(-1) == null) {
                    scrollY = 0f
                    break
                }
                onPageChange?.invoke(-1)
                val newPage = getPage(0) ?: return
                val newHeight = getPageHeight(newPage)
                anchorDocY -= newHeight
                currentPageHeight = newHeight
                if (newHeight <= 0f) break
                scrollY += newHeight
            }

            // Forwards, while it sits past the bottom of it. Stops at the last page rather than
            // stepping off the end, which would leave the position short instead of clamping.
            while (true) {
                val page = getPage(0) ?: return
                val pageHeight = getPageHeight(page)
                if (scrollY <= pageHeight || pageHeight <= 0f) break
                if (getPage(1) == null) {
                    scrollY = pageHeight
                    break
                }
                onPageChange?.invoke(1)
                anchorDocY += pageHeight
                val newPage = getPage(0) ?: return
                currentPageHeight = getPageHeight(newPage)
                scrollY -= pageHeight
            }

            if (getPage(1) == null) {
                val page = getPage(0) ?: return
                scrollY = scrollY.coerceAtMost(getPageHeight(page))
            }
            WgvLog.throttled(TAG, key = "scrollBy", intervalMs = 250L) {
                "scrollBy: $before -> $scrollY (delta=$deltaPixels, scale=$scale)"
            }
        }
    }

    fun animateScroll(deltaPixels: Float) {
        WgvLog.d(TAG, "animateScroll(deltaPixels=$deltaPixels)")
        animationJob?.cancel()
        animationJob = scope?.launch {
            var lastValue = 0f
            animate(
                0f, deltaPixels, animationSpec = spring(
                    stiffness = Spring.StiffnessMediumLow,
                    visibilityThreshold = 0.002f
                )
            ) { value, _ ->
                scrollBy(value - lastValue)
                lastValue = value
                invalidate()
            }
        }
    }

    /** One page visible this frame, with the document-space top [captureRenderState] found it at. */
    private class VisiblePage(val page: ImagePage, val docTop: Float, val pageHeight: Float)

    private class ContinuousRenderSnapshot(
        val pages: List<VisiblePage>,
        val scale: Float,
        val offsetX: Float,
        /** Document position (see [anchorDocY]) currently at the viewport's vertical centre. */
        val cameraDocY: Float,
        /** [isScaleAnimating] or [isFlinging] - either means "don't generate tiles right now". */
        val suppressGeneration: Boolean,
    )

    override fun captureRenderState(): Any = synchronized(scrollLock) {
        val screenH = height.toFloat()

        val y0 = getPage(0)?.let { page ->
            val pageHeight = getPageHeight(page)
            if (currentPageHeight != pageHeight && scrollY > 0) {
                currentPageHeight?.let { h -> scrollY -= pageHeight - h }
                currentPageHeight = pageHeight
            }
            -scrollY
        } ?: 0f

        // Document position at the viewport's centre - the point both the fast path and
        // TileRenderer's continuous overloads zoom around, so they always agree on where a page
        // belongs.
        val cameraDocY = anchorDocY - y0 + 0.5f * screenH

        val pages = mutableListOf<VisiblePage>()

        // Visible band in unscaled page space. Zoom is centered on the screen, so the
        // viewport covers screenH / scale of page space around the screen center.
        val visTop = 0.5f * screenH - screenH / (2f * scale)
        // +1 tile of margin, matching TileRenderer's own prefetch ring, so a boundary tile just
        // past the viewport has its page already discovered.
        val visBot = 0.5f * screenH + screenH / (2f * scale) + TILE_SIZE / scale

        // Backward: pages above page 0, needed once zoomed out enough that visTop goes negative -
        // i.e. the visible band reaches above where page 0 itself starts. Mirrors the forward
        // walk below, just toward negative indices.
        var yTop = y0
        var iBack = -1
        var docTopBack = anchorDocY
        val backPages = mutableListOf<VisiblePage>()
        while (yTop > visTop && iBack >= -MAX_VISIBLE_PAGES) {
            val page = getPage(iBack) ?: break
            val pageHeight = getPageHeight(page)
            docTopBack -= pageHeight
            yTop -= pageHeight
            if (page.isDecoded) {
                backPages.add(VisiblePage(page, docTopBack, pageHeight))
            }
            if (pageHeight <= 0f) break
            iBack--
        }
        pages.addAll(backPages.asReversed())

        // Walk forward until the viewport (plus margin) is covered or MAX_VISIBLE_PAGES
        // is reached, whichever comes first - zoomed out far enough (or with short enough pages),
        // the document-space bound alone would keep walking past it.
        // Purely local: nothing is written back to a page, so only [anchorDocY] needs to survive
        // across frames for this to stay correct.
        var y = y0
        var i = 0
        var docTop = anchorDocY
        var prevHeight = 0f
        var hasPrev = false
        while (y < visBot && i <= MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: break
            // Anchor to the previous page in this walk, never frozen: an undecoded page's height
            // is a guess, so re-deriving this fresh every frame self-corrects once it decodes.
            if (hasPrev) docTop += prevHeight
            hasPrev = true
            val pageHeight = getPageHeight(page)

            if (y + pageHeight > visTop && page.isDecoded) {
                pages.add(VisiblePage(page, docTop, pageHeight))
            }

            // A zero-height page never advances y, so stop rather than ask for pages forever.
            if (pageHeight <= 0f) break

            prevHeight = pageHeight
            y += pageHeight
            i++
        }

        WgvLog.throttled(TAG, key = "capture", intervalMs = 500L) {
            "captureRenderState: ${pages.size} visible page(s), scale=$scale, " +
                "cameraDocY=$cameraDocY, scrollY=$scrollY, " +
                "suppressGen=${isScaleAnimating || isFlinging}"
        }
        ContinuousRenderSnapshot(pages, scale, offsetX, cameraDocY, isScaleAnimating || isFlinging)
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        tiles.newFrame()
        if (s.pages.isEmpty()) {
            WgvLog.v(TAG, "renderSnapshot: no visible pages")
            return
        }
        WgvLog.throttled(TAG, key = "renderSnapshot", intervalMs = 500L) {
            "renderSnapshot: ${s.pages.size} page(s), scale=${s.scale}"
        }

        // Images pages batch into one shared pass (they never overlap vertically, so one clear
        // plus one draw per image writes each pixel once). A Render page (ImagePage.Render, e.g.
        // a loading placeholder) can't join that batch - it has no image/tile to draw, only its
        // own render() - so it draws afterward via its own renderLoaded call instead.
        // renderLoaded (unlike renderWith) loads rather than clears its pass, since this texture
        // is shared with every other visible page and clearing it would blank them too - which
        // relies on something having cleared the texture first. The Images batch's renderPass
        // does that whenever there is one; if every visible page turns out to be a Render page,
        // [ca.mpreg.webgpuviewer.draw.clear] does it instead so a Render page never has to paint
        // over stale content from prior frames.
        val imagePages = s.pages.filter { it.page is ImagePage.Images }
        val renderPages = s.pages.filter { it.page !is ImagePage.Images }

        val dstW = texture.width.toFloat()
        val dstH = texture.height.toFloat()
        // Screen position of document space's origin - mirrors TileRenderer's continuous anchor
        // exactly, so the fast path, tile cache, and Render pages below all agree on placement.
        val anchorX = dstW / 2f + s.scale * (s.offsetX * dstW + WebGpuRenderer.offsetX * dstW)
        val anchorY =
            dstH / 2f - s.scale * s.cameraDocY + s.scale * WebGpuRenderer.offsetY * dstH

        if (imagePages.isNotEmpty()) {
            renderPass(encoder, texture) { pass ->
                imagePages.forEach { vp ->
                    val page = vp.page as ImagePage.Images
                    // The snapshot was captured on the main thread; the page can have been
                    // evicted since, in which case its images' buffers are gone and drawing one
                    // throws.
                    if (page.destroyed || page.images.all { it == null } || page.width <= 0) return@forEach

                    val pageScale = dstW / page.width

                    // Tiles draw first, marking every pixel they cover in the stencil buffer
                    // tiles.draw() writes to; the sampler shader below then only shades what's
                    // left uncovered - skipped entirely once tiles alone cover this page.
                    // Animated pages never get tiles (they swap images every frame), so both
                    // calls are skipped outright rather than letting them no-op/report false
                    // every frame.
                    val covered = if (page.isAnimated) {
                        false
                    } else {
                        tiles.draw(
                            pass,
                            page,
                            texture,
                            s.cameraDocY,
                            vp.docTop,
                            s.offsetX,
                            s.scale,
                            s.suppressGeneration
                        )
                        tiles.isFullyCovered(
                            page,
                            texture,
                            s.cameraDocY,
                            vp.docTop,
                            s.offsetX,
                            s.scale,
                            s.suppressGeneration
                        )
                    }
                    if (!covered) {
                        val imageScale = pageScale * s.scale
                        page.images.forEach { image ->
                            image ?: return@forEach
                            if (image.mipmaps.isEmpty()) return@forEach
                            val docCenterX = pageScale * (image.spreadOffsetX + image.x)
                            val docCenterY =
                                vp.docTop + 0.5f * vp.pageHeight + pageScale * image.y
                            val targetX = anchorX + s.scale * docCenterX
                            val targetY = anchorY + s.scale * docCenterY
                            val (x, y) = solveImagePlacement(
                                targetX,
                                targetY,
                                imageScale,
                                image,
                                dstW,
                                dstH
                            )
                            // Content not worth linear-light correctness
                            // (ImagePage.Images.highQuality) gets the plain sampler - it never
                            // reaches the tile cache either. Animated pages are never highQuality
                            // but always want the fast sampler regardless, since they swap images
                            // every frame. Both are stencil-tested against the tile draw above,
                            // skipping pixels it already covered.
                            if (page.isAnimated || page.highQuality) {
                                RenderPage.renderFast(pass, image, texture, x, y, imageScale)
                            } else {
                                RenderPage.renderFast(
                                    pass,
                                    image,
                                    texture,
                                    x,
                                    y,
                                    imageScale,
                                    linear = false
                                )
                            }
                        }
                    }
                }
            }
        } else if (renderPages.isNotEmpty()) {
            Draw.clear(encoder, texture, 0)
        }

        renderPages.forEach { vp ->
            // Only ImagePage.Images overrides isDecoded to anything other than this fixed
            // "Render (or a subclass) with drawable content" default - the isDecoded filter in
            // captureRenderState already excludes anything else (e.g. Dummy).
            val page = vp.page as ImagePage.Render
            if (page.destroyed) return@forEach

            // Render's own render(dst, x, y, scale) convention has no fit-to-width factor to
            // undo - x/y/scale there are already fractions of dst (screen) size, not of this
            // page's own declared width/height (see getPageHeight's doc: unlike an Images page,
            // this page's size is never stretched to the viewer's width). So the only screen
            // scale in play is the pinch-zoom (s.scale) times whatever this page's own scale is -
            // folding in a dstW/page.width factor here (as an Images page's placement does) would
            // scale its content by that ratio for no reason, which is exactly what "overdrawing
            // its size" looked like. page.x/page.y are left out of the position for the same
            // reason: they're in that dst-fraction unit, not vp.docTop's doc-space-pixel one, so
            // they can't be combined with it - harmless since a locked-scale page like
            // TransitionPage never has them set to anything but 0 anyway.
            val renderScale = s.scale * page.scale
            val targetX = anchorX
            val targetY = anchorY + s.scale * (vp.docTop + 0.5f * vp.pageHeight)

            val x = (targetX - dstW / 2f) / (renderScale * dstW)
            val y = (targetY - dstH / 2f) / (renderScale * dstH)
            page.renderLoaded(encoder, x, y, renderScale, texture)
        }
    }
}
