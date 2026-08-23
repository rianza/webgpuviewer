package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import ca.mpreg.webgpuviewer.ImageUtil
import ca.mpreg.webgpuviewer.Trim
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.round

const val BUFFER_SIZE = 96L

class Image private constructor(
    val width: Int,
    val height: Int,
    var x: Float = 0f,
    var y: Float = 0f,
    var backgroundColor: Int = 0xFF000000.toInt()
) {

    /**
     * Trim bounds detected from image content, or null if not trimmed.
     */
    var trim: Rect? = null

    /**
     * Position of this image in a spread.
     */
    enum class Position {
        LEFT, RIGHT, SINGLE
    }

    var position: Position = Position.SINGLE

    /**
     * Pixel shift of a LEFT/RIGHT spread image from its page's anchor (0 for SINGLE) - the seam
     * sits at the anchor, so each side extends outward by half its own width. Callers scale this
     * into their own coordinate space rather than recomputing the formula.
     */
    val spreadOffsetX: Float
        get() = when (position) {
            Position.LEFT -> -0.5f * width
            Position.RIGHT -> 0.5f * width
            Position.SINGLE -> 0f
        }

    companion object {
        suspend operator fun invoke(
            pixels: ByteBuffer, width: Int, height: Int,
            createMipMaps: Boolean = true,
            trimColors: List<FloatArray>? = null,
            trimThreshold: Float = 0.05f,
            backgroundColor: Int? = null,
        ): Image {
            require(width > 0 && height > 0) { "Image dimensions must be positive" }
            require(trimColors == null || trimColors.all { it.size >= 3 }) {
                "each trimColor must have at least 3 elements [r, g, b]"
            }

            val image = Image(width, height)

            // Runs on a background dispatcher rather than as compute shaders - the GPU versions
            // would park on a buffer readback while holding the render thread, stalling every
            // queued frame (a stutter each time a page decodes).
            withContext(Dispatchers.Default) {
                var backgroundFromTrim = false

                val trimWith = trimColors?.takeIf { it.isNotEmpty() }
                if (trimWith != null) {
                    // Find trim for each color and pick the smallest rect
                    val rects = Trim.findAllCpu(pixels, width, height, trimWith, trimThreshold)
                    val best =
                        trimWith.zip(rects).minByOrNull { it.second.width() * it.second.height() }

                    if (best != null) {
                        image.trim = best.second
                        // Set background color from the winning trim color
                        if (backgroundColor == null) {
                            val c = best.first
                            image.backgroundColor =
                                0xFF000000.toInt() or ((c[0] * 255).toInt() shl 16) or ((c[1] * 255).toInt() shl 8) or (c[2] * 255).toInt()
                            backgroundFromTrim = true
                        }
                    }
                }

                // Probing the edges is only worth a pass when neither the caller nor trim has
                // already named a background colour.
                if (backgroundColor != null) {
                    image.backgroundColor = backgroundColor
                } else if (!backgroundFromTrim) {
                    image.backgroundColor =
                        Trim.detectBackgroundCpu(pixels, width, height, trimThreshold)
                }
            }

            val tilesize = 2048

            data class MipmapData(
                val pixels: ByteBuffer, val w: Int, val h: Int, val scale: Float
            )

            val mipmapDataList = mutableListOf<MipmapData>()
            mipmapDataList.add(MipmapData(pixels, width, height, 1f))

            if (createMipMaps) {
                var currentPixels = pixels
                var textureWidth = width
                var textureHeight = height
                var scale = 1f

                while (width * scale > tilesize || height * scale > tilesize) {
                    scale /= 2
                    val newWidth = floor(width * scale).toInt()
                    val newHeight = floor(height * scale).toInt()
                    Log.d("Renderer", "Create mipmap using CPU ${scale} ${newWidth} ${newHeight}")

                    currentPixels = withContext(Dispatchers.Default) {
                        ImageUtil.resize(currentPixels, textureWidth, textureHeight)
                    }
                    mipmapDataList.add(MipmapData(currentPixels, newWidth, newHeight, scale))
                    textureWidth = newWidth
                    textureHeight = newHeight
                }
            }

            // No render mutex: Mipmap.create yields between upload chunks so queued frames get
            // the thread back. Safe since the image isn't reachable from any page yet.
            WebGpuRenderer.onDispatcher { device ->
                try {
                    for (data in mipmapDataList) {
                        image.mipmaps.add(
                            Mipmap.create(data.pixels, data.w, data.h, data.scale, tilesize)
                        )
                    }
                } catch (e: Exception) {
                    Log.e("Renderer", "Error creating image", e)
                    image.mipmaps.forEach { it.cleanup() }
                    image.mipmaps.clear()
                    throw e
                }
            }

            return image
        }

        suspend operator fun invoke(width: Int, height: Int): Image {
            return Image(width, height).apply {
                WebGpuRenderer.withContext { _ ->
                    try {
                        mipmaps.add(Mipmap(width, height))
                    } catch (e: Exception) {
                        Log.e("Renderer", "Error creating drawable image", e)
                        throw e
                    }
                }
            }
        }
    }

    private var _buffer: GPUBuffer? = WebGpuRenderer.device.createBuffer(
        GPUBufferDescriptor(size = BUFFER_SIZE, usage = BufferUsage.CopyDst or BufferUsage.Uniform)
    ).also { Log.d("Renderer", "Image uniform buffer created (${BUFFER_SIZE} bytes)") }

    val buffer: GPUBuffer
        get() = _buffer ?: error("Image buffer accessed after cleanup")

    val mipmaps: MutableList<Mipmap> = mutableListOf()

    internal fun cleanup() {
        mipmaps.forEach { it.cleanup() }
        mipmaps.clear()
        _buffer?.destroy()
        Log.d("Renderer", "Image uniform buffer destroyed")
        _buffer = null
    }

    /**
     * Where this image's full extent lands in [dst], as normalised (x1, y1, x2, y2) surface
     * coordinates - the same placement [prepareForRender] resolves to, but without going through
     * a mip level or [Mipmap.getQuad]. For callers that only want geometry (a background rect,
     * [ca.mpreg.webgpuviewer.transition.Transition.pageRect]) with no reason to touch mip/tile
     * selection.
     */
    fun placement(dst: GPUTexture, x: Float, y: Float, scale: Float): FloatArray {
        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY
        val x1 = 0.5f + scale * (adjustedX - 0.5f * width / dst.width)
        val y1 = 0.5f + scale * (adjustedY - 0.5f * height / dst.height)
        return floatArrayOf(
            x1, y1, x1 + scale * width / dst.width, y1 + scale * height / dst.height
        )
    }

    class MipMapForDraw(
        val mipmap: Mipmap, val quad: Mipmap.Quad, val x: Float, val y: Float, val scale: Float
    )

    fun prepareForRender(dst: GPUTexture, x: Float, y: Float, scale: Float): MipMapForDraw? {
        if (mipmaps.isEmpty()) return null

        var level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        // Scale alone isn't enough: getQuad only promises half a tile either side of the view
        // centre, so the viewport must fit in one tile's texels. A <=2x2 grid binds in one go
        // regardless, so only larger ones need checking.
        while (level < mipmaps.size - 1) {
            val m = mipmaps[level]
            if (m.tilesCols <= 2 && m.tilesRows <= 2) break

            // Source texels the viewport covers at this level.
            val visibleW = dst.width * m.scale / scale
            val visibleH = dst.height * m.scale / scale
            if (visibleW <= m.tilesize && visibleH <= m.tilesize) break

            level++
        }

        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        // View centre in this level's pixels: scale the level-0 offset by mipmap.scale before
        // adding the level's half-size, or the window lands up to 2^level too far out.
        val vx = round(-adjustedX * dst.width * mipmap.scale + mipmap.width / 2).toInt()
        val vy = round(-adjustedY * dst.height * mipmap.scale + mipmap.height / 2).toInt()

        val quad = mipmap.getQuad(vx, vy)

        return MipMapForDraw(
            mipmap,
            quad,
            (0.5f / scale + adjustedX) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width,
            (0.5f / scale + adjustedY) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height,
            scale / mipmap.scale
        )
    }

    /** One physical tile, already placed for a single draw call - see [prepareTilesForRender]. */
    class TileForDraw(
        val texture: GPUTexture,
        val view: GPUTextureView,
        val uniform: GPUBuffer,
        val x: Float,
        val y: Float,
        val scale: Float
    )

    /**
     * Every tile needed to cover the current viewport, each already placed for its own draw call
     * - the fast/plain paths' answer to [prepareForRender]'s fixed one-window quad, which can
     * silently drop content once the viewport needs more than that window covers. No coarse-level
     * guard is needed here since any viewport is just whichever tiles it happens to overlap.
     */
    fun prepareTilesForRender(
        dst: GPUTexture, x: Float, y: Float, scale: Float
    ): List<TileForDraw> {
        if (mipmaps.isEmpty()) return emptyList()

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)
        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        // Same view-centre derivation as prepareForRender's vx/vy, kept unrounded since this is
        // now just a rect query rather than a single discrete window pick.
        val cx = -adjustedX * dst.width * mipmap.scale + mipmap.width / 2f
        val cy = -adjustedY * dst.height * mipmap.scale + mipmap.height / 2f
        val halfW = dst.width * mipmap.scale / (2f * scale)
        val halfH = dst.height * mipmap.scale / (2f * scale)

        return mipmap.tilesInRect(cx - halfW, cy - halfH, cx + halfW, cy + halfH).map { tile ->
            // Same reconstruction prepareForRender uses for quad.x/quad.y, evaluated at this
            // tile's own offset instead - the formula was already general, it just happened to
            // only ever be evaluated at one window's offset before.
            TileForDraw(
                tile.texture,
                tile.view,
                tile.uniform,
                (0.5f / scale + adjustedX) * mipmap.scale + (tile.x - 0.5f * mipmap.width) / dst.width,
                (0.5f / scale + adjustedY) * mipmap.scale + (tile.y - 0.5f * mipmap.height) / dst.height,
                scale / mipmap.scale
            )
        }
    }
}
