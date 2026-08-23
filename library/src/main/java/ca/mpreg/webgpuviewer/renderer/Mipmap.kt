package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUOrigin3D
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import kotlinx.coroutines.yield
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.min

class Mipmap(
    val width: Int,
    val height: Int,
    val scale: Float,
    val tilesCols: Int,
    val tilesRows: Int,
    val tilesize: Int,
) {
    companion object {
        private val device get() = WebGpuRenderer.device

        /**
         * Rough size of a single `writeTexture` call, in bytes.
         *
         * Big enough that per-call overhead stays noise, small enough that one copy fits in the
         * slack of a frame. A whole 2000x3000 page in one call is ~24MB of memcpy on the render
         * thread, which is several frames' worth.
         */
        private const val UPLOAD_CHUNK_BYTES = 1 shl 20

        /**
         * Build a mipmap level from [pixels] and upload it.
         *
         * Suspends between chunks, so it must run outside the render mutex (see
         * [WebGpuRenderer.onDispatcher]) for the yields to be worth anything. The level is only
         * returned once every chunk has landed, so no caller can sample a half-filled texture.
         */
        suspend fun create(
            pixels: ByteBuffer, width: Int, height: Int, scale: Float, tilesize: Int
        ): Mipmap {
            val mipmap = Mipmap(
                width = width,
                height = height,
                scale = scale,
                tilesCols = ceil(width.toFloat() / tilesize).toInt(),
                tilesRows = ceil(height.toFloat() / tilesize).toInt(),
                tilesize = tilesize,
            )
            try {
                mipmap.upload(pixels)
            } catch (e: Throwable) {
                // Yielding makes the upload cancellable, so a half-built level can now exist.
                // Free whatever landed before rethrowing - the caller never sees this instance
                // and so can't free it itself.
                mipmap.cleanup()
                throw e
            }
            return mipmap
        }
    }

    /** Allocate the tile textures and copy [pixels] into them a chunk at a time. */
    private suspend fun upload(pixels: ByteBuffer) {
        for (r in 0 until tilesRows) {
            val tileHeight = min((r + 1) * tilesize, height) - (r * tilesize)
            val y = r * tilesize
            for (c in 0 until tilesCols) {
                val x = c * tilesize
                val tileWidth = min((c + 1) * tilesize, width) - (c * tilesize)

                Log.i("Renderer", "Create tile $c $r $tileWidth $tileHeight $x $y")

                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        size = GPUExtent3D(tileWidth, tileHeight),
                        format = TextureFormat.RGBA8Unorm,
                        usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment,
                    )
                )

                val dstPitch = alignedPitch(tileWidth)
                val rowsPerChunk = (UPLOAD_CHUNK_BYTES / dstPitch).coerceAtLeast(1)
                val stage = ByteBuffer.allocateDirect(dstPitch * rowsPerChunk)

                var row = 0
                while (row < tileHeight) {
                    val rows = min(rowsPerChunk, tileHeight - row)

                    uploadTileRows(pixels, x, y, tileWidth, row, rows, texture, stage, dstPitch)

                    row += rows
                    yield()
                }

                textures.add(texture)
                textureViews.add(texture.createView())
            }
        }

        for (r in 0 until 2) {
            val row = r.coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + c.coerceAtMost(tilesCols - 1)
                tiles.add(textures[i])
                tileViews.add(textureViews[i])
            }
        }

        if (tilesCols <= 2 && tilesRows <= 2) {
            cachedQuad = Quad(tiles, tileViews, 0, 0)
        }
    }

    var textures: MutableList<GPUTexture> = mutableListOf()
    private var textureViews: MutableList<GPUTextureView> = mutableListOf()
    private var tiles: MutableList<GPUTexture> = mutableListOf()
    private var tileViews: MutableList<GPUTextureView> = mutableListOf()

    private var cachedQuad: Quad? = null

    constructor(texture: GPUTexture, scale: Float, tilesize: Int) : this(
        texture.width, texture.height, scale, 1, 1, tilesize
    ) {
        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    constructor(width: Int, height: Int) : this(width, height, 1f, 1, 1, 4096) {
        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(width, height),
                format = TextureFormat.RGBA8Unorm,
                usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment or TextureUsage.StorageBinding,
            )
        )

        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    internal fun cleanup() {
        cachedQuad = null
        lastQuad = null
        lastQuadTX = -1
        lastQuadTY = -1
        tileUniforms?.forEach { it?.destroy() }
        tileUniforms = null
        textureViews.clear()
        tileViews.clear()
        textures.forEach { tex -> tex.destroy() }
        textures.clear()
        tiles.clear()
    }

    /** Row pitch WebGPU demands for a multi-row write: bytesPerRow must be a multiple of 256. */
    private fun alignedPitch(tileWidth: Int): Int =
        (tileWidth * Int.SIZE_BYTES + 255) / 256 * 256

    /**
     * Copy [rows] rows of the ([x], [y])-anchored, [tileWidth]-wide subrect of [pixels] through
     * [stage] into [texture], starting at y = [dstRow]. The image's own row stride almost never
     * satisfies WebGPU's bytesPerRow-multiple-of-256 rule for multi-row writes (any page width
     * not divisible by 64 breaks it), so every source row is bulk-copied once into [stage],
     * whose [dstPitch] is aligned, and the upload reads from there.
     */
    private fun uploadTileRows(
        pixels: ByteBuffer, x: Int, y: Int, tileWidth: Int,
        dstRow: Int, rows: Int, texture: GPUTexture, stage: ByteBuffer, dstPitch: Int,
    ) {
        val srcRowBytes = tileWidth * Int.SIZE_BYTES
        val srcStride = width * Int.SIZE_BYTES
        // Long arithmetic: y * width overflows Int well before the byte offset does on a large
        // page - though every position used here fits back into an Int.
        var srcOffset = ((y + dstRow).toLong() * width + x) * Int.SIZE_BYTES
        val src = pixels.duplicate()
        stage.clear()
        for (r in 0 until rows) {
            src.position(srcOffset.toInt())
            src.limit(srcOffset.toInt() + srcRowBytes)
            stage.position(r * dstPitch)
            stage.put(src)
            srcOffset += srcStride
        }
        stage.position(0)

        device.queue.writeTexture(
            dataLayout = GPUTexelCopyBufferLayout(offset = 0L, bytesPerRow = dstPitch, rowsPerImage = rows),
            data = stage,
            destination = GPUTexelCopyTextureInfo(texture = texture, origin = GPUOrigin3D(y = dstRow)),
            writeSize = GPUExtent3D(tileWidth, rows),
        )
    }

    fun update(pixels: ByteBuffer) {
        var i = 0

        for (r in 0 until tilesRows) {
            val tileHeight = min((r + 1) * tilesize, height) - (r * tilesize)
            val y = r * tilesize
            for (c in 0 until tilesCols) {
                val x = c * tilesize
                val tileWidth = min((c + 1) * tilesize, width) - (c * tilesize)

                Log.d("Renderer", "Update tile $c $r")
                val texture = textures[i++]
                val dstPitch = alignedPitch(tileWidth)
                val rowsPerChunk = (UPLOAD_CHUNK_BYTES / dstPitch).coerceAtLeast(1)
                val stage = ByteBuffer.allocateDirect(dstPitch * rowsPerChunk)

                var row = 0
                while (row < tileHeight) {
                    val rows = min(rowsPerChunk, tileHeight - row)
                    uploadTileRows(pixels, x, y, tileWidth, row, rows, texture, stage, dstPitch)
                    row += rows
                }
            }
        }
    }

    class Quad(
        val tiles: List<GPUTexture>, val tileViews: List<GPUTextureView>, val x: Int, val y: Int
    )

    /**
     * One tile overlapping a queried rect, at its own pixel offset within the mipmap. [uniform] is
     * its own persistent placement buffer - see [tileUniforms] for why it needs one of its own.
     */
    class TileRect(
        val texture: GPUTexture,
        val view: GPUTextureView,
        val x: Int,
        val y: Int,
        val uniform: GPUBuffer
    )

    /**
     * One small uniform buffer per physical tile, created on first use and rewritten every frame
     * that tile is drawn - never shared between tiles. `writeBuffer` calls all land before any
     * command buffer submitted afterwards executes, so a shared buffer would let the last of
     * several tiles drawn in one frame win for all of them; one buffer per tile has no such race.
     */
    private var tileUniforms: Array<GPUBuffer?>? = null

    private fun tileUniformFor(index: Int): GPUBuffer {
        val arr = tileUniforms ?: arrayOfNulls<GPUBuffer>(textures.size).also { tileUniforms = it }
        return arr[index] ?: device.createBuffer(
            GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        ).also { arr[index] = it }
    }

    /**
     * Every tile overlapping [left]..[right] by [top]..[bottom] (mipmap pixels) - unlike [getQuad]
     * which always hands back exactly 2x2, letting a caller draw each tile separately instead of
     * needing everything to fit one window. See [Image.prepareTilesForRender].
     */
    fun tilesInRect(left: Float, top: Float, right: Float, bottom: Float): List<TileRect> {
        val l = left.coerceIn(0f, width.toFloat())
        val t = top.coerceIn(0f, height.toFloat())
        val r = right.coerceIn(0f, width.toFloat())
        val b = bottom.coerceIn(0f, height.toFloat())
        if (l >= r || t >= b) return emptyList()

        val c0 = (l / tilesize).toInt().coerceIn(0, tilesCols - 1)
        val c1 = ((r - 1f) / tilesize).toInt().coerceIn(0, tilesCols - 1)
        val row0 = (t / tilesize).toInt().coerceIn(0, tilesRows - 1)
        val row1 = ((b - 1f) / tilesize).toInt().coerceIn(0, tilesRows - 1)

        val result = ArrayList<TileRect>((c1 - c0 + 1) * (row1 - row0 + 1))
        for (row in row0..row1) {
            for (col in c0..c1) {
                val idx = row * tilesCols + col
                result.add(
                    TileRect(
                        textures[idx],
                        textureViews[idx],
                        col * tilesize,
                        row * tilesize,
                        tileUniformFor(idx)
                    )
                )
            }
        }
        return result
    }

    // Cache the last computed quad to avoid allocations when panning within the same tile region
    private var lastQuadTX = -1
    private var lastQuadTY = -1
    private var lastQuad: Quad? = null

    fun getQuad(centerX: Int, centerY: Int): Quad {
        cachedQuad?.let { return it }

        val cX = centerX.toFloat()
        val cY = centerY.toFloat()

        val c = (cX / tilesize).toInt()
        val tX = when {
            c >= tilesCols - 1 -> tilesCols - 2
            c <= 0 -> 0
            else -> {
                val xCenterRight = if (c + 1 == tilesCols - 1) {
                    ((tilesCols - 1) * tilesize + width) * 0.5
                } else {
                    (c + 1.5) * tilesize
                }

                if (cX - (c - 0.5) * tilesize < xCenterRight - cX) c - 1 else c
            }
        }.coerceIn(0, tilesCols - 1)

        val r = (cY / tilesize).toInt()
        val tY = when {
            r >= tilesRows - 1 -> tilesRows - 2
            r <= 0 -> 0
            else -> {
                val yCenterBottom = if (r + 1 == tilesRows - 1) {
                    ((tilesRows - 1) * tilesize + height) * 0.5
                } else {
                    (r + 1.5) * tilesize
                }

                if (cY - (r - 0.5) * tilesize < yCenterBottom - cY) r - 1 else r
            }
        }.coerceIn(0, tilesRows - 1)

        // Return cached quad if tile region hasn't changed
        lastQuad?.let { cached ->
            if (lastQuadTX == tX && lastQuadTY == tY) {
                return cached
            }
        }

        // Build a 2x2 quad from the tile grid at (tX, tY) without mutableList overhead
        val r0 = (tY).coerceAtMost(tilesRows - 1) * tilesCols
        val r1 = (tY + 1).coerceAtMost(tilesRows - 1) * tilesCols
        val c0 = (tX).coerceAtMost(tilesCols - 1)
        val c1 = (tX + 1).coerceAtMost(tilesCols - 1)

        val t00 = textures[r0 + c0]
        val v00 = textureViews[r0 + c0]
        val t01 = textures[r0 + c1]
        val v01 = textureViews[r0 + c1]
        val t10 = textures[r1 + c0]
        val v10 = textureViews[r1 + c0]
        val t11 = textures[r1 + c1]
        val v11 = textureViews[r1 + c1]

        val quad = Quad(
            listOf(t00, t01, t10, t11),
            listOf(v00, v01, v10, v11),
            tX * t00.width,
            tY * t00.height
        )
        lastQuadTX = tX
        lastQuadTY = tY
        lastQuad = quad
        return quad
    }
}
