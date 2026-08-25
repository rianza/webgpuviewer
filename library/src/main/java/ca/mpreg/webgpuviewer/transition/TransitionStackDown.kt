package ca.mpreg.webgpuviewer.transition

import ca.mpreg.webgpuviewer.log.WgvLog
import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage

private const val TAG = "WGV.StackDown"

object TransitionStackDown : Transition() {
    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        WgvLog.d(TAG, "StackDown: render frac=$frac")
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        val pass = beginClearedPass(encoder, dst)
        try {
            if (frac > 0f) {
                page2.drawBackgroundColumns(pass, dst, 0f, 0f)
                blitCached(pass, cached2, 0f, 0f)
                page1.drawBackgroundColumns(pass, dst, 0f, frac)
                blitCached(pass, cached1, 0f, frac)
            } else {
                page1.drawBackgroundColumns(pass, dst, 0f, 0f)
                blitCached(pass, cached1, 0f, 0f)
                page2.drawBackgroundColumns(pass, dst, 0f, frac + 1f)
                blitCached(pass, cached2, 0f, frac + 1f)
            }
        } finally {
            pass.end()
        }
    }
}
