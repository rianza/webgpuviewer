package ca.mpreg.webgpuviewer

import ca.mpreg.webgpuviewer.log.WgvLog
import java.nio.ByteBuffer

object ImageUtil {
    private const val TAG = "WGV.Resize"

    init {
        System.loadLibrary("resize")
        WgvLog.i(TAG, "native library 'resize' loaded")
    }

    external fun resizeLinearAreaNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int
    )

    fun resize(source: ByteBuffer, width: Int, height: Int): ByteBuffer {
        WgvLog.d(TAG, "resize -> ${width}x$height (${width * height} bytes out)")
        val output = ByteBuffer.allocateDirect(width * height /* * 4ch / 2width / 2height = 1 */)
        resizeLinearAreaNative(source, output, width, height)
        return output
    }
}
