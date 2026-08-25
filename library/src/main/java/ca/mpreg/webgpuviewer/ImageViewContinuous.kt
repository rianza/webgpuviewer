package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuous
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState

private const val TAG = "WGV.View"

class ImageViewContinuous(context: Context, attrs: AttributeSet? = null) :
    ImageView(context, attrs) {
    override val state = ImageViewerContinuousState()

    init {
        WgvLog.i(TAG, "ImageViewContinuous created (attrs=$attrs)")
    }

    @Composable
    override fun Content() {
        WgvLog.v(TAG, "ImageViewContinuous.Content composing")
        ImageViewerContinuous(state = state)
    }
}
