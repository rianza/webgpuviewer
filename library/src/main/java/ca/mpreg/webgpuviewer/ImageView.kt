package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.viewer.ImageViewer
import ca.mpreg.webgpuviewer.viewer.ImageViewerState

private const val TAG = "WGV.View"

open class ImageView(
    context: Context,
    attrs: AttributeSet? = null,
    isVertical: Boolean = false,
    isReversed: Boolean = false,
) : AbstractComposeView(context, attrs) {
    constructor(context: Context, attrs: AttributeSet? = null) : this(
        context, attrs, context.obtainStyledAttributes(
            attrs, intArrayOf(android.R.attr.orientation)
        ).let {
            val orientation = it.getInt(0, 0)
            it.recycle()
            orientation == 1
        })

    open val state: ImageViewerState = ImageViewerState(isVertical, isReversed)

    init {
        WgvLog.i(TAG, "ImageView created (vertical=$isVertical, reversed=$isReversed, attrs=$attrs)")
    }

    @Composable
    override fun Content() {
        WgvLog.v(TAG, "ImageView.Content composing")
        ImageViewer(state = state)
    }
}
