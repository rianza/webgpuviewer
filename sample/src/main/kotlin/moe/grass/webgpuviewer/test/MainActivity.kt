package ca.mpreg.webgpuviewer.test

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ca.mpreg.imagedecoder.ImageDecoder
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.viewer.ImagePage
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.test.databinding.MainActivityBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "WGV.Sample"

class MainActivity : AppCompatActivity() {
    private lateinit var binding: MainActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i(TAG, "onCreate")

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val width = resources.displayMetrics.widthPixels
        val height = resources.displayMetrics.heightPixels

        binding.composeView2.apply {
            layoutParams.width = width
            layoutParams.height = height
        }

        CoroutineScope(Dispatchers.Default).launch {
            Log.d(TAG, "decoding page1 (ref.png)...")
            val page1 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    ImagePage.Images(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView2.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }
            Log.d(TAG, "page1 ready: ${page1.width}x${page1.height}")

            val page2 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref2.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    ImagePage.Images(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView2.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }
            val page3 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref2.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    ImagePage.Images(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView2.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }
            val page4 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref2.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    ImagePage.Images(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView2.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }

            binding.composeView2.state.apply {
                dpi = resources.displayMetrics.densityDpi / 100f

//                isVertical = true
//                transition = TransitionBasic.Vertical

                haveNext = true
                havePrev = true

                fetchPage = { index ->
                    Log.d(TAG, "fetchPage($index)")
                    if(index == -1) {
                        page4
                    } else if (index == 0) {
                        page1
                    } else if (index == 1){
                        page2
                    } else {
                        page3
                    }
                }

                post {
                    Log.d(TAG, "post: first render requested")
                    render()
                }
            }
            Log.i(TAG, "sample state configured (4 pages)")
        }
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        super.onDestroy()
    }
}
