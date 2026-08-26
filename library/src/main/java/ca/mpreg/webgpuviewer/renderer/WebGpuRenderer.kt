package ca.mpreg.webgpuviewer.renderer

import android.view.Surface
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostException
import androidx.webgpu.FeatureLevel
import androidx.webgpu.FeatureName
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPUInstanceDescriptor
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPUSurface
import androidx.webgpu.GPUSurfaceConfiguration
import androidx.webgpu.GPUSurfaceDescriptor
import androidx.webgpu.GPUSurfaceSourceAndroidNativeWindow
import androidx.webgpu.GPUTexture
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.initLibrary
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.withContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

private const val TAG = "WGV.Render"

class WebGpuRenderer {
    companion object {
        var instance: GPUInstance
        var adapter: GPUAdapter
        var device: GPUDevice
        private val mutex = Mutex()

        var offsetX: Float = 0f
        var offsetY: Float = 0f

        val dispatcher = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "WebGPU-Render-Thread")
        }.asCoroutineDispatcher()

        // Frame time profiling
        var profilingEnabled = false
        private var frameCount = 0L
        private var totalFrameTimeNs = 0L
        private var minFrameTimeNs = Long.MAX_VALUE
        private var maxFrameTimeNs = 0L
        private var lastFrameTimeNs = 0L
        private val recentFrameTimes = LongArray(60)
        private var recentFrameIndex = 0

        val lastFrameTimeMs: Float get() = lastFrameTimeNs / 1_000_000f
        val avgFrameTimeMs: Float get() = if (frameCount > 0) totalFrameTimeNs / frameCount / 1_000_000f else 0f
        val minFrameTimeMs: Float get() = if (minFrameTimeNs == Long.MAX_VALUE) 0f else minFrameTimeNs / 1_000_000f
        val maxFrameTimeMs: Float get() = maxFrameTimeNs / 1_000_000f
        val recentAvgFrameTimeMs: Float
            get() {
                val count = minOf(frameCount.toInt(), 60)
                if (count == 0) return 0f
                var sum = 0L
                for (i in 0 until count) {
                    sum += recentFrameTimes[i]
                }
                return sum.toFloat() / count / 1_000_000f
            }
        val estimatedFps: Float get() = if (lastFrameTimeNs > 0) 1_000_000_000f / lastFrameTimeNs else 0f

        fun resetProfiling() {
            WgvLog.d(TAG, "resetProfiling()")
            frameCount = 0
            totalFrameTimeNs = 0
            minFrameTimeNs = Long.MAX_VALUE
            maxFrameTimeNs = 0
            lastFrameTimeNs = 0
            recentFrameIndex = 0
            recentFrameTimes.fill(0)
        }

        internal fun recordFrameTime(timeNs: Long) {
            if (!profilingEnabled) return
            frameCount++
            totalFrameTimeNs += timeNs
            lastFrameTimeNs = timeNs
            if (timeNs < minFrameTimeNs) minFrameTimeNs = timeNs
            if (timeNs > maxFrameTimeNs) maxFrameTimeNs = timeNs
            recentFrameTimes[recentFrameIndex] = timeNs
            recentFrameIndex = (recentFrameIndex + 1) % 60
        }

        init {
            runBlocking {
                WgvLog.i(TAG, "WebGPU bootstrap: initLibrary() + createInstance()...")
                initLibrary()

                instance = createInstance(GPUInstanceDescriptor())
                WgvLog.i(TAG, "GPUInstance created")

                adapter =
                    instance.requestAdapter(GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility))
                WgvLog.i(TAG, "GPUAdapter acquired (featureLevel=Compatibility)")

                val requiredFeatures =
                    if (adapter.hasFeature(FeatureName.TimestampQuery)) {
                        intArrayOf(FeatureName.TimestampQuery)
                    } else {
                        WgvLog.d(TAG, "TimestampQuery not supported by adapter")
                        intArrayOf()
                    }
                WgvLog.i(
                    TAG, "Requesting GPUDevice (requiredFeatures=${requiredFeatures.toList()})"
                )

                device = adapter.requestDevice(
                    GPUDeviceDescriptor(
                        deviceLostCallback = defaultDeviceLostCallback,
                        deviceLostCallbackExecutor = Executor(Runnable::run),
                        uncapturedErrorCallback = defaultUncapturedErrorCallback,
                        uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                        requiredFeatures = requiredFeatures,
                    )
                )
                WgvLog.i(TAG, "GPUDevice ready - WebGPU initialised")
            }
        }

        @JvmStatic
        suspend fun <R> withContext(block: suspend CoroutineScope.(GPUDevice) -> R): R {
            return withContext(dispatcher) {
                mutex.withLock {
                    block(this, device)
                }
            }
        }

        /**
         * Run [block] on the GPU thread *without* taking the render mutex.
         *
         * For long resource work that yields as it goes. [withContext] would defeat that: a
         * [render] call woken by the yield would just block on the mutex and hand the thread
         * straight back, so the work would still run to completion before the next frame. Without
         * the mutex the yield actually lets a frame through.
         *
         * Only safe for work that either owns its resources outright (an image still being built
         * and not yet reachable from a page) or that cannot be observed mid-flight. Anything that
         * has to appear atomically to the renderer belongs in [withContext].
         */
        @JvmStatic
        suspend fun <R> onDispatcher(block: suspend CoroutineScope.(GPUDevice) -> R): R {
            return withContext(dispatcher) {
                block(this, device)
            }
        }
    }

    @Volatile
    private var surface: GPUSurface? = null

    var width: Int = 0
    var height: Int = 0

    private var scope: CoroutineScope? = null

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        WgvLog.i(
            TAG,
            "init: surface ${width}x$height, thread=${Thread.currentThread().name}, scope=$scope"
        )
        this.scope = scope
        this.width = width
        this.height = height

        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val initSurface = {
            this@WebGpuRenderer.surface = surface.let {
                instance.createSurface(
                    GPUSurfaceDescriptor(
                        surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                            windowFromSurface(it)
                        )
                    )
                ).apply {
                    configure(
                        GPUSurfaceConfiguration(
                            device,
                            width,
                            height,
                            TextureFormat.RGBA8Unorm,
                            TextureUsage.RenderAttachment
                        )
                    )
                    WgvLog.i(TAG, "GPUSurface created + configured ${width}x$height (RGBA8Unorm)")
                }
            }
        }

        if (isOnDispatcherThread) {
            initSurface()
        } else {
            runBlocking(dispatcher) {
                initSurface()
            }
        }
    }

    suspend fun render(fn: suspend (GPUCommandEncoder, GPUTexture) -> Unit) {
        val startTime = if (profilingEnabled) System.nanoTime() else 0L

        mutex.withLock {
            val surface = surface
            if (surface == null) {
                WgvLog.v(TAG, "render skipped: surface not initialised yet")
                return
            }

            val texture = try {
                surface.getCurrentTexture().texture
            } catch (e: Exception) {
                WgvLog.w(TAG, "Failed to get current texture - dropping frame", e)
                return
            }

            try {
                val encoder = device.createCommandEncoder()
                fn(encoder, texture)
                device.queue.submit(arrayOf(encoder.finish()))
                surface.present()
            } catch (e: CancellationException) {
                WgvLog.v(TAG, "Frame cancelled mid-render: ${e.message}")
                throw e
            } catch (e: Exception) {
                WgvLog.e(TAG, "Render error - frame dropped, continuing", e)
                // Don't rethrow - allow the app to continue rendering next frame
            }
        }

        if (profilingEnabled) {
            val frameTime = System.nanoTime() - startTime
            recordFrameTime(frameTime)
            WgvLog.d(
                TAG, "Frame: %.2fms | Avg: %.2fms | FPS: %.1f".format(
                    frameTime / 1_000_000f, recentAvgFrameTimeMs, estimatedFps
                )
            )
        }
    }

    fun cleanup() {
        WgvLog.i(
            TAG, "cleanup: destroying surface, thread=${Thread.currentThread().name}"
        )
        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val doCleanup: suspend () -> Unit = {
            mutex.withLock {
                surface?.close()
                surface = null
                WgvLog.i(TAG, "cleanup: surface closed")
            }
        }

        if (isOnDispatcherThread) {
            // Already on dispatcher, run synchronously
            runBlocking {
                doCleanup()
            }
        } else {
            runBlocking(dispatcher) {
                doCleanup()
            }
        }
    }
}

private val defaultUncapturedErrorCallback
    get(): UncapturedErrorCallback {
        return UncapturedErrorCallback { _, type, message ->
            WgvLog.e(TAG, "Uncaptured WebGPU error: type=$type message=$message")
            throw WebGpuRuntimeException.create(type, message)
        }
    }

private val defaultDeviceLostCallback
    get(): DeviceLostCallback {
        return DeviceLostCallback { device, reason, message ->
            WgvLog.e(TAG, "GPU device lost: reason=$reason message=$message (device=$device)")
            throw DeviceLostException(device, reason, message)
        }
    }
