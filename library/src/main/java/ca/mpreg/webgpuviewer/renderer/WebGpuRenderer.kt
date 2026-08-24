package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import android.view.Surface
import androidx.webgpu.DeviceLostCallback
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
import androidx.webgpu.SurfaceGetCurrentTextureStatus
import androidx.webgpu.GPUTexture
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.initLibrary
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

class WebGpuRenderer {
    companion object {
        var instance: GPUInstance
        var adapter: GPUAdapter
        var device: GPUDevice
        @Volatile var deviceLost: Boolean = false
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
                Log.i("WebGpuRenderer", "Initializing Dawn library")
                initLibrary()

                instance = createInstance(GPUInstanceDescriptor())
                Log.i("WebGpuRenderer", "GPUInstance created")

                adapter =
                    instance.requestAdapter(GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility))

                if (adapter == null) {
                    val message = "requestAdapter returned null - no compatible GPU backend on this device"
                    Log.e("WebGpuRenderer", message)
                    error(message)
                }
                val timestampQuery = adapter.hasFeature(FeatureName.TimestampQuery)
                Log.i("WebGpuRenderer", "Adapter acquired: featureLevel=Compatibility timestampQuery=$timestampQuery")
                try {
                    val info = adapter.javaClass.getMethod("getInfo").invoke(adapter)
                    Log.i("WebGpuRenderer", "Adapter info/driver GPU: $info")
                } catch (e: Exception) {
                    Log.w("WebGpuRenderer", "Adapter info unavailable on this WebGPU runtime", e)
                }

                val requiredFeatures =
                    if (adapter.hasFeature(FeatureName.TimestampQuery)) {
                        intArrayOf(FeatureName.TimestampQuery)
                    } else {
                        intArrayOf()
                    }

                deviceLost = false
                device = adapter.requestDevice(
                    GPUDeviceDescriptor(
                        deviceLostCallback = defaultDeviceLostCallback,
                        deviceLostCallbackExecutor = Executor(Runnable::run),
                        uncapturedErrorCallback = defaultUncapturedErrorCallback,
                        uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                        requiredFeatures = requiredFeatures,
                    )
                )
                Log.i("WebGpuRenderer", "GPUDevice created; rendering runs on WebGPU-Render-Thread")
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

    // One-shot diagnostics: reset on every surface init so each reader session reports
    // its own "configured" and "first frame presented" markers.
    @Volatile
    private var loggedMissingSurface = false

    @Volatile
    private var firstFrameLogged = false

    private var presentedFrames = 0L

    var width: Int = 0
    var height: Int = 0

    private var scope: CoroutineScope? = null

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.scope = scope
        this.width = width
        this.height = height

        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val initSurface = {
            Log.i(
                "WebGpuRenderer",
                "Surface init ${width}x${height}: create GPUSurface + configure" +
                    "(RGBA8Unorm, RenderAttachment)"
            )
            this@WebGpuRenderer.surface = surface.let {
                instance.createSurface(
                    GPUSurfaceDescriptor(
                        surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(windowFromSurface(it))
                    )
                ).apply {
                    val capabilities = getCapabilities(adapter)
                    Log.i("WebGpuRenderer", "Surface capabilities: formats=${capabilities.formats.contentToString()} presentModes=${capabilities.presentModes.contentToString()} alphaModes=${capabilities.alphaModes.contentToString()}")
                    Log.i("WebGpuRenderer", "Surface configuration selected: format=${TextureFormat.RGBA8Unorm} size=${width}x${height} usage=${TextureUsage.RenderAttachment}")
                    try {
                        configure(GPUSurfaceConfiguration(device, width, height, TextureFormat.RGBA8Unorm, TextureUsage.RenderAttachment))
                    } catch (e: Exception) {
                        Log.e("WebGpuRenderer", "Surface configure failed", e)
                        throw e
                    }
                }
            }
            loggedMissingSurface = false
            firstFrameLogged = false
            Log.i("WebGpuRenderer", "Surface configured OK (${width}x${height})")
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
            if (deviceLost || surface == null) {
                if (!loggedMissingSurface) {
                    loggedMissingSurface = true
                    Log.w("WebGpuRenderer", "render() skipped: inactive renderer or no surface configured yet")
                }
                return
            }

            val surfaceTexture = try { surface.getCurrentTexture() } catch (e: Exception) {
                Log.w("WebGpuRenderer", "Failed to get current texture", e); return
            }
            Log.d("WebGpuRenderer", "getCurrentTexture status=${surfaceTexture.status}")
            if (surfaceTexture.status != SurfaceGetCurrentTextureStatus.SuccessOptimal &&
                surfaceTexture.status != SurfaceGetCurrentTextureStatus.SuccessSuboptimal) {
                Log.w("WebGpuRenderer", "Skipping frame: surface texture status=${surfaceTexture.status}")
                return
            }
            val texture = surfaceTexture.texture ?: run {
                Log.w("WebGpuRenderer", "Skipping frame: surface texture is null")
                return
            }

            try {
                val encoder = device.createCommandEncoder()
                fn(encoder, texture)
                device.queue.submit(arrayOf(encoder.finish()))
                surface.present()
                presentedFrames++
                Log.d("WebGpuRenderer", "Frame presented #$presentedFrames texture=${texture.width}x${texture.height}")
                if (!firstFrameLogged) {
                    firstFrameLogged = true
                    Log.i("WebGpuRenderer", "First frame presented (${width}x${height})")
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e("WebGpuRenderer", "Render error", e)
                // Don't rethrow - allow the app to continue rendering next frame
            }
        }

        if (profilingEnabled) {
            val frameTime = System.nanoTime() - startTime
            recordFrameTime(frameTime)
            Log.d(
                "WebGpuRenderer", "Frame: %.2fms | Avg: %.2fms | FPS: %.1f".format(
                    frameTime / 1_000_000f, recentAvgFrameTimeMs, estimatedFps
                )
            )
        }
    }

    fun cleanup() {
        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val doCleanup: suspend () -> Unit = {
            mutex.withLock {
                Log.i("WebGpuRenderer", "Surface released after $presentedFrames presented frames")
                presentedFrames = 0
                surface?.close()
                surface = null
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
            Log.e("WebGpuRenderer", "Uncaptured GPU error [$type]: $message")
            throw WebGpuRuntimeException.create(type, message)
        }
    }

private val defaultDeviceLostCallback
    get(): DeviceLostCallback {
        return DeviceLostCallback { _, reason, message ->
            Log.e("WebGpuRenderer", "GPU device lost: reason=$reason message=$message")
            WebGpuRenderer.deviceLost = true
        }
    }
