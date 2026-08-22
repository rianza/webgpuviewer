package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import android.view.Surface
import androidx.webgpu.BackendType
import androidx.webgpu.CompositeAlphaMode
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
        private const val TAG = "WebGpuRenderer"

        lateinit var instance: GPUInstance
        lateinit var adapter: GPUAdapter
        lateinit var device: GPUDevice
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

        private fun adapterDescription(a: GPUAdapter): String = try {
            val i = a.getInfo()
            "vendor=${i.vendor} arch=${i.architecture} device=${i.device} " +
                "backend=${BackendType.toString(i.backendType)}"
        } catch (e: Exception) {
            "unknown"
        }

        init {
            // Dawn's OpenGL backend binds its EGL context to the thread that creates the
            // device. Creating it on an arbitrary class-loading thread (usually main) makes
            // every later call from WebGPU-Render-Thread fail with EGL_BAD_ACCESS in
            // eglMakeCurrent, so run the whole setup on the render thread and block until
            // it completes.
            if (Thread.currentThread().name == "WebGPU-Render-Thread") {
                runBlocking {
                    setupDevice()
                }
            } else {
                runBlocking(dispatcher) {
                    setupDevice()
                }
            }
        }

        private suspend fun setupDevice() {
            initLibrary()

            instance = createInstance(GPUInstanceDescriptor())

            // Old Adreno Vulkan drivers present black frames without raising any validation
            // error; try the GLES backend first. If GLES can't produce an adapter on this
            // device, fall back to Dawn's own selection rather than crashing.
            Log.i(TAG, "Requesting adapter with backendType=OpenGLES")
            val requested: GPUAdapter? = try {
                instance.requestAdapter(
                    GPURequestAdapterOptions(
                        featureLevel = FeatureLevel.Compatibility,
                        backendType = BackendType.OpenGLES,
                    )
                )
            } catch (e: Exception) {
                Log.w(TAG, "requestAdapter(OpenGLES) failed", e)
                null
            }

            if (requested != null) {
                Log.i(TAG, "Using OpenGLES adapter: ${adapterDescription(requested)}")
                adapter = requested
            } else {
                Log.w(TAG, "No OpenGLES adapter available; falling back to default backend")
                adapter = instance.requestAdapter(
                    GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility)
                )
            }

            val requiredFeatures =
                if (adapter.hasFeature(FeatureName.TimestampQuery)) {
                    intArrayOf(FeatureName.TimestampQuery)
                } else {
                    intArrayOf()
                }

            device = adapter.requestDevice(
                GPUDeviceDescriptor(
                    deviceLostCallback = defaultDeviceLostCallback,
                    deviceLostCallbackExecutor = Executor(Runnable::run),
                    uncapturedErrorCallback = defaultUncapturedErrorCallback,
                    uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                    requiredFeatures = requiredFeatures,
                )
            )
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

    @Volatile
    private var surfaceConfigured = false

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.scope = scope
        this.width = width
        this.height = height
        this.surfaceConfigured = false

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
                    // Old Vulkan drivers accept only a subset of surface configurations:
                    // Adreno 610 (driver 12/2020) rejects both BGRA8Unorm and
                    // CompositeAlphaMode.Opaque with ValidationException on Configure.
                    // Try candidates in order until one is accepted.
                    val alphaCandidates = intArrayOf(
                        CompositeAlphaMode.Opaque,
                        CompositeAlphaMode.Premultiplied,
                        CompositeAlphaMode.Auto,
                        CompositeAlphaMode.Inherit,
                    )
                    for (alpha in alphaCandidates) {
                        try {
                            configure(
                                GPUSurfaceConfiguration(
                                    device,
                                    width,
                                    height,
                                    TextureFormat.RGBA8Unorm,
                                    TextureUsage.RenderAttachment,
                                    alphaMode = alpha,
                                )
                            )
                            surfaceConfigured = true
                            Log.i(TAG, "Surface configured, alphaMode=$alpha")
                            break
                        } catch (e: Exception) {
                            Log.w(TAG, "Surface configure rejected alphaMode=$alpha", e)
                        }
                    }
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
            // getCurrentTexture() on an unconfigured surface segfaults natively (uncatchable);
            // skip rendering entirely instead.
            if (!surfaceConfigured) return
            val surface = surface ?: return

            val texture = try {
                surface.getCurrentTexture().texture
            } catch (e: Exception) {
                Log.w("WebGpuRenderer", "Failed to get current texture", e)
                return
            }

            try {
                val encoder = device.createCommandEncoder()
                fn(encoder, texture)
                device.queue.submit(arrayOf(encoder.finish()))
                surface.present()
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
            throw WebGpuRuntimeException.create(type, message)
        }
    }

private val defaultDeviceLostCallback
    get(): DeviceLostCallback {
        return DeviceLostCallback { device, reason, message ->
            throw DeviceLostException(device, reason, message)
        }
    }
