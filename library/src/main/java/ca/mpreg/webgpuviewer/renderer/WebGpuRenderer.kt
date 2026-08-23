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
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class WebGpuRenderer {
    companion object {
        private const val TAG = "WebGpuRenderer"

        val instance get() = GpuContext.instance
        val adapter get() = GpuContext.adapter
        val device get() = GpuContext.device
        private val mutex = Mutex()

        var offsetX: Float = 0f
        var offsetY: Float = 0f

        private val renderExecutor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "WebGPU-Render-Thread")
        }
        val dispatcher = renderExecutor.asCoroutineDispatcher()

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
            Log.i(TAG, "Companion init starting on ${Thread.currentThread().name}")
            // Non-blocking on purpose: blocking here on the render thread while its task
            // touches this still-initializing companion deadlocks against the JVM class-init
            // lock (exp-d4: one log line, then silence). Device setup runs in the background;
            // consumers await GpuContext.ready.
            GpuContext.ensureStarted(dispatcher)
            Log.i(TAG, "Companion init complete")
        }

        @JvmStatic
        suspend fun <R> withContext(block: suspend CoroutineScope.(GPUDevice) -> R): R {
            return withContext(dispatcher) {
                GpuContext.ready.await()
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
                GpuContext.ready.await()
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

        // Device setup runs asynchronously on first class use; block until it finished
        // before creating or configuring any surface.
        if (Thread.currentThread().name == "WebGPU-Render-Thread") {
            runBlocking { GpuContext.ready.await() }
        } else {
            runBlocking(dispatcher) { GpuContext.ready.await() }
        }

        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val initSurface = {
            this@WebGpuRenderer.surface = surface.let {
                // Old Vulkan drivers accept only a subset of surface configurations:
                // Adreno 610 (driver 12/2020) rejects CompositeAlphaMode.Opaque with
                // ValidationException on Configure, and re-configuring that same surface
                // afterwards segfaults natively inside Dawn's Vulkan backend. So Auto -
                // which lets Dawn pick any supported mode itself - goes first, and every
                // further candidate gets a freshly created surface object.
                val window = windowFromSurface(it)
                val alphaCandidates = intArrayOf(
                    CompositeAlphaMode.Auto,
                    CompositeAlphaMode.Premultiplied,
                    CompositeAlphaMode.Opaque,
                )
                var configured: GPUSurface? = null
                for (alpha in alphaCandidates) {
                    val candidate = instance.createSurface(
                        GPUSurfaceDescriptor(
                            surfaceSourceAndroidNativeWindow =
                                GPUSurfaceSourceAndroidNativeWindow(window)
                        )
                    )
                    try {
                        candidate.configure(
                            GPUSurfaceConfiguration(
                                device,
                                width,
                                height,
                                TextureFormat.RGBA8Unorm,
                                TextureUsage.RenderAttachment,
                                alphaMode = alpha,
                            )
                        )
                        configured = candidate
                        surfaceConfigured = true
                        Log.i(TAG, "Surface configured, alphaMode=$alpha")
                        break
                    } catch (e: Exception) {
                        Log.w(TAG, "Surface configure rejected alphaMode=$alpha", e)
                    }
                }
                configured
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

/**
 * Holds the WebGPU instance/adapter/device.
 *
 * Kept separate from [WebGpuRenderer]'s companion on purpose. Dawn's OpenGL backend binds
 * its EGL context to the thread that creates the device, so setup must run on
 * "WebGPU-Render-Thread"; but a companion init block that blocks waiting for a task which
 * itself touches the still-initializing companion deadlocks on the JVM class-init lock
 * (exp-d4: "Companion init starting", then silence until the system kills the reader).
 * This object initializes under its own lock and never references [WebGpuRenderer] while
 * doing so.
 */
internal object GpuContext {
    private const val TAG = "WebGpuRenderer"

    lateinit var instance: GPUInstance
        private set
    lateinit var adapter: GPUAdapter
        private set
    lateinit var device: GPUDevice
        private set

    /** Completes once instance/adapter/device are usable; completes exceptionally if setup threw. */
    val ready = CompletableDeferred<Unit>()

    @Volatile
    private var started = false

    fun ensureStarted(dispatcher: CoroutineDispatcher) {
        if (started) return
        synchronized(this) {
            if (started) return
            started = true
            // Fire-and-forget on the render thread. SupervisorJob isolates a setup failure;
            // awaiters observe the outcome via [ready].
            CoroutineScope(SupervisorJob() + dispatcher).launch {
                try {
                    doSetup()
                    ready.complete(Unit)
                } catch (t: Throwable) {
                    Log.e(TAG, "Device setup FAILED", t)
                    ready.completeExceptionally(t)
                }
            }
        }
    }

    /** GLES-first worked around silent black frames on old Adreno Vulkan drivers, but the
     *  GL/EGL backend itself fails with eglMakeCurrent(EGL_BAD_ACCESS) on those same drivers
     *  (exp-d5). Flip to false to test whether the newer pinned Dawn fixes the Vulkan path. */
    private const val PREFER_GLES = false

    private suspend fun doSetup() {
        Log.i(TAG, "initLibrary() on ${Thread.currentThread().name}")
        initLibrary()

        instance = createInstance(GPUInstanceDescriptor())

        // Old Adreno Vulkan drivers present black frames without raising any validation
        // error; try the GLES backend first. If GLES can't produce an adapter on this
        // device, fall back to Dawn's own selection rather than crashing.
        Log.i(TAG, "Requesting adapter (preferGLES=$PREFER_GLES)")
        val requested: GPUAdapter? = if (PREFER_GLES) {
            try {
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
        } else {
            null
        }

        if (requested != null) {
            Log.i(TAG, "Using OpenGLES adapter: ${describe(requested)}")
            adapter = requested
        } else {
            Log.w(TAG, "No OpenGLES adapter available; falling back to default backend")
            adapter = instance.requestAdapter(
                GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility)
            )
            Log.i(TAG, "Fallback adapter: ${describe(adapter)}")
        }

        val requiredFeatures =
            if (adapter.hasFeature(FeatureName.TimestampQuery)) {
                intArrayOf(FeatureName.TimestampQuery)
            } else {
                intArrayOf()
            }

        Log.i(TAG, "Requesting device")
        device = adapter.requestDevice(
            GPUDeviceDescriptor(
                deviceLostCallback = defaultDeviceLostCallback,
                deviceLostCallbackExecutor = Executor(Runnable::run),
                uncapturedErrorCallback = defaultUncapturedErrorCallback,
                uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                requiredFeatures = requiredFeatures,
            )
        )
        Log.i(TAG, "Device ready")
    }

    private fun describe(a: GPUAdapter): String = try {
        val i = a.getInfo()
        "vendor=${i.vendor} arch=${i.architecture} device=${i.device} " +
            "backend=${BackendType.toString(i.backendType)}"
    } catch (e: Exception) {
        "unknown"
    }
}
