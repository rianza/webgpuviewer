package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoder
import ca.mpreg.webgpuviewer.log.WgvLog
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer

private const val TAG = "WGV.Draw"

object Draw {
    internal val device get() = WebGpuRenderer.device

    private val tempBuffers = ThreadLocal.withInitial { mutableListOf<GPUBuffer>() }

    fun submit(block: Draw.(GPUCommandEncoder) -> Unit) {
        val buffers = tempBuffers.get()
        val encoder = device.createCommandEncoder()
        block.invoke(this, encoder)
        device.queue.submit(arrayOf(encoder.finish()))
        WgvLog.v(TAG, "submit: ${buffers.size} scratch buffer(s) destroyed after queue.submit")
        buffers.forEach { it.destroy() }
        buffers.clear()
    }

    internal fun createBuffer(size: Long, usage: Int): GPUBuffer {
        return device.createBuffer(GPUBufferDescriptor(size = size, usage = usage)).also {
            WgvLog.v(TAG, "createBuffer: $size bytes (usage=$usage)")
            tempBuffers.get().add(it)
        }
    }
}
