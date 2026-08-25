package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil

private val device get() = WebGpuRenderer.device

private val pipeline: GPUComputePipeline by lazy {
    device.createComputePipeline(
        GPUComputePipelineDescriptor(
            GPUComputeState(
                device.createShaderModule(
                    GPUShaderModuleDescriptor(
                        shaderSourceWGSL = GPUShaderSourceWGSL(LINE_SHADER)
                    )
                )
            )
        )
    )
}

private const val LINE_SHADER = """
struct Params {
    start: vec2<f32>,
    end: vec2<f32>,
    color: vec4<f32>,
    width: f32,
}

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pos = vec2<f32>(f32(id.x), f32(id.y));
    let ab = params.end - params.start;
    let ap = pos - params.start;
    let len_sq = dot(ab, ab);
    let t = select(clamp(dot(ap, ab) / len_sq, 0.0, 1.0), 0.0, len_sq == 0.0);
    let closest = params.start + t * ab;
    let dist = length(pos - closest);

    let half_w = params.width * 0.5;
    if (dist <= half_w + 0.5) {
        let coverage = clamp(half_w - dist + 0.5, 0.0, 1.0);
        let alpha = params.color.a * coverage;
        textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(params.color.rgb, alpha));
    }
}
"""

// Thread-local ByteBuffer to avoid allocation per call
private val byteBufferLocal = ThreadLocal.withInitial {
    ByteBuffer.allocateDirect(48).order(ByteOrder.nativeOrder())
}

fun Draw.line(
    encoder: GPUCommandEncoder,
    texture: GPUTexture,
    x1: Float,
    y1: Float,
    x2: Float,
    y2: Float,
    color: Int,
    thickness: Float
) {
    ca.mpreg.webgpuviewer.log.WgvLog.v(
        "WGV.Draw", "line (%.3f,%.3f)-(%.3f,%.3f) #%08x w=%.3f".format(x1, y1, x2, y2, color, thickness)
    )
    val r = ((color shr 16) and 0xFF) / 255f
    val g = ((color shr 8) and 0xFF) / 255f
    val b = (color and 0xFF) / 255f
    val a = ((color ushr 24) and 0xFF) / 255f

    val px1 = x1 * texture.width
    val py1 = y1 * texture.height
    val px2 = x2 * texture.width
    val py2 = y2 * texture.height

    val byteBuffer = byteBufferLocal.get()
    byteBuffer.clear()
    byteBuffer.putFloat(0, px1)
    byteBuffer.putFloat(4, py1)
    byteBuffer.putFloat(8, px2)
    byteBuffer.putFloat(12, py2)
    byteBuffer.putFloat(16, r)
    byteBuffer.putFloat(20, g)
    byteBuffer.putFloat(24, b)
    byteBuffer.putFloat(28, a)
    byteBuffer.putFloat(32, thickness)

    val uniformBuffer = createBuffer(48L, BufferUsage.Uniform or BufferUsage.CopyDst)
    device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

    val dispatchW = ceil(texture.width / 8f).toInt()
    val dispatchH = ceil(texture.height / 8f).toInt()

    val pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(
        0, device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                    GPUBindGroupEntry(0, textureView = texture.createView()),
                    GPUBindGroupEntry(1, buffer = uniformBuffer),
                )
            )
        )
    )
    pass.dispatchWorkgroups(dispatchW, dispatchH)
    pass.end()
}
