#include "DenoiseEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QComboBox>
#include <QDebug>
#include <QLabel>
#include <QVBoxLayout>
#include <algorithm>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
//
// Two independent phases, each controlled by a slider:
//
//  Phase 1 — Luma denoising
//    Separable Gaussian blur → blend with original, attenuated in shadows.
//    Radius scales with Strength (1–5 pixels).
//    Shadow Preserve (0–1) reduces the blend fraction for dark pixels,
//    protecting shadow detail from over-smoothing.
//
//  Phase 2 — Colour noise reduction
//    A wider Gaussian blur is run. A chroma-only merge then keeps the
//    luminance (Y) of the current pixel but blends its Cb/Cr channels
//    toward the heavily-smoothed values, controlled by Color Noise (0–1).
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"
#include "blur_kernels.h"

namespace {

// Blur passes (blurH/blurV/blurH16/blurV16) come from the shared header.
static const char* GPU_KERNEL_SOURCE = SHARED_BLUR_KERNELS R"CL(

// ─── 8-bit path ─────────────────────────────────────────────────────────────

// Blend original → blurred, with shadow-based attenuation.
// strength:       0.0–1.0  overall blend fraction
// shadowPreserve: 0.0–1.0  reduces blend in dark regions (1 = full shadow protection)
__kernel void denoiseBlend(__global const uint* original,
                           __global const uint* blurred,
                           __global       uint* output,
                           int stride, int width, int height,
                           float strength, float shadowPreserve)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint op = original[y * stride + x];
    uint bp = blurred [y * stride + x];

    float or_ = (float)((op >> 16) & 0xFFu);
    float og  = (float)((op >>  8) & 0xFFu);
    float ob  = (float)( op        & 0xFFu);

    float br  = (float)((bp >> 16) & 0xFFu);
    float bg  = (float)((bp >>  8) & 0xFFu);
    float bb  = (float)( bp        & 0xFFu);

    // Perceived luminance (0–1); darker pixels get less denoising.
    float luma  = (0.299f * or_ + 0.587f * og + 0.114f * ob) * (1.0f / 255.0f);
    float blend = strength * (1.0f - shadowPreserve * (1.0f - luma));
    blend = clamp(blend, 0.0f, 1.0f);

    int rr = clamp((int)(or_ + blend * (br - or_) + 0.5f), 0, 255);
    int rg = clamp((int)(og  + blend * (bg - og)  + 0.5f), 0, 255);
    int rb = clamp((int)(ob  + blend * (bb - ob)  + 0.5f), 0, 255);
    output[y * stride + x] = 0xFF000000u | (rr << 16) | (rg << 8) | rb;
}

// Keep luminance (Y) from 'current'; blend Cb/Cr toward 'smoothed'.
// colorNoise: 0.0–1.0
__kernel void colorNoiseBlend(__global const uint* current,
                              __global const uint* smoothed,
                              __global       uint* output,
                              int stride, int width, int height,
                              float colorNoise)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint cp = current [y * stride + x];
    uint sp = smoothed[y * stride + x];

    float cr = (float)((cp >> 16) & 0xFFu);
    float cg = (float)((cp >>  8) & 0xFFu);
    float cb = (float)( cp        & 0xFFu);

    float sr = (float)((sp >> 16) & 0xFFu);
    float sg = (float)((sp >>  8) & 0xFFu);
    float sb = (float)( sp        & 0xFFu);

    // BT.601 RGB → YCbCr (Cb/Cr centred at 128)
    float Y   =  0.299f * cr + 0.587f * cg + 0.114f * cb;
    float Cb  = -0.169f * cr - 0.331f * cg + 0.500f * cb + 128.0f;
    float Cr  =  0.500f * cr - 0.419f * cg - 0.081f * cb + 128.0f;

    float sCb = -0.169f * sr - 0.331f * sg + 0.500f * sb + 128.0f;
    float sCr =  0.500f * sr - 0.419f * sg - 0.081f * sb + 128.0f;

    // Blend chroma toward smoother values; luma preserved exactly.
    float rCb = Cb + colorNoise * (sCb - Cb);
    float rCr = Cr + colorNoise * (sCr - Cr);

    // YCbCr → RGB
    float rr = Y + 1.402f  * (rCr - 128.0f);
    float rg = Y - 0.344f  * (rCb - 128.0f) - 0.714f * (rCr - 128.0f);
    float rb = Y + 1.772f  * (rCb - 128.0f);

    int ri = clamp((int)(rr + 0.5f), 0, 255);
    int gi = clamp((int)(rg + 0.5f), 0, 255);
    int bi = clamp((int)(rb + 0.5f), 0, 255);
    output[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

// ─── 16-bit path ─────────────────────────────────────────────────────────────
// blurH16/blurV16 come from the shared SHARED_BLUR_KERNELS header.

__kernel void denoiseBlend16(__global const ushort4* original,
                             __global const ushort4* blurred,
                             __global       ushort4* output,
                             int stride, int width, int height,
                             float strength, float shadowPreserve)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 op = original[y * stride + x];
    ushort4 bp = blurred [y * stride + x];

    float or_ = op.s0, og = op.s1, ob = op.s2;
    float br  = bp.s0, bg = bp.s1, bb = bp.s2;

    float luma  = (0.299f * or_ + 0.587f * og + 0.114f * ob) * (1.0f / 65535.0f);
    float blend = strength * (1.0f - shadowPreserve * (1.0f - luma));
    blend = clamp(blend, 0.0f, 1.0f);

    ushort4 res;
    res.s0 = (ushort)clamp((int)(or_ + blend * (br - or_) + 0.5f), 0, 65535);
    res.s1 = (ushort)clamp((int)(og  + blend * (bg - og)  + 0.5f), 0, 65535);
    res.s2 = (ushort)clamp((int)(ob  + blend * (bb - ob)  + 0.5f), 0, 65535);
    res.s3 = 65535;
    output[y * stride + x] = res;
}

__kernel void colorNoiseBlend16(__global const ushort4* current,
                                __global const ushort4* smoothed,
                                __global       ushort4* output,
                                int stride, int width, int height,
                                float colorNoise)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 cp = current [y * stride + x];
    ushort4 sp = smoothed[y * stride + x];

    float cr = cp.s0, cg = cp.s1, cb = cp.s2;
    float sr = sp.s0, sg = sp.s1, sb = sp.s2;

    // BT.601 YCbCr — 16-bit range; Cb/Cr centred at 32768
    float Y   =  0.299f * cr + 0.587f * cg + 0.114f * cb;
    float Cb  = -0.169f * cr - 0.331f * cg + 0.500f * cb + 32768.0f;
    float Cr  =  0.500f * cr - 0.419f * cg - 0.081f * cb + 32768.0f;

    float sCb = -0.169f * sr - 0.331f * sg + 0.500f * sb + 32768.0f;
    float sCr =  0.500f * sr - 0.419f * sg - 0.081f * sb + 32768.0f;

    float rCb = Cb + colorNoise * (sCb - Cb);
    float rCr = Cr + colorNoise * (sCr - Cr);

    float rr = Y + 1.402f  * (rCr - 32768.0f);
    float rg = Y - 0.344f  * (rCb - 32768.0f) - 0.714f * (rCr - 32768.0f);
    float rb = Y + 1.772f  * (rCb - 32768.0f);

    ushort4 res;
    res.s0 = (ushort)clamp((int)(rr + 0.5f), 0, 65535);
    res.s1 = (ushort)clamp((int)(rg + 0.5f), 0, 65535);
    res.s2 = (ushort)clamp((int)(rb + 0.5f), 0, 65535);
    res.s3 = 65535;
    output[y * stride + x] = res;
}

)CL";

// ─── GPU singleton ───────────────────────────────────────────────────────────

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernelBlurH;
    cl::Kernel kernelBlurV;
    cl::Kernel kernelDenoiseBlend;
    cl::Kernel kernelColorNoiseBlend;
    cl::Kernel kernelBlurH16;
    cl::Kernel kernelBlurV16;
    cl::Kernel kernelDenoiseBlend16;
    cl::Kernel kernelColorNoiseBlend16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Denoise")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernelBlurH             = cl::Kernel(prog, "blurH");
            kernelBlurV             = cl::Kernel(prog, "blurV");
            kernelDenoiseBlend      = cl::Kernel(prog, "denoiseBlend");
            kernelColorNoiseBlend   = cl::Kernel(prog, "colorNoiseBlend");
            kernelBlurH16           = cl::Kernel(prog, "blurH16");
            kernelBlurV16           = cl::Kernel(prog, "blurV16");
            kernelDenoiseBlend16    = cl::Kernel(prog, "denoiseBlend16");
            kernelColorNoiseBlend16 = cl::Kernel(prog, "colorNoiseBlend16");
            available = true;
            qDebug() << "[GPU] Denoise ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Denoise init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

// ─── Standalone GPU helpers ──────────────────────────────────────────────────

static QImage processImageGPU(const QImage& image,
                               float strength, float shadowPreserve, float colorNoise)
{
    QImage src = image.convertToFormat(QImage::Format_RGB32);
    const int    w      = src.width();
    const int    h      = src.height();
    const int    stride = src.bytesPerLine() / 4;
    const size_t bytes  = static_cast<size_t>(src.bytesPerLine()) * h;

    int lumRadius    = static_cast<int>(strength   * 5.0f + 0.5f);
    int chromaRadius = static_cast<int>(colorNoise * 10.0f + 0.5f);
    if (lumRadius    < 1) lumRadius    = 1;
    if (chromaRadius < 2) chromaRadius = 2;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufSrc (gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, src.bits());
        cl::Buffer bufWork(gpu.context, CL_MEM_READ_WRITE, bytes);
        cl::Buffer bufTemp(gpu.context, CL_MEM_READ_WRITE, bytes);

        const cl::NDRange global(w, h);

        // Phase 1: luma denoising
        if (strength > 0.0f) {
            // H blur: bufSrc → bufWork
            gpu.kernelBlurH.setArg(0, bufSrc);  gpu.kernelBlurH.setArg(1, bufWork);
            gpu.kernelBlurH.setArg(2, stride);  gpu.kernelBlurH.setArg(3, w);
            gpu.kernelBlurH.setArg(4, h);        gpu.kernelBlurH.setArg(5, lumRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurH, cl::NullRange, global, cl::NullRange);

            // V blur: bufWork → bufTemp
            gpu.kernelBlurV.setArg(0, bufWork); gpu.kernelBlurV.setArg(1, bufTemp);
            gpu.kernelBlurV.setArg(2, stride);  gpu.kernelBlurV.setArg(3, w);
            gpu.kernelBlurV.setArg(4, h);        gpu.kernelBlurV.setArg(5, lumRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurV, cl::NullRange, global, cl::NullRange);

            // Blend: (original=bufSrc, blurred=bufTemp) → bufWork
            gpu.kernelDenoiseBlend.setArg(0, bufSrc);  gpu.kernelDenoiseBlend.setArg(1, bufTemp);
            gpu.kernelDenoiseBlend.setArg(2, bufWork); gpu.kernelDenoiseBlend.setArg(3, stride);
            gpu.kernelDenoiseBlend.setArg(4, w);        gpu.kernelDenoiseBlend.setArg(5, h);
            gpu.kernelDenoiseBlend.setArg(6, strength); gpu.kernelDenoiseBlend.setArg(7, shadowPreserve);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelDenoiseBlend, cl::NullRange, global, cl::NullRange);

            gpu.queue.enqueueCopyBuffer(bufWork, bufSrc, 0, 0, bytes); // bufSrc = denoised
        }

        // Phase 2: colour noise reduction
        if (colorNoise > 0.0f) {
            // H blur (wide): bufSrc → bufWork
            gpu.kernelBlurH.setArg(0, bufSrc);  gpu.kernelBlurH.setArg(1, bufWork);
            gpu.kernelBlurH.setArg(2, stride);  gpu.kernelBlurH.setArg(3, w);
            gpu.kernelBlurH.setArg(4, h);        gpu.kernelBlurH.setArg(5, chromaRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurH, cl::NullRange, global, cl::NullRange);

            // V blur: bufWork → bufTemp
            gpu.kernelBlurV.setArg(0, bufWork); gpu.kernelBlurV.setArg(1, bufTemp);
            gpu.kernelBlurV.setArg(2, stride);  gpu.kernelBlurV.setArg(3, w);
            gpu.kernelBlurV.setArg(4, h);        gpu.kernelBlurV.setArg(5, chromaRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurV, cl::NullRange, global, cl::NullRange);

            // Chroma merge: (current=bufSrc, smoothed=bufTemp) → bufWork
            gpu.kernelColorNoiseBlend.setArg(0, bufSrc);    gpu.kernelColorNoiseBlend.setArg(1, bufTemp);
            gpu.kernelColorNoiseBlend.setArg(2, bufWork);   gpu.kernelColorNoiseBlend.setArg(3, stride);
            gpu.kernelColorNoiseBlend.setArg(4, w);          gpu.kernelColorNoiseBlend.setArg(5, h);
            gpu.kernelColorNoiseBlend.setArg(6, colorNoise);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelColorNoiseBlend, cl::NullRange, global, cl::NullRange);

            gpu.queue.enqueueCopyBuffer(bufWork, bufSrc, 0, 0, bytes); // bufSrc = final
        }

        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(bufSrc, CL_TRUE, 0, bytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Denoise kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

static QImage processImageGPU16(const QImage& image,
                                 float strength, float shadowPreserve, float colorNoise)
{
    QImage src = image; // already RGBX64
    const int    w      = src.width();
    const int    h      = src.height();
    const int    stride = src.bytesPerLine() / 8;
    const size_t bytes  = static_cast<size_t>(src.bytesPerLine()) * h;

    int lumRadius    = static_cast<int>(strength   * 5.0f + 0.5f);
    int chromaRadius = static_cast<int>(colorNoise * 10.0f + 0.5f);
    if (lumRadius    < 1) lumRadius    = 1;
    if (chromaRadius < 2) chromaRadius = 2;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufSrc (gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, src.bits());
        cl::Buffer bufWork(gpu.context, CL_MEM_READ_WRITE, bytes);
        cl::Buffer bufTemp(gpu.context, CL_MEM_READ_WRITE, bytes);

        const cl::NDRange global(w, h);

        // Phase 1: luma denoising
        if (strength > 0.0f) {
            gpu.kernelBlurH16.setArg(0, bufSrc);  gpu.kernelBlurH16.setArg(1, bufWork);
            gpu.kernelBlurH16.setArg(2, stride);  gpu.kernelBlurH16.setArg(3, w);
            gpu.kernelBlurH16.setArg(4, h);        gpu.kernelBlurH16.setArg(5, lumRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurH16, cl::NullRange, global, cl::NullRange);

            gpu.kernelBlurV16.setArg(0, bufWork); gpu.kernelBlurV16.setArg(1, bufTemp);
            gpu.kernelBlurV16.setArg(2, stride);  gpu.kernelBlurV16.setArg(3, w);
            gpu.kernelBlurV16.setArg(4, h);        gpu.kernelBlurV16.setArg(5, lumRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurV16, cl::NullRange, global, cl::NullRange);

            gpu.kernelDenoiseBlend16.setArg(0, bufSrc);       gpu.kernelDenoiseBlend16.setArg(1, bufTemp);
            gpu.kernelDenoiseBlend16.setArg(2, bufWork);      gpu.kernelDenoiseBlend16.setArg(3, stride);
            gpu.kernelDenoiseBlend16.setArg(4, w);             gpu.kernelDenoiseBlend16.setArg(5, h);
            gpu.kernelDenoiseBlend16.setArg(6, strength);      gpu.kernelDenoiseBlend16.setArg(7, shadowPreserve);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelDenoiseBlend16, cl::NullRange, global, cl::NullRange);

            gpu.queue.enqueueCopyBuffer(bufWork, bufSrc, 0, 0, bytes);
        }

        // Phase 2: colour noise reduction
        if (colorNoise > 0.0f) {
            gpu.kernelBlurH16.setArg(0, bufSrc);  gpu.kernelBlurH16.setArg(1, bufWork);
            gpu.kernelBlurH16.setArg(2, stride);  gpu.kernelBlurH16.setArg(3, w);
            gpu.kernelBlurH16.setArg(4, h);        gpu.kernelBlurH16.setArg(5, chromaRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurH16, cl::NullRange, global, cl::NullRange);

            gpu.kernelBlurV16.setArg(0, bufWork); gpu.kernelBlurV16.setArg(1, bufTemp);
            gpu.kernelBlurV16.setArg(2, stride);  gpu.kernelBlurV16.setArg(3, w);
            gpu.kernelBlurV16.setArg(4, h);        gpu.kernelBlurV16.setArg(5, chromaRadius);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelBlurV16, cl::NullRange, global, cl::NullRange);

            gpu.kernelColorNoiseBlend16.setArg(0, bufSrc);     gpu.kernelColorNoiseBlend16.setArg(1, bufTemp);
            gpu.kernelColorNoiseBlend16.setArg(2, bufWork);    gpu.kernelColorNoiseBlend16.setArg(3, stride);
            gpu.kernelColorNoiseBlend16.setArg(4, w);           gpu.kernelColorNoiseBlend16.setArg(5, h);
            gpu.kernelColorNoiseBlend16.setArg(6, colorNoise);
            gpu.queue.enqueueNDRangeKernel(gpu.kernelColorNoiseBlend16, cl::NullRange, global, cl::NullRange);

            gpu.queue.enqueueCopyBuffer(bufWork, bufSrc, 0, 0, bytes);
        }

        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(bufSrc, CL_TRUE, 0, bytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Denoise16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB.
//
// Two phases, identical in shape to the sRGB test path but in linear light:
//
//   Phase 1 (luma denoise):
//     H blur(buf→aux), V blur(aux→tempBuf), denoiseBlend(orig=buf,
//     blurred=tempBuf → aux), copy(aux→buf).
//     Shadow-protection mask uses sRGB-encoded luminance so its dark-vs-light
//     behaviour matches the test path's 0.299/0.587/0.114-in-sRGB weighting.
//
//   Phase 2 (colour noise):
//     H blur(buf→aux), V blur(aux→tempBuf), colorNoiseBlend(current=buf,
//     smoothed=tempBuf → aux), copy(aux→buf).
//     We keep linear luma exactly from the current pixel and blend
//     (Cb, Cr) toward the smoothed reference using BT.601 in linear.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC SHARED_BLUR_KERNELS_F4 R"CL(

// Phase 1: blend original -> blurred, with shadow-based attenuation.
__kernel void denoiseBlendLinear(__global const float4* original,
                                  __global const float4* blurred,
                                  __global       float4* output,
                                  int w, int h,
                                  float strength, float shadowPreserve)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 o = original[y * w + x];
    float4 b = blurred [y * w + x];

    // Shadow mask evaluated on sRGB-encoded luminance so "dark pixel"
    // means what it means perceptually (matches the test path).
    float L_linear = linear_luma(o);
    float L_srgb   = linear_to_srgb(L_linear);
    float blend = strength * (1.0f - shadowPreserve * (1.0f - L_srgb));
    blend = clamp(blend, 0.0f, 1.0f);

    float r  = o.x + blend * (b.x - o.x);
    float g  = o.y + blend * (b.y - o.y);
    float bl = o.z + blend * (b.z - o.z);
    output[y * w + x] = (float4)(r, g, bl, 1.0f);
}

// Phase 2: keep luma from `current`; blend Cb/Cr toward `smoothed`.
// BT.601 in linear RGB — the exact primaries matter less than round-tripping.
__kernel void colorNoiseBlendLinear(__global const float4* current,
                                     __global const float4* smoothed,
                                     __global       float4* output,
                                     int w, int h,
                                     float colorNoise)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 c = current [y * w + x];
    float4 s = smoothed[y * w + x];

    // BT.601 RGB -> YCbCr (centred at 0 for Cb/Cr; keep math in float).
    float Y   =  0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
    float Cb  = -0.169f * c.x - 0.331f * c.y + 0.500f * c.z;
    float Cr  =  0.500f * c.x - 0.419f * c.y - 0.081f * c.z;

    float sCb = -0.169f * s.x - 0.331f * s.y + 0.500f * s.z;
    float sCr =  0.500f * s.x - 0.419f * s.y - 0.081f * s.z;

    float rCb = Cb + colorNoise * (sCb - Cb);
    float rCr = Cr + colorNoise * (sCr - Cr);

    // YCbCr -> RGB (inverse BT.601)
    float r  = Y + 1.402f  * rCr;
    float g  = Y - 0.344f  * rCb - 0.714f * rCr;
    float bl = Y + 1.772f  * rCb;
    output[y * w + x] = (float4)(r, g, bl, 1.0f);
}

// Alternative Phase 1: bilateral filter (edge-aware luma denoise).
// Single 2D pass — spatial Gaussian × range Gaussian.  Taps inside the
// (2r+1)² window are weighted by both spatial distance and tonal distance
// from the centre pixel, so flat noisy regions smooth while edges survive.
// Output is blended back into the original with the same shadow-aware
// strength formula as the Gaussian path, so the two algorithms can be
// compared apples-to-apples at the same slider values.
__kernel void bilateralDenoiseLinear(__global const float4* in,
                                      __global       float4* out,
                                      int w, int h,
                                      int radius, float sigmaRange,
                                      float strength, float shadowPreserve)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 c = in[y * w + x];

    float sigmaSpatial = max((float)radius / 3.0f, 0.5f);
    float invSpatial2  = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
    float invRange2    = 1.0f / (2.0f * sigmaRange   * sigmaRange);

    float sr = 0.0f, sg = 0.0f, sb = 0.0f, wsum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp(y + dy, 0, h - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = clamp(x + dx, 0, w - 1);
            float4 q = in[sy * w + sx];
            float spatial = native_exp(-(float)(dx * dx + dy * dy) * invSpatial2);
            float d2 = (q.x - c.x) * (q.x - c.x)
                     + (q.y - c.y) * (q.y - c.y)
                     + (q.z - c.z) * (q.z - c.z);
            float range = native_exp(-d2 * invRange2);
            float ww = spatial * range;
            sr += ww * q.x; sg += ww * q.y; sb += ww * q.z;
            wsum += ww;
        }
    }
    float inv = 1.0f / wsum;
    float fr = sr * inv, fg = sg * inv, fb = sb * inv;

    float L_linear = linear_luma(c);
    float L_srgb   = linear_to_srgb(L_linear);
    float blend = strength * (1.0f - shadowPreserve * (1.0f - L_srgb));
    blend = clamp(blend, 0.0f, 1.0f);

    float r  = c.x + blend * (fr - c.x);
    float g  = c.y + blend * (fg - c.y);
    float bl = c.z + blend * (fb - c.z);
    out[y * w + x] = (float4)(r, g, bl, 1.0f);
}

)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool DenoiseEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelBlurHLinear            = cl::Kernel(prog, "blurHLinear");
        m_kernelBlurVLinear            = cl::Kernel(prog, "blurVLinear");
        m_kernelDenoiseBlendLinear     = cl::Kernel(prog, "denoiseBlendLinear");
        m_kernelColorNoiseBlendLinear  = cl::Kernel(prog, "colorNoiseBlendLinear");
        m_kernelBilateralDenoiseLinear = cl::Kernel(prog, "bilateralDenoiseLinear");
        m_pipelineCtx                  = ctx;
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Denoise initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool DenoiseEffect::enqueueGpu(cl::CommandQueue& queue,
                                cl::Buffer& buf, cl::Buffer& aux,
                                int w, int h,
                                const QMap<QString, QVariant>& params) {
    const float strength       = params.value("strength",       30).toInt() / 100.0f;
    const float shadowPreserve = params.value("shadowPreserve", 50).toInt() / 100.0f;
    const float colorNoise     = params.value("colorNoise",     50).toInt() / 100.0f;
    const int   algorithm      = params.value("algorithm", 0).toInt();

    if (strength == 0.0f && colorNoise == 0.0f) return true;

    // Scale the (source-pixel) radii to preview pixels.  The base formula
    // matches the test path (5 * strength, 10 * colorNoise source pixels).
    const double scale = params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble();
    const double invScale = 1.0 / std::max(scale, 1e-6);
    int lumRadiusSrc    = static_cast<int>(strength   * 5.0f + 0.5f);
    int chromaRadiusSrc = static_cast<int>(colorNoise * 10.0f + 0.5f);
    if (lumRadiusSrc    < 1) lumRadiusSrc    = 1;
    if (chromaRadiusSrc < 2) chromaRadiusSrc = 2;
    int lumRadius    = std::max(1, static_cast<int>(lumRadiusSrc    * invScale + 0.5));
    int chromaRadius = std::max(2, static_cast<int>(chromaRadiusSrc * invScale + 0.5));

    const size_t f4Bytes = static_cast<size_t>(w) * h * sizeof(cl_float4);
    cl::Buffer tempBuf(m_pipelineCtx, CL_MEM_READ_WRITE, f4Bytes);
    const cl::NDRange global(w, h);

    // Phase 1: luma denoising
    if (strength > 0.0f) {
        if (algorithm == 1) {
            // Bilateral: single 2D pass, edge-aware.  sigmaRange is driven
            // from Strength so the slider still controls overall aggression.
            const float sigmaRange = 0.02f + strength * 0.15f;
            m_kernelBilateralDenoiseLinear.setArg(0, buf);
            m_kernelBilateralDenoiseLinear.setArg(1, aux);
            m_kernelBilateralDenoiseLinear.setArg(2, w);
            m_kernelBilateralDenoiseLinear.setArg(3, h);
            m_kernelBilateralDenoiseLinear.setArg(4, lumRadius);
            m_kernelBilateralDenoiseLinear.setArg(5, sigmaRange);
            m_kernelBilateralDenoiseLinear.setArg(6, strength);
            m_kernelBilateralDenoiseLinear.setArg(7, shadowPreserve);
            queue.enqueueNDRangeKernel(m_kernelBilateralDenoiseLinear, cl::NullRange, global, cl::NullRange);
            queue.enqueueCopyBuffer(aux, buf, 0, 0, f4Bytes);
        } else {
            // H: buf → aux
            m_kernelBlurHLinear.setArg(0, buf);
            m_kernelBlurHLinear.setArg(1, aux);
            m_kernelBlurHLinear.setArg(2, w);
            m_kernelBlurHLinear.setArg(3, h);
            m_kernelBlurHLinear.setArg(4, lumRadius);
            m_kernelBlurHLinear.setArg(5, 1);
            queue.enqueueNDRangeKernel(m_kernelBlurHLinear, cl::NullRange, global, cl::NullRange);

            // V: aux → tempBuf
            m_kernelBlurVLinear.setArg(0, aux);
            m_kernelBlurVLinear.setArg(1, tempBuf);
            m_kernelBlurVLinear.setArg(2, w);
            m_kernelBlurVLinear.setArg(3, h);
            m_kernelBlurVLinear.setArg(4, lumRadius);
            m_kernelBlurVLinear.setArg(5, 1);
            queue.enqueueNDRangeKernel(m_kernelBlurVLinear, cl::NullRange, global, cl::NullRange);

            // Blend: (original=buf, blurred=tempBuf) → aux
            m_kernelDenoiseBlendLinear.setArg(0, buf);
            m_kernelDenoiseBlendLinear.setArg(1, tempBuf);
            m_kernelDenoiseBlendLinear.setArg(2, aux);
            m_kernelDenoiseBlendLinear.setArg(3, w);
            m_kernelDenoiseBlendLinear.setArg(4, h);
            m_kernelDenoiseBlendLinear.setArg(5, strength);
            m_kernelDenoiseBlendLinear.setArg(6, shadowPreserve);
            queue.enqueueNDRangeKernel(m_kernelDenoiseBlendLinear, cl::NullRange, global, cl::NullRange);

            // Fold denoised result back into buf so Phase 2 sees the updated image.
            queue.enqueueCopyBuffer(aux, buf, 0, 0, f4Bytes);
        }
    }

    // Phase 2: chroma smoothing
    if (colorNoise > 0.0f) {
        // H: buf → aux
        m_kernelBlurHLinear.setArg(0, buf);
        m_kernelBlurHLinear.setArg(1, aux);
        m_kernelBlurHLinear.setArg(2, w);
        m_kernelBlurHLinear.setArg(3, h);
        m_kernelBlurHLinear.setArg(4, chromaRadius);
        m_kernelBlurHLinear.setArg(5, 1);
        queue.enqueueNDRangeKernel(m_kernelBlurHLinear, cl::NullRange, global, cl::NullRange);

        // V: aux → tempBuf
        m_kernelBlurVLinear.setArg(0, aux);
        m_kernelBlurVLinear.setArg(1, tempBuf);
        m_kernelBlurVLinear.setArg(2, w);
        m_kernelBlurVLinear.setArg(3, h);
        m_kernelBlurVLinear.setArg(4, chromaRadius);
        m_kernelBlurVLinear.setArg(5, 1);
        queue.enqueueNDRangeKernel(m_kernelBlurVLinear, cl::NullRange, global, cl::NullRange);

        // Chroma merge: (current=buf, smoothed=tempBuf) → aux
        m_kernelColorNoiseBlendLinear.setArg(0, buf);
        m_kernelColorNoiseBlendLinear.setArg(1, tempBuf);
        m_kernelColorNoiseBlendLinear.setArg(2, aux);
        m_kernelColorNoiseBlendLinear.setArg(3, w);
        m_kernelColorNoiseBlendLinear.setArg(4, h);
        m_kernelColorNoiseBlendLinear.setArg(5, colorNoise);
        queue.enqueueNDRangeKernel(m_kernelColorNoiseBlendLinear, cl::NullRange, global, cl::NullRange);

        queue.enqueueCopyBuffer(aux, buf, 0, 0, f4Bytes);
    }

    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

DenoiseEffect::DenoiseEffect()
    : m_controls(nullptr),
      m_strengthParam(nullptr),
      m_shadowPreserveParam(nullptr),
      m_colorNoiseParam(nullptr),
      m_algorithmCombo(nullptr),
      m_algorithm(0)
{}

DenoiseEffect::~DenoiseEffect() {}

QString DenoiseEffect::getName()        const { return "Denoise"; }
QString DenoiseEffect::getDescription() const { return "Reduces luminance and colour noise"; }
QString DenoiseEffect::getVersion()     const { return "1.0.0"; }

bool DenoiseEffect::initialize() {
    qDebug() << "Denoise effect initialized";
    return true;
}

QWidget* DenoiseEffect::createControlsWidget() {
    if (m_controls) return m_controls;

    m_controls = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_controls);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    QLabel* algoLabel = new QLabel("Algorithm:");
    algoLabel->setStyleSheet("color: #2C2018;");
    layout->addWidget(algoLabel);

    m_algorithmCombo = new QComboBox();
    m_algorithmCombo->addItem("Gaussian blend");
    m_algorithmCombo->addItem("Bilateral");
    m_algorithmCombo->setToolTip(
        "Gaussian blend: blurs the image and mixes with the original — cheap but cannot tell noise from detail.\n"
        "Bilateral: edge-aware; smooths flat regions while preserving tonal edges. Slower.");
    m_algorithmCombo->setStyleSheet(
        "QComboBox { color: #2C2018; background-color: #F4F1EA;"
        "            border: 1px solid #CCC5B5; border-radius: 3px; padding: 3px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #2C2018; background-color: #F4F1EA; }");
    layout->addWidget(m_algorithmCombo);
    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        m_algorithm = index;
        emit parametersChanged();
    });

    m_strengthParam = new ParamSlider("Strength", 0, 100);
    m_strengthParam->setValue(30);
    m_strengthParam->setToolTip("Amount of luminance (brightness) noise reduction. Higher values blend more strongly toward the smoothed result.\nTip: keep below 60 to preserve fine texture.");
    connect(m_strengthParam, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
    connect(m_strengthParam, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    layout->addWidget(m_strengthParam);

    m_shadowPreserveParam = new ParamSlider("Shadow Preserve", 0, 100);
    m_shadowPreserveParam->setValue(50);
    m_shadowPreserveParam->setToolTip("Reduces denoising strength in dark areas to protect shadow detail from over-smoothing.\nAt 100, the darkest pixels receive almost no noise reduction.");
    connect(m_shadowPreserveParam, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
    connect(m_shadowPreserveParam, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    layout->addWidget(m_shadowPreserveParam);

    m_colorNoiseParam = new ParamSlider("Color Noise", 0, 100);
    m_colorNoiseParam->setValue(50);
    m_colorNoiseParam->setToolTip("Reduces chroma (colour speckle) noise by blending colour channels toward a heavily-smoothed reference while keeping luminance unchanged.");
    connect(m_colorNoiseParam, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
    connect(m_colorNoiseParam, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    layout->addWidget(m_colorNoiseParam);

    layout->addStretch();
    return m_controls;
}

QMap<QString, QVariant> DenoiseEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["strength"]       = m_strengthParam       ? static_cast<int>(m_strengthParam->value())       : 30;
    params["shadowPreserve"] = m_shadowPreserveParam ? static_cast<int>(m_shadowPreserveParam->value()) : 50;
    params["colorNoise"]     = m_colorNoiseParam     ? static_cast<int>(m_colorNoiseParam->value())     : 50;
    params["algorithm"]      = m_algorithm;
    return params;
}

QImage DenoiseEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const float strength       = parameters.value("strength",       30).toInt() / 100.0f;
    const float shadowPreserve = parameters.value("shadowPreserve", 50).toInt() / 100.0f;
    const float colorNoise     = parameters.value("colorNoise",     50).toInt() / 100.0f;

    if (strength == 0.0f && colorNoise == 0.0f) return image;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, strength, shadowPreserve, colorNoise);
    return processImageGPU(image, strength, shadowPreserve, colorNoise);
}
