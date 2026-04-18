#ifndef GPUPIPELINE_H
#define GPUPIPELINE_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include "PhotoEditorEffect.h"
#include <QImage>
#include <QMap>
#include <QPointF>
#include <QSize>
#include <QVariant>
#include <QVector>
#include <mutex>
#include <unordered_set>

struct GpuPipelineCall {
    PhotoEditorEffect*      effect;
    QMap<QString, QVariant> params;
};

struct ViewportRequest {
    QSize   displaySize;
    float   zoom   = 1.0f;
    QPointF center = {0.5, 0.5};
};

/**
 * @brief Shared GPU context with persistent source buffer.
 *
 * Owned as a shared_ptr by ImageProcessor so the QtConcurrent worker lambda
 * can safely capture it.
 *
 * run() detects the new constBits() pointer and uploads the full-res image to
 * srcBuf (once per unique image).  On every pipeline run it first downsamples
 * srcBuf → workBuf (preview-sized RGB32), then chains all effect kernels on
 * the small preview buffer, issues a single queue.finish(), and reads workBuf
 * back to the CPU as a QImage.  Effects run at preview resolution, not full
 * resolution, giving ~26× speedup on typical RAW files.
 *
 * Thread safety: run() is serialised by an internal mutex.
 */
class GpuPipeline {
public:
    GpuPipeline() = default;

    // Run the pipeline.  Returns {} on failure.
    // viewportOnly is accepted for API compatibility but ignored — the
    // downsample-first pipeline is always fast enough to run in full.
    QImage run(const QImage& image, const QVector<GpuPipelineCall>& calls,
               const ViewportRequest& viewport, bool viewportOnly = false);

private:
    // All must be called with m_mutex held.
    bool initContext();
    bool initDownsampleKernels();
    void uploadImageLocked(const QImage& image);

    std::mutex       m_mutex;
    cl::Context      m_context;
    cl::CommandQueue m_queue;
    cl::Device       m_device;

    cl::Buffer m_srcBuf;     // persistent original — written once per image load
    cl::Buffer m_workBuf;    // preview-sized cl_float4 linear; modified by effects in-place
    cl::Buffer m_auxBuf;     // preview-sized cl_float4 scratch (blur, unsharp ping-pong)
    cl::Buffer m_packedBuf;  // preview-sized uint (RGB32 sRGB) — final pack target, read back to CPU

    // Downsample kernels — output float4 linear sRGB (scene-linear, unclamped).
    cl::Kernel m_downsampleKernel8Srgb;     //  8-bit sRGB uint  → float4 linear (sRGB decode)
    cl::Kernel m_downsampleKernel16Srgb;    // 16-bit sRGB ushort → float4 linear (sRGB decode)
    cl::Kernel m_downsampleKernel16Linear;  // 16-bit linear ushort → float4 linear (divide by 65535)
    cl::Kernel m_packKernel;                // float4 linear → uint sRGB (clamp + gamma + pack RGB32)

    int        m_previewW = 0;
    int        m_previewH = 0;

    int    m_width    = 0;
    int    m_height   = 0;
    int    m_stride   = 0;
    size_t m_bufBytes = 0;
    bool   m_is16bit  = false;
    bool   m_inputIsLinear = false;  // true if source is scene-linear (RAW via LibRaw gamm=1)

    bool        m_available = false;
    int         m_revision  = -1;
    const void* m_lastBits  = nullptr;  // detect image changes between runs

    // Tracks which IGpuEffect instances have had initGpuKernels() called for
    // the current context.  Cleared on device change so re-enabled effects
    // get lazily re-compiled on their next appearance in a calls list.
    std::unordered_set<void*> m_initializedEffects;
};

#endif // GPUPIPELINE_H
