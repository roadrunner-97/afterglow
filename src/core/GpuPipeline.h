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

class IGpuEffect;

struct GpuPipelineCall {
    PhotoEditorEffect*      effect;
    IGpuEffect*             gpu;     // pre-resolved interface pointer; never null
    QMap<QString, QVariant> params;
};

struct ViewportRequest {
    QSize   displaySize;
    float   zoom   = 1.0f;
    QPointF center = {0.5, 0.5};
};

// Run mode selects how the pipeline handles the full-res post-effect cache.
//
//  Commit    Image load or slider-release: run effects on the full-res buffer,
//            (re)populate m_processedBuf, then downsample-from-cache for display.
//  LiveDrag  Live slider drag: bypass the cache and run effects on a preview
//            buffer (fast, approximate).  Invalidates the cache.
//  PanZoom   Pan/zoom: if the cache is valid, downsample from it with no effect
//            re-run.  If invalid, falls through to LiveDrag behaviour.
enum class RunMode { Commit, LiveDrag, PanZoom };

/**
 * @brief Shared GPU context with persistent source + full-res post-effect cache.
 *
 * Owned as a shared_ptr by ImageProcessor so the QtConcurrent worker lambda
 * can safely capture it.
 *
 * The pipeline holds two persistent full-resolution buffers on the GPU:
 *   * m_srcBuf       — the unmodified original, uploaded once per image load.
 *   * m_processedBuf — the post-effect result in linear float4, rebuilt by
 *                      Commit-mode runs and sampled by PanZoom-mode runs.
 *
 * On Commit the source is decoded 1:1 into m_processedBuf, all enabled effects
 * run on it at full resolution, and the cache is marked valid.  Pan/zoom then
 * reuses the cache: a single float4→float4 box-average downsample produces the
 * visible preview, skipping effects entirely.  Live slider drags use the
 * legacy small-preview pipeline, invalidating the cache; the next Commit
 * rebuilds it.
 *
 * Thread safety: run() is serialised by an internal mutex.
 */
class GpuPipeline {
public:
    GpuPipeline() = default;

    // Run the pipeline.  Returns {} on failure.  Default mode is Commit so
    // callers like exportImageAsync() get a full-fidelity pass without opt-in.
    QImage run(const QImage& image, const QVector<GpuPipelineCall>& calls,
               const ViewportRequest& viewport, RunMode mode = RunMode::Commit);

private:
    // All must be called with m_mutex held.
    bool initContext();
    bool initDownsampleKernels();
    void uploadImageLocked(const QImage& image);
    // Decode srcBuf → processedBuf at full resolution.  Allocates processedBuf
    // and fullAuxBuf on first use or when the image dimensions change.
    bool decodeFullResLocked();
    // float4 → uint sRGB pack + readback from the given source buffer.
    QImage packAndReadbackLocked(cl::Buffer& src, int w, int h);

    std::mutex       m_mutex;
    cl::Context      m_context;
    cl::CommandQueue m_queue;
    cl::Device       m_device;

    cl::Buffer m_srcBuf;        // persistent original — written once per image load
    cl::Buffer m_processedBuf;  // full-res cl_float4 linear post-effect cache
    cl::Buffer m_fullAuxBuf;    // full-res cl_float4 scratch for effect ping-pong
    cl::Buffer m_workBuf;       // preview-sized cl_float4 linear; sampled or edited by effects
    cl::Buffer m_auxBuf;        // preview-sized cl_float4 scratch (blur, unsharp ping-pong)
    cl::Buffer m_packedBuf;     // preview-sized uint (RGB32 sRGB) — final pack target, read back to CPU

    // Downsample kernels — output float4 linear sRGB (scene-linear, unclamped).
    cl::Kernel m_downsampleKernel8Srgb;     //  8-bit sRGB uint  → float4 linear (sRGB decode)
    cl::Kernel m_downsampleKernel16Srgb;    // 16-bit sRGB ushort → float4 linear (sRGB decode)
    cl::Kernel m_downsampleKernel16Linear;  // 16-bit linear ushort → float4 linear (divide by 65535)
    cl::Kernel m_downsampleKernelFloat4;    // float4 linear → float4 linear (crop + box average)
    cl::Kernel m_decodeKernel8Srgb;         // 1:1 decode:  8-bit sRGB uint   → float4 linear
    cl::Kernel m_decodeKernel16Srgb;        // 1:1 decode: 16-bit sRGB ushort → float4 linear
    cl::Kernel m_decodeKernel16Linear;      // 1:1 decode: 16-bit linear ushort → float4 linear
    cl::Kernel m_packKernel;                // float4 linear → uint sRGB (clamp + gamma + pack RGB32)

    int        m_previewW = 0;
    int        m_previewH = 0;

    int    m_width    = 0;
    int    m_height   = 0;
    int    m_stride   = 0;
    size_t m_bufBytes = 0;
    bool   m_is16bit  = false;
    bool   m_inputIsLinear = false;  // true if source is scene-linear (RAW via LibRaw gamm=1)

    // Full-res cache state.  m_processedBytes tracks the current allocation so
    // we can reallocate when a new image of a different size is loaded.
    bool   m_processedValid = false;
    size_t m_processedBytes = 0;

    bool        m_available = false;
    int         m_revision  = -1;
    const void* m_lastBits  = nullptr;  // detect image changes between runs

    // Tracks which IGpuEffect instances have had initGpuKernels() called for
    // the current context.  Cleared on device change so re-enabled effects
    // get lazily re-compiled on their next appearance in a calls list.
    std::unordered_set<void*> m_initializedEffects;
};

#endif // GPUPIPELINE_H
