#ifndef GPUPIPELINE_H
#define GPUPIPELINE_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include "PhotoEditorEffect.h"
#include <QImage>
#include <QMap>
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
    QSize displaySize;
    // future pan/zoom: QRectF sourceRegion; // normalised [0,1]² crop rect
};

/**
 * @brief Shared GPU context with persistent source buffer.
 *
 * Owned as a shared_ptr by ImageProcessor so the QtConcurrent worker lambda
 * can safely capture it.
 *
 * On image load: run() detects the new constBits() pointer and uploads the
 * image to srcBuf (once per unique image).  On every pipeline run it copies
 * srcBuf→workBuf (GPU-to-GPU, no PCIe), chains all effect kernels, issues a
 * single queue.finish(), then downsamples to the viewport size on-GPU and
 * reads back only the small preview buffer.
 *
 * Thread safety: run() is serialised by an internal mutex.
 */
class GpuPipeline {
public:
    GpuPipeline() = default;

    // Run the full pipeline.  Returns {} on failure; caller falls back to
    // the per-effect QImage chain.
    QImage run(const QImage& image, const QVector<GpuPipelineCall>& calls,
               const ViewportRequest& viewport);

private:
    // All must be called with m_mutex held.
    bool initContext();
    bool initPreviewKernels();
    void uploadImageLocked(const QImage& image);

    std::mutex       m_mutex;
    cl::Context      m_context;
    cl::CommandQueue m_queue;
    cl::Device       m_device;

    cl::Buffer m_srcBuf;   // persistent original — written once per image load
    cl::Buffer m_workBuf;  // scratch modified by each pipeline run
    cl::Buffer m_auxBuf;   // scratch for multi-pass effects (blur, unsharp)

    // Preview downsampling
    cl::Kernel m_previewKernel8;   // 8-bit source → 8-bit preview
    cl::Kernel m_previewKernel16;  // 16-bit source → 8-bit preview
    cl::Buffer m_previewBuf;       // write-only; always 8-bit ARGB
    int        m_previewW = 0;
    int        m_previewH = 0;

    int    m_width    = 0;
    int    m_height   = 0;
    int    m_stride   = 0;
    size_t m_bufBytes = 0;
    bool   m_is16bit  = false;

    bool        m_available = false;
    int         m_revision  = -1;
    const void* m_lastBits  = nullptr;  // detect image changes between runs

    // Tracks which IGpuEffect instances have had initGpuKernels() called for
    // the current context.  Cleared on device change so re-enabled effects
    // get lazily re-compiled on their next appearance in a calls list.
    std::unordered_set<void*> m_initializedEffects;
};

#endif // GPUPIPELINE_H
