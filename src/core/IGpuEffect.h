#ifndef IGPUEFFECT_H
#define IGPUEFFECT_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <QMap>
#include <QVariant>

/**
 * @brief Mixin interface for effects that participate in the shared GPU pipeline.
 *
 * The pipeline uploads the source image once on load, copies srcBuf→workBuf,
 * chains all effect kernels (no finish() between them), then does a single
 * readback.  This eliminates per-effect PCIe upload/readback overhead.
 *
 * Threading: initGpuKernels() and enqueueGpu() are always called from the
 * same mutex-protected worker thread — never concurrently.
 */
class IGpuEffect {
public:
    virtual ~IGpuEffect() = default;

    // Called when the shared pipeline context is (re)created.
    // Compile kernels into ctx and store them as member variables.
    // Effects that need to allocate temporary buffers in enqueueGpu()
    // should store a copy of ctx here.
    virtual bool initGpuKernels(cl::Context& ctx, cl::Device& dev) = 0;

    // Enqueue this effect's GPU work onto queue.
    //   buf   — main working buffer; read input, write output
    //   aux   — scratch buffer, same size as buf (may be used as ping-pong)
    //   stride — pixels per row (= bytesPerLine / bytes_per_pixel)
    //   is16bit — true: Format_RGBX64 (8 B/px), false: Format_RGB32 (4 B/px)
    // Do NOT call queue.finish() — GpuPipeline calls it once after all effects.
    // Return false on error (pipeline falls back to per-effect QImage chain).
    virtual bool enqueueGpu(cl::CommandQueue& queue,
                             cl::Buffer& buf, cl::Buffer& aux,
                             int w, int h, int stride, bool is16bit,
                             const QMap<QString, QVariant>& params) = 0;
};

#endif // IGPUEFFECT_H
