#include "ImageProcessor.h"
#include "GpuPipeline.h"
#include "ICropSource.h"
#include "IGpuEffect.h"
#include "PhotoEditorEffect.h"
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <QElapsedTimer>
#include <QDebug>

ImageProcessor::ImageProcessor(QObject *parent)
    : QObject(parent) {}

// Build the injection map for non-destructive crop.  Geometry-aware effects
// (vignette, film grain, ...) read these keys to operate on the cropped
// frame; effects that ignore them pay only a QMap lookup miss.  No-op when
// the supplied source is null.
static QMap<QString, QVariant> buildCropInjection(ICropSource* src) {
    if (!src) return {};
    const QRectF r = src->userCropRect();
    return {
        {"_userCropX0",    r.left()},
        {"_userCropY0",    r.top()},
        {"_userCropX1",    r.right()},
        {"_userCropY1",    r.bottom()},
        {"_userCropAngle", static_cast<double>(src->userCropAngle())},
    };
}

static void mergeInto(QMap<QString, QVariant>& dst,
                      const QMap<QString, QVariant>& src) {
    for (auto it = src.constBegin(); it != src.constEnd(); ++it)
        dst.insert(it.key(), it.value());
}

void ImageProcessor::processImageAsync(const QImage &originalImage,
                                       const EffectManager &effects,
                                       ViewportRequest viewport,
                                       RunMode mode) {
    auto genPtr = generationPtr;
    uint64_t myGen = ++(*genPtr);

    // Snapshot parameters on the calling (main) thread so effect QObjects
    // are never touched from the worker thread.
    using EffectCall = QPair<PhotoEditorEffect*, QMap<QString, QVariant>>;
    QVector<EffectCall> imageCalls;
    QVector<GpuPipelineCall> gpuCalls;
    imageCalls.reserve(effects.entries().size());
    gpuCalls.reserve(effects.entries().size());

    const QMap<QString, QVariant> cropInjected = buildCropInjection(effects.activeCropSource());

    bool any = false;
    bool allGpu = true;
    for (const EffectEntry& entry : effects.entries()) {
        if (!entry.enabled) continue;
        any = true;
        QMap<QString, QVariant> params = entry.effect->getParameters();
        mergeInto(params, cropInjected);
        imageCalls.append({entry.effect, params});
        if (entry.gpu)
            gpuCalls.append({entry.effect, entry.gpu, params});
        else
            allGpu = false;
    }
    if (!any) allGpu = false;

    emit processingStarted();

    QElapsedTimer timer;
    timer.start();

    auto *watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, myGen, timer, genPtr]() {
        if (myGen == genPtr->load(std::memory_order_relaxed)) {
            qint64 us = (timer.nsecsElapsed() + 500) / 1000;
            qDebug() << "Image reprocessing took" << us << "µs";
            emit processingComplete(watcher->result());
        }
        watcher->deleteLater();
    });

    if (allGpu) {
        // Fast path: single upload (if image changed), GPU downsample, single readback.
        auto pipeline = m_pipeline;
        watcher->setFuture(QtConcurrent::run(
            [image = originalImage, calls = std::move(gpuCalls),
             genPtr, myGen, pipeline, viewport, mode]() -> QImage {
                if (genPtr->load(std::memory_order_relaxed) != myGen) return {};
                return pipeline->run(image, calls, viewport, mode);
            }
        ));
    } else {
        // Fallback: per-effect QImage chain (upload + readback per effect).
        watcher->setFuture(QtConcurrent::run(
            [image = originalImage, calls = std::move(imageCalls),
             genPtr, myGen, timer, viewport]() -> QImage {
                QImage result = image;
                for (const auto &call : calls) {
                    if (genPtr->load(std::memory_order_relaxed) != myGen) return {};
                    qint64 t0 = (timer.nsecsElapsed() + 500) / 1000;
                    result = call.first->processImage(result, call.second);
                    qint64 t1 = (timer.nsecsElapsed() + 500) / 1000;
                    qDebug() << "  [pipeline]" << call.first->getName()
                             << ":" << (t1 - t0) << "µs  (at" << t0 << "µs)";
                }
                if (viewport.displaySize.isValid() && !result.isNull())
                    result = result.scaled(viewport.displaySize, Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation);
                return result;
            }
        ));
    }
}

void ImageProcessor::exportImageAsync(const QImage& originalImage,
                                      const EffectManager& effects,
                                      QString destinationPath) {
    using EffectCall = QPair<PhotoEditorEffect*, QMap<QString, QVariant>>;
    QVector<EffectCall> imageCalls;
    QVector<GpuPipelineCall> gpuCalls;

    const QMap<QString, QVariant> cropInjected = buildCropInjection(effects.activeCropSource());

    bool any = false;
    bool allGpu = true;
    for (const EffectEntry& entry : effects.entries()) {
        if (!entry.enabled) continue;
        any = true;
        QMap<QString, QVariant> params = entry.effect->getParameters();
        mergeInto(params, cropInjected);
        imageCalls.append({entry.effect, params});
        if (entry.gpu)
            gpuCalls.append({entry.effect, entry.gpu, params});
        else
            allGpu = false;
    }
    if (!any) allGpu = false;

    auto* watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, destinationPath]() {
        emit exportComplete(watcher->result(), destinationPath);
        watcher->deleteLater();
    });

    if (allGpu) {
        auto pipeline = m_pipeline;
        watcher->setFuture(QtConcurrent::run(
            [image = originalImage, calls = std::move(gpuCalls), pipeline]() -> QImage {
                return pipeline->run(image, calls, {}, RunMode::Commit);
            }
        ));
    } else {
        watcher->setFuture(QtConcurrent::run(
            [image = originalImage, calls = std::move(imageCalls)]() -> QImage {
                QImage result = image;
                for (const auto& call : calls)
                    result = call.first->processImage(result, call.second);
                return result;
            }
        ));
    }
}
