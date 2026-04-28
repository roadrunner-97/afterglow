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

// Builds the per-frame GPU call list from the manager's enabled entries.
// Every shipping effect implements IGpuEffect, so the cached entry.gpu
// pointer is always non-null and there is no QImage / CPU fallback path.
static QVector<GpuPipelineCall> buildGpuCalls(const EffectManager& effects) {
    const QMap<QString, QVariant> cropInjected =
        buildCropInjection(effects.activeCropSource());
    QVector<GpuPipelineCall> calls;
    calls.reserve(effects.entries().size());
    for (const EffectEntry& entry : effects.entries()) {
        if (!entry.enabled) continue;
        Q_ASSERT(entry.gpu);  // every shipping effect implements IGpuEffect
        QMap<QString, QVariant> params = entry.effect->getParameters();
        mergeInto(params, cropInjected);
        calls.append({entry.effect, entry.gpu, params});
    }
    return calls;
}

void ImageProcessor::processImageAsync(const QImage &originalImage,
                                       const EffectManager &effects,
                                       ViewportRequest viewport,
                                       RunMode mode) {
    auto genPtr = generationPtr;
    uint64_t myGen = ++(*genPtr);

    // Snapshot parameters on the calling (main) thread so effect QObjects
    // are never touched from the worker thread.
    QVector<GpuPipelineCall> calls = buildGpuCalls(effects);

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

    auto pipeline = m_pipeline;
    watcher->setFuture(QtConcurrent::run(
        [image = originalImage, calls = std::move(calls),
         genPtr, myGen, pipeline, viewport, mode]() -> QImage {
            if (genPtr->load(std::memory_order_relaxed) != myGen) return {};
            return pipeline->run(image, calls, viewport, mode);
        }
    ));
}

void ImageProcessor::exportImageAsync(const QImage& originalImage,
                                      const EffectManager& effects,
                                      QString destinationPath) {
    QVector<GpuPipelineCall> calls = buildGpuCalls(effects);

    auto* watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, destinationPath]() {
        emit exportComplete(watcher->result(), destinationPath);
        watcher->deleteLater();
    });

    auto pipeline = m_pipeline;
    watcher->setFuture(QtConcurrent::run(
        [image = originalImage, calls = std::move(calls), pipeline]() -> QImage {
            return pipeline->run(image, calls, {}, RunMode::Commit);
        }
    ));
}
