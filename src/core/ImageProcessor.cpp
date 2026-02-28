#include "ImageProcessor.h"
#include "GpuPipeline.h"
#include "PhotoEditorEffect.h"
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <QElapsedTimer>
#include <QDebug>

ImageProcessor::ImageProcessor(QObject *parent)
    : QObject(parent) {}

void ImageProcessor::cancelProcessing() {
    ++(*generationPtr);
    emit processingCancelled();
}

void ImageProcessor::processImageAsync(const QImage &originalImage,
                                       const QVector<PhotoEditorEffect*> &effects,
                                       ViewportRequest viewport,
                                       bool viewportOnly) {
    auto genPtr = generationPtr;
    uint64_t myGen = ++(*genPtr);

    // Snapshot parameters on the calling (main) thread so effect QObjects
    // are never touched from the worker thread.
    using EffectCall = QPair<PhotoEditorEffect*, QMap<QString, QVariant>>;
    QVector<EffectCall> imageCalls;
    imageCalls.reserve(effects.size());

    QVector<GpuPipelineCall> gpuCalls;
    gpuCalls.reserve(effects.size());

    bool allGpu = !effects.isEmpty();
    for (PhotoEditorEffect *e : effects) {
        QMap<QString, QVariant> params = e->getParameters();
        imageCalls.append({e, params});
        if (e->supportsGpuInPlace())
            gpuCalls.append({e, params});
        else
            allGpu = false;
    }

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
             genPtr, myGen, pipeline, viewport, viewportOnly]() -> QImage {
                if (genPtr->load(std::memory_order_relaxed) != myGen) return {};
                return pipeline->run(image, calls, viewport, viewportOnly);
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
