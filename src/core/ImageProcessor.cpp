#include "ImageProcessor.h"
#include "GpuPipeline.h"
#include "ICropSource.h"
#include "IGpuEffect.h"
#include "PhotoEditorEffect.h"
#include "RawLoader.h"
#include "SettingsExporter.h"
#include "StackFrameCache.h"
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <QColorSpace>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPointer>
#include <QTransform>
#include <algorithm>
#include <cmath>

ImageProcessor::ImageProcessor(QObject *parent) : QObject(parent) {}

ImageProcessor::~ImageProcessor() {
    ++(*m_stackGeneration);
    if (m_stackWatcher && m_stackWatcher->isRunning()) m_stackWatcher->future().waitForFinished();
}

// Build the injection map for non-destructive crop.  Geometry-aware effects
// (vignette, film grain, ...) read these keys to operate on the cropped
// frame; effects that ignore them pay only a QMap lookup miss.  No-op when
// the supplied source is null.
static QMap<QString, QVariant> buildCropInjection(ICropSource *src) {
    if (!src) return {};
    const QRectF r = src->userCropRect();
    return {
        {"_userCropX0", r.left()},
        {"_userCropY0", r.top()},
        {"_userCropX1", r.right()},
        {"_userCropY1", r.bottom()},
        {"_userCropAngle", static_cast<double>(src->userCropAngle())},
    };
}

static void mergeInto(QMap<QString, QVariant> &dst, const QMap<QString, QVariant> &src) {
    for (auto it = src.constBegin(); it != src.constEnd(); ++it) dst.insert(it.key(), it.value());
}

// Builds the per-frame GPU call list from the manager's enabled entries.
// Every shipping effect implements IGpuEffect, so the cached entry.gpu
// pointer is always non-null and there is no QImage / CPU fallback path.
static QVector<GpuPipelineCall> buildGpuCalls(const EffectManager &effects) {
    const QMap<QString, QVariant> cropInjected = buildCropInjection(effects.activeCropSource());
    QVector<GpuPipelineCall>      calls;
    calls.reserve(effects.entries().size());
    for (const EffectEntry &entry : effects.entries()) {
        Q_ASSERT(entry.gpu); // every shipping effect implements IGpuEffect
        QMap<QString, QVariant> params = effects.effectiveParameters(entry);
        mergeInto(params, cropInjected);
        calls.append({entry.effect, entry.gpu, params, entry.enabled});
    }
    return calls;
}

struct StackGeometry {
    QString committed;
    QRectF  crop{0.0, 0.0, 1.0, 1.0};
    double  angle        = 0.0;
    bool    applyPending = false;
};

static const SettingsImporter::EffectSettings *settingsForEntry(const EffectEntry                &entry,
                                                                const SettingsImporter::Settings &settings) {
    for (const auto &candidate : settings.effects) {
        if ((!candidate.id.isEmpty() && candidate.id == entry.effect->getId()) ||
            (candidate.id.isEmpty() && candidate.name == entry.effect->getName()))
            return &candidate;
    }
    return nullptr;
}

static QVector<GpuPipelineCall>
buildStackGpuCalls(const EffectManager &effects, const SettingsImporter::Settings &settings, StackGeometry *geometry) {
    QVector<GpuPipelineCall> calls;
    calls.reserve(effects.entries().size());
    QMap<QString, QVariant> cropParameters;
    bool                    cropEnabled = false;

    for (const EffectEntry &entry : effects.entries()) {
        Q_ASSERT(entry.gpu);
        const auto                   *snapshot = settingsForEntry(entry, settings);
        const bool                    enabled  = snapshot ? snapshot->enabled : entry.enabled;
        const QMap<QString, QVariant> params   = snapshot ? snapshot->parameters : effects.effectiveParameters(entry);
        calls.append({entry.effect, entry.gpu, params, enabled});
        if (entry.crop) {
            cropParameters = params;
            cropEnabled    = enabled;
        }
    }

    if (geometry && !cropParameters.isEmpty()) {
        geometry->committed = cropParameters.value("committedGeometry").toString();
        const double x0     = cropParameters.value("cropX0", 0.0).toDouble();
        const double y0     = cropParameters.value("cropY0", 0.0).toDouble();
        const double x1     = cropParameters.value("cropX1", 1.0).toDouble();
        const double y1     = cropParameters.value("cropY1", 1.0).toDouble();
        geometry->crop      = QRectF(x0, y0, x1 - x0, y1 - y0);
        geometry->angle =
            cropParameters.value("angle", 0.0).toDouble() + cropParameters.value("quarterTurns", 0).toInt() * 90.0;
        geometry->applyPending = cropEnabled;
    }

    if (cropEnabled) {
        const QMap<QString, QVariant> injection{
            {"_userCropX0", cropParameters.value("cropX0", 0.0)}, {"_userCropY0", cropParameters.value("cropY0", 0.0)},
            {"_userCropX1", cropParameters.value("cropX1", 1.0)}, {"_userCropY1", cropParameters.value("cropY1", 1.0)},
            {"_userCropAngle", geometry ? geometry->angle : 0.0},
        };
        for (GpuPipelineCall &call : calls) mergeInto(call.params, injection);
    }
    return calls;
}

static QImage decodeStackFrame(const QString &path) {
    if (RawLoader::isRawFile(path)) return RawLoader::load(path);
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}

static QImage applyGeometry(const QImage &source, const QRectF &crop, double angle) {
    if (source.isNull()) return {};
    const QSize dstSize(std::max(1, static_cast<int>(std::round(crop.width() * source.width()))),
                        std::max(1, static_cast<int>(std::round(crop.height() * source.height()))));
    QImage      result(dstSize, source.format());
    result.setColorSpace(source.colorSpace());
    for (const QString &key : source.textKeys()) result.setText(key, source.text(key));
    result.fill(Qt::black);
    QPainter painter(&result);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    QTransform transform;
    transform.translate(dstSize.width() * 0.5, dstSize.height() * 0.5);
    transform.rotate(-angle);
    transform.translate(-crop.center().x() * source.width(), -crop.center().y() * source.height());
    painter.setTransform(transform);
    painter.drawImage(0, 0, source);
    return result;
}

static QImage applyCommittedGeometry(const QImage &source, const QString &encodedOperations) {
    QImage           result     = source;
    const QJsonArray operations = QJsonDocument::fromJson(QByteArray::fromBase64(encodedOperations.toLatin1())).array();
    for (const QJsonValue &value : operations) {
        const QJsonObject op = value.toObject();
        result = applyGeometry(result,
                               QRectF(op["x"].toDouble(), op["y"].toDouble(), op["w"].toDouble(), op["h"].toDouble()),
                               op["angle"].toDouble());
    }
    return result;
}

void ImageProcessor::processImageAsync(const QImage &originalImage, const EffectManager &effects,
                                       ViewportRequest viewport, RunMode mode, bool bypassEffects,
                                       const QVector<LocalAdjustment> &localAdjustments) {
    auto     genPtr = generationPtr;
    uint64_t myGen  = ++(*genPtr);

    // Snapshot parameters on the calling (main) thread so effect QObjects
    // are never touched from the worker thread.
    QVector<GpuPipelineCall> calls  = bypassEffects ? QVector<GpuPipelineCall>{} : buildGpuCalls(effects);
    QVector<LocalAdjustment> locals = bypassEffects ? QVector<LocalAdjustment>{} : localAdjustments;

    emit processingStarted();

    auto *watcher = new QFutureWatcher<GpuPipelineResult>(this);
    connect(watcher, &QFutureWatcher<GpuPipelineResult>::finished, this, [this, watcher, myGen, genPtr]() {
        if (myGen == genPtr->load(std::memory_order_relaxed)) {
            const GpuPipelineResult r = watcher->result();
            emit                    processingComplete(r.image, r.offset);
        }
        watcher->deleteLater();
    });

    auto pipeline = m_pipeline;
    watcher->setFuture(QtConcurrent::run([image = originalImage, calls = std::move(calls), genPtr, myGen, pipeline,
                                          viewport, mode, locals = std::move(locals)]() -> GpuPipelineResult {
        if (genPtr->load(std::memory_order_relaxed) != myGen) return {};
        return pipeline->run(image, calls, viewport, mode, locals);
    }));
}

uint64_t ImageProcessor::exportImageAsync(const QImage &originalImage, const EffectManager &effects,
                                          QString destinationPath, const QVector<LocalAdjustment> &localAdjustments) {
    QVector<GpuPipelineCall> calls     = buildGpuCalls(effects);
    const uint64_t           requestId = ++m_nextExportRequestId;

    auto *watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher, requestId, destinationPath]() {
        emit exportComplete(requestId, watcher->result(), destinationPath);
        watcher->deleteLater();
    });

    auto pipeline = m_pipeline;
    watcher->setFuture(QtConcurrent::run(
        [image = originalImage, calls = std::move(calls), pipeline, locals = localAdjustments]() -> QImage {
            // Export has no viewport, so the pipeline returns the full-resolution
            // image with offset (0, 0).  Strip the offset; exportComplete only
            // needs the pixels.
            return pipeline->run(image, calls, {}, RunMode::Commit, locals).image;
        }));
    return requestId;
}

void ImageProcessor::processStackAsync(const QVector<StackFrame> &frames, const EffectManager &effects,
                                       const SettingsImporter::Settings &referenceSettings) {
    if (isStackProcessing()) return;

    QVector<QString> includedPaths;
    includedPaths.reserve(frames.size());
    for (const StackFrame &frame : frames)
        if (frame.decision == StackFrameDecision::Include) includedPaths.append(frame.path);

    if (includedPaths.isEmpty()) {
        emit stackProcessingComplete({}, QStringLiteral("Include at least one frame before rebuilding the stack."),
                                     false, 0);
        return;
    }

    StackGeometry              geometry;
    QVector<GpuPipelineCall>   calls        = buildStackGpuCalls(effects, referenceSettings, &geometry);
    QVector<LocalAdjustment>   locals       = referenceSettings.localAdjustments;
    const int                  total        = static_cast<int>(includedPaths.size());
    auto                       pipeline     = m_pipeline;
    auto                       generation   = m_stackGeneration;
    const uint64_t             myGeneration = ++(*generation);
    QPointer<ImageProcessor>   target(this);
    SettingsImporter::Settings effectiveSettings;
    effectiveSettings.localAdjustments = locals;
    for (const GpuPipelineCall &call : calls) {
        effectiveSettings.effects.append({call.effect->getId(), call.effect->getName(), call.enabled, call.params});
    }
    QByteArray settingsSignature = SettingsExporter::toYaml(effectiveSettings).toUtf8();
    for (const GpuPipelineCall &call : calls) settingsSignature.append(call.effect->getVersion().toUtf8());

    emit stackProcessingStarted(total);
    m_stackWatcher = new QFutureWatcher<StackProcessingResult>(this);
    auto *watcher  = m_stackWatcher;
    connect(watcher, &QFutureWatcher<StackProcessingResult>::finished, this,
            [this, watcher, generation, myGeneration]() {
                const StackProcessingResult result = watcher->result();
                if (m_stackWatcher == watcher) m_stackWatcher = nullptr;
                watcher->deleteLater();
                // A newer request owns the UI; an older worker finishing must
                // not replace its progress or result.
                if (generation->load(std::memory_order_relaxed) != myGeneration && !result.cancelled) return;
                emit stackProcessingComplete(result.image, result.error, result.cancelled, result.cachedFrames);
            });

    watcher->setFuture(
        QtConcurrent::run([paths = std::move(includedPaths), calls = std::move(calls), locals = std::move(locals),
                           geometry, pipeline = std::move(pipeline), generation, myGeneration, target, total,
                           settingsSignature = std::move(settingsSignature)]() -> StackProcessingResult {
            QImage accumulator;
            int    cachedFrames = 0;
            for (int i = 0; i < paths.size(); ++i) {
                if (generation->load(std::memory_order_relaxed) != myGeneration) return {{}, {}, true};

                const QString    path              = paths[i];
                const QByteArray renderFingerprint = StackFrameCache::fingerprint(path, settingsSignature);
                QImage           rendered          = StackFrameCache::load(path, renderFingerprint);
                if (!rendered.isNull()) {
                    ++cachedFrames;
                } else {
                    QImage frame = decodeStackFrame(path);
                    if (frame.isNull()) return {{}, QStringLiteral("Could not decode %1.").arg(path), false};
                    frame    = applyCommittedGeometry(frame, geometry.committed);
                    rendered = pipeline->run(frame, calls, {}, RunMode::Commit, locals).image;
                    if (rendered.isNull()) return {{}, QStringLiteral("Processing failed for %1.").arg(path), false};
                    StackFrameCache::store(path, renderFingerprint, rendered);
                }

                QString blendError;
                if (!LongExposureStack::blendLighten(&accumulator, rendered, &blendError))
                    return {{}, QStringLiteral("Could not add %1: %2").arg(path, blendError), false};

                if (target) {
                    QMetaObject::invokeMethod(
                        target,
                        [target, completed = i + 1, total, path]() {
                            if (target) emit target->stackProcessingProgress(completed, total, path);
                        },
                        Qt::QueuedConnection);
                }
            }

            const QRectF fullFrame(0.0, 0.0, 1.0, 1.0);
            if (geometry.applyPending && (geometry.crop != fullFrame || std::abs(geometry.angle) > 0.0001))
                accumulator = applyGeometry(accumulator, geometry.crop, geometry.angle);
            return {accumulator, {}, false, cachedFrames};
        }));
}

void ImageProcessor::cancelStackProcessing() {
    if (!isStackProcessing()) return;
    ++(*m_stackGeneration);
}

bool ImageProcessor::isStackProcessing() const {
    return m_stackWatcher && m_stackWatcher->isRunning();
}
