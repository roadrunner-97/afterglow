#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QVector>
#include <atomic>
#include <cstdint>
#include <memory>
#include "EffectManager.h"
#include "GpuPipeline.h"
#include "LongExposureStack.h"
#include "PhotoEditorEffect.h"
#include "SettingsImporter.h"

template <typename T> class QFutureWatcher;
struct StagedStackMaster;

struct StackProcessingResult {
    QImage  image;
    QString error;
    bool    cancelled    = false;
    int     cachedFrames = 0;
    // Keeps the unpublished file alive until delivery (or processor teardown).
    std::shared_ptr<StagedStackMaster> stagedMaster = {};
};

/**
 * @brief Runs the effect pipeline asynchronously via QtConcurrent.
 *
 * A generation counter lets processImageAsync() be called any number of times
 * in quick succession: only the latest result is delivered via processingComplete.
 */
class ImageProcessor : public QObject {
    Q_OBJECT

public:
    explicit ImageProcessor(QObject *parent = nullptr);
    ~ImageProcessor() override;

    // bypassEffects=true skips the entire effect list — used by the
    // \-key "before" preview so the viewport falls back to the raw,
    // un-edited image without disturbing per-effect enabled flags.
    void processImageAsync(const QImage &originalImage, const EffectManager &effects, ViewportRequest viewport = {},
                           RunMode mode = RunMode::Commit, bool bypassEffects = false,
                           const QVector<LocalAdjustment> &localAdjustments = {});

    uint64_t exportImageAsync(const QImage &originalImage, const EffectManager &effects, QString destinationPath,
                              const QVector<LocalAdjustment> &localAdjustments = {});

    // Renders every included frame with one immutable snapshot of the golden
    // reference's settings, then combines the rendered frames with a lighten
    // blend. Frames are decoded and released one at a time.
    void processStackAsync(const QVector<StackFrame> &frames, const EffectManager &effects,
                           const SettingsImporter::Settings &referenceSettings,
                           const StackAggregationConfig &aggregation, const QString &projectFolder,
                           const QString &masterPath);
    void cancelStackProcessing();
    bool isStackProcessing() const;

signals:
    void processingStarted();
    // `offset` is the top-left position of `result` within the requested
    // viewport.  When the image fills the viewport (or for export, where no
    // viewport is requested), offset is (0, 0).  Receivers blit `result` at
    // `offset` and leave the surrounding letterbox to the viewport's clear.
    void processingComplete(QImage result, QPoint offset);
    void exportComplete(uint64_t requestId, QImage result, QString destinationPath);
    void stackProcessingStarted(int totalFrames);
    void stackProcessingProgress(int completedFrames, int totalFrames, QString path);
    void stackProcessingComplete(QImage result, QString error, bool cancelled, int cachedFrames);

private:
    std::shared_ptr<std::atomic<uint64_t>> generationPtr         = std::make_shared<std::atomic<uint64_t>>(0);
    std::shared_ptr<std::atomic<uint64_t>> m_stackGeneration     = std::make_shared<std::atomic<uint64_t>>(0);
    uint64_t                               m_nextExportRequestId = 0;
    QFutureWatcher<StackProcessingResult> *m_stackWatcher        = nullptr;

    std::shared_ptr<GpuPipeline> m_pipeline = std::make_shared<GpuPipeline>();
};

#endif // IMAGEPROCESSOR_H
