#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QVector>
#include <atomic>
#include <memory>
#include "EffectManager.h"
#include "GpuPipeline.h"
#include "PhotoEditorEffect.h"

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

    // bypassEffects=true skips the entire effect list — used by the
    // \-key "before" preview so the viewport falls back to the raw,
    // un-edited image without disturbing per-effect enabled flags.
    void processImageAsync(const QImage &originalImage, const EffectManager &effects, ViewportRequest viewport = {},
                           RunMode mode = RunMode::Commit, bool bypassEffects = false);

    void exportImageAsync(const QImage &originalImage, const EffectManager &effects, QString destinationPath);

signals:
    void processingStarted();
    // `offset` is the top-left position of `result` within the requested
    // viewport.  When the image fills the viewport (or for export, where no
    // viewport is requested), offset is (0, 0).  Receivers blit `result` at
    // `offset` and leave the surrounding letterbox to the viewport's clear.
    void processingComplete(QImage result, QPoint offset);
    void exportComplete(QImage result, QString destinationPath);

private:
    std::shared_ptr<std::atomic<uint64_t>> generationPtr = std::make_shared<std::atomic<uint64_t>>(0);

    std::shared_ptr<GpuPipeline> m_pipeline = std::make_shared<GpuPipeline>();
};

#endif // IMAGEPROCESSOR_H
