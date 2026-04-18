#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QVector>
#include <atomic>
#include <memory>
#include "PhotoEditorEffect.h"
#include "GpuPipeline.h"

/**
 * @brief Runs the effect pipeline asynchronously via QtConcurrent.
 *
 * A generation counter lets processImageAsync() be called any number of times
 * in quick succession: only the latest result is delivered via processingComplete.
 */
class ImageProcessor : public QObject {
    Q_OBJECT

public:
    explicit ImageProcessor(QObject* parent = nullptr);

    void processImageAsync(const QImage& originalImage,
                           const QVector<PhotoEditorEffect*>& effects,
                           ViewportRequest viewport = {},
                           bool viewportOnly = false);

    void exportImageAsync(const QImage& originalImage,
                          const QVector<PhotoEditorEffect*>& effects);

signals:
    void processingStarted();
    void processingComplete(QImage result);
    void exportComplete(QImage result);

private:
    std::shared_ptr<std::atomic<uint64_t>> generationPtr =
        std::make_shared<std::atomic<uint64_t>>(0);

    std::shared_ptr<GpuPipeline> m_pipeline = std::make_shared<GpuPipeline>();
};

#endif // IMAGEPROCESSOR_H
