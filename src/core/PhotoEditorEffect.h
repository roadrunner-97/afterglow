#ifndef PHOTOEDITOREFFECT_H
#define PHOTOEDITOREFFECT_H

#include <QString>
#include <QImage>
#include <QMap>
#include <QVariant>
#include <QObject>
#include <QWidget>

/**
 * @brief Abstract base class for all photo editing effects.
 *
 * Effects are statically linked into the binary and instantiated in main().
 * Each effect implements image processing logic and optionally provides a
 * UI controls widget.
 */
class PhotoEditorEffect : public QObject {
    Q_OBJECT

public:
    virtual ~PhotoEditorEffect() = default;

    virtual QString getName() const = 0;
    virtual QString getDescription() const = 0;
    virtual QString getVersion() const = 0;
    virtual bool initialize() = 0;

    virtual QImage processImage(const QImage &image,
                                const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) = 0;

    virtual QWidget* createControlsWidget() { return nullptr; }
    virtual QMap<QString, QVariant> getParameters() const { return {}; }

    // Returns true if this effect implements IGpuEffect and can participate
    // in the shared GPU pipeline (single upload + single readback).
    virtual bool supportsGpuInPlace() const { return false; }

signals:
    void parametersChanged();     // committed change (slider release / spinbox commit)
    void liveParametersChanged(); // real-time change (every drag tick)
    void processingStarted();
    void processingProgress(int progress);
    void processingFinished();
};

#endif // PHOTOEDITOREFFECT_H
