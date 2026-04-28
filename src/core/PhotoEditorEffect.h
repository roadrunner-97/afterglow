#ifndef PHOTOEDITOREFFECT_H
#define PHOTOEDITOREFFECT_H

#include <QString>
#include <QImage>
#include <QMap>
#include <QVariant>
#include <QObject>
#include <QWidget>
#include "ImageMetadata.h"

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

    // Reverse of getParameters(): pushes a previously-saved parameter map back
    // onto the effect's controls.  Default no-op so effects without parameters
    // (or that haven't yet implemented load support) don't have to override.
    // Implementations should ignore unknown keys and tolerate missing ones.
    virtual void applyParameters(const QMap<QString, QVariant>& /*parameters*/) {}

    // Called after a new image is loaded.  Effects that want to adapt their
    // defaults to the image's metadata (e.g. white balance) override this.
    virtual void onImageLoaded(const ImageMetadata& /*meta*/) {}

signals:
    void parametersChanged();     // committed change (slider release / spinbox commit)
    void liveParametersChanged(); // real-time change (every drag tick)
};

#endif // PHOTOEDITOREFFECT_H
