#ifndef GRAYSCALEPLUGIN_H
#define GRAYSCALEPLUGIN_H

#include "../../src/core/PhotoEditorPlugin.h"

class QCheckBox;

/**
 * @brief A grayscale filter plugin with enable/disable checkbox
 */
class GrayscalePlugin : public PhotoEditorPlugin {
    Q_OBJECT

public:
    GrayscalePlugin();
    ~GrayscalePlugin() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

private slots:
    void onEnabledChanged(bool checked);

private:
    QWidget* controlsWidget;
    QCheckBox* enableCheckBox;
    bool isEnabled;
};

#endif // GRAYSCALEPLUGIN_H
