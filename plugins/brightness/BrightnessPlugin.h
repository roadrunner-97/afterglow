#ifndef BRIGHTNESSPLUGIN_H
#define BRIGHTNESSPLUGIN_H

#include "../../src/core/PhotoEditorPlugin.h"

class QSlider;
class QSpinBox;
class QLabel;

/**
 * @brief A brightness and contrast adjustment plugin with UI controls
 */
class BrightnessPlugin : public PhotoEditorPlugin {
    Q_OBJECT

public:
    BrightnessPlugin();
    ~BrightnessPlugin() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

private slots:
    void onBrightnessChanged(int value);
    void onContrastChanged(int value);

private:
    QWidget* controlsWidget;
    QSlider* brightnessSlider;
    QSlider* contrastSlider;
    QSpinBox* brightnessSpinBox;
    QSpinBox* contrastSpinBox;
    QLabel* brightnessLabel;
    QLabel* contrastLabel;

    int brightnessValue;
    int contrastValue;
};

#endif // BRIGHTNESSPLUGIN_H
