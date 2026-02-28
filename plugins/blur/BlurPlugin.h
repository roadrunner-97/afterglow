#ifndef BLURPLUGIN_H
#define BLURPLUGIN_H

#include "../../src/core/PhotoEditorPlugin.h"

class ParamSlider;
class QComboBox;
class QLabel;

class BlurPlugin : public PhotoEditorPlugin {
    Q_OBJECT

public:
    BlurPlugin();
    ~BlurPlugin() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

private:
    QWidget*     controlsWidget;
    QComboBox*   blurTypeCombo;
    ParamSlider* radiusParam;

    int blurType;  // 0 = Gaussian, 1 = Box
};

#endif // BLURPLUGIN_H
