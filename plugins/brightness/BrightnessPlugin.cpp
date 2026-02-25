#include "BrightnessPlugin.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <algorithm>


BrightnessPlugin::BrightnessPlugin()
    : controlsWidget(nullptr), brightnessSlider(nullptr), contrastSlider(nullptr),
      brightnessSpinBox(nullptr), contrastSpinBox(nullptr),
      brightnessLabel(nullptr), contrastLabel(nullptr),
      brightnessValue(0), contrastValue(0) {
}

BrightnessPlugin::~BrightnessPlugin() {
}

QString BrightnessPlugin::getName() const {
    return "Brightness & Contrast";
}

QString BrightnessPlugin::getDescription() const {
    return "Adjusts brightness and contrast of the image";
}

QString BrightnessPlugin::getVersion() const {
    return "2.0.0";
}

bool BrightnessPlugin::initialize() {
    qDebug() << "Brightness & Contrast plugin initialized";
    return true;
}

QWidget* BrightnessPlugin::createControlsWidget() {
    if (controlsWidget) {
        return controlsWidget;
    }

    controlsWidget = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(controlsWidget);

    // Brightness control
    brightnessLabel = new QLabel("Brightness: 0");
    mainLayout->addWidget(brightnessLabel);

    QHBoxLayout *brightnessLayout = new QHBoxLayout();
    brightnessSlider = new QSlider(Qt::Horizontal);
    brightnessSlider->setRange(-100, 100);
    brightnessSlider->setValue(0);
    brightnessSpinBox = new QSpinBox();
    brightnessSpinBox->setRange(-100, 100);
    brightnessSpinBox->setValue(0);
    
    connect(brightnessSlider, QOverload<int>::of(&QSlider::valueChanged),
            brightnessSpinBox, &QSpinBox::setValue);
    connect(brightnessSlider, &QSlider::sliderReleased, this, [this]() {
        onBrightnessChanged(0);
    });
    connect(brightnessSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            brightnessSlider, &QSlider::setValue);
    connect(brightnessSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() {
        onBrightnessChanged(0);
    });

    brightnessLayout->addWidget(brightnessSlider);
    brightnessLayout->addWidget(brightnessSpinBox);
    mainLayout->addLayout(brightnessLayout);

    // Contrast control
    contrastLabel = new QLabel("Contrast: 0");
    mainLayout->addWidget(contrastLabel);

    QHBoxLayout *contrastLayout = new QHBoxLayout();
    contrastSlider = new QSlider(Qt::Horizontal);
    contrastSlider->setRange(-50, 50);
    contrastSlider->setValue(0);
    contrastSpinBox = new QSpinBox();
    contrastSpinBox->setRange(-50, 50);
    contrastSpinBox->setValue(0);
    
    connect(contrastSlider, QOverload<int>::of(&QSlider::valueChanged),
            contrastSpinBox, &QSpinBox::setValue);
    connect(contrastSlider, &QSlider::sliderReleased, this, [this]() {
        onContrastChanged(0);
    });
    connect(contrastSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            contrastSlider, &QSlider::setValue);
    connect(contrastSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() {
        onContrastChanged(0);
    });

    contrastLayout->addWidget(contrastSlider);
    contrastLayout->addWidget(contrastSpinBox);
    mainLayout->addLayout(contrastLayout);

    mainLayout->addStretch();

    return controlsWidget;
}

QMap<QString, QVariant> BrightnessPlugin::getParameters() const {
    QMap<QString, QVariant> params;
    params["brightness"] = brightnessValue;
    params["contrast"] = contrastValue;
    return params;
}

void BrightnessPlugin::onBrightnessChanged(int) {
    brightnessValue = brightnessSlider->value();
    brightnessLabel->setText(QString("Brightness: %1").arg(brightnessValue));
    brightnessSpinBox->blockSignals(true);
    brightnessSpinBox->setValue(brightnessValue);
    brightnessSpinBox->blockSignals(false);
    emit parametersChanged();
}

void BrightnessPlugin::onContrastChanged(int) {
    contrastValue = contrastSlider->value();
    contrastLabel->setText(QString("Contrast: %1").arg(contrastValue));
    contrastSpinBox->blockSignals(true);
    contrastSpinBox->setValue(contrastValue);
    contrastSpinBox->blockSignals(false);
    emit parametersChanged();
}

QImage BrightnessPlugin::processImage(const QImage &image, const QMap<QString, QVariant> &parameters) {
    if (image.isNull()) {
        return image;
    }

    int brightnessFactor = parameters.value("brightness", 0).toInt();
    int contrastFactor = parameters.value("contrast", 0).toInt();
    float contrastFactor_f = (contrastFactor + 100.0f) / 100.0f;

    QImage result = image.convertToFormat(QImage::Format_RGB32);

    for (int y = 0; y < result.height(); ++y) {
        for (int x = 0; x < result.width(); ++x) {
            QRgb pixel = result.pixel(x, y);
            int r = qRed(pixel);
            int g = qGreen(pixel);
            int b = qBlue(pixel);
            int a = qAlpha(pixel);

            // Apply brightness
            r = std::min(255, std::max(0, r + brightnessFactor));
            g = std::min(255, std::max(0, g + brightnessFactor));
            b = std::min(255, std::max(0, b + brightnessFactor));

            // Apply contrast
            if (contrastFactor != 0) {
                r = std::min(255, std::max(0, static_cast<int>((r - 128) * contrastFactor_f + 128)));
                g = std::min(255, std::max(0, static_cast<int>((g - 128) * contrastFactor_f + 128)));
                b = std::min(255, std::max(0, static_cast<int>((b - 128) * contrastFactor_f + 128)));
            }

            result.setPixel(x, y, qRgba(r, g, b, a));
        }
    }

    return result;
}

// Export the plugin factory functions
extern "C" {
    PhotoEditorPlugin* createPlugin() {
        return new BrightnessPlugin();
    }

    void destroyPlugin(PhotoEditorPlugin* plugin) {
        delete plugin;
    }
}
