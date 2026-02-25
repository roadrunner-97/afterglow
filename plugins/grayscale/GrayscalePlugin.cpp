#include "GrayscalePlugin.h"
#include <QDebug>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

GrayscalePlugin::GrayscalePlugin() 
    : controlsWidget(nullptr), enableCheckBox(nullptr), isEnabled(false) {
}

GrayscalePlugin::~GrayscalePlugin() {
}

QString GrayscalePlugin::getName() const {
    return "Grayscale";
}

QString GrayscalePlugin::getDescription() const {
    return "Converts the image to grayscale";
}

QString GrayscalePlugin::getVersion() const {
    return "2.0.0";
}

bool GrayscalePlugin::initialize() {
    qDebug() << "Grayscale plugin initialized";
    return true;
}

QWidget* GrayscalePlugin::createControlsWidget() {
    if (controlsWidget) {
        return controlsWidget;
    }

    controlsWidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(5);

    QHBoxLayout *checkLayout = new QHBoxLayout();
    QLabel *enableLabel = new QLabel("Enable Grayscale:");
    enableLabel->setStyleSheet("color: #e0e0e0;");
    enableCheckBox = new QCheckBox();
    enableCheckBox->setChecked(false);
    enableCheckBox->setStyleSheet(
        "QCheckBox { color: #e0e0e0; }"
        "QCheckBox::indicator { width: 16px; height: 16px; }"
    );

    connect(enableCheckBox, &QCheckBox::checkStateChanged, this, [this]() {
        onEnabledChanged(enableCheckBox->isChecked());
    });

    checkLayout->addWidget(enableLabel);
    checkLayout->addWidget(enableCheckBox);
    checkLayout->addStretch();
    layout->addLayout(checkLayout);

    return controlsWidget;
}

QMap<QString, QVariant> GrayscalePlugin::getParameters() const {
    QMap<QString, QVariant> params;
    params["enabled"] = isEnabled;
    return params;
}

void GrayscalePlugin::onEnabledChanged(bool checked) {
    isEnabled = checked;
    emit parametersChanged();
}

QImage GrayscalePlugin::processImage(const QImage &image, const QMap<QString, QVariant> &parameters) {
    // Check if grayscale is enabled
    bool enabled = parameters.value("enabled", false).toBool();
    
    if (!enabled || image.isNull()) {
        return image;
    }

    QImage result = image.convertToFormat(QImage::Format_RGB32);

    for (int y = 0; y < result.height(); ++y) {
        for (int x = 0; x < result.width(); ++x) {
            QRgb pixel = result.pixel(x, y);
            int r = qRed(pixel);
            int g = qGreen(pixel);
            int b = qBlue(pixel);

            // Calculate grayscale value using luminosity formula
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);

            result.setPixel(x, y, qRgb(gray, gray, gray));
        }
    }

    return result;
}

// Export the plugin factory functions
extern "C" {
    PhotoEditorPlugin* createPlugin() {
        return new GrayscalePlugin();
    }

    void destroyPlugin(PhotoEditorPlugin* plugin) {
        delete plugin;
    }
}
