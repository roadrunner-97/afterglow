#include "MetadataTray.h"

#include <QFont>
#include <QFormLayout>
#include <QFrame>
#include <QLabel>
#include <QVBoxLayout>

MetadataTray::MetadataTray(QWidget *parent) : QWidget(parent) {
    setAutoFillBackground(true);
    setBackgroundRole(QPalette::Window);
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);

    auto *header = new QLabel("Metadata");
    QFont hf     = header->font();
    hf.setBold(true);
    header->setFont(hf);
    header->setContentsMargins(6, 4, 6, 4);
    outer->addWidget(header);

    auto *sep = new QFrame();
    sep->setFrameShape(QFrame::HLine);
    outer->addWidget(sep);

    auto *form = new QWidget();
    auto *fl   = new QFormLayout(form);
    fl->setContentsMargins(6, 6, 6, 6);
    fl->setHorizontalSpacing(8);
    fl->setVerticalSpacing(4);
    fl->setLabelAlignment(Qt::AlignRight | Qt::AlignTop);

    auto addRow = [&](const QString &key, QLabel *&valSlot) {
        auto *k  = new QLabel(key);
        QFont kf = k->font();
        kf.setPointSizeF(kf.pointSizeF() * 0.9);
        k->setFont(kf);
        k->setForegroundRole(QPalette::PlaceholderText);
        valSlot = new QLabel("\xe2\x80\x94"); // em-dash placeholder
        valSlot->setWordWrap(true);
        fl->addRow(k, valSlot);
    };

    auto addSection = [&](const QString &text) {
        auto *label = new QLabel(text.toUpper());
        label->setProperty("metadataSection", true);
        QFont font = label->font();
        font.setBold(true);
        font.setPointSizeF(font.pointSizeF() * 0.82);
        label->setFont(font);
        label->setForegroundRole(QPalette::Mid);
        label->setContentsMargins(0, 7, 0, 1);
        fl->addRow(label);
    };

    addSection("File");
    addRow("File", m_valFilename);
    addRow("Type", m_valFileType);
    addRow("File Size", m_valFileSize);
    addRow("Size", m_valDimensions);
    addSection("Camera");
    addRow("Camera", m_valCamera);
    addRow("Lens", m_valLens);
    addRow("Serial", m_valSerial);
    addSection("Capture");
    addRow("Exposure", m_valExposure);
    addRow("Focal Length", m_valFocalLength);
    addRow("Exposure Bias", m_valExposureBias);
    addRow("Program", m_valExposureProgram);
    addRow("Metering", m_valMeteringMode);
    addRow("Flash", m_valFlash);
    addRow("White Balance", m_valWhiteBalance);
    addRow("Color Temp", m_valColorTemperature);
    addRow("Captured", m_valCaptured);
    addRow("Location", m_valLocation);
    addSection("Description");
    addRow("Creator", m_valCreator);
    addRow("Copyright", m_valCopyright);
    addRow("Caption", m_valDescription);
    addRow("Software", m_valSoftware);

    outer->addWidget(form);
    outer->addStretch();
}

void MetadataTray::setInfo(const Info &info) {
    m_valFilename->setText(info.filename.isEmpty() ? "\xe2\x80\x94" : info.filename);
    m_valFileType->setText(info.fileType.isEmpty() ? "\xe2\x80\x94" : info.fileType);
    m_valFileSize->setText(info.fileSize.isEmpty() ? "\xe2\x80\x94" : info.fileSize);
    m_valDimensions->setText(info.dimensions.isEmpty() ? "\xe2\x80\x94" : info.dimensions);
    m_valCamera->setText(info.camera.isEmpty() ? "\xe2\x80\x94" : info.camera);
    m_valLens->setText(info.lens.isEmpty() ? "\xe2\x80\x94" : info.lens);
    m_valSerial->setText(info.serial.isEmpty() ? "\xe2\x80\x94" : info.serial);
    m_valExposure->setText(info.exposure.isEmpty() ? "\xe2\x80\x94" : info.exposure);
    m_valFocalLength->setText(info.focalLength.isEmpty() ? "\xe2\x80\x94" : info.focalLength);
    m_valExposureBias->setText(info.exposureBias.isEmpty() ? "\xe2\x80\x94" : info.exposureBias);
    m_valExposureProgram->setText(info.exposureProgram.isEmpty() ? "\xe2\x80\x94" : info.exposureProgram);
    m_valMeteringMode->setText(info.meteringMode.isEmpty() ? "\xe2\x80\x94" : info.meteringMode);
    m_valFlash->setText(info.flash.isEmpty() ? "\xe2\x80\x94" : info.flash);
    m_valWhiteBalance->setText(info.whiteBalance.isEmpty() ? "\xe2\x80\x94" : info.whiteBalance);
    m_valColorTemperature->setText(info.colorTemperature.isEmpty() ? "\xe2\x80\x94" : info.colorTemperature);
    m_valCaptured->setText(info.captured.isEmpty() ? "\xe2\x80\x94" : info.captured);
    m_valLocation->setText(info.location.isEmpty() ? "\xe2\x80\x94" : info.location);
    m_valCreator->setText(info.creator.isEmpty() ? "\xe2\x80\x94" : info.creator);
    m_valCopyright->setText(info.copyright.isEmpty() ? "\xe2\x80\x94" : info.copyright);
    m_valDescription->setText(info.description.isEmpty() ? "\xe2\x80\x94" : info.description);
    m_valSoftware->setText(info.software.isEmpty() ? "\xe2\x80\x94" : info.software);
}

void MetadataTray::clear() {
    setInfo({});
}
