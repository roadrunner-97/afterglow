#include "MetadataTray.h"

#include <QFont>
#include <QFormLayout>
#include <QFrame>
#include <QLabel>
#include <QVBoxLayout>

MetadataTray::MetadataTray(QWidget *parent) : QWidget(parent) {
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
        auto *k = new QLabel(key);
        QFont kf = k->font();
        kf.setPointSizeF(kf.pointSizeF() * 0.9);
        k->setFont(kf);
        k->setForegroundRole(QPalette::Mid);
        valSlot = new QLabel("\xe2\x80\x94"); // em-dash placeholder
        valSlot->setWordWrap(true);
        fl->addRow(k, valSlot);
    };

    addRow("File", m_valFilename);
    addRow("Size", m_valDimensions);
    addRow("Camera", m_valCamera);
    addRow("Lens", m_valLens);
    addRow("Exposure", m_valExposure);
    addRow("Captured", m_valCaptured);

    outer->addWidget(form);
    outer->addStretch();
}

void MetadataTray::setInfo(const Info &info) {
    m_valFilename->setText(info.filename.isEmpty() ? "\xe2\x80\x94" : info.filename);
    m_valDimensions->setText(info.dimensions.isEmpty() ? "\xe2\x80\x94" : info.dimensions);
    m_valCamera->setText(info.camera.isEmpty() ? "\xe2\x80\x94" : info.camera);
    m_valLens->setText(info.lens.isEmpty() ? "\xe2\x80\x94" : info.lens);
    m_valExposure->setText(info.exposure.isEmpty() ? "\xe2\x80\x94" : info.exposure);
    m_valCaptured->setText(info.captured.isEmpty() ? "\xe2\x80\x94" : info.captured);
}

void MetadataTray::clear() {
    const QString dash = "\xe2\x80\x94";
    m_valFilename->setText(dash);
    m_valDimensions->setText(dash);
    m_valCamera->setText(dash);
    m_valLens->setText(dash);
    m_valExposure->setText(dash);
    m_valCaptured->setText(dash);
}
