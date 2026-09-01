#include "PreferencesDialog.h"
#include "EffectOrganizerDialog.h"
#include "EffectManager.h"
#include "GpuDeviceRegistry.h"
#include <QComboBox>
#include <QApplication>
#include <QDialogButtonBox>
#include <QFontComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>

namespace {
QString deviceKey(const GpuDeviceInfo &device) {
    return device.platformName + QChar('\n') + device.name;
}
} // namespace

PreferencesDialog::PreferencesDialog(EffectManager *effects, QWidget *parent)
    : QDialog(parent), m_pages(new QListWidget(this)), m_stack(new QStackedWidget(this)),
      m_gpuSelector(new QComboBox(this)) {
    setObjectName("preferencesDialog");
    setWindowTitle("Preferences");
    setModal(false);
    resize(720, 480);

    m_pages->setObjectName("preferencesPages");
    m_pages->setFixedWidth(140);
    m_pages->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_pages->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_pages->setSpacing(2);
    m_pages->setStyleSheet("QListWidget { padding: 6px; } QListWidget::item { padding: 6px 8px; }");
    m_pages->addItems({"Effects", "Processing", "Appearance"});

    auto *effectsPage = new EffectOrganizerWidget(effects, this);
    connect(effectsPage, &EffectOrganizerWidget::organizationChanged, this,
            &PreferencesDialog::effectOrganizationChanged);
    m_stack->addWidget(effectsPage);

    auto *processingPage   = new QWidget(this);
    auto *processingLayout = new QVBoxLayout(processingPage);
    auto *heading          = new QLabel("Processing");
    QFont headingFont      = heading->font();
    headingFont.setBold(true);
    heading->setFont(headingFont);
    processingLayout->addWidget(heading);
    auto *description = new QLabel("Choose the OpenCL device used for image processing. Changes apply immediately.");
    description->setWordWrap(true);
    processingLayout->addWidget(description);
    auto *form = new QFormLayout();
    m_gpuSelector->setObjectName("gpuDeviceSelector");
    const auto &devices = GpuDeviceRegistry::instance().devices();
    if (devices.empty()) {
        m_gpuSelector->addItem("No OpenCL devices found");
        m_gpuSelector->setEnabled(false);
    } else {
        for (const auto &device : devices) {
            m_gpuSelector->addItem(device.name + " [" + device.platformName + " · " + device.typeName + "]",
                                   deviceKey(device));
        }
        m_gpuSelector->setCurrentIndex(GpuDeviceRegistry::instance().currentIndex());
    }
    form->addRow("OpenCL device", m_gpuSelector);
    processingLayout->addLayout(form);
    processingLayout->addStretch();
    m_stack->addWidget(processingPage);

    auto *appearancePage   = new QWidget(this);
    auto *appearanceLayout = new QVBoxLayout(appearancePage);
    auto *appearanceHeading = new QLabel("Appearance");
    QFont appearanceHeadingFont = appearanceHeading->font();
    appearanceHeadingFont.setBold(true);
    appearanceHeading->setFont(appearanceHeadingFont);
    appearanceLayout->addWidget(appearanceHeading);

    auto *appearanceDescription = new QLabel(
        "Choose how Afterglow looks and how large its interface text appears.");
    appearanceDescription->setWordWrap(true);
    appearanceLayout->addWidget(appearanceDescription);

    auto *appearanceForm = new QFormLayout();
    auto *themeSelector  = new QComboBox(appearancePage);
    themeSelector->setObjectName("themeSelector");
    themeSelector->addItems({"Follow system", "Light", "Dark"});
    appearanceForm->addRow("Theme", themeSelector);

    auto *fontSelector = new QFontComboBox(appearancePage);
    fontSelector->setObjectName("interfaceFontSelector");
    fontSelector->setCurrentFont(QApplication::font());
    appearanceForm->addRow("Interface font", fontSelector);

    auto *fontSizeSelector = new QSpinBox(appearancePage);
    fontSizeSelector->setObjectName("interfaceFontSizeSelector");
    fontSizeSelector->setRange(8, 24);
    fontSizeSelector->setSuffix(" pt");
    fontSizeSelector->setValue(QApplication::font().pointSize() > 0 ? QApplication::font().pointSize() : 10);
    appearanceForm->addRow("Text size", fontSizeSelector);
    appearanceLayout->addLayout(appearanceForm);

    auto *preview = new QGroupBox("Preview", appearancePage);
    preview->setObjectName("appearancePreview");
    auto *previewLayout = new QVBoxLayout(preview);
    auto *previewTitle  = new QLabel("Exposure", preview);
    previewTitle->setObjectName("appearancePreviewTitle");
    QFont previewTitleFont = previewTitle->font();
    previewTitleFont.setBold(true);
    previewTitle->setFont(previewTitleFont);
    auto *previewText = new QLabel("Adjust the brightness of your photograph without changing its colours.", preview);
    previewText->setWordWrap(true);
    auto *previewButton = new QPushButton("Sample button", preview);
    previewButton->setEnabled(false);
    previewLayout->addWidget(previewTitle);
    previewLayout->addWidget(previewText);
    previewLayout->addWidget(previewButton, 0, Qt::AlignLeft);
    appearanceLayout->addWidget(preview);

    auto *resetAppearance = new QPushButton("Reset appearance defaults", appearancePage);
    resetAppearance->setObjectName("resetAppearanceButton");
    resetAppearance->setToolTip("Restore the system theme, default interface font, and default text size.");
    appearanceLayout->addWidget(resetAppearance, 0, Qt::AlignLeft);
    appearanceLayout->addStretch();
    m_stack->addWidget(appearancePage);

    const QPalette systemPreviewPalette = preview->palette();
    auto updatePreview = [themeSelector, fontSelector, fontSizeSelector, preview, previewTitle, previewText,
                          previewButton, systemPreviewPalette]() {
        QFont previewFont(fontSelector->currentFont());
        previewFont.setPointSize(fontSizeSelector->value());
        preview->setFont(previewFont);
        previewTitle->setFont(QFont(previewFont.family(), previewFont.pointSize(), QFont::Bold));
        previewText->setFont(previewFont);
        previewButton->setFont(previewFont);

        QPalette palette = systemPreviewPalette;
        if (themeSelector->currentIndex() == 1) {
            palette.setColor(QPalette::Window, QColor("#F4F1EA"));
            palette.setColor(QPalette::WindowText, QColor("#2C2018"));
            palette.setColor(QPalette::Button, QColor("#E6E0D4"));
            palette.setColor(QPalette::ButtonText, QColor("#2C2018"));
        } else if (themeSelector->currentIndex() == 2) {
            palette.setColor(QPalette::Window, QColor("#25272B"));
            palette.setColor(QPalette::WindowText, QColor("#ECEDEF"));
            palette.setColor(QPalette::Button, QColor("#35383E"));
            palette.setColor(QPalette::ButtonText, QColor("#ECEDEF"));
        }
        preview->setAutoFillBackground(true);
        preview->setPalette(palette);
        previewTitle->setPalette(palette);
        previewText->setPalette(palette);
        previewButton->setPalette(palette);
    };
    connect(themeSelector, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [updatePreview](int) { updatePreview(); });
    connect(fontSelector, &QFontComboBox::currentFontChanged, this,
            [updatePreview](const QFont &) { updatePreview(); });
    connect(fontSizeSelector, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [updatePreview](int) { updatePreview(); });
    connect(resetAppearance, &QPushButton::clicked, this,
            [themeSelector, fontSelector, fontSizeSelector, updatePreview]() {
                themeSelector->setCurrentIndex(0);
                fontSelector->setCurrentFont(QApplication::font());
                fontSizeSelector->setValue(QApplication::font().pointSize() > 0 ? QApplication::font().pointSize()
                                                                                 : 10);
                updatePreview();
            });
    updatePreview();

    auto *content = new QHBoxLayout();
    content->addWidget(m_pages);
    content->addWidget(m_stack, 1);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::close);
    auto *layout = new QVBoxLayout(this);
    layout->addLayout(content, 1);
    layout->addWidget(buttons);

    connect(m_pages, &QListWidget::currentRowChanged, m_stack, &QStackedWidget::setCurrentIndex);
    connect(m_gpuSelector, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        GpuDeviceRegistry::instance().setDevice(index);
        QSettings("Afterglow", "Afterglow").setValue("processing/openclDevice", m_gpuSelector->currentData());
        emit gpuDeviceChanged();
    });
    showPage(0);
}

void PreferencesDialog::showPage(int index) {
    if (index < 0 || index >= m_stack->count()) index = 0;
    m_pages->setCurrentRow(index);
    m_stack->setCurrentIndex(index);
}
