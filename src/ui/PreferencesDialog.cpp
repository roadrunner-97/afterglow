#include "PreferencesDialog.h"
#include "EffectOrganizerDialog.h"
#include "EffectManager.h"
#include "GpuDeviceRegistry.h"
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QListWidget>
#include <QSettings>
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
    m_pages->setMaximumWidth(160);
    m_pages->addItems({"Effects", "Processing"});

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
