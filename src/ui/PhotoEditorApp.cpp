#include "PhotoEditorApp.h"
#include "GpuDeviceRegistry.h"
#include "RawLoader.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QComboBox>
#include <QFrame>
#include <QResizeEvent>
#include <QDebug>
#include <memory>

PhotoEditorApp::PhotoEditorApp(EffectManager* effectManager, QWidget* parent)
    : QMainWindow(parent)
    , m_effects(effectManager)
    , m_processor(new ImageProcessor(this))
    , m_imageLabel(nullptr)
    , m_scrollArea(nullptr)
    , m_gpuSelector(nullptr)
{
    connect(m_processor, &ImageProcessor::processingComplete,
            this, &PhotoEditorApp::onProcessingComplete);

    setupUI();
    setWindowTitle("Lightroom Clone");
    setGeometry(100, 100, 1400, 900);
}

void PhotoEditorApp::setupUI() {
    setupMenuBar();

    QWidget* central = new QWidget();
    setCentralWidget(central);
    QHBoxLayout* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // ── Image view ──────────────────────────────────────────────────────────
    m_scrollArea = new QScrollArea();
    m_scrollArea->setStyleSheet("background-color: #1e1e1e;");
    m_imageLabel = new QLabel("Open an image to begin");
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setStyleSheet("background-color: #2a2a2a; color: #808080;");
    m_scrollArea->setWidget(m_imageLabel);
    m_scrollArea->setWidgetResizable(false);
    m_scrollArea->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_scrollArea, 3);

    // ── Right panel ──────────────────────────────────────────────────────────
    QWidget* rightPanel = new QWidget();
    rightPanel->setStyleSheet("background-color: #252525;");
    rightPanel->setFixedWidth(300);
    QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(8, 8, 8, 8);
    rightLayout->setSpacing(6);

    setupGpuSelector(rightLayout);

    QFrame* sep = new QFrame();
    sep->setFrameShape(QFrame::HLine);
    sep->setStyleSheet("color: #444444;");
    rightLayout->addWidget(sep);

    QScrollArea* effectsScroll = new QScrollArea();
    effectsScroll->setWidgetResizable(true);
    effectsScroll->setStyleSheet("QScrollArea { background: transparent; border: none; }");

    QWidget* effectsContainer = new QWidget();
    effectsContainer->setStyleSheet("background: transparent;");
    QVBoxLayout* effectsLayout = new QVBoxLayout(effectsContainer);
    effectsLayout->setContentsMargins(0, 0, 0, 0);
    effectsLayout->setSpacing(4);

    setupEffectPanels(effectsLayout);
    effectsLayout->addStretch();

    effectsScroll->setWidget(effectsContainer);
    rightLayout->addWidget(effectsScroll, 1);

    mainLayout->addWidget(rightPanel);
}

void PhotoEditorApp::setupMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu("File");

    QAction* openAct = fileMenu->addAction("Open Image…");
    openAct->setShortcut(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &PhotoEditorApp::openImage);

    QAction* saveAct = fileMenu->addAction("Save Image…");
    saveAct->setShortcut(QKeySequence::Save);
    connect(saveAct, &QAction::triggered, this, &PhotoEditorApp::saveImage);

    fileMenu->addSeparator();

    QAction* exitAct = fileMenu->addAction("Exit");
    connect(exitAct, &QAction::triggered, this, &QWidget::close);

    // View → Effects — enable/disable individual effects
    QMenu* viewMenu = menuBar()->addMenu("View");
    QMenu* effectsMenu = viewMenu->addMenu("Effects");

    const auto& entries = m_effects->entries();
    for (int i = 0; i < entries.size(); ++i) {
        QAction* act = effectsMenu->addAction(entries[i].effect->getName());
        act->setCheckable(true);
        act->setChecked(entries[i].enabled);
        connect(act, &QAction::toggled, this, [this, i](bool on) {
            m_effects->setEnabled(i, on);
            triggerReprocess();
        });
    }
}

void PhotoEditorApp::setupGpuSelector(QVBoxLayout* rightLayout) {
    QLabel* label = new QLabel("GPU Device");
    label->setStyleSheet("color: #a0a0a0; font-size: 10px; text-transform: uppercase;");
    rightLayout->addWidget(label);

    m_gpuSelector = new QComboBox();
    m_gpuSelector->setStyleSheet(
        "QComboBox { color: #e0e0e0; background-color: #3a3a3a;"
        "  border: 1px solid #555; border-radius: 3px; padding: 4px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #e0e0e0; background-color: #3a3a3a; }");

    const auto& devs = GpuDeviceRegistry::instance().devices();
    if (devs.empty()) {
        m_gpuSelector->addItem("No GPU devices found");
        m_gpuSelector->setEnabled(false);
    } else {
        for (const auto& d : devs)
            m_gpuSelector->addItem(d.name + " [" + d.platformName + "]");
        m_gpuSelector->setCurrentIndex(GpuDeviceRegistry::instance().currentIndex());
    }

    connect(m_gpuSelector, QOverload<int>::of(&QComboBox::activated), this, [this](int idx) {
        GpuDeviceRegistry::instance().setDevice(idx);
        triggerReprocess();
    });

    rightLayout->addWidget(m_gpuSelector);
}

void PhotoEditorApp::setupEffectPanels(QVBoxLayout* effectsLayout) {
    const auto& entries = m_effects->entries();
    for (int i = 0; i < entries.size(); ++i) {
        PhotoEditorEffect* effect = entries[i].effect;

        // Container
        QWidget* panel = new QWidget();
        panel->setStyleSheet(
            "QWidget { background-color: #333333; border-radius: 4px; }");
        QVBoxLayout* panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(6, 4, 6, 6);
        panelLayout->setSpacing(4);

        // Title bar
        QWidget* titleBar = new QWidget();
        titleBar->setStyleSheet("background: transparent;");
        QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(0, 0, 0, 0);

        QLabel* title = new QLabel(QString("<b>%1</b>").arg(effect->getName()));
        title->setStyleSheet("color: #e0e0e0; background: transparent;");
        titleLayout->addWidget(title, 1);

        QPushButton* collapseBtn = new QPushButton("−");
        collapseBtn->setStyleSheet(
            "QPushButton { background:#555; color:#fff; border:none;"
            "  border-radius:3px; padding:1px 5px; font-weight:bold; }"
            "QPushButton:hover { background:#777; }");
        collapseBtn->setMaximumWidth(28);
        titleLayout->addWidget(collapseBtn);
        panelLayout->addWidget(titleBar);

        // Controls
        QWidget* controls = effect->createControlsWidget();
        if (controls) {
            panelLayout->addWidget(controls);
        }

        // Collapse toggle — shared_ptr so the lambda stays valid after panel is reparented
        auto expanded = std::make_shared<bool>(true);
        connect(collapseBtn, &QPushButton::clicked, this,
                [controls, collapseBtn, expanded]() {
            *expanded = !*expanded;
            if (controls) controls->setVisible(*expanded);
            collapseBtn->setText(*expanded ? "−" : "+");
        });

        // Show/hide panel when effect is toggled from the View menu
        panel->setVisible(entries[i].enabled);
        connect(m_effects, &EffectManager::effectToggled, panel,
                [panel, i](int idx, bool on) {
            if (idx == i) panel->setVisible(on);
        });

        // Wire parametersChanged
        connect(effect, &PhotoEditorEffect::parametersChanged,
                this, &PhotoEditorApp::onParametersChanged);

        effectsLayout->addWidget(panel);
    }
}

void PhotoEditorApp::openImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open Image", m_lastDir,
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif "
        "*.cr2 *.cr3 *.nef *.nrw *.arw *.dng *.raf *.orf *.rw2);;"
        "All Files (*)");

    if (fileName.isEmpty()) return;
    m_lastDir = QFileInfo(fileName).absolutePath();

    QImage img;
    if (RawLoader::isRawFile(fileName)) {
        img = RawLoader::load(fileName);
        if (img.isNull())
            qWarning() << "RawLoader failed for" << fileName << "— trying QImage::load";
    }
    if (img.isNull())
        img = QImage(fileName);

    if (img.isNull()) {
        qWarning() << "Failed to load image:" << fileName;
        return;
    }

    m_originalImage = img;
    triggerReprocess();
}

void PhotoEditorApp::saveImage() {
    if (m_originalImage.isNull()) return;

    QString fileName = QFileDialog::getSaveFileName(
        this, "Save Image", m_lastDir,
        "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;All Files (*)");

    if (fileName.isEmpty()) return;
    // Save the last-displayed pixmap converted back — or just save originalImage
    // if no processed result is available yet.
    m_imageLabel->pixmap().toImage().save(fileName);
    m_lastDir = QFileInfo(fileName).absolutePath();
}

void PhotoEditorApp::onParametersChanged() {
    triggerReprocess();
}

void PhotoEditorApp::triggerReprocess() {
    if (m_originalImage.isNull()) return;

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    QSize previewSize = m_scrollArea->viewport()->size();
    m_processor->processImageAsync(m_originalImage, active, previewSize);
}

void PhotoEditorApp::onProcessingComplete(QImage result) {
    if (!result.isNull())
        displayImage(result);
}

void PhotoEditorApp::displayImage(const QImage& image) {
    QPixmap px = QPixmap::fromImage(image);
    m_imageLabel->setPixmap(px);
    m_imageLabel->setFixedSize(px.size());
}

void PhotoEditorApp::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    triggerReprocess();
}
