#include "PhotoEditorApp.h"
#include "Theme.h"
#include "GpuDeviceRegistry.h"
#include "Histogram.h"
#include "RawLoader.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QComboBox>
#include <QFrame>
#include <QLabel>
#include <QResizeEvent>
#include <QCloseEvent>
#include <QToolBar>
#include <QSettings>
#include <QDir>
#include <QDebug>
#include <memory>

PhotoEditorApp::PhotoEditorApp(EffectManager* effectManager, QWidget* parent)
    : QMainWindow(parent)
    , m_effects(effectManager)
    , m_processor(new ImageProcessor(this))
    , m_resizeDebounce(new QTimer(this))
{
    connect(m_processor, &ImageProcessor::processingComplete,
            this, &PhotoEditorApp::onProcessingComplete);
    connect(m_processor, &ImageProcessor::processingStarted,
            this, &PhotoEditorApp::onProcessingStarted);
    connect(m_processor, &ImageProcessor::exportComplete,
            this, &PhotoEditorApp::onExportComplete);

    m_resizeDebounce->setSingleShot(true);
    m_resizeDebounce->setInterval(150);
    connect(m_resizeDebounce, &QTimer::timeout, this, &PhotoEditorApp::triggerReprocess);

    setupToolBar();
    setupUI();
    setWindowTitle("Lightroom Clone");

    // Restore geometry and last-used directory from previous session
    QSettings settings("LightroomClone", "LightroomClone");
    if (settings.contains("geometry"))
        restoreGeometry(settings.value("geometry").toByteArray());
    else
        setGeometry(100, 100, 1400, 900);
    m_lastDir = settings.value("lastDir", QDir::homePath()).toString();
}

PhotoEditorApp::~PhotoEditorApp() = default;

void PhotoEditorApp::setupToolBar() {
    QToolBar* toolbar = addToolBar("Preview");
    toolbar->setMovable(false);
    toolbar->setStyleSheet(
        QString("QToolBar { background: %1; border-bottom: 1px solid %2; spacing: 6px; padding: 2px 6px; }"
                "QToolButton { color: %3; background: transparent; border: 1px solid %2;"
                "  border-radius: 3px; padding: 3px 10px; }"
                "QToolButton:checked { color: %4; background: %5; border-color: %5; }"
                "QToolButton:hover { background: %6; }")
        .arg(Theme::BG_MAIN, Theme::BORDER,
             Theme::TEXT_PRIMARY, Theme::CHECKED_TEXT,
             Theme::CHECKED_BG, Theme::COLLAPSE_HOVER));

    QAction* liveAct = new QAction("Live Preview", this);
    liveAct->setCheckable(true);
    liveAct->setChecked(false);
    liveAct->setToolTip("Update preview in real-time while dragging sliders");
    connect(liveAct, &QAction::toggled, this, [this](bool on) {
        m_liveUpdate = on;
    });
    toolbar->addAction(liveAct);

    // Spacer + processing indicator label on the right side of the toolbar
    QWidget* spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(spacer);

    m_processingLabel = new QLabel("Processing…");
    m_processingLabel->setStyleSheet("color: #6E5E46; font-style: italic; padding: 0 6px;");
    m_processingLabel->setVisible(false);
    toolbar->addWidget(m_processingLabel);
}

void PhotoEditorApp::setupUI() {
    setupMenuBar();

    QWidget* central = new QWidget();
    central->setStyleSheet(QString("background: %1;").arg(Theme::BG_MAIN));
    setCentralWidget(central);
    QHBoxLayout* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // ── Image view ──────────────────────────────────────────────────────────
    m_viewport = new ViewportWidget();
    connect(m_viewport, &ViewportWidget::viewportChanged,
            this, &PhotoEditorApp::triggerViewportUpdate);
    mainLayout->addWidget(m_viewport, 3);

    // ── Right panel ──────────────────────────────────────────────────────────
    QWidget* rightPanel = new QWidget();
    rightPanel->setStyleSheet(QString("background-color: %1;").arg(Theme::BG_RIGHT_PANEL));
    rightPanel->setFixedWidth(300);
    QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(8, 8, 8, 8);
    rightLayout->setSpacing(6);

    setupGpuSelector(rightLayout);

    QFrame* sep = new QFrame();
    sep->setFrameShape(QFrame::HLine);
    sep->setStyleSheet("color: #CBBFAE;");
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
    menuBar()->setStyleSheet(
        QString("QMenuBar { background: %1; color: %2; border-bottom: 1px solid %3; }"
                "QMenuBar::item { padding: 4px 8px; }"
                "QMenuBar::item:selected { background: %4; border-radius: 3px; }"
                "QMenu { background: %5; color: %2; border: 1px solid %3; }"
                "QMenu::item { padding: 4px 20px; }"
                "QMenu::item:selected { background: %6; color: %7; }"
                "QMenu::separator { height: 1px; background: %3; margin: 2px 0; }")
        .arg(Theme::BG_MAIN, Theme::TEXT_PRIMARY, Theme::BORDER,
             Theme::COLLAPSE_HOVER, Theme::BG_EFFECT_PANEL,
             Theme::CHECKED_BG, Theme::CHECKED_TEXT));

    QMenu* fileMenu = menuBar()->addMenu("File");

    QAction* openAct = fileMenu->addAction("Open Image…");
    openAct->setShortcut(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &PhotoEditorApp::openImage);

    QAction* saveAct = fileMenu->addAction("Save Image…");
    saveAct->setShortcut(QKeySequence::Save);
    connect(saveAct, &QAction::triggered, this, &PhotoEditorApp::saveImage);

    fileMenu->addSeparator();

    QAction* exitAct = fileMenu->addAction("Exit");
    exitAct->setShortcut(QKeySequence("Ctrl+Q"));
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
    label->setStyleSheet(QString("color: %1; font-size: 10px; text-transform: uppercase;").arg(Theme::TEXT_SECONDARY));
    rightLayout->addWidget(label);

    m_gpuSelector = new QComboBox();
    m_gpuSelector->setToolTip("Select the OpenCL compute device used to accelerate all image processing effects.\nChanging device reinitialises all GPU kernels and triggers a full reprocess.");
    m_gpuSelector->setStyleSheet(
        QString("QComboBox { color: %1; background-color: %2;"
                "  border: 1px solid %3; border-radius: 3px; padding: 4px; }"
                "QComboBox::drop-down { border: none; }"
                "QComboBox QAbstractItemView { color: %1; background-color: %2; }")
        .arg(Theme::TEXT_PRIMARY, Theme::BG_EFFECT_PANEL, Theme::BORDER));

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
            QString("QWidget { background-color: %1; border-radius: 4px; }").arg(Theme::BG_EFFECT_PANEL));
        QVBoxLayout* panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(6, 4, 6, 6);
        panelLayout->setSpacing(4);

        // Title bar
        QWidget* titleBar = new QWidget();
        titleBar->setStyleSheet("background: transparent;");
        QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(0, 0, 0, 0);

        QLabel* title = new QLabel(QString("<b>%1</b>").arg(effect->getName()));
        title->setStyleSheet(QString("color: %1; background: transparent;").arg(Theme::TEXT_PRIMARY));
        titleLayout->addWidget(title, 1);

        QPushButton* collapseBtn = new QPushButton("−");
        collapseBtn->setStyleSheet(
            QString("QPushButton { background: %1; color: %2; border: none;"
                    "  border-radius: 3px; padding: 1px 5px; font-weight: bold; }"
                    "QPushButton:hover { background: %3; }")
            .arg(Theme::COLLAPSE_BG, Theme::TEXT_PRIMARY, Theme::COLLAPSE_HOVER));
        collapseBtn->setToolTip("Collapse or expand this effect's controls.");
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

        // Wire parametersChanged (committed) and liveParametersChanged (drag)
        connect(effect, &PhotoEditorEffect::parametersChanged,
                this, &PhotoEditorApp::onParametersChanged);
        connect(effect, &PhotoEditorEffect::liveParametersChanged,
                this, &PhotoEditorApp::onLiveParametersChanged);

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
    ImageMetadata meta;
    if (RawLoader::isRawFile(fileName)) {
        img = RawLoader::load(fileName, &meta);
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
    m_viewport->setImageSize(img.size());
    m_viewport->resetView();
    meta.luminanceHistogram = computeLuminanceHistogram(img);
    for (const auto& e : m_effects->entries())
        e.effect->onImageLoaded(meta);
    triggerReprocess();
}

void PhotoEditorApp::saveImage() {
    if (m_originalImage.isNull()) return;

    QString fileName = QFileDialog::getSaveFileName(
        this, "Save Image", m_lastDir,
        "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;All Files (*)");

    if (fileName.isEmpty()) return;
    m_lastDir = QFileInfo(fileName).absolutePath();
    m_pendingSavePath = fileName;

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);
    m_processor->exportImageAsync(m_originalImage, active);
}

void PhotoEditorApp::onExportComplete(QImage result) {
    if (m_pendingSavePath.isEmpty()) return;
    const QString path = m_pendingSavePath;
    m_pendingSavePath.clear();

    if (result.isNull() || !result.save(path)) {
        QMessageBox::warning(this, "Save Failed",
            QString("Could not save image to:\n%1\n\n"
                    "Check that the directory is writable and you have sufficient disk space.")
            .arg(path));
    }
}

void PhotoEditorApp::onParametersChanged() {
    triggerReprocess();
}

void PhotoEditorApp::onLiveParametersChanged() {
    if (m_liveUpdate) triggerReprocess();
}

void PhotoEditorApp::triggerReprocess() {
    if (m_originalImage.isNull()) return;

    m_pendingFullRun = true;

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    m_processor->processImageAsync(m_originalImage, active, m_viewport->viewportRequest());
}

void PhotoEditorApp::triggerViewportUpdate() {
    if (m_originalImage.isNull()) return;

    // If a full reprocess is already in-flight (params changed), promote this
    // viewport change to a full run so the displayed result is never stale.
    if (m_pendingFullRun) {
        triggerReprocess();
        return;
    }

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    m_processor->processImageAsync(m_originalImage, active,
                                   m_viewport->viewportRequest(),
                                   /*viewportOnly=*/true);
}

void PhotoEditorApp::onProcessingStarted() {
    m_processingLabel->setVisible(true);
}

void PhotoEditorApp::onProcessingComplete(QImage result) {
    m_processingLabel->setVisible(false);
    m_pendingFullRun = false;
    if (result.isNull()) {
        m_viewport->update();
    } else {
        m_viewport->setImage(result);
    }
}

void PhotoEditorApp::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    // Debounce: avoid firing a full GPU reprocess on every pixel of a window drag.
    m_resizeDebounce->start();
}

void PhotoEditorApp::closeEvent(QCloseEvent* event) {
    QSettings settings("LightroomClone", "LightroomClone");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("lastDir",  m_lastDir);
    QMainWindow::closeEvent(event);
}
