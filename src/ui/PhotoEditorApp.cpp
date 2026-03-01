#include "PhotoEditorApp.h"
#include "GpuDeviceRegistry.h"
#include "RawLoader.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QComboBox>
#include <QFrame>
#include <QLabel>
#include <QResizeEvent>
#include <QToolBar>
#include <QDebug>
#include <memory>

PhotoEditorApp::PhotoEditorApp(EffectManager* effectManager, QWidget* parent)
    : QMainWindow(parent)
    , m_effects(effectManager)
    , m_processor(new ImageProcessor(this))
{
    connect(m_processor, &ImageProcessor::processingComplete,
            this, &PhotoEditorApp::onProcessingComplete);

    setupToolBar();
    setupUI();
    setWindowTitle("Lightroom Clone");
    setGeometry(100, 100, 1400, 900);
}

void PhotoEditorApp::setupToolBar() {
    QToolBar* toolbar = addToolBar("Preview");
    toolbar->setMovable(false);
    toolbar->setStyleSheet(
        "QToolBar { background: #F0EDE5; border-bottom: 1px solid #CCC5B5; spacing: 6px; padding: 2px 6px; }"
        "QToolButton { color: #2C2018; background: transparent; border: 1px solid #CCC5B5;"
        "  border-radius: 3px; padding: 3px 10px; }"
        "QToolButton:checked { color: #F5F2EA; background: #C0802C; border-color: #C0802C; }"
        "QToolButton:hover { background: #E0D8CC; }");

    QAction* liveAct = new QAction("Live Preview", this);
    liveAct->setCheckable(true);
    liveAct->setChecked(false);
    liveAct->setToolTip("Update preview in real-time while dragging sliders");
    connect(liveAct, &QAction::toggled, this, [this](bool on) {
        m_liveUpdate = on;
    });
    toolbar->addAction(liveAct);
}

void PhotoEditorApp::setupUI() {
    setupMenuBar();

    QWidget* central = new QWidget();
    central->setStyleSheet("background: #F0EDE5;");
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
    rightPanel->setStyleSheet("background-color: #EDEADE;");
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
        "QMenuBar { background: #F0EDE5; color: #2C2018; border-bottom: 1px solid #CCC5B5; }"
        "QMenuBar::item { padding: 4px 8px; }"
        "QMenuBar::item:selected { background: #E0D8CC; border-radius: 3px; }"
        "QMenu { background: #F4F1EA; color: #2C2018; border: 1px solid #CCC5B5; }"
        "QMenu::item { padding: 4px 20px; }"
        "QMenu::item:selected { background: #C0802C; color: #F5F2EA; }"
        "QMenu::separator { height: 1px; background: #CCC5B5; margin: 2px 0; }");

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
    label->setStyleSheet("color: #6E5E46; font-size: 10px; text-transform: uppercase;");
    rightLayout->addWidget(label);

    m_gpuSelector = new QComboBox();
    m_gpuSelector->setToolTip("Select the OpenCL compute device used to accelerate all image processing effects.\nChanging device reinitialises all GPU kernels and triggers a full reprocess.");
    m_gpuSelector->setStyleSheet(
        "QComboBox { color: #2C2018; background-color: #F4F1EA;"
        "  border: 1px solid #CCC5B5; border-radius: 3px; padding: 4px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #2C2018; background-color: #F4F1EA; }");

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
            "QWidget { background-color: #F4F1EA; border-radius: 4px; }");
        QVBoxLayout* panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(6, 4, 6, 6);
        panelLayout->setSpacing(4);

        // Title bar
        QWidget* titleBar = new QWidget();
        titleBar->setStyleSheet("background: transparent;");
        QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(0, 0, 0, 0);

        QLabel* title = new QLabel(QString("<b>%1</b>").arg(effect->getName()));
        title->setStyleSheet("color: #2C2018; background: transparent;");
        titleLayout->addWidget(title, 1);

        QPushButton* collapseBtn = new QPushButton("−");
        collapseBtn->setStyleSheet(
            "QPushButton { background: #D0C8B8; color: #2C2018; border: none;"
            "  border-radius: 3px; padding: 1px 5px; font-weight: bold; }"
            "QPushButton:hover { background: #BEB8A8; }");
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
    m_viewport->setImageSize(img.size());
    m_viewport->resetView();
    triggerReprocess();
}

void PhotoEditorApp::saveImage() {
    if (m_originalImage.isNull()) return;

    QString fileName = QFileDialog::getSaveFileName(
        this, "Save Image", m_lastDir,
        "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;All Files (*)");

    if (fileName.isEmpty()) return;
    m_viewport->currentImage().save(fileName);
    m_lastDir = QFileInfo(fileName).absolutePath();
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

void PhotoEditorApp::onProcessingComplete(QImage result) {
    if (result.isNull()) {
        m_viewport->update();   // CL-GL path: result already in GL texture
    } else {
        m_pendingFullRun = false;
        m_viewport->setImage(result);
    }
}

void PhotoEditorApp::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    triggerReprocess();
}
