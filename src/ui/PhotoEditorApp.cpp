#include "PhotoEditorApp.h"
#include "Theme.h"
#include "GpuDeviceRegistry.h"
#include "Histogram.h"
#include "ICropSource.h"
#include "IInteractiveEffect.h"
#include "RawLoader.h"
#include <QPainter>
#include <QTransform>
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

    // Pan throttle: coalesces mouseMove bursts (which fire at >100Hz on modern
    // mice) into at most one pipeline dispatch per ~16ms.  Leading edge fires
    // immediately; trailing edge covers the final state after a burst ends.
    m_panThrottle = new QTimer(this);
    m_panThrottle->setSingleShot(true);
    connect(m_panThrottle, &QTimer::timeout, this, &PhotoEditorApp::dispatchViewportUpdate);

    setupToolBar();
    setupUI();
    setWindowTitle("Afterglow");

    // Restore geometry and last-used directory from previous session
    QSettings settings("Afterglow", "Afterglow");
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

        // If this effect owns an on-canvas tool (crop handles, etc.), track it so
        // expanding/collapsing the panel activates/deactivates the overlay.
        IInteractiveEffect* interactive = dynamic_cast<IInteractiveEffect*>(effect);

        // Collapse toggle — shared_ptr so the lambda stays valid after panel is reparented
        auto expanded = std::make_shared<bool>(true);
        connect(collapseBtn, &QPushButton::clicked, this,
                [this, controls, collapseBtn, expanded, interactive]() {
            *expanded = !*expanded;
            if (controls) controls->setVisible(*expanded);
            collapseBtn->setText(*expanded ? "−" : "+");
            if (interactive)
                m_viewport->setActiveInteractiveEffect(*expanded ? interactive : nullptr);
        });

        // Show/hide panel when effect is toggled from the View menu
        panel->setVisible(entries[i].enabled);
        connect(m_effects, &EffectManager::effectToggled, panel,
                [this, panel, i, interactive](int idx, bool on) {
            if (idx != i) return;
            panel->setVisible(on);
            if (interactive && !on)
                m_viewport->setActiveInteractiveEffect(nullptr);
        });

        // Initial activation: if an interactive effect starts enabled + expanded,
        // attach it to the viewport so the overlay shows up on first image load.
        if (interactive && entries[i].enabled)
            m_viewport->setActiveInteractiveEffect(interactive);

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
    for (const auto& e : m_effects->entries()) {
        e.effect->onImageLoaded(meta);
        if (auto* cs = dynamic_cast<ICropSource*>(e.effect))
            cs->setSourceImageSize(img.size());
    }
    syncViewportRotation();
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

// Bake the user's non-destructive crop + rotation into the exported QImage.
// Pipeline output is still full-frame because crop/rotate is metadata; this
// is the one place where those metadata choices become real pixels.
static QImage applyCropAndRotate(const QImage& image, const ICropSource& cs) {
    if (image.isNull()) return image;

    const QRectF cropN = cs.userCropRect();
    const double cx = cropN.center().x() * image.width();
    const double cy = cropN.center().y() * image.height();
    const QSize dstSize(static_cast<int>(std::round(cropN.width()  * image.width())),
                        static_cast<int>(std::round(cropN.height() * image.height())));
    if (dstSize.isEmpty()) return image;

    // Map source→dst: translate crop centre to origin, rotate by -angle (Qt
    // rotates CW by default; our angle convention is CCW-positive), translate
    // out to the centre of the destination canvas.
    QTransform t;
    t.translate(dstSize.width() * 0.5, dstSize.height() * 0.5);
    t.rotate(-static_cast<double>(cs.userCropAngle()));
    t.translate(-cx, -cy);

    QImage dst(dstSize, image.format());
    dst.fill(Qt::black);
    QPainter p(&dst);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.setTransform(t);
    p.drawImage(0, 0, image);
    p.end();
    return dst;
}

void PhotoEditorApp::onExportComplete(QImage result) {
    if (m_pendingSavePath.isEmpty()) return;
    const QString path = m_pendingSavePath;
    m_pendingSavePath.clear();

    if (!result.isNull()) {
        for (const auto& e : m_effects->entries()) {
            if (!e.enabled) continue;
            if (auto* cs = dynamic_cast<ICropSource*>(e.effect)) {
                result = applyCropAndRotate(result, *cs);
                break;
            }
        }
    }

    if (result.isNull() || !result.save(path)) {
        QMessageBox::warning(this, "Save Failed",
            QString("Could not save image to:\n%1\n\n"
                    "Check that the directory is writable and you have sufficient disk space.")
            .arg(path));
    }
}

void PhotoEditorApp::onParametersChanged() {
    syncViewportRotation();
    triggerReprocess();
}

void PhotoEditorApp::onLiveParametersChanged() {
    syncViewportRotation();
    if (m_liveUpdate) triggerLiveReprocess();
}

void PhotoEditorApp::syncViewportRotation() {
    // Push the user's crop angle/centre to the viewport so the GL shader can
    // rotate the displayed image around the crop centre (Lightroom-style).
    // Updates immediately, independently of pipeline reprocessing — so live
    // dragging the rotation slider feels instant even on a slow GPU.
    for (const auto& e : m_effects->entries()) {
        if (auto* cs = dynamic_cast<ICropSource*>(e.effect)) {
            const QRectF c = cs->userCropRect();
            m_viewport->setImageRotation(cs->userCropAngle(), c.center());
            return;
        }
    }
}

void PhotoEditorApp::triggerReprocess() {
    if (m_originalImage.isNull()) return;

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    m_processor->processImageAsync(m_originalImage, active,
                                   m_viewport->viewportRequest(),
                                   RunMode::Commit);
}

void PhotoEditorApp::triggerLiveReprocess() {
    if (m_originalImage.isNull()) return;

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    m_processor->processImageAsync(m_originalImage, active,
                                   m_viewport->viewportRequest(),
                                   RunMode::LiveDrag);
}

void PhotoEditorApp::triggerViewportUpdate() {
    if (m_originalImage.isNull()) return;

    // Leading/trailing throttle — dispatch at most once per ~16ms so rapid
    // mouseMove events (1000Hz gaming mice, trackpads) don't saturate the
    // pipeline.  Zoom events go through the same path but are naturally rare
    // (one wheel tick = one event), so they aren't affected.
    constexpr int kIntervalMs = 16;
    if (!m_lastPanDispatch.isValid() || m_lastPanDispatch.elapsed() >= kIntervalMs) {
        dispatchViewportUpdate();
        return;
    }
    if (!m_panThrottle->isActive()) {
        const int remaining = kIntervalMs - static_cast<int>(m_lastPanDispatch.elapsed());
        m_panThrottle->start(remaining > 0 ? remaining : 1);
    }
}

void PhotoEditorApp::dispatchViewportUpdate() {
    if (m_originalImage.isNull()) return;
    m_lastPanDispatch.start();

    QVector<PhotoEditorEffect*> active;
    for (const auto& e : m_effects->entries())
        if (e.enabled) active.append(e.effect);

    m_processor->processImageAsync(m_originalImage, active,
                                   m_viewport->viewportRequest(),
                                   RunMode::PanZoom);
}

void PhotoEditorApp::onProcessingStarted() {
    m_processingLabel->setVisible(true);
}

void PhotoEditorApp::onProcessingComplete(QImage result) {
    m_processingLabel->setVisible(false);
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
    QSettings settings("Afterglow", "Afterglow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("lastDir",  m_lastDir);
    QMainWindow::closeEvent(event);
}
