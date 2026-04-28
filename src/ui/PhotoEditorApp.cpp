#include "PhotoEditorApp.h"
#include "Theme.h"
#include "GpuDeviceRegistry.h"
#include "Histogram.h"
#include "ICropSource.h"
#include "IInteractiveEffect.h"
#include "RawLoader.h"
#include "SettingsExporter.h"
#include "SettingsImporter.h"
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
#include <QScreen>
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
    // Width scales with the user's font / DPI: ParamSlider rows need room for
    // a label, slider track, and spinbox without wrapping.  Empirically ~36
    // characters of the body font fits the widest control we ship.
    rightPanel->setFixedWidth(fontMetrics().averageCharWidth() * 36);
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

    // Debug menu — import/export YAML presets.  Hidden behind its own menu so
    // it stays out of the way of the everyday File workflow but is also the
    // foundation for end-to-end tests and the future per-image edit-history
    // library system (sidecar YAMLs detected on image load).
    QMenu* debugMenu = menuBar()->addMenu("Debug");

    QAction* importAct = debugMenu->addAction("Load Settings…");
    connect(importAct, &QAction::triggered, this, &PhotoEditorApp::importSettings);

    QAction* exportAct = debugMenu->addAction("Save Settings…");
    connect(exportAct, &QAction::triggered, this, &PhotoEditorApp::exportSettings);

    debugMenu->addSeparator();

    QAction* testCaseAct = debugMenu->addAction("Save Test Case…");
    connect(testCaseAct, &QAction::triggered, this, &PhotoEditorApp::saveTestCase);
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
        m_gpuSelector->addItem("No OpenCL devices found");
        m_gpuSelector->setEnabled(false);
    } else {
        for (const auto& d : devs)
            m_gpuSelector->addItem(d.name + " [" + d.platformName + " · " + d.typeName + "]");
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
        IInteractiveEffect* interactive = entries[i].interactive;

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
    m_currentImagePath = fileName;
    m_viewport->setImageSize(img.size());
    m_viewport->resetView();
    meta.luminanceHistogram = computeLuminanceHistogram(img);
    for (const auto& e : m_effects->entries())
        e.effect->onImageLoaded(meta);
    if (auto* cs = m_effects->cropSource())
        cs->setSourceImageSize(img.size());
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

    m_processor->exportImageAsync(m_originalImage, *m_effects, fileName);
}

void PhotoEditorApp::importSettings() {
    QString suggested = m_lastDir;
    if (!m_currentImagePath.isEmpty()) {
        const QFileInfo fi(m_currentImagePath);
        const QString sidecar = fi.absoluteDir().filePath(fi.completeBaseName() + ".yml");
        if (QFile::exists(sidecar)) suggested = sidecar;
    }

    const QString fileName = QFileDialog::getOpenFileName(
        this, "Load Settings", suggested,
        "YAML (*.yml *.yaml);;All Files (*)");
    if (fileName.isEmpty()) return;
    m_lastDir = QFileInfo(fileName).absolutePath();

    SettingsImporter::Settings parsed;
    QString error;
    if (!SettingsImporter::readYaml(fileName, &parsed, &error)) {
        QMessageBox::warning(this, "Load Failed",
            QString("Could not read settings from:\n%1\n\n%2").arg(fileName, error));
        return;
    }

    SettingsImporter::applyToManager(parsed, *m_effects);

    // applyToManager blocks parametersChanged on each effect; fire one
    // definitive reprocess now that the full state is in place.
    triggerReprocess();
}

void PhotoEditorApp::saveTestCase() {
    if (m_originalImage.isNull() || m_currentImagePath.isEmpty()) {
        QMessageBox::warning(this, "Save Test Case",
            "Open an image first — a test case bundles the source image, the "
            "current settings, and the rendered output.");
        return;
    }

    const QString dir = QFileDialog::getExistingDirectory(
        this, "Save Test Case To Folder", m_lastDir,
        QFileDialog::ShowDirsOnly);
    if (dir.isEmpty()) return;
    m_lastDir = dir;

    const QFileInfo srcInfo(m_currentImagePath);
    const QString inputDest = QDir(dir).filePath("input." + srcInfo.suffix().toLower());
    if (QFile::exists(inputDest)) QFile::remove(inputDest);
    if (!QFile::copy(m_currentImagePath, inputDest)) {
        QMessageBox::warning(this, "Save Test Case",
            QString("Could not copy source image to:\n%1").arg(inputDest));
        return;
    }

    QString error;
    const QString yamlPath = QDir(dir).filePath("settings.yaml");
    if (!SettingsExporter::writeYaml(yamlPath, *m_effects, m_currentImagePath, &error)) {
        QMessageBox::warning(this, "Save Test Case",
            QString("Could not write settings to:\n%1\n\n%2").arg(yamlPath, error));
        return;
    }

    // Reuse the normal export path: onExportComplete bakes crop + rotate and
    // writes the destination passed in here.  PNG keeps the rendered output
    // bit-exact for the SSIM check that test_golden does at runtime.
    m_processor->exportImageAsync(m_originalImage, *m_effects,
                                  QDir(dir).filePath("expected.png"));
}

void PhotoEditorApp::exportSettings() {
    // Default the dump filename to <imagebasename>.yml next to the image.
    QString suggested;
    if (!m_currentImagePath.isEmpty()) {
        const QFileInfo fi(m_currentImagePath);
        suggested = fi.absoluteDir().filePath(fi.completeBaseName() + ".yml");
    } else {
        suggested = QDir(m_lastDir).filePath("settings.yml");
    }

    const QString fileName = QFileDialog::getSaveFileName(
        this, "Export Settings", suggested,
        "YAML (*.yml *.yaml);;All Files (*)");
    if (fileName.isEmpty()) return;
    m_lastDir = QFileInfo(fileName).absolutePath();

    QString error;
    if (!SettingsExporter::writeYaml(fileName, *m_effects, m_currentImagePath, &error)) {
        QMessageBox::warning(this, "Export Failed",
            QString("Could not write settings to:\n%1\n\n%2").arg(fileName, error));
    }
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

void PhotoEditorApp::onExportComplete(QImage result, QString destinationPath) {
    if (destinationPath.isEmpty()) return;
    const QString path = destinationPath;

    if (!result.isNull()) {
        if (auto* cs = m_effects->activeCropSource())
            result = applyCropAndRotate(result, *cs);
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
    if (auto* cs = m_effects->cropSource()) {
        const QRectF c = cs->userCropRect();
        m_viewport->setImageRotation(cs->userCropAngle(), c.center());
    }
}

void PhotoEditorApp::triggerReprocess() {
    if (m_originalImage.isNull()) return;

    m_processor->processImageAsync(m_originalImage, *m_effects,
                                   m_viewport->viewportRequest(),
                                   RunMode::Commit);
}

void PhotoEditorApp::triggerLiveReprocess() {
    if (m_originalImage.isNull()) return;

    m_processor->processImageAsync(m_originalImage, *m_effects,
                                   m_viewport->viewportRequest(),
                                   RunMode::LiveDrag);
}

void PhotoEditorApp::triggerViewportUpdate() {
    if (m_originalImage.isNull()) return;

    // Leading/trailing throttle — dispatch at most once per display frame so
    // rapid mouseMove events (1000Hz gaming mice, trackpads) don't saturate
    // the pipeline.  Use the active screen's refresh rate so 144Hz/240Hz
    // panels get smoother feedback than the old hard-coded 16ms (60Hz).
    // Zoom events go through the same path but are naturally rare (one wheel
    // tick = one event), so they aren't affected.
    const QScreen* s = screen();
    const double hz = (s && s->refreshRate() > 0.0) ? s->refreshRate() : 60.0;
    const int intervalMs = std::max(1, static_cast<int>(1000.0 / hz));
    if (!m_lastPanDispatch.isValid() || m_lastPanDispatch.elapsed() >= intervalMs) {
        dispatchViewportUpdate();
        return;
    }
    if (!m_panThrottle->isActive()) {
        const int remaining = intervalMs - static_cast<int>(m_lastPanDispatch.elapsed());
        m_panThrottle->start(remaining > 0 ? remaining : 1);
    }
}

void PhotoEditorApp::dispatchViewportUpdate() {
    if (m_originalImage.isNull()) return;
    m_lastPanDispatch.start();

    m_processor->processImageAsync(m_originalImage, *m_effects,
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
