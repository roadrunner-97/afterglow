#include "PhotoEditorApp.h"
#include "ExportDialog.h"
#include "ExportPath.h"
#include "LoupeView.h"
#include "Stylesheets.h"
#include "Theme.h"
#include "GpuDeviceRegistry.h"
#include "Histogram.h"
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
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
#include <QStackedWidget>
#include <QActionGroup>
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
#include <QDirIterator>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QPointer>
#include <QApplication>
#include <QThreadPool>
#include <QImageReader>
#include <QSet>
#include <QDebug>
#include <memory>

// Decode a non-RAW image with EXIF auto-orientation applied. QImage(path)
// honours no orientation tag, so portrait-shot JPEGs come out sideways
// without this. RAW files go through RawLoader, which handles flip itself.
static QImage decodeOriented(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}

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
    toolbar->setStyleSheet(Stylesheets::toolbar());

    // Mode switcher: Gallery (grid) / Loupe (preview) / Develop (editor).
    // Mirrors Lightroom's module picker — user double-clicks a thumbnail to
    // step through to Loupe, then Enter (or another double-click) to Develop.
    m_modeGroup = new QActionGroup(this);
    m_modeGroup->setExclusive(true);
    auto addModeAction = [&](const QString& label, Mode m) {
        QAction* act = new QAction(label, this);
        act->setCheckable(true);
        act->setData(static_cast<int>(m));
        m_modeGroup->addAction(act);
        toolbar->addAction(act);
        connect(act, &QAction::triggered, this, [this, m]() { setMode(m); });
        return act;
    };
    addModeAction("Gallery", Mode::Gallery)->setChecked(true);
    addModeAction("Loupe",   Mode::Loupe);
    QAction* developAct = addModeAction("Develop", Mode::Develop);
    // The default addModeAction handler only switches the page, which leaves
    // the editor empty when the user clicks Develop after browsing in
    // Loupe.  Run the full develop flow (load + reprocess) on top so the
    // toolbar button matches what double-click or Enter does in Loupe.
    connect(developAct, &QAction::triggered, this, [this]() {
        if (!m_currentImagePath.isEmpty() && m_currentImagePath != m_developedPath)
            loadFullImage(m_currentImagePath);
    });

    // Spacer + processing indicator label on the right side of the toolbar
    QWidget* spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(spacer);

    m_processingLabel = new QLabel("Processing…");
    m_processingLabel->setStyleSheet(Stylesheets::processingLabel());
    m_processingLabel->setVisible(false);
    toolbar->addWidget(m_processingLabel);
}

void PhotoEditorApp::setupUI() {
    setupMenuBar();

    // The central widget is a stacked widget with three pages:
    //   0 = Gallery (grid of thumbnails for browsing/triage)
    //   1 = Loupe   (full-size single-image preview, no GPU pipeline)
    //   2 = Develop (existing viewport + right panel — the editor)
    m_stack = new QStackedWidget();
    m_stack->setStyleSheet(QString("background: %1;").arg(Theme::BG_MAIN));
    setCentralWidget(m_stack);

    // ── Gallery page ────────────────────────────────────────────────────────
    m_gridView = new GridView();
    connect(m_gridView, &GridView::photoActivated,
            this, &PhotoEditorApp::onPhotoActivated);
    connect(m_gridView, &GridView::markChanged,
            this, &PhotoEditorApp::onMarkChanged);
    // Single-click / arrow keys in the grid track m_currentImagePath so
    // the toolbar Develop / Loupe buttons act on the highlighted photo.
    connect(m_gridView, &GridView::currentPathChanged,
            this, [this](const QString& path) { m_currentImagePath = path; });
    m_stack->addWidget(m_gridView);

    // ── Loupe page ──────────────────────────────────────────────────────────
    m_loupeView = new LoupeView();
    connect(m_loupeView, &LoupeView::developRequested,
            this, &PhotoEditorApp::onDevelopRequested);
    connect(m_loupeView, &LoupeView::previousRequested,
            this, [this]() { onLoupeNavigate(-1); });
    connect(m_loupeView, &LoupeView::nextRequested,
            this, [this]() { onLoupeNavigate(+1); });
    m_stack->addWidget(m_loupeView);

    // ── Develop page (existing editor: viewport + right panel) ─────────────
    QWidget* develop = new QWidget();
    develop->setStyleSheet(QString("background: %1;").arg(Theme::BG_MAIN));
    QHBoxLayout* mainLayout = new QHBoxLayout(develop);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    m_viewport = new ViewportWidget();
    connect(m_viewport, &ViewportWidget::viewportChanged,
            this, &PhotoEditorApp::triggerViewportUpdate);
    mainLayout->addWidget(m_viewport, 3);

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
    sep->setStyleSheet(Stylesheets::panelSeparator());
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
    m_stack->addWidget(develop);

    setMode(Mode::Gallery);
}

void PhotoEditorApp::setupMenuBar() {
    menuBar()->setStyleSheet(Stylesheets::menuBar());

    QMenu* fileMenu = menuBar()->addMenu("File");

    QAction* openAct = fileMenu->addAction("Open Image…");
    openAct->setShortcut(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &PhotoEditorApp::openImage);

    QAction* openFolderAct = fileMenu->addAction("Open Folder…");
    openFolderAct->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(openFolderAct, &QAction::triggered, this, &PhotoEditorApp::openFolder);

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
    label->setStyleSheet(Stylesheets::gpuSelectorLabel());
    rightLayout->addWidget(label);

    m_gpuSelector = new QComboBox();
    m_gpuSelector->setToolTip("Select the OpenCL compute device used to accelerate all image processing effects.\nChanging device reinitialises all GPU kernels and triggers a full reprocess.");
    m_gpuSelector->setStyleSheet(Stylesheets::gpuSelector());

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
        panel->setStyleSheet(Stylesheets::effectPanel());
        QVBoxLayout* panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(6, 4, 6, 6);
        panelLayout->setSpacing(4);

        // Title bar
        QWidget* titleBar = new QWidget();
        titleBar->setStyleSheet("background: transparent;");
        QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(0, 0, 0, 0);

        QLabel* title = new QLabel(QString("<b>%1</b>").arg(effect->getName()));
        title->setStyleSheet(Stylesheets::effectTitle());
        titleLayout->addWidget(title, 1);

        QPushButton* collapseBtn = new QPushButton("−");
        collapseBtn->setStyleSheet(Stylesheets::collapseButton());
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
    setMode(Mode::Develop);
    loadFullImage(fileName);
}

void PhotoEditorApp::loadFullImage(const QString& path) {
    m_lastDir = QFileInfo(path).absolutePath();

    QImage img;
    ImageMetadata meta;
    if (RawLoader::isRawFile(path)) {
        img = RawLoader::load(path, &meta);
        if (img.isNull())
            qWarning() << "RawLoader failed for" << path << "— trying QImage::load";
    }
    if (img.isNull())
        img = decodeOriented(path);

    if (img.isNull()) {
        qWarning() << "Failed to load image:" << path;
        return;
    }

    m_originalImage = img;
    m_currentImagePath = path;
    m_developedPath = path;
    m_viewport->setImageSize(img.size());
    m_viewport->resetView();
    // Notify effects with whatever metadata is already cheap to provide
    // (RAW colorTempK from LibRaw); the luminance histogram follows from
    // a worker thread because computing it on a 60MP RAW would otherwise
    // freeze the UI for hundreds of milliseconds.
    for (const auto& e : m_effects->entries())
        e.effect->onImageLoaded(meta);
    if (auto* cs = m_effects->cropSource())
        cs->setSourceImageSize(img.size());
    syncViewportRotation();
    triggerReprocess();

    auto* watcher = new QFutureWatcher<std::vector<uint32_t>>(this);
    connect(watcher, &QFutureWatcher<std::vector<uint32_t>>::finished, this,
            [this, watcher, expectedBits = img.constBits(), tempK = meta.colorTempK]() {
        // Drop the result if the user opened a different image while we
        // were computing.  constBits() identity is stable for the
        // lifetime of m_originalImage's underlying data buffer.
        if (m_originalImage.constBits() == expectedBits) {
            ImageMetadata fullMeta;
            fullMeta.colorTempK         = tempK;
            fullMeta.luminanceHistogram = watcher->result();
            for (const auto& e : m_effects->entries())
                e.effect->onImageLoaded(fullMeta);
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(
        [image = img]() { return computeLuminanceHistogram(image); }
    ));
}

void PhotoEditorApp::saveImage() {
    if (m_originalImage.isNull()) return;

    ExportDialog dlg(this);
    dlg.setDefaultDestinationDir(m_lastDir);
    if (dlg.exec() != QDialog::Accepted) return;

    const ExportOptions::Options opts = dlg.options();
    if (opts.destinationDir.isEmpty()) {
        QMessageBox::warning(this, "Export",
            "Please choose a destination folder.");
        return;
    }

    // batchIndex = 1 today; when batch export lands, the caller iterates and
    // bumps the index for the {n} token.  chooseDestination() handles the
    // overwrite policy (skip / suffix / overwrite) consistently here and there.
    const QString destPath = ExportPath::chooseDestination(
        opts, m_currentImagePath, /*batchIndex=*/1);
    if (destPath.isEmpty()) {
        // Skip-on-conflict — surface it so the user knows nothing was written.
        QMessageBox::information(this, "Export Skipped",
            "A file with that name already exists. "
            "Change the pattern or pick a different policy.");
        return;
    }

    m_lastDir = opts.destinationDir;
    m_pendingExportOpts = opts;
    m_processor->exportImageAsync(m_originalImage, *m_effects, destPath);
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
    // Take the pending options unconditionally so a failed/early-returning
    // export doesn't leak state into the next one.
    const auto opts = std::exchange(m_pendingExportOpts, std::nullopt);
    if (destinationPath.isEmpty()) return;
    const QString path = destinationPath;

    if (!result.isNull()) {
        if (auto* cs = m_effects->activeCropSource())
            result = applyCropAndRotate(result, *cs);
    }

    // With opts: explicit format hint + quality (saveImage path).
    // Without: legacy QImage::save behaviour, used by saveTestCase().
    const bool ok = !result.isNull() && (opts
        ? result.save(path, ExportOptions::qImageFormatHint(opts->format),
                            ExportOptions::qualityFor(*opts))
        : result.save(path));

    if (!ok) {
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
    triggerLiveReprocess();
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

// ─── Gallery / Loupe / Develop mode switching ───────────────────────────────

void PhotoEditorApp::setMode(Mode m) {
    m_stack->setCurrentIndex(static_cast<int>(m));
    // Keep the toolbar checkmark in sync with programmatic transitions
    // (e.g. double-click in the grid jumps us to Loupe).
    for (QAction* a : m_modeGroup->actions()) {
        if (a->data().toInt() == static_cast<int>(m)) {
            a->setChecked(true);
            break;
        }
    }
}

void PhotoEditorApp::openFolder() {
    const QString folder = QFileDialog::getExistingDirectory(
        this, "Open Folder", m_lastDir, QFileDialog::ShowDirsOnly);
    if (folder.isEmpty()) return;
    m_lastDir = folder;
    loadFolderIntoGrid(folder);
    setMode(Mode::Gallery);
}

// Per-folder JPEG cache for grid thumbnails. The first folder-open decodes
// each RAW's embedded preview (or full QImage for non-RAW) and writes a
// quality-85 JPEG here; subsequent opens read straight from disk if the
// source file's mtime hasn't moved past the cache file's mtime.
static QString thumbCachePath(const QString& sourcePath) {
    const QFileInfo fi(sourcePath);
    return fi.absoluteDir().filePath(".afterglow-thumbs/" + fi.fileName() + ".jpg");
}

static QImage tryLoadCachedThumb(const QString& sourcePath) {
    const QFileInfo cacheFi(thumbCachePath(sourcePath));
    if (!cacheFi.exists()) return {};
    const QFileInfo srcFi(sourcePath);
    // Stale cache: source has been re-saved since we last decoded.
    if (srcFi.lastModified() > cacheFi.lastModified()) return {};
    return QImage(cacheFi.absoluteFilePath());
}

static void writeCachedThumb(const QString& sourcePath, const QImage& thumb) {
    const QString out = thumbCachePath(sourcePath);
    QDir().mkpath(QFileInfo(out).absolutePath());
    thumb.save(out, "JPEG", 85);
}

// Recognised image extensions: same set the single-file dialog accepts. Kept
// here as a static QStringList so the lookup is amortised across all photos.
static const QStringList& imageExtensions() {
    static const QStringList exts = {
        "png", "jpg", "jpeg", "bmp", "tiff", "tif",
        "cr2", "cr3", "nef", "nrw", "arw", "sr2", "srf", "dng",
        "raf", "orf", "rw2", "pef", "srw", "x3f", "rwl", "mrw",
        "3fr", "kdc", "dcr", "erf",
    };
    return exts;
}

void PhotoEditorApp::loadFolderIntoGrid(const QString& folder) {
    QStringList allPaths;
    QDirIterator it(folder, QDir::Files | QDir::Readable, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
        const QString p = it.next();
        if (imageExtensions().contains(QFileInfo(p).suffix().toLower()))
            allPaths.append(p);
    }

    // Cameras shoot RAW + JPEG side-by-side; the JPEG is just the in-camera
    // preview of the RAW so we'd be triaging two views of the same photo.
    // Drop the JPEG sibling whenever a RAW with the same basename exists.
    QSet<QString> rawBases;
    for (const QString& p : allPaths) {
        if (RawLoader::isRawFile(p))
            rawBases.insert(QFileInfo(p).completeBaseName());
    }
    QStringList paths;
    for (const QString& p : allPaths) {
        const QFileInfo fi(p);
        if (!RawLoader::isRawFile(p) && rawBases.contains(fi.completeBaseName()))
            continue;
        paths.append(p);
    }
    paths.sort(Qt::CaseInsensitive);

    m_currentFolder = folder;
    m_currentPaths = paths;
    m_gridView->setPhotos(paths);
    readCatalog(folder);

    // Decode thumbnails on the global thread pool. Each finished decode posts
    // back to the GUI thread via QueuedConnection. Stale results from a
    // previous folder are dropped via the m_currentFolder guard.
    QPointer<PhotoEditorApp> self(this);
    const QString tag = folder;
    for (const QString& path : paths) {
        QThreadPool::globalInstance()->start([self, path, tag]() {
            QImage thumb = tryLoadCachedThumb(path);
            if (thumb.isNull()) {
                if (RawLoader::isRawFile(path)) thumb = RawLoader::loadThumbnail(path);
                else                            thumb = decodeOriented(path);
                if (thumb.isNull()) return;
                // Cap the side at 512px — saves memory when the grid is showing
                // hundreds of thumbnails and avoids holding full-res JPEGs alive.
                if (thumb.width() > 512 || thumb.height() > 512)
                    thumb = thumb.scaled(512, 512, Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation);
                writeCachedThumb(path, thumb);
            }
            QMetaObject::invokeMethod(qApp, [self, path, thumb, tag]() {
                if (!self) return;
                if (self->m_currentFolder != tag) return;
                self->m_gridView->setThumbnail(path, thumb);
            }, Qt::QueuedConnection);
        });
    }
}

void PhotoEditorApp::onPhotoActivated(const QString& path) {
    // Fast preview: prefer the embedded JPEG for RAW files; fall back to
    // QImage decode for non-RAW. Loaded synchronously since the embedded
    // JPEG is a few MB at most — measured later if it shows up as jank.
    QImage preview;
    if (RawLoader::isRawFile(path)) preview = RawLoader::loadThumbnail(path);
    if (preview.isNull())            preview = decodeOriented(path);
    if (preview.isNull()) {
        qWarning() << "No preview available for" << path;
        return;
    }
    m_currentImagePath = path;
    m_loupeView->setImage(preview);
    setMode(Mode::Loupe);
}

void PhotoEditorApp::onDevelopRequested() {
    if (m_currentImagePath.isEmpty()) return;
    setMode(Mode::Develop);
    if (m_currentImagePath != m_developedPath)
        loadFullImage(m_currentImagePath);
}

void PhotoEditorApp::onLoupeNavigate(int direction) {
    if (m_currentPaths.isEmpty() || m_currentImagePath.isEmpty()) return;
    const int idx = m_currentPaths.indexOf(m_currentImagePath);
    const int next = idx + direction;
    if (idx < 0 || next < 0 || next >= m_currentPaths.size()) return;
    onPhotoActivated(m_currentPaths[next]);
}

void PhotoEditorApp::onMarkChanged(const QString& path, GridView::Mark mark) {
    m_gridView->setMark(path, mark);
    writeCatalog();
}

// ─── Per-folder catalog (triage marks) ──────────────────────────────────────
//
// Stored as a flat JSON object next to the photos: <folder>/.afterglow-catalog.json
// Keys are basenames (so the file survives the folder being moved); values
// are single-character mark codes ('P', 'X', 'U').

QString PhotoEditorApp::catalogPath(const QString& folder) const {
    return QDir(folder).filePath(".afterglow-catalog.json");
}

void PhotoEditorApp::readCatalog(const QString& folder) {
    QFile f(catalogPath(folder));
    if (!f.open(QIODevice::ReadOnly)) return;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject()) return;
    const QJsonObject obj = doc.object();
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const QString fullPath = QDir(folder).filePath(it.key());
        const QString s = it.value().toString();
        if (s.isEmpty()) continue;
        m_gridView->setMark(fullPath, static_cast<GridView::Mark>(s.at(0).toLatin1()));
    }
}

void PhotoEditorApp::writeCatalog() const {
    if (m_currentFolder.isEmpty()) return;
    QJsonObject obj;
    QDirIterator it(m_currentFolder, QDir::Files, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
        const QString p = it.next();
        if (!imageExtensions().contains(QFileInfo(p).suffix().toLower())) continue;
        const auto m = m_gridView->mark(p);
        if (m == GridView::Mark::None) continue;  // unflagged is the default
        obj.insert(QFileInfo(p).fileName(), QString(QChar(static_cast<char>(m))));
    }
    QFile f(catalogPath(m_currentFolder));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
}
