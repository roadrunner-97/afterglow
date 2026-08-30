#include "PhotoEditorApp.h"
#include "ExportDialog.h"
#include "HistorySerializer.h"
#include "CachePurger.h"
#include "ImageMetadata.h"
#include "ExportPath.h"
#include "ExportResize.h"
#include "LoupeView.h"
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
#include <QSplitter>
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
#include <QKeyEvent>
#include <QEvent>
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
#include <QDateTime>
#include <QFileInfo>
#include <QLocale>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <memory>

namespace {
struct LoupeLoadResult {
    QImage        cameraJpeg;
    ImageMetadata metadata;
};

// RawLoader returns scene-linear RGBX64 so effects retain the sensor's range.
// QPainter expects display-encoded pixels, however, so an unedited RAW needs
// the same final linear-to-sRGB transfer used by GpuPipeline's pack kernel.
QImage linearRawToDisplay(const QImage &raw) {
    if (raw.isNull() || raw.format() != QImage::Format_RGBX64 || raw.text("color_space") != QStringLiteral("linear"))
        return raw;

    static const std::array<uint8_t, 65536> lut = [] {
        std::array<uint8_t, 65536> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float linear = static_cast<float>(i) / 65535.0f;
            const float srgb = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            values[i]        = static_cast<uint8_t>(std::clamp(std::lround(srgb * 255.0f), 0L, 255L));
        }
        return values;
    }();

    QImage display(raw.size(), QImage::Format_RGB32);
    for (int y = 0; y < raw.height(); ++y) {
        const auto *src = reinterpret_cast<const uint16_t *>(raw.constScanLine(y));
        auto       *dst = reinterpret_cast<QRgb *>(display.scanLine(y));
        for (int x = 0; x < raw.width(); ++x) {
            const uint16_t *pixel = src + 4 * x;
            dst[x]                = qRgb(lut[pixel[0]], lut[pixel[1]], lut[pixel[2]]);
        }
    }
    return display;
}

// ── Metadata format helpers ────────────────────────────────────────────────

static QString fmtShutter(float s) {
    if (s <= 0.0f) return "\xe2\x80\x94";
    if (s >= 1.0f) return QString::number(s, 'f', 1) + " s";
    return QString("1/%1 s").arg(static_cast<int>(std::round(1.0f / s)));
}

static QString fmtAperture(float f) {
    if (f <= 0.0f) return "\xe2\x80\x94";
    return QString("f/%1").arg(QString::number(f, 'f', f < 10.0f ? 1 : 0));
}

static QString fmtIso(float iso) {
    if (iso <= 0.0f) return "\xe2\x80\x94";
    return "ISO\xc2\xa0" + QString::number(static_cast<int>(std::round(iso)));
}

static QString fmtCamera(const ImageMetadata &m) {
    if (m.cameraMake.isEmpty() && m.cameraModel.isEmpty()) return "\xe2\x80\x94";
    if (!m.cameraMake.isEmpty() && m.cameraModel.startsWith(m.cameraMake, Qt::CaseInsensitive)) return m.cameraModel;
    if (m.cameraMake.isEmpty()) return m.cameraModel;
    if (m.cameraModel.isEmpty()) return m.cameraMake;
    return m.cameraMake + " " + m.cameraModel;
}

static MetadataTray::Info buildMetadataInfo(const QString &path, const QSize &size, const ImageMetadata &meta) {
    MetadataTray::Info info;
    info.filename   = QFileInfo(path).fileName();
    info.dimensions = QString("%1 \xc3\x97 %2").arg(size.width()).arg(size.height());
    info.camera     = fmtCamera(meta);
    info.lens       = meta.lens.isEmpty() ? QString("\xe2\x80\x94") : meta.lens;

    QStringList exp;
    if (meta.isoSpeed > 0.0f) exp << fmtIso(meta.isoSpeed);
    if (meta.shutterSec > 0.0f) exp << fmtShutter(meta.shutterSec);
    if (meta.aperture > 0.0f) exp << fmtAperture(meta.aperture);
    info.exposure = exp.isEmpty() ? QString("\xe2\x80\x94") : exp.join(" \xc2\xb7 ");

    info.captured = meta.captureTime.isValid() ? QLocale::system().toString(meta.captureTime, QLocale::ShortFormat)
                                               : QString("\xe2\x80\x94");
    return info;
}

// ── History label builder ──────────────────────────────────────────────────

// Build a human-readable label for a history entry given the effect's display name.
QString entryLabel(const UndoHistory::Entry &e, const QString &effectName) {
    QString result;

    if (e.enabled.has_value()) result = effectName + (e.enabled->second ? " on" : " off");

    if (!e.params.isEmpty()) {
        QString paramPart;
        if (e.params.size() == 1) {
            const auto    it = e.params.cbegin();
            const QString f  = it.value().from.isValid() ? it.value().from.toString() : "set";
            const QString t  = it.value().to.isValid() ? it.value().to.toString() : "removed";
            paramPart        = f + " → " + t;
        } else {
            paramPart = "(" + QString::number(e.params.size()) + " changes)";
        }
        if (result.isEmpty()) result = effectName + " " + paramPart;
        else result += " · " + paramPart;
    }

    return result.isEmpty() ? effectName : result;
}
} // namespace

// Decode a non-RAW image with EXIF auto-orientation applied. QImage(path)
// honours no orientation tag, so portrait-shot JPEGs come out sideways
// without this. RAW files go through RawLoader, which handles flip itself.
static QImage decodeOriented(const QString &path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}

static QImage decodeThumbnailOriented(const QString &path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QSize sourceSize = reader.size();
    if (sourceSize.isValid()) reader.setScaledSize(sourceSize.scaled(512, 512, Qt::KeepAspectRatio));
    return reader.read();
}

PhotoEditorApp::PhotoEditorApp(EffectManager *effectManager, QWidget *parent)
    : QMainWindow(parent), m_effects(effectManager), m_processor(new ImageProcessor(this)),
      m_resizeDebounce(new QTimer(this)) {
    m_history = new UndoHistory(200, this);
    connect(m_processor, &ImageProcessor::processingComplete, this, &PhotoEditorApp::onProcessingComplete);
    connect(m_processor, &ImageProcessor::processingStarted, this, &PhotoEditorApp::onProcessingStarted);
    connect(m_processor, &ImageProcessor::exportComplete, this, &PhotoEditorApp::onExportComplete);

    m_resizeDebounce->setSingleShot(true);
    m_resizeDebounce->setInterval(150);
    connect(m_resizeDebounce, &QTimer::timeout, this, &PhotoEditorApp::triggerReprocess);

    // Pan throttle: coalesces mouseMove bursts (which fire at >100Hz on modern
    // mice) into at most one pipeline dispatch per ~16ms.  Leading edge fires
    // immediately; trailing edge covers the final state after a burst ends.
    m_panThrottle = new QTimer(this);
    m_panThrottle->setSingleShot(true);
    connect(m_panThrottle, &QTimer::timeout, this, &PhotoEditorApp::dispatchViewportUpdate);

    m_thumbnailPool = new QThreadPool(this);
    m_thumbnailPool->setMaxThreadCount(2);

    setupToolBar();
    setupUI();
    snapshotDefaults();
    setWindowTitle("Afterglow");

    // Restore geometry and last-used directory from previous session
    QSettings settings("Afterglow", "Afterglow");
    if (settings.contains("geometry")) restoreGeometry(settings.value("geometry").toByteArray());
    else setGeometry(100, 100, 1400, 900);
    m_lastDir = settings.value("lastDir", QDir::homePath()).toString();

    // Global \-key "before edits" preview.  Filter on qApp so the binding
    // works regardless of which child widget currently holds focus
    // (Develop's viewport, Loupe, sidebar buttons, etc.).
    qApp->installEventFilter(this);
}

PhotoEditorApp::~PhotoEditorApp() = default;

void PhotoEditorApp::initProofer(std::unique_ptr<EffectManager> prooferEffects) {
    m_proofCache = new ProofCache(this);
    m_proofer    = new Proofer(std::move(prooferEffects), m_defaults, m_proofCache, this);

    connect(m_proofer, &Proofer::proofStarted, this, [this](const QString &path) {
        m_gridView->setProofStatus(path, GridView::ProofStatus::Proofing);
        if (m_loupePath == path) m_loupeView->setProofingState(true);
    });

    connect(m_proofer, &Proofer::proofFinished, this, [this](const QString &path, const QImage &proof) {
        m_gridView->setProofStatus(path, GridView::ProofStatus::Proofed);
        m_gridView->setThumbnail(path, proof);
        if (m_loupePath == path) {
            m_loupeView->setProofingState(false);
            m_loupeView->setProofImage(proof);
        }
        if (m_stack->currentIndex() == static_cast<int>(Mode::Develop) && m_proofer->pendingCount() == 0)
            m_proofer->pause();
    });

    connect(m_proofer, &Proofer::proofFailed, this, [this](const QString &path, const QString & /*error*/) {
        m_gridView->setProofStatus(path, GridView::ProofStatus::NotProofed);
        if (m_loupePath == path) m_loupeView->setProofingState(false);
        if (m_stack->currentIndex() == static_cast<int>(Mode::Develop) && m_proofer->pendingCount() == 0)
            m_proofer->pause();
    });
}

void PhotoEditorApp::setupToolBar() {
    QToolBar *toolbar = addToolBar("Preview");
    toolbar->setMovable(false);
    // Mode switcher: Gallery (grid) / Loupe (preview) / Develop (editor).
    // Mirrors Lightroom's module picker — user double-clicks a thumbnail to
    // step through to Loupe, then Enter (or another double-click) to Develop.
    m_modeGroup = new QActionGroup(this);
    m_modeGroup->setExclusive(true);
    auto addModeAction = [&](const QString &label, Mode m) {
        QAction *act = new QAction(label, this);
        act->setCheckable(true);
        act->setData(static_cast<int>(m));
        m_modeGroup->addAction(act);
        toolbar->addAction(act);
        connect(act, &QAction::triggered, this, [this, m]() {
            if (m == Mode::Gallery) {
                setMode(m);
            } else if (m == Mode::Loupe) {
                if (!m_currentImagePath.isEmpty()) loadLoupeImage(m_currentImagePath);
                else setMode(static_cast<Mode>(m_stack->currentIndex()));
            } else {
                if (m_currentImagePath.isEmpty()) {
                    setMode(static_cast<Mode>(m_stack->currentIndex()));
                    return;
                }
                setMode(m);
                if (m_currentImagePath != m_developedPath) loadFullImage(m_currentImagePath);
            }
        });
        return act;
    };
    addModeAction("Gallery", Mode::Gallery)->setChecked(true);
    addModeAction("Loupe", Mode::Loupe);
    addModeAction("Develop", Mode::Develop);

    // Spacer + processing indicator label on the right side of the toolbar
    QWidget *spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(spacer);

    m_processingLabel = new QLabel("Processing…");
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
    setCentralWidget(m_stack);

    // ── Gallery page ────────────────────────────────────────────────────────
    m_gridView = new GridView();
    connect(m_gridView, &GridView::photoActivated, this, &PhotoEditorApp::onPhotoActivated);
    connect(m_gridView, &GridView::markChanged, this, &PhotoEditorApp::onMarkChanged);
    // Single-click / arrow keys in the grid track m_currentImagePath so
    // the toolbar Develop / Loupe buttons act on the highlighted photo.
    connect(m_gridView, &GridView::currentPathChanged, this,
            [this](const QString &path) { m_currentImagePath = path; });
    m_stack->addWidget(m_gridView);

    // ── Loupe page ──────────────────────────────────────────────────────────
    m_loupeView = new LoupeView();
    connect(m_loupeView, &LoupeView::developRequested, this, &PhotoEditorApp::onDevelopRequested);
    connect(m_loupeView, &LoupeView::previousRequested, this, [this]() { onLoupeNavigate(-1); });
    connect(m_loupeView, &LoupeView::nextRequested, this, [this]() { onLoupeNavigate(+1); });
    // Marks set from the Loupe sidebar / A-R-D keys flow through the same
    // catalog-write path as marks set from the grid.  LoupeView doesn't
    // know the path; we inject the currently-displayed one here.
    connect(m_loupeView, &LoupeView::markChanged, this, [this](GridView::Mark m) {
        if (!m_currentImagePath.isEmpty()) onMarkChanged(m_currentImagePath, m);
    });
    m_stack->addWidget(m_loupeView);

    // ── Develop page (existing editor: viewport + right panel) ─────────────
    QSplitter *develop = new QSplitter(Qt::Horizontal);
    develop->setContentsMargins(0, 0, 0, 0);
    develop->setHandleWidth(4);

    // ── Left panel (metadata) ──────────────────────────────────────────────
    QWidget     *leftPanel  = new QWidget();
    QVBoxLayout *leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(8, 8, 8, 8);
    leftLayout->setSpacing(0);
    leftPanel->setMinimumWidth(fontMetrics().averageCharWidth() * 20);
    leftPanel->setMaximumWidth(fontMetrics().averageCharWidth() * 50);

    m_metadataTray = new MetadataTray();
    leftLayout->addWidget(m_metadataTray);

    auto *leftSep = new QFrame();
    leftSep->setFrameShape(QFrame::HLine);
    leftLayout->addWidget(leftSep);

    m_historyTray = new HistoryTray();
    leftLayout->addWidget(m_historyTray, 1);

    develop->addWidget(leftPanel);

    m_viewport = new ViewportWidget();
    connect(m_viewport, &ViewportWidget::viewportChanged, this, &PhotoEditorApp::triggerViewportUpdate);
    develop->addWidget(m_viewport);

    connect(m_history, &UndoHistory::historyChanged, this, &PhotoEditorApp::refreshHistoryTray);
    connect(m_historyTray, &HistoryTray::rowActivated, this, [this](int index) {
        // Defer the jump to the next event-loop iteration.  itemClicked fires
        // inside QListWidget's mouseReleaseEvent; if we call m_list->clear()
        // (via refreshHistoryTray) synchronously, QListWidget never finishes
        // its release path and its implicit mouse grab stays active — breaking
        // subsequent viewport panning.
        QTimer::singleShot(0, this, [this, index]() {
            const int target = index;
            m_history->setApplying(true);
            while (m_history->cursor() > target) {
                if (auto e = m_history->undo()) applyHistoryEntry(*e, /*applyFrom=*/true);
                else break;
            }
            while (m_history->cursor() < target) {
                if (auto e = m_history->redo()) applyHistoryEntry(*e, /*applyFrom=*/false);
                else break;
            }
            m_history->setApplying(false);
            syncViewportRotation();
            triggerReprocess();
            writeSidecar();
        });
    });

    QWidget *rightPanel = new QWidget();
    rightPanel->setMinimumWidth(fontMetrics().averageCharWidth() * 24);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(8, 8, 8, 8);
    rightLayout->setSpacing(6);

    setupGpuSelector(rightLayout);

    QFrame *sep = new QFrame();
    sep->setFrameShape(QFrame::HLine);
    rightLayout->addWidget(sep);

    QScrollArea *effectsScroll = new QScrollArea();
    effectsScroll->setWidgetResizable(true);

    QWidget     *effectsContainer = new QWidget();
    QVBoxLayout *effectsLayout    = new QVBoxLayout(effectsContainer);
    effectsLayout->setContentsMargins(0, 0, 0, 0);
    effectsLayout->setSpacing(4);

    setupEffectPanels(effectsLayout);
    effectsLayout->addStretch();

    effectsScroll->setWidget(effectsContainer);
    rightLayout->addWidget(effectsScroll, 1);

    develop->addWidget(rightPanel);

    develop->setStretchFactor(0, 0);
    develop->setStretchFactor(1, 1);
    develop->setStretchFactor(2, 0);

    m_stack->addWidget(develop);

    setMode(Mode::Gallery);
}

void PhotoEditorApp::setupMenuBar() {
    QMenu *fileMenu = menuBar()->addMenu("File");

    QAction *openAct = fileMenu->addAction("Open Image…");
    openAct->setShortcut(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &PhotoEditorApp::openImage);

    QAction *openFolderAct = fileMenu->addAction("Open Folder…");
    openFolderAct->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(openFolderAct, &QAction::triggered, this, &PhotoEditorApp::openFolder);

    QAction *saveAct = fileMenu->addAction("Save Image…");
    saveAct->setShortcut(QKeySequence::Save);
    connect(saveAct, &QAction::triggered, this, &PhotoEditorApp::saveImage);

    fileMenu->addSeparator();

    QAction *exitAct = fileMenu->addAction("Exit");
    exitAct->setShortcut(QKeySequence("Ctrl+Q"));
    connect(exitAct, &QAction::triggered, this, &QWidget::close);

    // Edit → Undo / Redo
    QMenu *editMenu = menuBar()->addMenu("Edit");

    m_undoAct = editMenu->addAction("Undo");
    m_undoAct->setShortcut(QKeySequence::Undo);
    m_undoAct->setEnabled(false);
    connect(m_undoAct, &QAction::triggered, this, [this]() {
        m_history->setApplying(true);
        if (auto entry = m_history->undo()) {
            applyHistoryEntry(*entry, /*applyFrom=*/true);
            syncCommittedGeometry();
            syncViewportRotation();
            triggerReprocess();
            writeSidecar();
            refreshEditedState();
        }
        m_history->setApplying(false);
    });
    connect(m_history, &UndoHistory::canUndoChanged, m_undoAct, &QAction::setEnabled);

    m_redoAct = editMenu->addAction("Redo");
    m_redoAct->setShortcuts({QKeySequence::Redo, QKeySequence("Ctrl+Y")});
    m_redoAct->setEnabled(false);
    connect(m_redoAct, &QAction::triggered, this, [this]() {
        m_history->setApplying(true);
        if (auto entry = m_history->redo()) {
            applyHistoryEntry(*entry, /*applyFrom=*/false);
            syncCommittedGeometry();
            syncViewportRotation();
            triggerReprocess();
            writeSidecar();
            refreshEditedState();
        }
        m_history->setApplying(false);
    });
    connect(m_history, &UndoHistory::canRedoChanged, m_redoAct, &QAction::setEnabled);

    // View → Effects — enable/disable individual effects
    QMenu *viewMenu    = menuBar()->addMenu("View");
    QMenu *effectsMenu = viewMenu->addMenu("Effects");

    const auto &entries = m_effects->entries();
    m_effectMenuActions.clear();
    for (int i = 0; i < entries.size(); ++i) {
        QAction *act = effectsMenu->addAction(entries[i].effect->getName());
        act->setCheckable(true);
        act->setChecked(entries[i].enabled);
        m_effectMenuActions.append(act);
        connect(act, &QAction::toggled, this, [this, i](bool on) {
            m_effects->setEnabled(i, on);
            triggerReprocess();
            writeSidecar();
            m_history->recordFromCurrent(currentSnapshot());
            refreshEditedState();
        });
    }

    // Debug menu — import/export YAML presets.  Hidden behind its own menu so
    // it stays out of the way of the everyday File workflow but is also the
    // foundation for end-to-end tests and the future per-image edit-history
    // library system (sidecar YAMLs detected on image load).
    QMenu *debugMenu = menuBar()->addMenu("Debug");

    QAction *importAct = debugMenu->addAction("Load Settings…");
    connect(importAct, &QAction::triggered, this, &PhotoEditorApp::importSettings);

    QAction *exportAct = debugMenu->addAction("Save Settings…");
    connect(exportAct, &QAction::triggered, this, &PhotoEditorApp::exportSettings);

    debugMenu->addSeparator();

    QAction *testCaseAct = debugMenu->addAction("Save Test Case…");
    connect(testCaseAct, &QAction::triggered, this, &PhotoEditorApp::saveTestCase);

    debugMenu->addSeparator();
    QAction *rebuildPreviewsAct = debugMenu->addAction("Rebuild Edited Previews");
    connect(rebuildPreviewsAct, &QAction::triggered, this, &PhotoEditorApp::rebuildEditedPreviews);

    QAction *purgeCachesAct = debugMenu->addAction("Purge Photo Caches…");
    connect(purgeCachesAct, &QAction::triggered, this, &PhotoEditorApp::purgeCaches);
}

void PhotoEditorApp::setupGpuSelector(QVBoxLayout *rightLayout) {
    QLabel *label = new QLabel("GPU Device");
    rightLayout->addWidget(label);

    m_gpuSelector = new QComboBox();
    m_gpuSelector->setToolTip("Select the OpenCL compute device used to accelerate all image processing "
                              "effects.\nChanging device reinitialises all GPU kernels and triggers a full reprocess.");

    const auto &devs = GpuDeviceRegistry::instance().devices();
    if (devs.empty()) {
        m_gpuSelector->addItem("No OpenCL devices found");
        m_gpuSelector->setEnabled(false);
    } else {
        for (const auto &d : devs) m_gpuSelector->addItem(d.name + " [" + d.platformName + " · " + d.typeName + "]");
        m_gpuSelector->setCurrentIndex(GpuDeviceRegistry::instance().currentIndex());
    }

    connect(m_gpuSelector, QOverload<int>::of(&QComboBox::activated), this, [this](int idx) {
        GpuDeviceRegistry::instance().setDevice(idx);
        triggerReprocess();
    });

    rightLayout->addWidget(m_gpuSelector);
}

void PhotoEditorApp::setupEffectPanels(QVBoxLayout *effectsLayout) {
    const auto &entries              = m_effects->entries();
    bool        hasActiveInteractive = false;
    for (int i = 0; i < entries.size(); ++i) {
        PhotoEditorEffect  *effect      = entries[i].effect;
        IInteractiveEffect *interactive = entries[i].interactive;

        // Container
        QWidget     *panel       = new QWidget();
        QVBoxLayout *panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(6, 4, 6, 6);
        panelLayout->setSpacing(4);

        // Title bar
        QWidget     *titleBar    = new QWidget();
        QHBoxLayout *titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(0, 0, 0, 0);

        QLabel *title = new QLabel(QString("<b>%1</b>").arg(effect->getName()));
        titleLayout->addWidget(title, 1);

        if (interactive) {
            QPushButton *editOnImageBtn = new QPushButton("◎");
            editOnImageBtn->setToolTip("Activate this effect's on-image controls.");
            editOnImageBtn->setMaximumWidth(28);
            connect(editOnImageBtn, &QPushButton::clicked, this,
                    [this, interactive]() { m_viewport->setActiveInteractiveEffect(interactive); });
            titleLayout->addWidget(editOnImageBtn);
        }

        QPushButton *collapseBtn = new QPushButton("−");
        collapseBtn->setToolTip("Collapse or expand this effect's controls.");
        collapseBtn->setMaximumWidth(28);
        titleLayout->addWidget(collapseBtn);
        panelLayout->addWidget(titleBar);

        // Controls
        QWidget *controls = effect->createControlsWidget();
        if (controls) {
            panelLayout->addWidget(controls);
        }

        // If this effect owns an on-canvas tool (crop handles, etc.), track it so
        // expanding/collapsing the panel activates/deactivates the overlay.
        // Collapse toggle — shared_ptr so the lambda stays valid after panel is reparented
        auto expanded = std::make_shared<bool>(true);
        connect(collapseBtn, &QPushButton::clicked, this, [this, controls, collapseBtn, expanded, interactive]() {
            *expanded = !*expanded;
            if (controls) controls->setVisible(*expanded);
            collapseBtn->setText(*expanded ? "−" : "+");
            if (interactive) {
                if (*expanded) m_viewport->setActiveInteractiveEffect(interactive);
                else m_viewport->clearActiveInteractiveEffect(interactive);
            }
        });

        // Show/hide panel when effect is toggled from the View menu
        panel->setVisible(entries[i].enabled);
        connect(m_effects, &EffectManager::effectToggled, panel, [this, panel, i, interactive](int idx, bool on) {
            if (idx != i) return;
            panel->setVisible(on);
            if (interactive && !on) m_viewport->clearActiveInteractiveEffect(interactive);
        });

        // Initial activation: if an interactive effect starts enabled + expanded,
        // attach it to the viewport so the overlay shows up on first image load.
        if (interactive && entries[i].enabled && !hasActiveInteractive) {
            m_viewport->setActiveInteractiveEffect(interactive);
            hasActiveInteractive = true;
        }

        // Wire parametersChanged (committed) and liveParametersChanged (drag)
        connect(effect, &PhotoEditorEffect::parametersChanged, this, &PhotoEditorApp::onParametersChanged);
        connect(effect, &PhotoEditorEffect::liveParametersChanged, this, &PhotoEditorApp::onLiveParametersChanged);

        effectsLayout->addWidget(panel);
    }
}

void PhotoEditorApp::openImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", m_lastDir,
                                                    "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif "
                                                    "*.cr2 *.cr3 *.nef *.nrw *.arw *.dng *.raf *.orf *.rw2);;"
                                                    "All Files (*)");

    if (fileName.isEmpty()) return;
    setMode(Mode::Develop);
    loadFullImage(fileName);
}

void PhotoEditorApp::loadFullImage(const QString &path) {
    m_lastDir = QFileInfo(path).absolutePath();

    QImage        img;
    ImageMetadata meta;
    if (RawLoader::isRawFile(path)) {
        img = RawLoader::load(path, &meta);
        if (img.isNull()) qWarning() << "RawLoader failed for" << path << "— trying QImage::load";
    }
    if (img.isNull()) img = decodeOriented(path);

    if (img.isNull()) {
        qWarning() << "Failed to load image:" << path;
        return;
    }

    // Flush history for the outgoing image before we replace m_developedPath.
    flushHistorySidecar();

    m_loadedImage   = img;
    m_originalImage = img;
    m_committedGeometryState.clear();
    m_latestDevelopPreview = {};
    m_latestDevelopPreviewPath.clear();
    m_currentImagePath = path;
    m_developedPath    = path;
    m_viewport->setImageSize(img.size());
    m_viewport->resetView();

    // Reset every effect to its constructor-time state before touching the
    // new image: otherwise a brightness/crop tweak from the previous photo
    // would silently apply to this one too.  onImageLoaded then layers on
    // any image-aware adjustments (e.g. as-shot WB), and the sidecar (if
    // present) overrides on top of that.
    SettingsImporter::applyToManager(m_defaults, *m_effects);

    // Notify effects with whatever metadata is already cheap to provide
    // (RAW colorTempK from LibRaw); the luminance histogram follows from
    // a worker thread because computing it on a 60MP RAW would otherwise
    // freeze the UI for hundreds of milliseconds.
    for (const auto &e : m_effects->entries()) e.effect->onImageLoaded(meta);
    if (auto *cs = m_effects->cropSource()) cs->setSourceImageSize(img.size());

    const QString sidecar = sidecarPathFor(path);
    if (QFile::exists(sidecar)) {
        SettingsImporter::Settings parsed;
        QString                    error;
        if (SettingsImporter::readYaml(sidecar, &parsed, &error)) SettingsImporter::applyToManager(parsed, *m_effects);
        else qWarning() << "Sidecar parse failed for" << sidecar << ":" << error;
    } else {
        writeSidecar();
    }

    syncCommittedGeometry();

    // Load or seed undo history for the new image.
    const QString histPath = historySidecarPathFor(path);
    if (QFile::exists(histPath)) {
        HistorySerializer::HistoryData hdata;
        QString                        hErr;
        if (HistorySerializer::readYaml(histPath, &hdata, &hErr))
            m_history->load(std::move(hdata.entries), hdata.cursor, std::move(hdata.shadow));
        else {
            qWarning() << "History parse failed for" << histPath << ":" << hErr;
            m_history->seed(currentSnapshot());
        }
    } else {
        m_history->seed(currentSnapshot());
    }
    refreshEditedState();

    m_metadataTray->setInfo(buildMetadataInfo(path, img.size(), meta));

    syncViewportRotation();
    triggerReprocess();

    auto *watcher = new QFutureWatcher<std::vector<uint32_t>>(this);
    connect(watcher, &QFutureWatcher<std::vector<uint32_t>>::finished, this,
            [this, watcher, expectedImageKey = img.cacheKey(), tempK = meta.colorTempK]() {
                // Drop the result if the user opened or mutated a different
                // image while we were computing. Unlike a bits pointer, the
                // cache key cannot alias a recycled allocation.
                if (m_originalImage.cacheKey() == expectedImageKey) {
                    ImageMetadata fullMeta;
                    fullMeta.colorTempK         = tempK;
                    fullMeta.luminanceHistogram = watcher->result();
                    for (const auto &e : m_effects->entries()) e.effect->onImageLoaded(fullMeta);
                }
                watcher->deleteLater();
            });
    watcher->setFuture(QtConcurrent::run([image = img]() { return computeLuminanceHistogram(image); }));
}

void PhotoEditorApp::saveImage() {
    if (m_originalImage.isNull()) return;

    ExportDialog dlg(this);
    dlg.setDefaultDestinationDir(m_lastDir);
    if (dlg.exec() != QDialog::Accepted) return;

    const ExportOptions::Options opts = dlg.options();
    if (opts.destinationDir.isEmpty()) {
        QMessageBox::warning(this, "Export", "Please choose a destination folder.");
        return;
    }

    // batchIndex = 1 today; when batch export lands, the caller iterates and
    // bumps the index for the {n} token.  chooseDestination() handles the
    // overwrite policy (skip / suffix / overwrite) consistently here and there.
    const QString destPath = ExportPath::chooseDestination(opts, m_currentImagePath, /*batchIndex=*/1);
    if (destPath.isEmpty()) {
        // Skip-on-conflict — surface it so the user knows nothing was written.
        QMessageBox::information(this, "Export Skipped",
                                 "A file with that name already exists. "
                                 "Change the pattern or pick a different policy.");
        return;
    }

    m_lastDir = opts.destinationDir;
    PendingExport pending{opts, std::nullopt};
    if (auto *cs = m_effects->activeCropSource()) pending.crop = CropSnapshot{cs->userCropRect(), cs->userCropAngle()};
    const uint64_t requestId = m_processor->exportImageAsync(m_originalImage, *m_effects, destPath);
    m_pendingExports.insert(requestId, std::move(pending));
}

void PhotoEditorApp::importSettings() {
    QString suggested = m_lastDir;
    if (!m_currentImagePath.isEmpty()) {
        const QFileInfo fi(m_currentImagePath);
        const QString   sidecar = fi.absoluteDir().filePath(fi.completeBaseName() + ".yml");
        if (QFile::exists(sidecar)) suggested = sidecar;
    }

    const QString fileName =
        QFileDialog::getOpenFileName(this, "Load Settings", suggested, "YAML (*.yml *.yaml);;All Files (*)");
    if (fileName.isEmpty()) return;
    m_lastDir = QFileInfo(fileName).absolutePath();

    SettingsImporter::Settings parsed;
    QString                    error;
    if (!SettingsImporter::readYaml(fileName, &parsed, &error)) {
        QMessageBox::warning(this, "Load Failed",
                             QString("Could not read settings from:\n%1\n\n%2").arg(fileName, error));
        return;
    }

    SettingsImporter::applyToManager(parsed, *m_effects);

    // applyToManager blocks parametersChanged on each effect; fire one
    // definitive reprocess now that the full state is in place.
    syncCommittedGeometry();
    triggerReprocess();
    writeSidecar();
    m_history->recordFromCurrent(currentSnapshot());
    refreshEditedState();
}

void PhotoEditorApp::saveTestCase() {
    if (m_originalImage.isNull() || m_currentImagePath.isEmpty()) {
        QMessageBox::warning(this, "Save Test Case",
                             "Open an image first — a test case bundles the source image, the "
                             "current settings, and the rendered output.");
        return;
    }

    const QString dir =
        QFileDialog::getExistingDirectory(this, "Save Test Case To Folder", m_lastDir, QFileDialog::ShowDirsOnly);
    if (dir.isEmpty()) return;
    m_lastDir = dir;

    const QFileInfo srcInfo(m_currentImagePath);
    const QString   inputDest = QDir(dir).filePath("input." + srcInfo.suffix().toLower());
    if (QFile::exists(inputDest)) QFile::remove(inputDest);
    if (!QFile::copy(m_currentImagePath, inputDest)) {
        QMessageBox::warning(this, "Save Test Case", QString("Could not copy source image to:\n%1").arg(inputDest));
        return;
    }

    QString       error;
    const QString yamlPath = QDir(dir).filePath("settings.yaml");
    if (!SettingsExporter::writeYaml(yamlPath, *m_effects, m_currentImagePath, &error)) {
        QMessageBox::warning(this, "Save Test Case",
                             QString("Could not write settings to:\n%1\n\n%2").arg(yamlPath, error));
        return;
    }

    // Reuse the normal export path: onExportComplete bakes crop + rotate and
    // writes the destination passed in here.  PNG keeps the rendered output
    // bit-exact for the SSIM check that test_golden does at runtime.
    PendingExport pending;
    if (auto *cs = m_effects->activeCropSource()) pending.crop = CropSnapshot{cs->userCropRect(), cs->userCropAngle()};
    const uint64_t requestId =
        m_processor->exportImageAsync(m_originalImage, *m_effects, QDir(dir).filePath("expected.png"));
    m_pendingExports.insert(requestId, std::move(pending));
}

void PhotoEditorApp::purgeCaches() {
    QString folder;
    if (!m_currentImagePath.isEmpty()) folder = QFileInfo(m_currentImagePath).absolutePath();
    else folder = m_currentFolder;
    if (folder.isEmpty()) {
        QMessageBox::information(this, "Purge Photo Caches", "Open a photo folder first.");
        return;
    }

    const auto answer = QMessageBox::question(
        this, "Purge Photo Caches",
        QString("Remove generated thumbnails and rendered proof JPEGs from:\n%1\n\n"
                "Source photos, edits, history, marks, and application settings will not be changed.")
            .arg(folder));
    if (answer != QMessageBox::Yes) return;

    const CachePurger::Result result = CachePurger::purgePhotoCaches(folder);
    if (!result.success) {
        QMessageBox::warning(this, "Purge Failed", result.error);
        return;
    }

    if (m_proofCache) m_proofCache->clear();
    if (m_proofer) m_proofer->clear();

    QStringList thumbnailsToRegenerate;
    for (const QString &path : m_currentPaths) {
        if (QFileInfo(path).absolutePath() == folder) {
            m_gridView->setProofStatus(path, GridView::ProofStatus::NotProofed);
            thumbnailsToRegenerate.append(path);
        }
    }
    if (!thumbnailsToRegenerate.isEmpty()) scheduleThumbnails(thumbnailsToRegenerate, folder);

    if (!m_loupePath.isEmpty() && QFileInfo(m_loupePath).absolutePath() == folder) {
        m_loupeView->setProofImage({});
        if (m_stack->currentWidget() == m_loupeView && m_proofer) {
            m_loupeView->setProofingState(true);
            m_proofer->promote(m_loupePath);
        } else {
            m_loupeView->setProofingState(false);
        }
    }

    QMessageBox::information(
        this, "Caches Purged",
        QString("Removed %1 generated cache file(s). Thumbnail regeneration is running in the background.")
            .arg(result.filesRemoved));
}

void PhotoEditorApp::rebuildEditedPreviews() {
    if (!m_proofer || !m_proofCache) return;

    // Include the current Develop photo even when it was opened directly
    // rather than through Gallery, and persist its latest history first.
    flushHistorySidecar();
    QStringList candidates = m_currentPaths;
    if (!m_developedPath.isEmpty() && !candidates.contains(m_developedPath)) candidates.append(m_developedPath);

    QStringList editedPaths;
    for (const QString &path : candidates) {
        const QString historyPath = historySidecarPathFor(path);
        if (!QFile::exists(historyPath)) continue;

        HistorySerializer::HistoryData data;
        QString                        error;
        if (!HistorySerializer::readYaml(historyPath, &data, &error)) {
            qWarning() << "History parse failed for" << historyPath << ":" << error;
            continue;
        }
        if (data.cursor <= 0) continue;

        m_proofCache->invalidate(path);
        m_gridView->setProofStatus(path, GridView::ProofStatus::NotProofed);
        editedPaths.append(path);
        if (m_loupePath == path) {
            m_loupeView->setProofImage({});
            m_loupeView->setProofingState(true);
        }
    }

    m_proofer->setQueue(editedPaths);
    // This is an explicit developer command, so run even while Develop would
    // normally keep automatic background proofing paused.
    m_proofer->resume();
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

    const QString fileName =
        QFileDialog::getSaveFileName(this, "Export Settings", suggested, "YAML (*.yml *.yaml);;All Files (*)");
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
static QImage applyCropAndRotate(const QImage &image, const QRectF &cropN, float angle) {
    if (image.isNull()) return image;

    const double cx = cropN.center().x() * image.width();
    const double cy = cropN.center().y() * image.height();
    const QSize  dstSize(static_cast<int>(std::round(cropN.width() * image.width())),
                         static_cast<int>(std::round(cropN.height() * image.height())));
    if (dstSize.isEmpty()) return image;

    // Map source→dst: translate crop centre to origin, rotate by -angle (Qt
    // rotates CW by default; our angle convention is CCW-positive), translate
    // out to the centre of the destination canvas.
    QTransform t;
    t.translate(dstSize.width() * 0.5, dstSize.height() * 0.5);
    t.rotate(-static_cast<double>(angle));
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

void PhotoEditorApp::onExportComplete(uint64_t requestId, QImage result, QString destinationPath) {
    const PendingExport pending = m_pendingExports.take(requestId);
    if (destinationPath.isEmpty()) return;
    const QString path = destinationPath;

    if (!result.isNull()) {
        if (pending.crop) result = applyCropAndRotate(result, pending.crop->rect, pending.crop->angle);
        if (pending.options) result = ExportResize::apply(result, pending.options->resize);
    }

    // chooseDestination() resolves opts.subfolder into the path but never
    // touches the filesystem.  Create the parent dir here so a token-driven
    // subfolder (e.g. {date}) materialises on first export.
    QDir().mkpath(QFileInfo(path).absolutePath());

    // With opts: explicit format hint + quality (saveImage path).
    // Without: legacy QImage::save behaviour, used by saveTestCase().
    const bool ok = !result.isNull() &&
                    (pending.options ? result.save(path, ExportOptions::qImageFormatHint(pending.options->format),
                                                   ExportOptions::qualityFor(*pending.options))
                                     : result.save(path));

    if (!ok) {
        QMessageBox::warning(this, "Save Failed",
                             QString("Could not save image to:\n%1\n\n"
                                     "Check that the directory is writable and you have sufficient disk space.")
                                 .arg(path));
    }
}

void PhotoEditorApp::onParametersChanged() {
    syncCommittedGeometry();
    syncViewportRotation();
    triggerReprocess();
    writeSidecar();
    m_history->recordFromCurrent(currentSnapshot());
    refreshEditedState();
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
    if (auto *cs = m_effects->cropSource()) {
        const QRectF c = cs->userCropRect();
        m_viewport->setImageRotation(cs->userCropAngle(), c.center());
    }
}

void PhotoEditorApp::syncCommittedGeometry() {
    auto *cs = m_effects->cropSource();
    if (!cs || m_loadedImage.isNull()) return;
    const QString state = cs->committedGeometryState();
    if (state == m_committedGeometryState) return;
    m_committedGeometryState = state;
    m_originalImage          = cs->applyCommittedGeometry(m_loadedImage);
    cs->setSourceImageSize(m_originalImage.size());
    m_viewport->setImageSize(m_originalImage.size());
    m_viewport->resetView();
}

void PhotoEditorApp::triggerReprocess() {
    if (m_originalImage.isNull()) return;

    m_processor->processImageAsync(m_originalImage, *m_effects, m_viewport->viewportRequest(), RunMode::Commit);
}

void PhotoEditorApp::triggerLiveReprocess() {
    if (m_originalImage.isNull()) return;

    m_processor->processImageAsync(m_originalImage, *m_effects, m_viewport->viewportRequest(), RunMode::LiveDrag);
}

void PhotoEditorApp::triggerViewportUpdate() {
    if (m_originalImage.isNull()) return;

    // Leading/trailing throttle — dispatch at most once per display frame so
    // rapid mouseMove events (1000Hz gaming mice, trackpads) don't saturate
    // the pipeline.  Use the active screen's refresh rate so 144Hz/240Hz
    // panels get smoother feedback than the old hard-coded 16ms (60Hz).
    // Zoom events go through the same path but are naturally rare (one wheel
    // tick = one event), so they aren't affected.
    const QScreen *s          = screen();
    const double   hz         = (s && s->refreshRate() > 0.0) ? s->refreshRate() : 60.0;
    const int      intervalMs = std::max(1, static_cast<int>(1000.0 / hz));
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

    m_processor->processImageAsync(m_originalImage, *m_effects, m_viewport->viewportRequest(), RunMode::PanZoom);
}

bool PhotoEditorApp::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::KeyPress || event->type() == QEvent::KeyRelease) {
        auto *ke = static_cast<QKeyEvent *>(event);
        if (ke->key() == Qt::Key_Backslash && !ke->isAutoRepeat()) {
            // Don't steal the key from text input fields (path bars, etc.).
            QWidget *fw = QApplication::focusWidget();
            if (fw && (fw->inherits("QLineEdit") || fw->inherits("QAbstractSpinBox") || fw->inherits("QTextEdit") ||
                       fw->inherits("QPlainTextEdit"))) {
                return QMainWindow::eventFilter(obj, event);
            }
            if (event->type() == QEvent::KeyPress && !m_beforeViewActive) {
                enterBeforeView();
                return true;
            }
            if (event->type() == QEvent::KeyRelease && m_beforeViewActive) {
                exitBeforeView();
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

void PhotoEditorApp::enterBeforeView() {
    m_beforeViewActive = true;
    if (m_stack && m_stack->currentWidget() == m_loupeView) {
        m_loupeView->setShowBefore(true);
    } else if (!m_originalImage.isNull()) {
        m_processor->processImageAsync(m_originalImage, *m_effects, m_viewport->viewportRequest(), RunMode::Commit,
                                       /*bypassEffects=*/true);
    }
}

void PhotoEditorApp::exitBeforeView() {
    m_beforeViewActive = false;
    if (m_stack && m_stack->currentWidget() == m_loupeView) {
        m_loupeView->setShowBefore(false);
    } else {
        triggerReprocess();
    }
}

void PhotoEditorApp::onProcessingStarted() {
    m_processingLabel->setVisible(true);
}

void PhotoEditorApp::onProcessingComplete(QImage result, QPoint offset) {
    m_processingLabel->setVisible(false);
    if (result.isNull()) {
        m_viewport->update();
    } else {
        m_viewport->setImage(result, offset);
        if (!m_beforeViewActive && !m_developedPath.isEmpty()) {
            m_latestDevelopPreview     = result;
            m_latestDevelopPreviewPath = m_developedPath;
        }
    }
}

void PhotoEditorApp::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    // Debounce: avoid firing a full GPU reprocess on every pixel of a window drag.
    m_resizeDebounce->start();
}

void PhotoEditorApp::closeEvent(QCloseEvent *event) {
    QSettings settings("Afterglow", "Afterglow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("lastDir", m_lastDir);
    flushHistorySidecar();
    persistLatestDevelopPreview();
    QMainWindow::closeEvent(event);
}

// ─── Gallery / Loupe / Develop mode switching ───────────────────────────────

void PhotoEditorApp::setMode(Mode m) {
    const bool leavingDevelop = m_stack->currentIndex() == static_cast<int>(Mode::Develop) && m != Mode::Develop;
    if (leavingDevelop) {
        flushHistorySidecar();
        if (!m_latestDevelopPreview.isNull() && m_latestDevelopPreviewPath == m_currentImagePath) {
            // Give Gallery/Loupe the exact render the user was just viewing.
            // The background proofer will replace this viewport-sized image
            // with its full-frame render when it finishes.
            persistLatestDevelopPreview();
            m_gridView->setThumbnail(m_currentImagePath, m_latestDevelopPreview);
            if (m == Mode::Loupe) {
                m_loupeView->setProofImage(m_latestDevelopPreview);
                m_loupeView->setProofingState(m_proofer != nullptr);
            }
        }
    }
    if (m == Mode::Gallery && !m_gridView->setCurrentPath(m_currentImagePath))
        m_currentImagePath = m_gridView->currentPath();
    m_stack->setCurrentIndex(static_cast<int>(m));
    for (QAction *a : m_modeGroup->actions()) {
        if (a->data().toInt() == static_cast<int>(m)) {
            a->setChecked(true);
            break;
        }
    }
    if (m_proofer) {
        if (m == Mode::Develop) m_proofer->pause();
        else m_proofer->resume();
    }
}

void PhotoEditorApp::openFolder() {
    const QString folder = QFileDialog::getExistingDirectory(this, "Open Folder", m_lastDir, QFileDialog::ShowDirsOnly);
    if (folder.isEmpty()) return;
    m_lastDir = folder;
    loadFolderIntoGrid(folder);
    setMode(Mode::Gallery);
}

// Per-folder JPEG cache for grid thumbnails. The first folder-open decodes
// each RAW's embedded preview (or full QImage for non-RAW) and writes a
// quality-85 JPEG here; subsequent opens read straight from disk if the
// source file's mtime hasn't moved past the cache file's mtime.
static QString thumbCachePath(const QString &sourcePath) {
    const QFileInfo fi(sourcePath);
    return fi.absoluteDir().filePath(".afterglow-thumbs/" + fi.fileName() + ".jpg");
}

static QImage tryLoadCachedThumb(const QString &sourcePath) {
    const QFileInfo cacheFi(thumbCachePath(sourcePath));
    if (!cacheFi.exists()) return {};
    const QFileInfo srcFi(sourcePath);
    // Stale cache: source has been re-saved since we last decoded.
    if (srcFi.lastModified() > cacheFi.lastModified()) return {};
    return QImage(cacheFi.absoluteFilePath());
}

static void writeCachedThumb(const QString &sourcePath, const QImage &thumb) {
    const QString out = thumbCachePath(sourcePath);
    QDir().mkpath(QFileInfo(out).absolutePath());
    thumb.save(out, "JPEG", 85);
}

// Recognised image extensions: same set the single-file dialog accepts. Kept
// here as a static QStringList so the lookup is amortised across all photos.
static const QStringList &imageExtensions() {
    static const QStringList exts = {
        "png", "jpg", "jpeg", "bmp", "tiff", "tif", "cr2", "cr3", "nef", "nrw", "arw", "sr2", "srf",
        "dng", "raf", "orf",  "rw2", "pef",  "srw", "x3f", "rwl", "mrw", "3fr", "kdc", "dcr", "erf",
    };
    return exts;
}

void PhotoEditorApp::loadFolderIntoGrid(const QString &folder) {
    QStringList  allPaths;
    QDirIterator it(folder, QDir::Files | QDir::Readable, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
        const QString p = it.next();
        if (imageExtensions().contains(QFileInfo(p).suffix().toLower())) allPaths.append(p);
    }

    // Cameras shoot RAW + JPEG side-by-side; the JPEG is just the in-camera
    // preview of the RAW so we'd be triaging two views of the same photo.
    // Drop the JPEG sibling whenever a RAW with the same basename exists.
    QSet<QString> rawBases;
    for (const QString &p : allPaths) {
        if (RawLoader::isRawFile(p)) rawBases.insert(QFileInfo(p).completeBaseName());
    }
    QStringList paths;
    for (const QString &p : allPaths) {
        const QFileInfo fi(p);
        if (!RawLoader::isRawFile(p) && rawBases.contains(fi.completeBaseName())) continue;
        paths.append(p);
    }
    paths.sort(Qt::CaseInsensitive);

    m_currentFolder = folder;
    m_currentPaths  = paths;
    m_gridView->setPhotos(paths);
    readCatalog(folder);

    for (const QString &path : paths) {
        const QString historyPath = historySidecarPathFor(path);
        if (!QFile::exists(historyPath)) continue;
        HistorySerializer::HistoryData data;
        QString                        error;
        if (HistorySerializer::readYaml(historyPath, &data, &error)) m_gridView->setEdited(path, data.cursor > 0);
    }

    if (m_proofCache && m_proofer) {
        m_proofCache->clear();
        m_proofer->clear();

        QStringList unproofed;
        for (const QString &path : paths) {
            if (m_proofCache->isProofed(path)) {
                m_gridView->setProofStatus(path, GridView::ProofStatus::Proofed);
                const QImage proof = m_proofCache->proof(path);
                if (!proof.isNull()) m_gridView->setThumbnail(path, proof);
            } else {
                unproofed.append(path);
            }
        }
        m_proofer->setQueue(unproofed);
    }

    scheduleThumbnails(paths, folder);
}

void PhotoEditorApp::scheduleThumbnails(const QStringList &paths, const QString &folder) {
    const uint64_t thumbnailGeneration = ++(*m_thumbnailGeneration);
    m_thumbnailPool->clear();

    // Decode thumbnails on a bounded, dedicated pool. A new batch clears
    // queued work; generation checks cheaply cancel tasks already running.
    QPointer<PhotoEditorApp> self(this);
    const QString            tag        = folder;
    const auto               generation = m_thumbnailGeneration;
    for (const QString &path : paths) {
        m_thumbnailPool->start([self, path, tag, generation, thumbnailGeneration]() {
            if (generation->load(std::memory_order_relaxed) != thumbnailGeneration) return;
            QImage thumb = tryLoadCachedThumb(path);
            if (thumb.isNull()) {
                if (RawLoader::isRawFile(path)) thumb = RawLoader::loadThumbnail(path);
                else thumb = decodeThumbnailOriented(path);
                if (thumb.isNull()) return;
                // Cap the side at 512px — saves memory when the grid is showing
                // hundreds of thumbnails and avoids holding full-res JPEGs alive.
                if (thumb.width() > 512 || thumb.height() > 512)
                    thumb = thumb.scaled(512, 512, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                if (generation->load(std::memory_order_relaxed) != thumbnailGeneration) return;
                writeCachedThumb(path, thumb);
            }
            QMetaObject::invokeMethod(
                qApp,
                [self, path, thumb, tag, thumbnailGeneration]() {
                    if (!self) return;
                    if (self->m_currentFolder != tag) return;
                    if (self->m_thumbnailGeneration->load(std::memory_order_relaxed) != thumbnailGeneration) return;
                    QImage displayed = thumb;
                    if (self->m_proofCache) {
                        const QImage proof = self->m_proofCache->proof(path);
                        if (!proof.isNull()) displayed = proof;
                    }
                    self->m_gridView->setThumbnail(path, displayed);
                },
                Qt::QueuedConnection);
        });
    }
}

void PhotoEditorApp::onPhotoActivated(const QString &path) {
    loadLoupeImage(path);
}

void PhotoEditorApp::loadLoupeImage(const QString &path) {
    if (path.isEmpty()) return;
    const uint64_t generation = ++m_loupeLoadGeneration;
    m_currentImagePath        = path;
    m_loupePath               = path;
    m_gridView->setCurrentPath(path);
    m_loupeView->beginPhoto(m_gridView->thumbnail(path));
    m_loupeView->setMetadata({});
    m_loupeView->setCurrentMark(m_gridView->mark(path));

    if (m_proofCache) {
        const QImage proof = m_proofCache->proof(path);
        if (!proof.isNull()) {
            m_loupeView->setProofImage(proof);
            m_loupeView->setProofingState(false);
        } else {
            m_loupeView->setProofImage({});
            m_loupeView->setProofingState(true);
            if (m_proofer) m_proofer->promote(path);
        }
    }

    setMode(Mode::Loupe);

    auto *watcher = new QFutureWatcher<LoupeLoadResult>(this);
    connect(watcher, &QFutureWatcher<LoupeLoadResult>::finished, this, [this, watcher, path, generation]() {
        const LoupeLoadResult loaded = watcher->result();
        if (generation == m_loupeLoadGeneration && path == m_loupePath) {
            if (!loaded.cameraJpeg.isNull()) m_loupeView->setCameraJpegImage(loaded.cameraJpeg);
            else qWarning() << "No preview available for" << path;
            m_loupeView->setMetadata(loaded.metadata);
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([path]() {
        LoupeLoadResult loaded;
        if (RawLoader::isRawFile(path)) loaded.cameraJpeg = RawLoader::loadThumbnail(path, &loaded.metadata);
        if (loaded.cameraJpeg.isNull()) loaded.cameraJpeg = decodeOriented(path);
        return loaded;
    }));

    // Demosaicing can be much slower than extracting the embedded JPEG, so
    // keep it on a separate future: Camera JPEG remains responsive while the
    // Original RAW choice becomes available when its decode completes.
    if (RawLoader::isRawFile(path)) {
        auto *rawWatcher = new QFutureWatcher<QImage>(this);
        connect(rawWatcher, &QFutureWatcher<QImage>::finished, this, [this, rawWatcher, path, generation]() {
            if (generation == m_loupeLoadGeneration && path == m_loupePath)
                m_loupeView->setOriginalRawImage(rawWatcher->result());
            rawWatcher->deleteLater();
        });
        rawWatcher->setFuture(QtConcurrent::run([path]() { return linearRawToDisplay(RawLoader::load(path)); }));
    }
}

void PhotoEditorApp::onDevelopRequested() {
    if (m_currentImagePath.isEmpty()) return;
    setMode(Mode::Develop);
    if (m_currentImagePath != m_developedPath) loadFullImage(m_currentImagePath);
}

void PhotoEditorApp::onLoupeNavigate(int direction) {
    if (m_currentPaths.isEmpty() || m_currentImagePath.isEmpty()) return;
    const int idx  = static_cast<int>(m_currentPaths.indexOf(m_currentImagePath));
    const int next = idx + direction;
    if (idx < 0 || next < 0 || next >= m_currentPaths.size()) return;
    onPhotoActivated(m_currentPaths[next]);
}

void PhotoEditorApp::onMarkChanged(const QString &path, GridView::Mark mark) {
    m_gridView->setMark(path, mark);
    writeCatalog();
}

// ─── Per-image sidecar (.yml) ───────────────────────────────────────────────

QString PhotoEditorApp::sidecarPathFor(const QString &imagePath) const {
    const QFileInfo fi(imagePath);
    return fi.absoluteDir().filePath(fi.completeBaseName() + ".yml");
}

void PhotoEditorApp::refreshHistoryTray() {
    if (!m_historyTray) return;
    const auto               &entries = m_history->entries();
    QVector<HistoryTray::Row> rows;
    rows.reserve(entries.size());
    for (const auto &e : entries) {
        QString effectName = e.effectId;
        for (const auto &eff : m_effects->entries()) {
            if (eff.effect && eff.effect->getId() == e.effectId) {
                effectName = eff.effect->getName();
                break;
            }
        }
        rows.append({entryLabel(e, effectName)});
    }
    m_historyTray->setHistory(rows, m_history->cursor());
}

QVector<SettingsImporter::EffectSettings> PhotoEditorApp::currentSnapshot() const {
    const auto                               &entries = m_effects->entries();
    QVector<SettingsImporter::EffectSettings> snap;
    snap.reserve(entries.size());
    for (const auto &e : entries) {
        if (!e.effect) continue;
        SettingsImporter::EffectSettings es;
        es.id         = e.effect->getId();
        es.name       = e.effect->getName();
        es.enabled    = e.enabled;
        es.parameters = e.effect->getParameters();
        snap.append(es);
    }
    return snap;
}

QString PhotoEditorApp::historySidecarPathFor(const QString &imagePath) const {
    const QFileInfo fi(imagePath);
    return fi.absoluteDir().filePath(fi.completeBaseName() + ".history.yml");
}

void PhotoEditorApp::flushHistorySidecar() {
    if (m_developedPath.isEmpty() || m_history->entries().isEmpty()) return;
    const QString path = historySidecarPathFor(m_developedPath);
    QString       error;
    if (!HistorySerializer::writeYaml(path, m_history->entries(), m_history->cursor(), currentSnapshot(), &error))
        qWarning() << "History sidecar write failed for" << path << ":" << error;
}

void PhotoEditorApp::applyHistoryEntry(const UndoHistory::Entry &e, bool applyFrom) {
    const auto &entries = m_effects->entries();
    for (int i = 0; i < entries.size(); ++i) {
        if (!entries[i].effect || entries[i].effect->getId() != e.effectId) continue;

        PhotoEditorEffect *effect = entries[i].effect;

        if (e.enabled) {
            const bool val = applyFrom ? e.enabled->first : e.enabled->second;
            m_effects->setEnabled(i, val);
            if (i < m_effectMenuActions.size()) {
                QSignalBlocker block(m_effectMenuActions[i]);
                m_effectMenuActions[i]->setChecked(val);
            }
        }

        if (!e.params.isEmpty()) {
            auto params = effect->getParameters();
            for (auto pit = e.params.cbegin(); pit != e.params.cend(); ++pit) {
                const QVariant &val = applyFrom ? pit.value().from : pit.value().to;
                if (val.isValid()) params.insert(pit.key(), val);
                else params.remove(pit.key());
            }
            QSignalBlocker block(effect);
            effect->applyParameters(params);
        }
        break;
    }
}

void PhotoEditorApp::snapshotDefaults() {
    m_defaults.image.clear();
    m_defaults.effects.clear();
    const auto &entries = m_effects->entries();
    m_defaults.effects.reserve(entries.size());
    for (const auto &e : entries) {
        SettingsImporter::EffectSettings es;
        es.id         = e.effect->getId();
        es.name       = e.effect->getName();
        es.enabled    = e.enabled;
        es.parameters = e.effect->getParameters();
        m_defaults.effects.append(es);
    }
}

void PhotoEditorApp::writeSidecar() {
    if (m_currentImagePath.isEmpty()) return;
    const QString path = sidecarPathFor(m_currentImagePath);
    QString       error;
    if (!SettingsExporter::writeYaml(path, *m_effects, m_currentImagePath, &error))
        qWarning() << "Sidecar write failed for" << path << ":" << error;

    // Invalidate the proof for this photo: the pipeline output has changed.
    // Any in-flight result is versioned out, and a replacement is queued for
    // when the proofer resumes after leaving Develop.
    if (m_proofCache) {
        m_proofCache->invalidate(m_currentImagePath);
        m_gridView->setProofStatus(m_currentImagePath, GridView::ProofStatus::NotProofed);
        if (m_loupePath == m_currentImagePath) {
            m_loupeView->setProofImage({});
            m_loupeView->setProofingState(true);
        }
        if (m_proofer) m_proofer->refresh(m_currentImagePath);
    }
}

void PhotoEditorApp::refreshEditedState() {
    if (m_currentImagePath.isEmpty() || !m_gridView) return;
    m_gridView->setEdited(m_currentImagePath, m_history->cursor() > 0);
}

void PhotoEditorApp::persistLatestDevelopPreview() {
    if (!m_proofCache || m_latestDevelopPreview.isNull() || m_latestDevelopPreviewPath.isEmpty()) return;
    m_proofCache->store(m_latestDevelopPreviewPath, m_latestDevelopPreview);
}

// ─── Per-folder catalog (triage marks) ──────────────────────────────────────
//
// Stored as a flat JSON object next to the photos: <folder>/.afterglow-catalog.json
// Keys are basenames (so the file survives the folder being moved); values
// are single-character mark codes ('P', 'X', 'U').

QString PhotoEditorApp::catalogPath(const QString &folder) const {
    return QDir(folder).filePath(".afterglow-catalog.json");
}

void PhotoEditorApp::readCatalog(const QString &folder) {
    QFile f(catalogPath(folder));
    if (!f.open(QIODevice::ReadOnly)) return;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject()) return;
    const QJsonObject obj = doc.object();
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const QString fullPath = QDir(folder).filePath(it.key());
        const QString s        = it.value().toString();
        if (s.isEmpty()) continue;
        // Only accept the current code set — a catalog written by an older
        // build (with 'P'/'X'/'U') simply loses its marks rather than
        // populating the map with bogus enum values.
        const char c = s.at(0).toLatin1();
        if (c == 'A' || c == 'R' || c == 'D') m_gridView->setMark(fullPath, static_cast<GridView::Mark>(c));
    }
}

void PhotoEditorApp::writeCatalog() const {
    if (m_currentFolder.isEmpty()) return;
    QJsonObject  obj;
    QDirIterator it(m_currentFolder, QDir::Files, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
        const QString p = it.next();
        if (!imageExtensions().contains(QFileInfo(p).suffix().toLower())) continue;
        const auto m = m_gridView->mark(p);
        if (m == GridView::Mark::None) continue; // unflagged is the default
        obj.insert(QFileInfo(p).fileName(), QString(QChar(static_cast<char>(m))));
    }
    QFile f(catalogPath(m_currentFolder));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
}
