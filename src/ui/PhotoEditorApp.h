#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include <QRectF>
#include <QTimer>
#include <QElapsedTimer>
#include <QHash>
#include <memory>
#include <cstdint>
#include <atomic>
#include <optional>
#include "EffectManager.h"
#include "EditorUiState.h"
#include "ExportOptions.h"
#include "GridView.h"
#include "ImageProcessor.h"
#include "ProofCache.h"
#include "Proofer.h"
#include "SettingsImporter.h"
#include "HistoryTray.h"
#include "MetadataTray.h"
#include "LocalAdjustment.h"
#include "UndoHistory.h"
#include "ViewportWidget.h"

class QVBoxLayout;
class QLabel;
class QStackedWidget;
class QActionGroup;
class LoupeView;
class QThreadPool;
class UiServices;
class PreferencesDialog;
class LinearGradientTool;
class ParamSlider;

class PhotoEditorApp : public QMainWindow {
    Q_OBJECT

public:
    explicit PhotoEditorApp(EffectManager *effectManager, QWidget *parent = nullptr);
    ~PhotoEditorApp() override;

    // Replaces native modal dialogs with a caller-owned service. Intended for
    // deterministic workflow tests; call before triggering any UI action.
    void setUiServices(UiServices *services);

    // Call once after construction with a dedicated EffectManager (separate
    // instances from the Develop pipeline) to enable proof cache generation.
    void initProofer(std::unique_ptr<EffectManager> prooferEffects);

    // Opens an image without showing the native picker. Used by command-line
    // launches and deterministic UI workflows.
    void openImagePath(const QString &path);

protected:
    void resizeEvent(QResizeEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void openImage();
    void openFolder();
    void saveImage();
    void exportSettings();
    void importSettings();
    void saveTestCase();
    void purgeCaches();
    void rebuildEditedPreviews();
    void onParametersChanged();
    void onLiveParametersChanged();
    void onProcessingComplete(QImage result, QPoint offset);
    void onProcessingStarted();
    void onExportComplete(uint64_t requestId, QImage result, QString destinationPath);
    void onPhotoActivated(const QString &path);
    void onDevelopRequested();
    void onLoupeNavigate(int direction);
    void onMarkChanged(const QString &path, GridView::Mark mark);
    void copyDevelopSettings();
    void pasteDevelopSettings();

private:
    using Mode = EditorUiState::Mode;

    void                                      setupUI();
    void                                      setupToolBar();
    void                                      setupMenuBar();
    void                                      setupEffectPanels(QVBoxLayout *rightLayout);
    void                                      reorderEffectPanels();
    void                                      showPreferences(int page = 0);
    void                                      saveEffectPreferences();
    void                                      updateDefaultEffectOrganization();
    void                                      commitLinearGradient();
    void                                      applyLocalAdjustments(const QVector<LocalAdjustment> &adjustments);
    SettingsImporter::Settings                currentSettings() const;
    QVector<SettingsImporter::EffectSettings> currentSnapshot() const;
    QString                                   historySidecarPathFor(const QString &imagePath) const;
    void                                      flushHistorySidecar();
    void                                      applyHistoryEntry(const UndoHistory::Entry &e, bool applyFrom);
    void                                      triggerReprocess(); // Commit: rebuild full-res post-effect cache
    void triggerLiveReprocess();                                  // LiveDrag: preview-sized pipeline, bypasses cache
    void triggerViewportUpdate();  // PanZoom: throttled entry; coalesces mouseMove bursts
    void dispatchViewportUpdate(); // actual PanZoom dispatch — fires from throttle
    void syncViewportRotation();   // push the user's crop angle/centre to the viewport
    void syncCommittedGeometry();  // rebuild working source after Apply/Undo/Redo

    void refreshHistoryTray();

    void setMode(Mode m);
    // Backslash-held "before edits" preview: bypass the pipeline in Develop
    // and fall back to the camera JPEG in Loupe; restore on key release.
    void    enterBeforeView();
    void    exitBeforeView();
    void    loadFolderIntoGrid(const QString &folder);
    void    scheduleThumbnails(const QStringList &paths, const QString &folder);
    void    loadFullImage(const QString &path);
    void    loadLoupeImage(const QString &path);
    QString catalogPath(const QString &folder) const;
    void    readCatalog(const QString &folder);
    void    writeCatalog() const;

    // Per-image edit state lives in <basename>.yml next to the source.  Always
    // present once an image has been opened: loadFullImage creates one filled
    // with defaults if missing, and onParametersChanged rewrites it on every
    // committed edit so the sidecar tracks the live editor state.
    QString                    sidecarPathFor(const QString &imagePath) const;
    void                       writeSidecar();
    void                       refreshEditedState();
    void                       persistLatestDevelopPreview();
    void                       snapshotDefaults();
    SettingsImporter::Settings settingsForPath(const QString &path) const;
    void                       copyDevelopSettingsFrom(const QString &path);
    void                       pasteDevelopSettingsTo(const QString &path);

    MetadataTray                         *m_metadataTray  = nullptr;
    HistoryTray                          *m_historyTray   = nullptr;
    UndoHistory                          *m_history       = nullptr;
    QAction                              *m_undoAct       = nullptr;
    QAction                              *m_redoAct       = nullptr;
    PreferencesDialog                    *m_preferences   = nullptr;
    QVBoxLayout                          *m_effectsLayout = nullptr;
    QHash<PhotoEditorEffect *, QWidget *> m_effectPanels;

    EffectManager              *m_effects;
    ImageProcessor             *m_processor;
    EditorUiState               m_uiState;
    std::unique_ptr<UiServices> m_ownedUiServices;
    UiServices                 *m_uiServices = nullptr;
    QImage                      m_originalImage;
    QImage                      m_loadedImage; // immutable decoded source used to rebuild applied geometry
    QString                     m_committedGeometryState;
    QImage                      m_latestDevelopPreview;
    QString                     m_currentImagePath;
    QString                     m_latestDevelopPreviewPath;
    QString                     m_lastDir;

    QStackedWidget      *m_stack              = nullptr;
    GridView            *m_gridView           = nullptr;
    LoupeView           *m_loupeView          = nullptr;
    ViewportWidget      *m_viewport           = nullptr;
    LinearGradientTool  *m_linearGradientTool = nullptr;
    LocalAdjustmentStack m_localAdjustments;
    ParamSlider         *m_localExposure   = nullptr;
    QActionGroup        *m_modeGroup       = nullptr;
    QLabel              *m_processingLabel = nullptr;
    QTimer              *m_resizeDebounce  = nullptr;
    QTimer              *m_panThrottle     = nullptr; // trailing edge of pan throttle
    QThreadPool         *m_thumbnailPool   = nullptr;
    QElapsedTimer        m_lastPanDispatch; // invalid until first dispatch

    QString                                m_currentFolder;
    QStringList                            m_currentPaths;  // photos shown in the gallery, in display order
    QString                                m_developedPath; // path currently loaded in m_originalImage
    QString                                m_loupePath;     // path currently represented by LoupeView
    uint64_t                               m_loupeLoadGeneration = 0;
    std::shared_ptr<std::atomic<uint64_t>> m_thumbnailGeneration = std::make_shared<std::atomic<uint64_t>>(0);

    ProofCache *m_proofCache = nullptr;
    Proofer    *m_proofer    = nullptr;

    // Constructor-time snapshot of every effect's enabled flag and parameter
    // map.  Reapplied at the start of every loadFullImage so edits from a
    // previously opened photo can't bleed onto the next one.
    SettingsImporter::Settings m_defaults;

    struct CropSnapshot {
        QRectF rect;
        float  angle = 0.0f;
    };
    struct PendingExport {
        std::optional<ExportOptions::Options> options;
        std::optional<CropSnapshot>           crop;
    };
    QHash<uint64_t, PendingExport> m_pendingExports;
};

#endif // PHOTOEDITORAPP_H
