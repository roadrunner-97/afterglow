#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include <QElapsedTimer>
#include <memory>
#include <optional>
#include "EffectManager.h"
#include "ExportOptions.h"
#include "GridView.h"
#include "ImageProcessor.h"
#include "ProofCache.h"
#include "Proofer.h"
#include "SettingsImporter.h"
#include "UndoHistory.h"
#include "ViewportWidget.h"

class QVBoxLayout;
class QComboBox;
class QLabel;
class QStackedWidget;
class QActionGroup;
class LoupeView;

class PhotoEditorApp : public QMainWindow {
    Q_OBJECT

public:
    explicit PhotoEditorApp(EffectManager* effectManager, QWidget* parent = nullptr);
    ~PhotoEditorApp() override;

    // Call once after construction with a dedicated EffectManager (separate
    // instances from the Develop pipeline) to enable proof cache generation.
    void initProofer(std::unique_ptr<EffectManager> prooferEffects);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void openImage();
    void openFolder();
    void saveImage();
    void exportSettings();
    void importSettings();
    void saveTestCase();
    void onParametersChanged();
    void onLiveParametersChanged();
    void onProcessingComplete(QImage result, QPoint offset);
    void onProcessingStarted();
    void onExportComplete(QImage result, QString destinationPath);
    void onPhotoActivated(const QString& path);
    void onDevelopRequested();
    void onLoupeNavigate(int direction);
    void onMarkChanged(const QString& path, GridView::Mark mark);

private:
    enum class Mode { Gallery = 0, Loupe = 1, Develop = 2 };

    void setupUI();
    void setupToolBar();
    void setupMenuBar();
    void setupGpuSelector(QVBoxLayout* rightLayout);
    void setupEffectPanels(QVBoxLayout* rightLayout);
    QVector<SettingsImporter::EffectSettings> currentSnapshot() const;
    QString historySidecarPathFor(const QString& imagePath) const;
    void flushHistorySidecar();
    void applyHistoryEntry(const UndoHistory::Entry& e, bool applyFrom);
    void triggerReprocess();        // Commit: rebuild full-res post-effect cache
    void triggerLiveReprocess();    // LiveDrag: preview-sized pipeline, bypasses cache
    void triggerViewportUpdate();   // PanZoom: throttled entry; coalesces mouseMove bursts
    void dispatchViewportUpdate();  // actual PanZoom dispatch — fires from throttle
    void syncViewportRotation();    // push the user's crop angle/centre to the viewport

    void setMode(Mode m);
    // Backslash-held "before edits" preview: bypass the pipeline in Develop
    // and fall back to the camera JPEG in Loupe; restore on key release.
    void enterBeforeView();
    void exitBeforeView();
    void loadFolderIntoGrid(const QString& folder);
    void loadFullImage(const QString& path);
    QString catalogPath(const QString& folder) const;
    void readCatalog(const QString& folder);
    void writeCatalog() const;

    // Per-image edit state lives in <basename>.yml next to the source.  Always
    // present once an image has been opened: loadFullImage creates one filled
    // with defaults if missing, and onParametersChanged rewrites it on every
    // committed edit so the sidecar tracks the live editor state.
    QString sidecarPathFor(const QString& imagePath) const;
    void writeSidecar();
    void snapshotDefaults();

    UndoHistory*    m_history          = nullptr;
    QAction*        m_undoAct          = nullptr;
    QAction*        m_redoAct          = nullptr;
    QVector<QAction*> m_effectMenuActions;

    EffectManager*  m_effects;
    ImageProcessor* m_processor;
    QImage          m_originalImage;
    QString         m_currentImagePath;
    QString         m_lastDir;

    QStackedWidget* m_stack            = nullptr;
    GridView*       m_gridView         = nullptr;
    LoupeView*      m_loupeView        = nullptr;
    ViewportWidget* m_viewport         = nullptr;
    QActionGroup*   m_modeGroup        = nullptr;
    QComboBox*      m_gpuSelector      = nullptr;
    QLabel*         m_processingLabel  = nullptr;
    QTimer*         m_resizeDebounce   = nullptr;
    QTimer*         m_panThrottle      = nullptr;  // trailing edge of pan throttle
    QElapsedTimer   m_lastPanDispatch;              // invalid until first dispatch

    QString         m_currentFolder;
    QStringList     m_currentPaths;   // photos shown in the gallery, in display order
    QString         m_developedPath;  // path currently loaded in m_originalImage

    ProofCache*     m_proofCache = nullptr;
    Proofer*        m_proofer    = nullptr;

    // Constructor-time snapshot of every effect's enabled flag and parameter
    // map.  Reapplied at the start of every loadFullImage so edits from a
    // previously opened photo can't bleed onto the next one.
    SettingsImporter::Settings m_defaults;

    // Set by saveImage() before kicking off the async export, consumed (and
    // cleared) by onExportComplete().  std::nullopt means "no options" — that
    // path is reserved for saveTestCase(), which writes a fixed PNG and wants
    // the default QImage::save() behaviour.
    std::optional<ExportOptions::Options> m_pendingExportOpts;

    bool            m_beforeViewActive = false;
};

#endif // PHOTOEDITORAPP_H
