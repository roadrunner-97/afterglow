#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include <QElapsedTimer>
#include "EffectManager.h"
#include "ImageProcessor.h"
#include "ViewportWidget.h"

class QVBoxLayout;
class QComboBox;
class QLabel;

class PhotoEditorApp : public QMainWindow {
    Q_OBJECT

public:
    explicit PhotoEditorApp(EffectManager* effectManager, QWidget* parent = nullptr);
    ~PhotoEditorApp() override;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private slots:
    void openImage();
    void saveImage();
    void exportSettings();
    void importSettings();
    void saveTestCase();
    void onParametersChanged();
    void onLiveParametersChanged();
    void onProcessingComplete(QImage result);
    void onProcessingStarted();
    void onExportComplete(QImage result, QString destinationPath);

private:
    void setupUI();
    void setupToolBar();
    void setupMenuBar();
    void setupGpuSelector(QVBoxLayout* rightLayout);
    void setupEffectPanels(QVBoxLayout* rightLayout);
    void triggerReprocess();        // Commit: rebuild full-res post-effect cache
    void triggerLiveReprocess();    // LiveDrag: preview-sized pipeline, bypasses cache
    void triggerViewportUpdate();   // PanZoom: throttled entry; coalesces mouseMove bursts
    void dispatchViewportUpdate();  // actual PanZoom dispatch — fires from throttle
    void syncViewportRotation();    // push the user's crop angle/centre to the viewport

    EffectManager*  m_effects;
    ImageProcessor* m_processor;
    QImage          m_originalImage;
    QString         m_currentImagePath;
    QString         m_lastDir;
    bool            m_liveUpdate = false;

    ViewportWidget* m_viewport         = nullptr;
    QComboBox*      m_gpuSelector      = nullptr;
    QLabel*         m_processingLabel  = nullptr;
    QTimer*         m_resizeDebounce   = nullptr;
    QTimer*         m_panThrottle      = nullptr;  // trailing edge of pan throttle
    QElapsedTimer   m_lastPanDispatch;              // invalid until first dispatch
};

#endif // PHOTOEDITORAPP_H
