#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include "EffectManager.h"
#include "ImageProcessor.h"
#include "ViewportWidget.h"

class QVBoxLayout;
class QComboBox;

class PhotoEditorApp : public QMainWindow {
    Q_OBJECT

public:
    explicit PhotoEditorApp(EffectManager* effectManager, QWidget* parent = nullptr);
    ~PhotoEditorApp() override = default;

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void openImage();
    void saveImage();
    void onParametersChanged();
    void onLiveParametersChanged();
    void onProcessingComplete(QImage result);

private:
    void setupUI();
    void setupToolBar();
    void setupMenuBar();
    void setupGpuSelector(QVBoxLayout* rightLayout);
    void setupEffectPanels(QVBoxLayout* rightLayout);
    void triggerReprocess();       // full pipeline run (effects + downsample)
    void triggerViewportUpdate();  // viewport-only run (downsample only, no effects)

    EffectManager*  m_effects;
    ImageProcessor* m_processor;
    QImage          m_originalImage;
    QString         m_lastDir;
    bool            m_liveUpdate = false;

    ViewportWidget* m_viewport        = nullptr;
    QComboBox*      m_gpuSelector     = nullptr;
    // True from the moment triggerReprocess() is called until the resulting
    // processingComplete() delivers a non-null image.  While set, viewport
    // changes also trigger full reruns so pan/zoom never shows a stale frame.
    bool            m_pendingFullRun  = false;
};

#endif // PHOTOEDITORAPP_H
