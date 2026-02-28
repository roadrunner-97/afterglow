#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include "EffectManager.h"
#include "ImageProcessor.h"

class QLabel;
class QScrollArea;
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
    void triggerReprocess();
    void displayImage(const QImage& image);

    EffectManager*  m_effects;
    ImageProcessor* m_processor;
    QImage          m_originalImage;
    QString         m_lastDir;
    bool            m_liveUpdate = false;

    QLabel*      m_imageLabel;
    QScrollArea* m_scrollArea;
    QComboBox*   m_gpuSelector;
};

#endif // PHOTOEDITORAPP_H
