#ifndef PHOTOEDITORAPP_H
#define PHOTOEDITORAPP_H

#include <QMainWindow>
#include <QImage>
#include <memory>
#include "PluginManager.h"

class QLabel;
class QScrollArea;
class QVBoxLayout;
class QHBoxLayout;
class PhotoEditorPlugin;

/**
 * @brief Main application window for the photo editor
 */
class PhotoEditorApp : public QMainWindow {
    Q_OBJECT

public:
    PhotoEditorApp(QWidget *parent = nullptr);
    ~PhotoEditorApp();

private slots:
    void openImage();
    void saveImage();
    void onPluginParametersChanged();
    void updateImagePreview();

private:
    void setupUI();
    void setupMenuBar();
    void createCentralWidget();
    void loadPlugins();
    void displayImage(const QImage &image);
    void applyAllPlugins();
    QImage applyPluginStack(const QImage &sourceImage);

    // UI Components
    QLabel *imageLabel;
    QScrollArea *scrollArea;
    QVBoxLayout *pluginControlsLayout;
    QVBoxLayout *controlsLayout;

    // Data members
    QImage originalImage;
    QImage currentImage;
    std::unique_ptr<PluginManager> pluginManager;
    QString lastOpenedPath;
    
    // Store loaded plugins in order
    QVector<PhotoEditorPlugin*> loadedPlugins;
};

#endif // PHOTOEDITORAPP_H
