#include "PhotoEditorApp.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QDebug>
#include <QDir>
#include <QFrame>
#include <QElapsedTimer>
#include "../core/PhotoEditorPlugin.h"

PhotoEditorApp::PhotoEditorApp(QWidget *parent)
    : QMainWindow(parent), pluginManager(std::make_unique<PluginManager>()) {
    setupUI();
    loadPlugins();
    setWindowTitle("Lightroom Clone - Photo Editor");
    setGeometry(100, 100, 1400, 900);
}

PhotoEditorApp::~PhotoEditorApp() {
}

void PhotoEditorApp::setupUI() {
    setupMenuBar();
    createCentralWidget();
}

void PhotoEditorApp::setupMenuBar() {
    QMenu *fileMenu = menuBar()->addMenu("File");

    QAction *openAction = fileMenu->addAction("Open Image");
    connect(openAction, &QAction::triggered, this, &PhotoEditorApp::openImage);

    QAction *saveAction = fileMenu->addAction("Save Image");
    connect(saveAction, &QAction::triggered, this, &PhotoEditorApp::saveImage);

    fileMenu->addSeparator();

    QAction *exitAction = fileMenu->addAction("Exit");
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
}

void PhotoEditorApp::createCentralWidget() {
    QWidget *centralWidget = new QWidget();
    setCentralWidget(centralWidget);

    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

    // Left side: Image display (3/4 width)
    scrollArea = new QScrollArea();
    imageLabel = new QLabel();
    imageLabel->setPixmap(QPixmap(400, 300));
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setStyleSheet("background-color: #2a2a2a; color: white;");
    imageLabel->setText("Open an image to start editing");
    
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setWidget(imageLabel);
    scrollArea->setWidgetResizable(false);
    scrollArea->setAlignment(Qt::AlignCenter);
    scrollArea->setStyleSheet("background-color: #1e1e1e;");

    mainLayout->addWidget(scrollArea, 3);

    // Right side: Plugin controls vertical list (1/4 width)
    QWidget *controlPanel = new QWidget();
    controlsLayout = new QVBoxLayout(controlPanel);

    // Header
    controlsLayout->addWidget(new QLabel("<b>Plugin Controls:</b>"), 0);

    // Scrollable area for plugins
    QScrollArea *pluginScrollArea = new QScrollArea();
    pluginScrollArea->setWidgetResizable(true);
    pluginScrollArea->setStyleSheet("QScrollArea { background-color: #2a2a2a; }");
    
    QWidget *pluginContainer = new QWidget();
    pluginControlsLayout = new QVBoxLayout(pluginContainer);
    pluginControlsLayout->setSpacing(10);
    pluginControlsLayout->setContentsMargins(0, 0, 0, 0);
    
    pluginScrollArea->setWidget(pluginContainer);
    controlsLayout->addWidget(pluginScrollArea, 1);

    controlsLayout->addStretch();

    mainLayout->addWidget(controlPanel, 1);
}

void PhotoEditorApp::openImage() {
    QString fileName = QFileDialog::getOpenFileName(this,
        "Open Image", lastOpenedPath,
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)");

    if (!fileName.isEmpty()) {
        originalImage.load(fileName);
        if (!originalImage.isNull()) {
            lastOpenedPath = QFileInfo(fileName).absolutePath();
            currentImage = originalImage;
            applyAllPlugins();
        }
    }
}

void PhotoEditorApp::saveImage() {
    if (currentImage.isNull()) {
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,
        "Save Image", lastOpenedPath,
        "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp);;All Files (*)");

    if (!fileName.isEmpty()) {
        currentImage.save(fileName);
        lastOpenedPath = QFileInfo(fileName).absolutePath();
    }
}

void PhotoEditorApp::onPluginParametersChanged() {
    // When plugin parameters change, re-apply all plugins
    QElapsedTimer timer;
    timer.start();
    
    applyAllPlugins();
    
    qDebug() << "Image reprocessing took" << timer.nsecsElapsed() / 1000.0 << "µs";
}

void PhotoEditorApp::updateImagePreview() {
    displayImage(currentImage);
}

void PhotoEditorApp::loadPlugins() {
    // Load all plugins from the plugins directory
    // Try the build directory first, then fall back to ./plugins
    QString pluginPath = "./plugins";
    if (QDir("./build/plugins").exists()) {
        pluginPath = "./build/plugins";
    }
    pluginManager->loadPluginsFromDirectory(pluginPath);
    
    // Get all loaded plugins
    const auto &pluginsMap = pluginManager->getLoadedPlugins();
    
    int pluginCount = 0;
    for (auto it = pluginsMap.begin(); it != pluginsMap.end(); ++it) {
        PhotoEditorPlugin* plugin = it.value();
        loadedPlugins.append(plugin);
        
        // Add separator if not the first plugin
        if (pluginCount > 0) {
            QFrame *separator = new QFrame();
            separator->setFrameShape(QFrame::HLine);
            separator->setFrameShadow(QFrame::Sunken);
            separator->setStyleSheet("background-color: #555555; margin: 5px 0px;");
            separator->setMaximumHeight(1);
            pluginControlsLayout->addWidget(separator);
        }
        
        // Create a container for this plugin with title and collapse button
        QWidget *pluginWidget = new QWidget();
        pluginWidget->setStyleSheet("background-color: #333333; border-radius: 4px;");
        QVBoxLayout *pluginLayout = new QVBoxLayout(pluginWidget);
        pluginLayout->setContentsMargins(5, 5, 5, 5);
        pluginLayout->setSpacing(5);
        
        // Title bar with collapse button
        QWidget *titleBarWidget = new QWidget();
        QHBoxLayout *titleLayout = new QHBoxLayout(titleBarWidget);
        titleLayout->setContentsMargins(0, 0, 0, 0);
        
        QLabel *titleLabel = new QLabel(QString("<b>%1</b>").arg(plugin->getName()));
        titleLabel->setStyleSheet("color: #e0e0e0;");
        titleLayout->addWidget(titleLabel, 1);
        
        QPushButton *collapseBtn = new QPushButton("−");
        collapseBtn->setStyleSheet(
            "QPushButton {"
            "  background-color: #555555;"
            "  color: #ffffff;"
            "  border: none;"
            "  border-radius: 3px;"
            "  padding: 2px 6px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background-color: #777777;"
            "}"
        );
        collapseBtn->setMaximumWidth(30);
        titleLayout->addWidget(collapseBtn);
        
        pluginLayout->addWidget(titleBarWidget);
        
        // Create controls widget for this plugin
        QWidget* controlsWidget = plugin->createControlsWidget();
        
        if (controlsWidget) {
            pluginLayout->addWidget(controlsWidget);
        } else {
            // Empty space for plugins without controls
            QLabel *emptyLabel = new QLabel("<i>No parameters</i>");
            emptyLabel->setStyleSheet("color: #999999; font-size: 10pt;");
            pluginLayout->addWidget(emptyLabel);
        }
        
        // Store the controls widget and its visibility state
        struct PluginUIState {
            QWidget* controlsWidget;
            bool isExpanded;
        };
        
        PluginUIState* state = new PluginUIState{controlsWidget ? controlsWidget : nullptr, true};
        
        // Connect collapse button
        connect(collapseBtn, &QPushButton::clicked, [state, collapseBtn]() {
            if (state->isExpanded) {
                // Collapse
                if (state->controlsWidget) {
                    state->controlsWidget->hide();
                }
                collapseBtn->setText("+");
                state->isExpanded = false;
            } else {
                // Expand
                if (state->controlsWidget) {
                    state->controlsWidget->show();
                }
                collapseBtn->setText("−");
                state->isExpanded = true;
            }
        });
        
        pluginControlsLayout->addWidget(pluginWidget);
        
        // Connect parameter changes to preview update
        connect(plugin, &PhotoEditorPlugin::parametersChanged,
                this, &PhotoEditorApp::onPluginParametersChanged);
        
        qDebug() << "Loaded plugin:" << plugin->getName();
        pluginCount++;
    }
    
    // Add stretch at the bottom so plugins don't take up all space
    pluginControlsLayout->addStretch();
}

void PhotoEditorApp::displayImage(const QImage &image) {
    QPixmap pixmap = QPixmap::fromImage(image);
    
    // Get available space in scroll area
    int maxWidth = scrollArea->viewport()->width() - 10;
    int maxHeight = scrollArea->viewport()->height() - 10;
    
    // Ensure we have valid dimensions
    if (maxWidth <= 0) maxWidth = 800;
    if (maxHeight <= 0) maxHeight = 600;

    // Scale the image to fit within the viewport while keeping aspect ratio
    if (pixmap.width() > maxWidth || pixmap.height() > maxHeight) {
        pixmap = pixmap.scaled(maxWidth, maxHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    // Set the pixmap and size the label to fit
    imageLabel->setPixmap(pixmap);
    imageLabel->setFixedSize(pixmap.width(), pixmap.height());
}

void PhotoEditorApp::applyAllPlugins() {
    if (originalImage.isNull()) {
        return;
    }

    currentImage = applyPluginStack(originalImage);
    displayImage(currentImage);
}

QImage PhotoEditorApp::applyPluginStack(const QImage &sourceImage) {
    QImage result = sourceImage;
    
    // Apply each plugin in sequence
    for (PhotoEditorPlugin* plugin : loadedPlugins) {
        QMap<QString, QVariant> params = plugin->getParameters();
        result = plugin->processImage(result, params);
    }
    
    return result;
}
