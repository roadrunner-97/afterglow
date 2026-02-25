#ifndef PHOTOEDITORPLUGIN_H
#define PHOTOEDITORPLUGIN_H

#include <QString>
#include <QImage>
#include <QMap>
#include <QVariant>
#include <QObject>
#include <QWidget>
#include <memory>

class QWidget;

/**
 * @brief Abstract base class for all photo editing plugins
 * 
 * All plugins must inherit from this class and implement the required methods.
 * Plugins can be loaded dynamically from shared libraries.
 */
class PhotoEditorPlugin : public QObject {
    Q_OBJECT

public:
    virtual ~PhotoEditorPlugin() = default;

    /**
     * @brief Get the name of the plugin
     */
    virtual QString getName() const = 0;

    /**
     * @brief Get the description of the plugin
     */
    virtual QString getDescription() const = 0;

    /**
     * @brief Get the version of the plugin
     */
    virtual QString getVersion() const = 0;

    /**
     * @brief Initialize the plugin
     * @return true if initialization was successful, false otherwise
     */
    virtual bool initialize() = 0;

    /**
     * @brief Process an image with this plugin
     * @param image The input image to process
     * @param parameters Optional parameters for processing
     * @return The processed image
     */
    virtual QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) = 0;

    /**
     * @brief Create the UI widget for this plugin
     * @return A QWidget containing the plugin's controls, or nullptr if no UI is needed
     */
    virtual QWidget* createControlsWidget() {
        return nullptr;  // Default: no UI
    }

    /**
     * @brief Get current parameters from the UI
     * @return Map of parameter names to values
     */
    virtual QMap<QString, QVariant> getParameters() const {
        return QMap<QString, QVariant>();  // Default: no parameters
    }

signals:
    /**
     * @brief Signal emitted when plugin parameters change
     */
    void parametersChanged();

    /**
     * @brief Signal emitted when processing starts
     */
    void processingStarted();

    /**
     * @brief Signal emitted during processing with progress (0-100)
     */
    void processingProgress(int progress);

    /**
     * @brief Signal emitted when processing completes
     */
    void processingFinished();
};

// Plugin factory function that must be exported from plugins
extern "C" {
    PhotoEditorPlugin* createPlugin();
    void destroyPlugin(PhotoEditorPlugin* plugin);
}

#endif // PHOTOEDITORPLUGIN_H
