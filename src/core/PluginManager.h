#ifndef PLUGINMANAGER_H
#define PLUGINMANAGER_H

#include "PhotoEditorPlugin.h"
#include <QString>
#include <QMap>
#include <memory>

// Type definition for plugin factory functions
typedef PhotoEditorPlugin* (*CreatePluginFunc)();
typedef void (*DestroyPluginFunc)(PhotoEditorPlugin*);

/**
 * @brief Manages loading and unloading of plugins
 */
class PluginManager {
public:
    PluginManager();
    ~PluginManager();

    /**
     * @brief Load a plugin from a file
     * @param filePath Path to the plugin library file
     * @return true if the plugin was loaded successfully
     */
    bool loadPlugin(const QString &filePath);

    /**
     * @brief Unload a plugin
     * @param pluginName The name of the plugin to unload
     */
    bool unloadPlugin(const QString &pluginName);

    /**
     * @brief Get a plugin by name
     * @param pluginName The name of the plugin
     * @return Pointer to the plugin, or nullptr if not found
     */
    PhotoEditorPlugin* getPlugin(const QString &pluginName) const;

    /**
     * @brief Get all loaded plugins
     * @return Map of plugin names to plugin pointers
     */
    const QMap<QString, PhotoEditorPlugin*>& getLoadedPlugins() const;

    /**
     * @brief Load all plugins from a directory
     * @param directoryPath Path to the directory containing plugins
     */
    void loadPluginsFromDirectory(const QString &directoryPath);

private:
    struct PluginInfo {
        PhotoEditorPlugin* plugin;
        void* handle;
        DestroyPluginFunc destroyFunc;
    };

    QMap<QString, PluginInfo> loadedPlugins;
};

#endif // PLUGINMANAGER_H
