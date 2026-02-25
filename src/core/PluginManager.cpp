#include "PluginManager.h"
#include <QDir>
#include <QDebug>
#include <dlfcn.h>

PluginManager::PluginManager() {
}

PluginManager::~PluginManager() {
    // Unload all plugins
    for (auto it = loadedPlugins.begin(); it != loadedPlugins.end(); ++it) {
        const PluginInfo& info = it.value();
        if (info.destroyFunc && info.plugin) {
            info.destroyFunc(info.plugin);
        }
        if (info.handle) {
            dlclose(info.handle);
        }
    }
    loadedPlugins.clear();
}

bool PluginManager::loadPlugin(const QString &filePath) {
    // Load the shared library
    void* handle = dlopen(filePath.toStdString().c_str(), RTLD_LAZY);
    if (!handle) {
        qWarning() << "Failed to load plugin:" << filePath << "-" << dlerror();
        return false;
    }

    // Get the plugin factory function
    CreatePluginFunc createPlugin = (CreatePluginFunc)dlsym(handle, "createPlugin");

    if (!createPlugin) {
        qWarning() << "Plugin does not export createPlugin function:" << filePath;
        dlclose(handle);
        return false;
    }

    // Get the destroy function
    DestroyPluginFunc destroyFunc = (DestroyPluginFunc)dlsym(handle, "destroyPlugin");

    if (!destroyFunc) {
        qWarning() << "Plugin does not export destroyPlugin function:" << filePath;
        dlclose(handle);
        return false;
    }

    // Create the plugin instance
    PhotoEditorPlugin* plugin = createPlugin();
    if (!plugin) {
        qWarning() << "Failed to create plugin instance:" << filePath;
        dlclose(handle);
        return false;
    }

    // Initialize the plugin
    if (!plugin->initialize()) {
        qWarning() << "Plugin initialization failed:" << filePath;
        destroyFunc(plugin);
        dlclose(handle);
        return false;
    }

    // Store the plugin and handle
    QString pluginName = plugin->getName();
    loadedPlugins[pluginName] = {plugin, handle, destroyFunc};

    qDebug() << "Successfully loaded plugin:" << pluginName << "v" << plugin->getVersion();
    return true;
}

bool PluginManager::unloadPlugin(const QString &pluginName) {
    auto it = loadedPlugins.find(pluginName);
    if (it == loadedPlugins.end()) {
        return false;
    }

    const PluginInfo& info = it.value();
    if (info.destroyFunc && info.plugin) {
        info.destroyFunc(info.plugin);
    }
    if (info.handle) {
        dlclose(info.handle);
    }

    loadedPlugins.erase(it);
    return true;
}

PhotoEditorPlugin* PluginManager::getPlugin(const QString &pluginName) const {
    auto it = loadedPlugins.find(pluginName);
    if (it != loadedPlugins.end()) {
        return it.value().plugin;
    }
    return nullptr;
}

const QMap<QString, PhotoEditorPlugin*>& PluginManager::getLoadedPlugins() const {
    // Create a static map to return - this is a workaround for returning a different type
    static QMap<QString, PhotoEditorPlugin*> result;
    result.clear();
    for (auto it = loadedPlugins.begin(); it != loadedPlugins.end(); ++it) {
        result[it.key()] = it.value().plugin;
    }
    return result;
}

void PluginManager::loadPluginsFromDirectory(const QString &directoryPath) {
    QDir dir(directoryPath);
    if (!dir.exists()) {
        qWarning() << "Plugin directory does not exist:" << directoryPath;
        return;
    }

    QStringList filters;
    filters << "*.so" << "*.dll" << "*.dylib";
    QStringList files = dir.entryList(filters, QDir::Files);

    for (const QString &file : files) {
        QString fullPath = dir.absoluteFilePath(file);
        loadPlugin(fullPath);
    }
}
