# Lightroom Clone - Photo Editor

A modular, plugin-based photo editing application built with C++ and Qt. This project demonstrates how to create a flexible photo editing software with a plugin architecture that allows for easy extension.

## Features

- **Qt GUI Framework**: Modern, cross-platform desktop application interface
- **Plugin Architecture**: Dynamic plugin loading from shared libraries
- **Image Processing**: Built-in support for various image formats (PNG, JPG, BMP, TIFF)
- **Modular Design**: Core components separated from UI for flexibility
- **Extensible**: Easy to add new plugins for different image processing effects

## Project Structure

```
lightroom_clone/
├── src/
│   ├── core/                    # Core library components
│   │   ├── PhotoEditorPlugin.h  # Plugin interface (base class)
│   │   ├── PluginManager.h      # Plugin manager (header)
│   │   └── PluginManager.cpp    # Plugin manager (implementation)
│   ├── ui/                      # UI components
│   │   ├── PhotoEditorApp.h     # Main application window (header)
│   │   └── PhotoEditorApp.cpp   # Main application window (implementation)
│   ├── plugins/                 # Empty directory for plugin subdirectory references
│   └── main.cpp                 # Entry point
├── plugins/                     # Plugin source directories
│   ├── CMakeLists.txt           # Master plugin CMakeLists
│   ├── grayscale/               # Grayscale filter plugin
│   │   ├── CMakeLists.txt       # Plugin-specific build config
│   │   ├── GrayscalePlugin.h
│   │   └── GrayscalePlugin.cpp
│   └── brightness/              # Brightness adjustment plugin
│       ├── CMakeLists.txt       # Plugin-specific build config
│       ├── BrightnessPlugin.h
│       └── BrightnessPlugin.cpp
├── CMakeLists.txt               # Main CMake build configuration
└── build/                       # Build output directory
```

## Prerequisites

Before building, make sure you have:

- **CMake** 3.16 or higher
- **Qt6** development libraries (Core, Gui, Widgets)
- **C++17** compatible compiler (GCC, Clang, MSVC)
- **Linux/macOS/Windows** system

### Installing Qt6

**Ubuntu/Debian:**
```bash
sudo apt-get install qt6-base-dev qt6-tools-dev qt6-l10n-tools
```

**macOS (with Homebrew):**
```bash
brew install qt@6
```

**Windows (with vcpkg):**
```bash
vcpkg install qt6:x64-windows
```

## Building the Project

1. **Create and navigate to the build directory:**
   ```bash
   mkdir -p build && cd build
   ```

2. **Configure the project with CMake:**
   ```bash
   cmake ..
   ```

3. **Build the project:**
   ```bash
   cmake --build . --config Release
   ```

4. **Run the application:**
   ```bash
   ./bin/lightroom_clone
   ```

## Usage

1. Launch the application
2. Click "File" → "Open Image" to load a photo
3. Select a plugin from the dropdown menu on the right panel
4. Click "Apply Plugin" to process the image
5. Click "File" → "Save Image" to save your edited photo

## Plugin Architecture

### Creating a Custom Plugin

To create a new plugin using modern CMake practices:

1. **Create a plugin directory:**
   ```bash
   mkdir -p plugins/myplugin
   cd plugins/myplugin
   ```

2. **Create a class that inherits from `PhotoEditorPlugin` (MyPlugin.h):**

```cpp
#ifndef MYPLUGIN_H
#define MYPLUGIN_H

#include "../../src/core/PhotoEditorPlugin.h"

class MyPlugin : public PhotoEditorPlugin {
    Q_OBJECT
public:
    MyPlugin();
    ~MyPlugin() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;
};

#endif // MYPLUGIN_H
```

3. **Implement the required methods (MyPlugin.cpp):**
   - `getName()`: Return the plugin name
   - `getDescription()`: Return a brief description
   - `getVersion()`: Return the plugin version
   - `initialize()`: Initialize the plugin (return true on success)
   - `processImage()`: Process and return the modified image

```cpp
#include "MyPlugin.h"
#include <QDebug>

MyPlugin::MyPlugin() {}
MyPlugin::~MyPlugin() {}

QString MyPlugin::getName() const { return "My Plugin"; }
QString MyPlugin::getDescription() const { return "My plugin description"; }
QString MyPlugin::getVersion() const { return "1.0.0"; }
bool MyPlugin::initialize() { 
    qDebug() << "MyPlugin initialized"; 
    return true; 
}

QImage MyPlugin::processImage(const QImage &image, const QMap<QString, QVariant> &parameters) {
    // Your image processing code here
    return image;
}

// Export the plugin factory functions
extern "C" {
    PhotoEditorPlugin* createPlugin() {
        return new MyPlugin();
    }

    void destroyPlugin(PhotoEditorPlugin* plugin) {
        delete plugin;
    }
}
```

4. **Create `plugins/myplugin/CMakeLists.txt` with the build configuration:**

```cmake
cmake_minimum_required(VERSION 3.16)

set(MYPLUGIN_SOURCES
    MyPlugin.cpp
)

set(MYPLUGIN_HEADERS
    MyPlugin.h
)

add_library(my_plugin SHARED ${MYPLUGIN_SOURCES} ${MYPLUGIN_HEADERS})

target_link_libraries(my_plugin
    PRIVATE
        photoeditor_core
        Qt6::Core
        Qt6::Gui
)

target_include_directories(my_plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(my_plugin PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    PREFIX ""
    SUFFIX ".so"
)
```

5. **Add your plugin to the master `plugins/CMakeLists.txt`:**

```cmake
add_subdirectory(myplugin)
```

6. **Rebuild the project:**
   ```bash
   cd build
   cmake ..
   cmake --build .
   ```

Your plugin will appear in the dropdown automatically on next run!

## Plugin Interface

The `PhotoEditorPlugin` base class provides:

- **Signals:**
  - `processingStarted()`: Emitted when image processing begins
  - `processingProgress(int)`: Emitted with progress percentage (0-100)
  - `processingFinished()`: Emitted when processing completes

- **Methods:**
  - All virtual, must be implemented by plugins

## Included Sample Plugins

### GrayscalePlugin
Converts the image to grayscale using the luminosity formula:
```
Gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### BrightnessPlugin
Adjusts the brightness of the image. Accepts a `brightness` parameter (-100 to 100).

## Building on Different Platforms

### Linux
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### macOS
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/opt/qt@6 ..
cmake --build .
```

### Windows (Visual Studio)
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## Troubleshooting

### Qt not found
- Ensure Qt6 is installed and in your system PATH
- Try: `export CMAKE_PREFIX_PATH=/path/to/qt6`

### Plugin not loading
- Check that plugins are built in `build/plugins/` directory
- Verify plugin names are unique
- Check application console output for error messages

## Architecture Benefits

- **Modularity**: Core functionality separated from UI
- **Extensibility**: Add new effects without modifying core code
- **Reusability**: Core library can be used independently
- **Dynamic Loading**: Plugins loaded at runtime from files
- **Type Safety**: C++ with Qt's type system

## Future Enhancements

- [ ] Plugin parameter UI auto-generation
- [ ] Undo/redo system
- [ ] Layer support
- [ ] Batch processing
- [ ] Plugin marketplace
- [ ] Scripting support
- [ ] Real-time preview
- [ ] More built-in plugins (blur, sharpen, color correction, etc.)

## License

This project is provided as-is for educational purposes.

## Contributing

Feel free to extend this project by:
1. Creating new plugins
2. Improving the UI
3. Adding more features to the core
4. Optimizing image processing

## Support

For issues or questions, refer to the Qt documentation and the included comments in the source code.
