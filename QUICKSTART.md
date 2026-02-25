# QUICKSTART GUIDE - Lightroom Clone

## What You've Built

A **modular, plugin-based photo editing application** in C++ with Qt 6. The architecture uses dynamic library loading to allow plugins to be loaded at runtime without recompiling the main application.

## Project Structure at a Glance

```
lightroom_clone/
├── src/
│   ├── core/              → Core plugin system
│   │   ├── PhotoEditorPlugin.h      (Plugin interface)
│   │   ├── PluginManager.h/cpp      (Plugin loader)
│   ├── ui/                → User interface
│   │   ├── PhotoEditorApp.h/cpp     (Main window)
│   ├── plugins/           → Empty (for reference only)
│   └── main.cpp           → Application entry point
├── plugins/               → Plugin implementations
│   ├── CMakeLists.txt     → Master plugin builder
│   ├── grayscale/         → Grayscale filter plugin
│   │   ├── CMakeLists.txt → Grayscale-specific build config
│   │   ├── GrayscalePlugin.h/cpp
│   └── brightness/        → Brightness adjustment plugin
│       ├── CMakeLists.txt → Brightness-specific build config
│       ├── BrightnessPlugin.h/cpp
├── CMakeLists.txt         → Main build configuration
└── build/                 → Build output
```

## Key Components

### 1. **PhotoEditorPlugin.h** (Plugin Interface)
The base class that all plugins must inherit from. Defines the contract:
- `getName()` - Plugin name
- `getDescription()` - What it does
- `getVersion()` - Version string
- `initialize()` - Setup when loaded
- `processImage()` - Main processing function

### 2. **PluginManager** (Plugin Loader)
Handles loading/unloading of `.so` files from disk using:
- `dlopen()` - Load shared library
- `dlsym()` - Find the factory function
- `dlclose()` - Cleanup

### 3. **PhotoEditorApp** (Main Window)
Qt GUI with:
- Image viewer with scroll
- Plugin selector dropdown
- Apply button
- File open/save dialogs

## How the Plugin System Works

```
User opens image
    ↓
User selects plugin from dropdown
    ↓
User clicks "Apply Plugin"
    ↓
PluginManager::getPlugin() retrieves the plugin
    ↓
plugin->processImage() is called
    ↓
Result is displayed and can be saved
```

## How Plugins Load

1. Application starts
2. `loadPluginsFromDirectory()` scans `./plugins/` for `.so` files
3. For each plugin:
   - `dlopen()` loads the library
   - `dlsym()` finds the `createPlugin` function
   - Factory function creates the plugin instance
   - `initialize()` is called
   - Plugin appears in dropdown

## Building & Running

### Quick Build
```bash
cd /home/raffi/programming/lightroom_clone
chmod +x build.sh
./build.sh
```

### Manual Build
```bash
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

### Run
```bash
./build/bin/lightroom_clone
```

## What Happens During Build

```
CMake Configuration
├── Finds Qt6 installation
├── Configures MOC (Meta-Object Compiler)
└── Sets up compiler flags

Build Phase
├── Compiles core library (PluginManager, Plugin interface)
├── Compiles UI library (PhotoEditorApp with Qt MOC)
├── Compiles plugins (GrayscalePlugin, BrightnessPlugin)
└── Links main executable

Output
├── build/bin/lightroom_clone       ← Main executable
├── build/lib/libphotoeditor_core.so
├── build/lib/libphotoeditor_ui.so
├── build/plugins/grayscale_plugin.so
└── build/plugins/brightness_plugin.so
```

## Creating Your First Plugin

### Step 1: Create the header (`plugins/MyFilter.h`)
```cpp
#ifndef MYFILTER_H
#define MYFILTER_H
#include "../src/core/PhotoEditorPlugin.h"

class MyFilter : public PhotoEditorPlugin {
    Q_OBJECT
public:
    QString getName() const override { return "My Filter"; }
    QString getDescription() const override { return "My description"; }
    QString getVersion() const override { return "1.0.0"; }
    bool initialize() override { return true; }
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &params = {}) override;
};
#endif
```

### Step 2: Implement (`plugins/MyFilter.cpp`)
```cpp
#include "MyFilter.h"

QImage MyFilter::processImage(const QImage &image, const QMap<QString, QVariant> &params) {
    // Process and return modified image
    QImage result = image;
    // Your processing code here
    return result;
}

extern "C" {
    PhotoEditorPlugin* createPlugin() { return new MyFilter(); }
    void destroyPlugin(PhotoEditorPlugin* p) { delete p; }
}
```

### Step 3: Add to CMakeLists.txt
Add this to `src/plugins/CMakeLists.txt`:
```cmake
add_library(my_filter SHARED plugins/MyFilter.cpp plugins/MyFilter.h)
target_link_libraries(my_filter photoeditor_core Qt6::Core Qt6::Gui)
target_include_directories(my_filter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/core)
set_target_properties(my_filter PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    PREFIX ""
    SUFFIX ".so"
)
```

### Step 4: Rebuild
```bash
cd build
cmake ..
cmake --build .
```

Your plugin will appear in the dropdown automatically on next run!

## Example: Understanding the Grayscale Plugin

```cpp
// Convert each pixel to grayscale
for (int y = 0; y < result.height(); ++y) {
    for (int x = 0; x < result.width(); ++x) {
        QRgb pixel = result.pixel(x, y);
        int r = qRed(pixel), g = qGreen(pixel), b = qBlue(pixel);
        
        // Luminosity formula (human eye perceives green > red > blue)
        int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
        
        result.setPixel(x, y, qRgb(gray, gray, gray));
    }
}
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Qt6 not found | `sudo apt install qt6-base-dev` (Ubuntu) or `brew install qt@6` (macOS) |
| CMake error | Make sure CMake 3.16+ is installed: `cmake --version` |
| Plugin not loading | Check permissions: `chmod +x build/plugins/*.so` |
| MOC error | Delete `build/` and try again: `rm -rf build && cmake ..` |
| Missing includes | Verify relative paths in `#include` statements |

## Architecture Benefits

✅ **Modularity** - Core, UI, and effects are separate  
✅ **Extensibility** - Add effects without modifying core  
✅ **Flexibility** - Plugins loaded dynamically at runtime  
✅ **Reusability** - Core can be used in other projects  
✅ **Performance** - C++ with direct image pixel access  

## What You Can Extend

- More filters (blur, sharpen, color correction, etc.)
- Parameter sliders for plugins (Brightness already supports this)
- Batch processing
- Plugin preview window
- Undo/redo system
- Layers support
- Real-time preview

## Qt Concepts Used

- **Q_OBJECT** - Macro for Qt meta-object system
- **QImage** - Image data structure
- **QMainWindow** - Top-level window widget
- **QLayout** - Automatic widget positioning
- **QFileDialog** - Open/save file dialogs
- **AUTOMOC** - Automatic Meta-Object Compiler in CMake
- **Signals/Slots** - Event system

## Resources

- Qt6 Documentation: https://doc.qt.io/qt-6/
- CMake Guide: https://cmake.org/cmake/help/latest/
- C++17 Features: https://en.cppreference.com/
- Image Processing: Look at GrayscalePlugin.cpp for examples

## Next Steps

1. ✅ Build and run the application
2. Try applying the grayscale and brightness filters
3. Create a new simple plugin (e.g., invert colors)
4. Add a parameter slider for brightness control
5. Explore adding more complex filters

Happy coding! 🎨
