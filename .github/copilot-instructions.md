# Lightroom Clone - Project Setup

## Project Overview

A modular, plugin-based photo editing application built with C++ and Qt. The project demonstrates dynamic plugin loading, clean architecture, and modern C++ practices.

**Technology Stack:**
- Language: C++ (C++17)
- GUI Framework: Qt 6
- Build System: CMake 3.16+
- Plugin System: Dynamic library loading (dlopen/dlsym)

## Setup Progress

### ✅ Framework Selection & Project Scaffolding
- Framework: **Qt 6** (excellent for desktop GUI, cross-platform, built-in image processing)
- Project structure created with modular design:
  - `src/core/`: Core library with plugin interface and manager
  - `src/ui/`: UI components using Qt
  - `plugins/`: Sample plugins demonstrating the architecture

### ✅ Core Components Created
- `PhotoEditorPlugin.h`: Plugin interface (abstract base class)
- `PluginManager.h/cpp`: Handles dynamic plugin loading/unloading
- `PhotoEditorApp.h/cpp`: Main application window
- `main.cpp`: Application entry point

### ✅ CMake Configuration
- Main `CMakeLists.txt` with Qt6 integration
- Plugin subdirectory with separate `CMakeLists.txt`
- Proper library output directories configured
- Qt MOC (Meta-Object Compiler) automation enabled

### ✅ Sample Plugins
- **GrayscalePlugin**: Converts images to grayscale
- **BrightnessPlugin**: Adjusts image brightness
- Both plugins properly export factory functions

### ✅ Documentation
- Comprehensive `README.md` with:
  - Feature overview
  - Installation instructions for all platforms
  - Plugin development guide
  - Build instructions
  - Troubleshooting section

## Next Steps to Run the Project

### 1. Install Qt6
**Ubuntu/Debian:**
```bash
sudo apt-get install qt6-base-dev qt6-tools-dev
```

**macOS:**
```bash
brew install qt@6
```

### 2. Build the Project
```bash
cd /home/raffi/programming/lightroom_clone
mkdir -p build && cd build
cmake ..
cmake --build .
```

### 3. Run the Application
```bash
./bin/lightroom_clone
```

## Project Features

✅ **Plugin Architecture**
- Dynamic loading of plugins from shared libraries
- Clean plugin interface using virtual methods
- Factory pattern for plugin instantiation

✅ **Qt Integration**
- Modern Qt6 with MOC support
- Image display with scrolling
- File dialogs for open/save
- Menu bar navigation

✅ **Modular Design**
- Separate core, UI, and plugins
- Plugin manager handles all lifecycle operations
- Easy to extend with new effects

✅ **Sample Plugins**
- Working examples for developers
- Demonstrates best practices
- Ready to use immediately

## Extension Guide

To add a new plugin:

1. Create `YourPlugin.h` and `YourPlugin.cpp` in `plugins/`
2. Inherit from `PhotoEditorPlugin` and implement virtual methods
3. Export `createPlugin()` and `destroyPlugin()` functions
4. Add CMake target to `src/plugins/CMakeLists.txt`
5. Rebuild with `cmake --build .`

Plugins will automatically appear in the application dropdown on next run.

## Known Limitations & Future Work

- UI parameter controls not yet implemented (can be added)
- No undo/redo system (complex but feasible)
- Single-threaded processing (could add threading)
- No layer support yet
- Basic effects only (can expand with more plugins)

## Files Created

```
lightroom_clone/
├── src/
│   ├── core/
│   │   ├── PhotoEditorPlugin.h
│   │   ├── PluginManager.h
│   │   └── PluginManager.cpp
│   ├── ui/
│   │   ├── PhotoEditorApp.h
│   │   └── PhotoEditorApp.cpp
│   ├── plugins/CMakeLists.txt
│   └── main.cpp
├── plugins/
│   ├── GrayscalePlugin.h
│   ├── GrayscalePlugin.cpp
│   ├── BrightnessPlugin.h
│   └── BrightnessPlugin.cpp
├── CMakeLists.txt
├── README.md
└── .github/copilot-instructions.md (this file)
```

## Build Output Structure

After building, you'll have:
```
build/
├── bin/
│   └── lightroom_clone (executable)
├── lib/
│   ├── libphotoeditor_core.so
│   └── libphotoeditor_ui.so
└── plugins/
    ├── grayscale_plugin.so
    └── brightness_plugin.so
```

## Compilation Notes

- Qt MOC is automatically configured via CMake
- All required Qt modules linked (Core, Gui, Widgets)
- C++17 standard enforced
- Position-independent code enabled for plugins
