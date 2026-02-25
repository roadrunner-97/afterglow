# Installation Guide for Lightroom Clone

## System Requirements

- **OS**: Linux, macOS, or Windows
- **CMake**: 3.16 or newer
- **Compiler**: GCC 7+, Clang 6+, or MSVC 2017+
- **Qt6**: Development libraries (Core, Gui, Widgets)
- **RAM**: 2GB minimum for compilation

## Installation Instructions

### Ubuntu/Debian Linux

```bash
# Install Qt6 development libraries
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    qt6-base-dev \
    qt6-tools-dev \
    git

# Navigate to the project directory
cd /home/raffi/programming/lightroom_clone

# Build the project
mkdir -p build && cd build
cmake ..
cmake --build . --config Release

# Run the application
./bin/lightroom_clone
```

### macOS (with Homebrew)

```bash
# Install dependencies
brew install cmake qt@6

# Navigate to the project
cd /home/raffi/programming/lightroom_clone

# Build
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/opt/qt@6 ..
cmake --build . --config Release

# Run
./bin/lightroom_clone
```

### Windows (with Visual Studio 2019/2022)

```bash
# Install Qt6 from https://www.qt.io/download
# Install Visual Studio with C++ tools

# Navigate to project
cd C:\Users\YourUser\lightroom_clone

# Build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release

# Run
.\bin\Release\lightroom_clone.exe
```

### Windows (with MinGW)

```bash
# Install Qt6 (MinGW version from qt.io)
# Install CMake and MinGW

mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build . --config Release

.\bin\lightroom_clone.exe
```

## Verifying Installation

After building successfully, verify the output:

```bash
cd build
ls -la bin/
ls -la lib/
ls -la plugins/
```

You should see:
- `bin/lightroom_clone` - Main executable
- `lib/libphotoeditor_core.so` - Core library
- `lib/libphotoeditor_ui.so` - UI library
- `plugins/grayscale_plugin.so` - Grayscale filter
- `plugins/brightness_plugin.so` - Brightness adjustment

## Running from the Build Directory

Once built, you can run directly from the build folder:

```bash
cd /home/raffi/programming/lightroom_clone/build
./bin/lightroom_clone
```

The application will automatically find plugins in `./plugins/` directory.

## Troubleshooting

### Problem: "Qt6 not found"

**Solution**: Install Qt6 development libraries

```bash
# Ubuntu/Debian
sudo apt-get install qt6-base-dev qt6-tools-dev

# macOS
brew install qt@6

# Or set CMAKE_PREFIX_PATH manually
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt6
```

### Problem: "qmake not found"

**Solution**: Ensure Qt development tools are installed

```bash
# Ubuntu/Debian
sudo apt-get install qt6-tools-dev

# macOS
brew install qt@6
```

### Problem: CMake configuration fails

**Solution**: Clear CMake cache and try again

```bash
cd build
rm CMakeCache.txt
rm -rf CMakeFiles/
cmake ..
```

### Problem: Plugins not loading

**Solution**: Check plugin directory and permissions

```bash
# Verify plugins exist
ls -la build/plugins/

# Make them executable
chmod +x build/plugins/*.so

# Check if core library is loadable
ldd build/lib/libphotoeditor_core.so
```

### Problem: "AUTOMOC: error"

**Solution**: Qt MOC setup issue - clean and rebuild

```bash
cd build
rm -rf CMakeFiles
cmake ..
cmake --build .
```

## Optional: System-Wide Installation

To install the application system-wide (advanced):

```bash
cd build
sudo cmake --install . --prefix /usr/local
```

This installs:
- `/usr/local/bin/lightroom_clone` - Executable
- `/usr/local/lib/` - Libraries
- `/usr/local/share/lightroom_clone/` - Plugins

Then run from anywhere:
```bash
lightroom_clone
```

## Development Setup

For development and debugging:

```bash
# Build with debug symbols
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .

# Run with debug information
gdb ./bin/lightroom_clone
```

## Uninstallation

To clean up:

```bash
# Remove build directory
rm -rf /home/raffi/programming/lightroom_clone/build

# If installed system-wide
sudo rm /usr/local/bin/lightroom_clone
sudo rm -rf /usr/local/lib/libphotoeditor*
sudo rm -rf /usr/local/share/lightroom_clone
```

## Getting Help

1. Check `README.md` for general information
2. Check `QUICKSTART.md` for quick reference
3. Review `.github/copilot-instructions.md` for architecture details
4. Check source code comments
5. Consult Qt documentation: https://doc.qt.io/qt-6/

## Next Steps

- [x] Install dependencies
- [x] Build the project
- [ ] Run `./build/bin/lightroom_clone`
- [ ] Open a sample image
- [ ] Apply grayscale filter
- [ ] Create a custom plugin
- [ ] Explore Qt documentation for advanced features
