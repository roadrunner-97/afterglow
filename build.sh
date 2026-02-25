#!/bin/bash
# Build script for Lightroom Clone

set -e

echo "=========================================="
echo "Lightroom Clone - Build Script"
echo "=========================================="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

cd build

echo "Running CMake..."
cmake ..

echo "Building project..."
cmake --build . --config Release --parallel $(nproc)

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "To run the application, execute:"
echo "  ./bin/lightroom_clone"
echo ""
echo "The following files were created:"
echo "  - bin/lightroom_clone (main executable)"
echo "  - lib/libphotoeditor_core.so (core library)"
echo "  - lib/libphotoeditor_ui.so (UI library)"
echo "  - plugins/grayscale_plugin.so (grayscale filter)"
echo "  - plugins/brightness_plugin.so (brightness adjustment)"
echo ""
