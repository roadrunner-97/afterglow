#!/bin/bash
# Build verification script

echo "=========================================="
echo "Lightroom Clone - Build Verification"
echo "=========================================="
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "❌ Build directory not found!"
    echo "Run: ./build.sh"
    exit 1
fi

echo "✅ Build directory exists"
echo ""

# Check executables
echo "Checking executables:"
if [ -f "build/bin/lightroom_clone" ]; then
    SIZE=$(ls -lh build/bin/lightroom_clone | awk '{print $5}')
    echo "  ✅ build/bin/lightroom_clone ($SIZE)"
else
    echo "  ❌ build/bin/lightroom_clone not found"
fi
echo ""

# Check core libraries
echo "Checking core libraries:"
if [ -f "build/lib/libphotoeditor_core.so" ]; then
    SIZE=$(ls -lh build/lib/libphotoeditor_core.so | awk '{print $5}')
    echo "  ✅ build/lib/libphotoeditor_core.so ($SIZE)"
else
    echo "  ❌ build/lib/libphotoeditor_core.so not found"
fi

if [ -f "build/lib/libphotoeditor_ui.so" ]; then
    SIZE=$(ls -lh build/lib/libphotoeditor_ui.so | awk '{print $5}')
    echo "  ✅ build/lib/libphotoeditor_ui.so ($SIZE)"
else
    echo "  ❌ build/lib/libphotoeditor_ui.so not found"
fi
echo ""

# Check plugins
echo "Checking plugins:"
if [ -f "build/plugins/grayscale_plugin.so" ]; then
    SIZE=$(ls -lh build/plugins/grayscale_plugin.so | awk '{print $5}')
    echo "  ✅ build/plugins/grayscale_plugin.so ($SIZE)"
else
    echo "  ❌ build/plugins/grayscale_plugin.so not found"
fi

if [ -f "build/plugins/brightness_plugin.so" ]; then
    SIZE=$(ls -lh build/plugins/brightness_plugin.so | awk '{print $5}')
    echo "  ✅ build/plugins/brightness_plugin.so ($SIZE)"
else
    echo "  ❌ build/plugins/brightness_plugin.so not found"
fi
echo ""

# Check library dependencies
echo "Checking library dependencies:"
echo ""
echo "libphotoeditor_core.so depends on:"
ldd build/lib/libphotoeditor_core.so | grep Qt | head -3
echo ""
echo "lightroom_clone depends on:"
ldd build/bin/lightroom_clone | grep -E "(libphotoeditor|Qt)" | head -5
echo ""

echo "=========================================="
echo "Build Verification Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  cd /home/raffi/programming/lightroom_clone"
echo "  ./build/bin/lightroom_clone"
echo ""
