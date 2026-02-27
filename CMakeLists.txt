cmake_minimum_required(VERSION 3.16)
project(LightroomClone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC")

find_package(Qt6 COMPONENTS Core Gui Widgets Concurrent REQUIRED)
find_package(OpenCL REQUIRED)

# Optional: LibRaw for RAW file support
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBRAW QUIET libraw)
endif()

# ============================================================================
# Widgets Library
# ============================================================================

add_library(photoeditor_widgets SHARED
    src/widgets/ParamSlider.cpp
    src/widgets/ParamSlider.h
)

target_link_libraries(photoeditor_widgets
    PUBLIC Qt6::Core Qt6::Gui Qt6::Widgets
)

target_include_directories(photoeditor_widgets
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/widgets>
)

set_target_properties(photoeditor_widgets PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# Core Library
# ============================================================================

add_library(photoeditor_core SHARED
    src/core/EffectManager.cpp
    src/core/EffectManager.h
    src/core/ImageProcessor.cpp
    src/core/ImageProcessor.h
    src/core/GpuDeviceRegistry.cpp
    src/core/GpuDeviceRegistry.h
    src/core/GpuPipeline.cpp
    src/core/GpuPipeline.h
    src/core/IGpuEffect.h
    src/core/RawLoader.cpp
    src/core/RawLoader.h
    src/core/PhotoEditorEffect.h
)

target_link_libraries(photoeditor_core
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Concurrent
        OpenCL::OpenCL
)

target_compile_definitions(photoeditor_core PUBLIC HAVE_OPENCL)

if(LIBRAW_FOUND)
    target_link_libraries(photoeditor_core PRIVATE ${LIBRAW_LIBRARIES})
    target_compile_definitions(photoeditor_core PRIVATE HAVE_LIBRAW)
    target_include_directories(photoeditor_core PRIVATE ${LIBRAW_INCLUDE_DIRS})
    message(STATUS "LibRaw found — RAW file support enabled")
else()
    message(STATUS "LibRaw not found — RAW file support disabled")
endif()

target_include_directories(photoeditor_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
)

set_target_properties(photoeditor_core PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# UI Library
# ============================================================================

add_library(photoeditor_ui SHARED
    src/ui/PhotoEditorApp.cpp
    src/ui/PhotoEditorApp.h
)

target_link_libraries(photoeditor_ui
    PRIVATE
        photoeditor_core
        Qt6::Widgets
        Qt6::Gui
)

set_target_properties(photoeditor_ui PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# Effect Plugins (static libs) — defines all_effects INTERFACE target
# ============================================================================

add_subdirectory(plugins)

# ============================================================================
# Main Executable
# ============================================================================

add_executable(lightroom_clone src/main.cpp)

target_link_libraries(lightroom_clone
    PRIVATE
        photoeditor_ui
        photoeditor_core
        photoeditor_widgets
        all_effects
        Qt6::Widgets
        Qt6::Core
)

set_target_properties(lightroom_clone PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
