cmake_minimum_required(VERSION 3.16)
project(Afterglow VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC \
    -Wshadow=local \
    -Wnon-virtual-dtor \
    -Woverloaded-virtual \
    -Wuseless-cast \
    -Wimplicit-fallthrough \
    -Wduplicated-cond \
    -Wduplicated-branches \
    -Werror")

# Coverage instrumentation — overrides -O3 with -O0 for accurate results
option(COVERAGE "Enable gcov coverage instrumentation (requires gcovr)" OFF)
if(COVERAGE)
    string(REPLACE "-O3" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# Export compile_commands.json for clangd/LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "ccache found — using ${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found — builds will not be cached")
endif()

find_package(Qt6 COMPONENTS Core Gui Widgets Concurrent OpenGLWidgets REQUIRED)
find_package(OpenGL REQUIRED)
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
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/widgets>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/ui>
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
    src/core/GpuContextBase.h
    src/core/blur_kernels.h
    src/core/Histogram.cpp
    src/core/Histogram.h
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
    src/ui/ViewportWidget.cpp
    src/ui/ViewportWidget.h
)

target_link_libraries(photoeditor_ui
    PRIVATE
        photoeditor_core
        Qt6::Widgets
        Qt6::Gui
        Qt6::OpenGLWidgets
        OpenGL::GL
)

target_include_directories(photoeditor_ui
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/ui>
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

add_executable(afterglow src/main.cpp)

target_link_libraries(afterglow
    PRIVATE
        photoeditor_ui
        photoeditor_core
        photoeditor_widgets
        all_effects
        Qt6::Widgets
        Qt6::Core
        Qt6::OpenGLWidgets
)

set_target_properties(afterglow PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# Tests
# ============================================================================

enable_testing()
add_subdirectory(tests)

# ============================================================================
# Coverage report (gcovr)
# ============================================================================

if(COVERAGE)
    find_program(GCOVR gcovr REQUIRED)
    add_custom_target(coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${GCOVR}
            --root ${CMAKE_SOURCE_DIR}
            --object-directory ${CMAKE_BINARY_DIR}
            # Generated / build artefacts
            --exclude "${CMAKE_BINARY_DIR}/"
            # Test sources (we measure what's tested, not the tests themselves)
            --exclude "${CMAKE_SOURCE_DIR}/tests/"
            # UI and app-level code that requires a display + event loop
            --exclude "${CMAKE_SOURCE_DIR}/src/ui/"
            --exclude "${CMAKE_SOURCE_DIR}/src/core/ImageProcessor\\.cpp"
            --exclude "${CMAKE_SOURCE_DIR}/src/core/RawLoader\\.cpp"
            --exclude "${CMAKE_SOURCE_DIR}/src/main\\.cpp"
            # Drop compiler-generated exception branches and proven-unreachable
            # branches; combined with GCOVR_EXCL_START/STOP markers around GPU
            # failure catches, this stops infeasible paths from skewing the total.
            --exclude-throw-branches
            --exclude-unreachable-branches
            # Tolerate gcov reporting the same function at two line numbers
            # (happens after split-line catch blocks); pick the earliest.
            --merge-mode-functions=merge-use-line-min
            --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
            --xml ${CMAKE_BINARY_DIR}/coverage/coverage.xml
            --print-summary
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating coverage report -> build/coverage/index.html"
    )
endif()
