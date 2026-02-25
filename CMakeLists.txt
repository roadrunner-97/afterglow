cmake_minimum_required(VERSION 3.16)
project(LightroomClone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C++ flags with -O3 optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC")

# Find Qt6
find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)

# ============================================================================
# Core Library
# ============================================================================

set(CORE_SOURCES
    src/core/PluginManager.cpp
)

set(CORE_HEADERS
    src/core/PhotoEditorPlugin.h
    src/core/PluginManager.h
)

add_library(photoeditor_core SHARED ${CORE_SOURCES} ${CORE_HEADERS})

target_link_libraries(photoeditor_core
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
)

target_include_directories(photoeditor_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(photoeditor_core PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# UI Library
# ============================================================================

set(UI_SOURCES
    src/ui/PhotoEditorApp.cpp
)

set(UI_HEADERS
    src/ui/PhotoEditorApp.h
)

add_library(photoeditor_ui SHARED ${UI_SOURCES} ${UI_HEADERS})

target_link_libraries(photoeditor_ui
    PRIVATE
        photoeditor_core
        Qt6::Widgets
        Qt6::Gui
)

target_include_directories(photoeditor_ui
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

set_target_properties(photoeditor_ui PROPERTIES
    AUTOMOC ON
    AUTORCC ON
    AUTOUIC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# Main Executable
# ============================================================================

add_executable(lightroom_clone src/main.cpp)

target_link_libraries(lightroom_clone
    PRIVATE
        photoeditor_ui
        photoeditor_core
        Qt6::Widgets
        Qt6::Core
)

target_include_directories(lightroom_clone
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ui
)

set_target_properties(lightroom_clone PROPERTIES
    AUTOMOC ON
    AUTORCC ON
    AUTOUIC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# Plugins
# ============================================================================

add_subdirectory(plugins)
