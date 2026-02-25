cmake_minimum_required(VERSION 3.16)

set(BRIGHTNESS_SOURCES
    BrightnessPlugin.cpp
)

set(BRIGHTNESS_HEADERS
    BrightnessPlugin.h
)

add_library(brightness_plugin SHARED ${BRIGHTNESS_SOURCES} ${BRIGHTNESS_HEADERS})

target_link_libraries(brightness_plugin
    PRIVATE
        photoeditor_core
        Qt6::Core
        Qt6::Gui
)

target_include_directories(brightness_plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(brightness_plugin PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    PREFIX ""
    SUFFIX ".so"
)
