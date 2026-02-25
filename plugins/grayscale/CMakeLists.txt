cmake_minimum_required(VERSION 3.16)

set(GRAYSCALE_SOURCES
    GrayscalePlugin.cpp
)

set(GRAYSCALE_HEADERS
    GrayscalePlugin.h
)

add_library(grayscale_plugin SHARED ${GRAYSCALE_SOURCES} ${GRAYSCALE_HEADERS})

target_link_libraries(grayscale_plugin
    PRIVATE
        photoeditor_core
        Qt6::Core
        Qt6::Gui
)

target_include_directories(grayscale_plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(grayscale_plugin PROPERTIES
    AUTOMOC ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    PREFIX ""
    SUFFIX ".so"
)
