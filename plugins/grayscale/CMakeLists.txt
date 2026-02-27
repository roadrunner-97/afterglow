cmake_minimum_required(VERSION 3.16)

add_library(grayscale_effect STATIC GrayscaleEffect.cpp GrayscaleEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(grayscale_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(grayscale_effect PRIVATE HAVE_OPENCL)

target_link_libraries(grayscale_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(grayscale_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(grayscale_effect PROPERTIES AUTOMOC ON)
