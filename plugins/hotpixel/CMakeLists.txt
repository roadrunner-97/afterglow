cmake_minimum_required(VERSION 3.16)

add_library(hotpixel_effect STATIC HotPixelEffect.cpp HotPixelEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(hotpixel_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(hotpixel_effect PRIVATE HAVE_OPENCL)

target_link_libraries(hotpixel_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(hotpixel_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(hotpixel_effect PROPERTIES AUTOMOC ON)
