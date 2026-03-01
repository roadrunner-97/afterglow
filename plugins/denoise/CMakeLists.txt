cmake_minimum_required(VERSION 3.16)

add_library(denoise_effect STATIC DenoiseEffect.cpp DenoiseEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(denoise_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(denoise_effect PRIVATE HAVE_OPENCL)

target_link_libraries(denoise_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(denoise_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(denoise_effect PROPERTIES AUTOMOC ON)
