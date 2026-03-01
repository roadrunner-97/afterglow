cmake_minimum_required(VERSION 3.16)

add_library(exposure_effect STATIC ExposureEffect.cpp ExposureEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(exposure_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(exposure_effect PRIVATE HAVE_OPENCL)

target_link_libraries(exposure_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(exposure_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(exposure_effect PROPERTIES AUTOMOC ON)
