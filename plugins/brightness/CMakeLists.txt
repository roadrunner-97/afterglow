cmake_minimum_required(VERSION 3.16)

add_library(brightness_effect STATIC BrightnessEffect.cpp BrightnessEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(brightness_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(brightness_effect PRIVATE HAVE_OPENCL)

target_link_libraries(brightness_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(brightness_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(brightness_effect PROPERTIES AUTOMOC ON)
