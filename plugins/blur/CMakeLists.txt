cmake_minimum_required(VERSION 3.16)

add_library(blur_effect STATIC BlurEffect.cpp BlurEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(blur_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(blur_effect PRIVATE HAVE_OPENCL)

target_link_libraries(blur_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(blur_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(blur_effect PROPERTIES AUTOMOC ON)
