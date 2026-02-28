cmake_minimum_required(VERSION 3.16)

add_library(unsharp_effect STATIC UnsharpEffect.cpp UnsharpEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(unsharp_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(unsharp_effect PRIVATE HAVE_OPENCL)

target_link_libraries(unsharp_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(unsharp_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(unsharp_effect PROPERTIES AUTOMOC ON)
