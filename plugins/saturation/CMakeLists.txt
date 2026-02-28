cmake_minimum_required(VERSION 3.16)

add_library(saturation_effect STATIC SaturationEffect.cpp SaturationEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(saturation_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(saturation_effect PRIVATE HAVE_OPENCL)

target_link_libraries(saturation_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(saturation_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(saturation_effect PROPERTIES AUTOMOC ON)
