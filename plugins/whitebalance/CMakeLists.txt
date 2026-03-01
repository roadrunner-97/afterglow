cmake_minimum_required(VERSION 3.16)

add_library(whitebalance_effect STATIC WhiteBalanceEffect.cpp WhiteBalanceEffect.h)

find_package(OpenCL REQUIRED)
target_link_libraries(whitebalance_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(whitebalance_effect PRIVATE HAVE_OPENCL)

target_link_libraries(whitebalance_effect
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(whitebalance_effect
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(whitebalance_effect PROPERTIES AUTOMOC ON)
