find_package(benchmark CONFIG REQUIRED)

add_executable(afterglow_effect_benchmarks effect_benchmarks.cpp)
target_link_libraries(afterglow_effect_benchmarks PRIVATE
    benchmark::benchmark
    photoeditor_core
    photoeditor_widgets
    all_effects
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Concurrent
    OpenCL::OpenCL
)
set_target_properties(afterglow_effect_benchmarks PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
