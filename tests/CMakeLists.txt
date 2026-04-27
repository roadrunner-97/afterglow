find_package(Qt6 COMPONENTS Test REQUIRED)

# ---------------------------------------------------------------------------
# EffectManager tests — no GPU, no display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_effect_manager
    core/test_effect_manager.cpp
)

# ---------------------------------------------------------------------------
# GpuPipeline tests — exercises run(), initGpuKernels(), enqueueGpu() for
# all 7 effects plus the viewportOnly fast path.
# ---------------------------------------------------------------------------
add_executable(test_gpu_pipeline
    core/test_gpu_pipeline.cpp
)
target_link_libraries(test_gpu_pipeline PRIVATE
    photoeditor_core
    photoeditor_widgets
    all_effects
    Qt6::Test Qt6::Widgets Qt6::Gui Qt6::Core
)
set_target_properties(test_gpu_pipeline PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME GpuPipeline COMMAND test_gpu_pipeline)
set_tests_properties(GpuPipeline PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)
target_link_libraries(test_effect_manager
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
)
set_target_properties(test_effect_manager PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME EffectManager COMMAND test_effect_manager)

# ---------------------------------------------------------------------------
# SettingsExporter tests — pure CPU, no GPU/display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_settings_exporter
    core/test_settings_exporter.cpp
)
target_link_libraries(test_settings_exporter
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
        Qt6::Gui
)
set_target_properties(test_settings_exporter PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME SettingsExporter COMMAND test_settings_exporter)

# ---------------------------------------------------------------------------
# SettingsImporter tests — pure CPU, no GPU/display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_settings_importer
    core/test_settings_importer.cpp
)
target_link_libraries(test_settings_importer
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
        Qt6::Gui
)
set_target_properties(test_settings_importer PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME SettingsImporter COMMAND test_settings_importer)

# ---------------------------------------------------------------------------
# Effect applyParameters() round-trip tests — widget tests, offscreen
# ---------------------------------------------------------------------------
add_executable(test_effect_apply_parameters
    core/test_effect_apply_parameters.cpp
)
target_link_libraries(test_effect_apply_parameters
    PRIVATE
        photoeditor_core
        photoeditor_widgets
        all_effects
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
)
set_target_properties(test_effect_apply_parameters PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME EffectApplyParameters COMMAND test_effect_apply_parameters)
set_tests_properties(EffectApplyParameters PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# Histogram tests — no GPU, no display; pure CPU image math
# ---------------------------------------------------------------------------
add_executable(test_histogram
    core/test_histogram.cpp
)
target_link_libraries(test_histogram
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
        Qt6::Gui
)
set_target_properties(test_histogram PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Histogram COMMAND test_histogram)

# ---------------------------------------------------------------------------
# ParamSlider tests — widget tests, need offscreen QApplication
# ---------------------------------------------------------------------------
add_executable(test_param_slider
    widgets/test_param_slider.cpp
)
target_link_libraries(test_param_slider
    PRIVATE
        photoeditor_widgets
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
)
set_target_properties(test_param_slider PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME ParamSlider COMMAND test_param_slider)
set_tests_properties(ParamSlider PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# Effect integration tests — GPU (OpenCL) required; skip gracefully if absent
# ---------------------------------------------------------------------------
add_subdirectory(effects)
