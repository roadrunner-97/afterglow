find_package(Qt6 COMPONENTS Test REQUIRED)

add_executable(test_linear_gradient_mask core/test_linear_gradient_mask.cpp)
target_link_libraries(test_linear_gradient_mask PRIVATE photoeditor_core Qt6::Test Qt6::Core)
set_target_properties(test_linear_gradient_mask PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME LinearGradientMask COMMAND test_linear_gradient_mask)

add_executable(test_local_adjustment core/test_local_adjustment.cpp)
target_link_libraries(test_local_adjustment PRIVATE photoeditor_core Qt6::Test Qt6::Core)
set_target_properties(test_local_adjustment PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME LocalAdjustment COMMAND test_local_adjustment)

add_executable(test_linear_gradient_tool ui/test_linear_gradient_tool.cpp)
target_link_libraries(test_linear_gradient_tool PRIVATE photoeditor_ui photoeditor_core Qt6::Test Qt6::Widgets Qt6::Gui)
target_include_directories(test_linear_gradient_tool PRIVATE ${CMAKE_SOURCE_DIR}/src/ui)
set_target_properties(test_linear_gradient_tool PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME LinearGradientTool COMMAND test_linear_gradient_tool)
set_tests_properties(LinearGradientTool PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_cache_purger core/test_cache_purger.cpp)
target_link_libraries(test_cache_purger PRIVATE photoeditor_core Qt6::Test Qt6::Core)
set_target_properties(test_cache_purger PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME CachePurger COMMAND test_cache_purger)

add_executable(test_gallery_loupe
    ui/test_gallery_loupe.cpp
)
target_link_libraries(test_gallery_loupe PRIVATE
    photoeditor_ui photoeditor_core photoeditor_widgets
    Qt6::Test Qt6::Widgets Qt6::Gui Qt6::Core Qt6::OpenGLWidgets OpenGL::GL
)
target_include_directories(test_gallery_loupe PRIVATE ${CMAKE_SOURCE_DIR}/src/ui)
set_target_properties(test_gallery_loupe PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME GalleryLoupe COMMAND test_gallery_loupe)
set_tests_properties(GalleryLoupe PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    RESOURCE_LOCK opencl_jit
)

# ---------------------------------------------------------------------------
# EffectManager tests — no GPU, no display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_effect_manager
    core/test_effect_manager.cpp
)

# ---------------------------------------------------------------------------
# ProofCache tests — disk + in-memory LRU, no GPU required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_proof_cache
    core/test_proof_cache.cpp
)
target_link_libraries(test_proof_cache
    PRIVATE
        photoeditor_core
        Qt6::Test Qt6::Core Qt6::Gui
)
set_target_properties(test_proof_cache PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME ProofCache COMMAND test_proof_cache)

# ---------------------------------------------------------------------------
# Proofer queue-management tests — no GPU required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_proofer
    core/test_proofer.cpp
)
target_link_libraries(test_proofer
    PRIVATE
        photoeditor_core
        brightness_effect
        croprotate_effect
        Qt6::Test Qt6::Core Qt6::Gui
)
set_target_properties(test_proofer PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Proofer COMMAND test_proofer)

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
    RESOURCE_LOCK opencl_jit
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
# Editor UI transition policy — pure headless state-machine tests
# ---------------------------------------------------------------------------
add_executable(test_editor_ui_state core/test_editor_ui_state.cpp)
target_link_libraries(test_editor_ui_state PRIVATE photoeditor_core Qt6::Test Qt6::Core)
set_target_properties(test_editor_ui_state PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME EditorUiState COMMAND test_editor_ui_state)

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
# ExportPath tests — pure CPU path-resolution logic (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_export_path
    core/test_export_path.cpp
)
target_link_libraries(test_export_path
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
)
set_target_properties(test_export_path PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME ExportPath COMMAND test_export_path)

# ---------------------------------------------------------------------------
# ExportResize tests — pure CPU resize sizing logic
# ---------------------------------------------------------------------------
add_executable(test_export_resize
    core/test_export_resize.cpp
)
target_link_libraries(test_export_resize
    PRIVATE
        photoeditor_core
        Qt6::Test
        Qt6::Core
        Qt6::Gui
)
set_target_properties(test_export_resize PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME ExportResize COMMAND test_export_resize)

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
# LogarithmParamSlider tests — widget tests, need offscreen QApplication
# ---------------------------------------------------------------------------
add_executable(test_logarithm_param_slider
    widgets/test_logarithm_param_slider.cpp
)
target_link_libraries(test_logarithm_param_slider
    PRIVATE
        photoeditor_widgets
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
)
set_target_properties(test_logarithm_param_slider PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME LogarithmParamSlider COMMAND test_logarithm_param_slider)
set_tests_properties(LogarithmParamSlider PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# UndoHistory tests — pure CPU, no GPU/display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_undo_history
    core/test_undo_history.cpp
)
target_link_libraries(test_undo_history
    PRIVATE
        photoeditor_core
        Qt6::Test Qt6::Core Qt6::Gui
)
set_target_properties(test_undo_history PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME UndoHistory COMMAND test_undo_history)

# ---------------------------------------------------------------------------
# HistorySerializer tests — pure CPU, no GPU/display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_history_serializer
    core/test_history_serializer.cpp
)
target_link_libraries(test_history_serializer
    PRIVATE
        photoeditor_core
        Qt6::Test Qt6::Core Qt6::Gui
)
set_target_properties(test_history_serializer PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME HistorySerializer COMMAND test_history_serializer)

# ---------------------------------------------------------------------------
# HistoryTray tests — widget tests, need offscreen QApplication
# ---------------------------------------------------------------------------
add_executable(test_history_tray
    widgets/test_history_tray.cpp
)
target_link_libraries(test_history_tray
    PRIVATE
        photoeditor_widgets
        Qt6::Test Qt6::Core Qt6::Gui Qt6::Widgets
)
set_target_properties(test_history_tray PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME HistoryTray COMMAND test_history_tray)
set_tests_properties(HistoryTray PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# MetadataTray tests — widget tests, need offscreen QApplication
# ---------------------------------------------------------------------------
add_executable(test_metadata_tray
    widgets/test_metadata_tray.cpp
)
target_link_libraries(test_metadata_tray
    PRIVATE
        photoeditor_widgets
        Qt6::Test Qt6::Core Qt6::Gui Qt6::Widgets
)
set_target_properties(test_metadata_tray PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME MetadataTray COMMAND test_metadata_tray)
set_tests_properties(MetadataTray PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# Effect integration tests — GPU (OpenCL) required; skip gracefully if absent
# ---------------------------------------------------------------------------
add_subdirectory(effects)

# ---------------------------------------------------------------------------
# Golden-image regression tests — fixtures in tests/golden/cases/<name>/
# ---------------------------------------------------------------------------
add_subdirectory(golden)
