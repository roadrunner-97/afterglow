find_package(Qt6 COMPONENTS Test REQUIRED)

# ---------------------------------------------------------------------------
# EffectManager tests — no GPU, no display required (GUILESS)
# ---------------------------------------------------------------------------
add_executable(test_effect_manager
    core/test_effect_manager.cpp
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
