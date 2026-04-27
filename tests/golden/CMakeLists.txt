find_package(Qt6 COMPONENTS Test REQUIRED)

# Golden-image regression tests — fixtures live in cases/<name>/.
# The test binary loads each case at runtime; GOLDEN_CASES_DIR points at the
# source-tree fixtures so adding/editing cases doesn't require a rebuild.
add_executable(test_golden test_golden.cpp)
target_link_libraries(test_golden PRIVATE
    photoeditor_core
    photoeditor_widgets
    all_effects
    Qt6::Test Qt6::Widgets Qt6::Gui Qt6::Core
)
target_compile_definitions(test_golden PRIVATE
    GOLDEN_CASES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cases"
)
set_target_properties(test_golden PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Golden COMMAND test_golden)
set_tests_properties(Golden PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)
