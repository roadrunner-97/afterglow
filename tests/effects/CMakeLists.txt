find_package(Qt6 COMPONENTS Test REQUIRED)

# Shared helper: bump COMPILE_DEFINITIONS so HAVE_OPENCL is visible
# when effect headers are pulled in through IGpuEffect.h.
# (photoeditor_core already declares it PUBLIC, but be explicit here.)

# ---------------------------------------------------------------------------
# Helper macro to reduce repetition
# ---------------------------------------------------------------------------
# add_effect_test(name sources... LINK_LIBS extra_libs...)
# All tests share:
#   - photoeditor_core  (propagates OpenCL::OpenCL + Qt6 transitive deps)
#   - Qt6::Test + Qt6::Gui
#   - AUTOMOC ON
#   - RUNTIME_OUTPUT_DIRECTORY in build/bin/tests
# ---------------------------------------------------------------------------

# Brightness
add_executable(test_brightness test_brightness.cpp)
target_link_libraries(test_brightness PRIVATE
    brightness_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_brightness PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Brightness COMMAND test_brightness)

# Saturation
add_executable(test_saturation test_saturation.cpp)
target_link_libraries(test_saturation PRIVATE
    saturation_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_saturation PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Saturation COMMAND test_saturation)

# Grayscale — requires full QApplication for QWidget; uses offscreen platform.
add_executable(test_grayscale test_grayscale.cpp)
target_link_libraries(test_grayscale PRIVATE
    grayscale_effect
    photoeditor_core
    photoeditor_widgets
    Qt6::Test Qt6::Widgets Qt6::Gui Qt6::Core
)
set_target_properties(test_grayscale PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Grayscale COMMAND test_grayscale)
set_tests_properties(Effects_Grayscale PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# Blur
add_executable(test_blur test_blur.cpp)
target_link_libraries(test_blur PRIVATE
    blur_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_blur PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Blur COMMAND test_blur)

# Exposure
add_executable(test_exposure test_exposure.cpp)
target_link_libraries(test_exposure PRIVATE
    exposure_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_exposure PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Exposure COMMAND test_exposure)

# HotPixel
add_executable(test_hotpixel test_hotpixel.cpp)
target_link_libraries(test_hotpixel PRIVATE
    hotpixel_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_hotpixel PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_HotPixel COMMAND test_hotpixel)

# Unsharp
add_executable(test_unsharp test_unsharp.cpp)
target_link_libraries(test_unsharp PRIVATE
    unsharp_effect
    photoeditor_core
    Qt6::Test Qt6::Gui Qt6::Core
)
set_target_properties(test_unsharp PROPERTIES
    AUTOMOC ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
add_test(NAME Effects_Unsharp COMMAND test_unsharp)
