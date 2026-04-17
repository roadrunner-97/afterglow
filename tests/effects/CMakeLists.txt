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

# ---------------------------------------------------------------------------
# Helper macro — all effect tests now use QTEST_MAIN (QApplication) with
# QT_QPA_PLATFORM=offscreen so createControlsWidget() can run without a display.
# ---------------------------------------------------------------------------
macro(add_effect_test name src effect_lib)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE
        ${effect_lib}
        photoeditor_core
        photoeditor_widgets
        Qt6::Test Qt6::Widgets Qt6::Gui Qt6::Core
    )
    set_target_properties(${name} PROPERTIES
        AUTOMOC ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
    add_test(NAME ${ARGV3} COMMAND ${name})
    set_tests_properties(${ARGV3} PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    )
endmacro()

add_effect_test(test_brightness   test_brightness.cpp   brightness_effect   Effects_Brightness)
add_effect_test(test_saturation   test_saturation.cpp   saturation_effect   Effects_Saturation)
add_effect_test(test_grayscale    test_grayscale.cpp    grayscale_effect    Effects_Grayscale)
add_effect_test(test_blur         test_blur.cpp         blur_effect         Effects_Blur)
add_effect_test(test_exposure     test_exposure.cpp     exposure_effect     Effects_Exposure)
add_effect_test(test_hotpixel     test_hotpixel.cpp     hotpixel_effect     Effects_HotPixel)
add_effect_test(test_unsharp      test_unsharp.cpp      unsharp_effect      Effects_Unsharp)
add_effect_test(test_denoise      test_denoise.cpp      denoise_effect      Effects_Denoise)
add_effect_test(test_whitebalance test_whitebalance.cpp whitebalance_effect Effects_WhiteBalance)
add_effect_test(test_vignette     test_vignette.cpp     vignette_effect     Effects_Vignette)
add_effect_test(test_filmgrain    test_filmgrain.cpp    filmgrain_effect    Effects_FilmGrain)
