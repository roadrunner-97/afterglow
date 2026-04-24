# ──────────────────────────────────────────────────────────────────────────────
# add_effect_plugin(ClassName)
#   Creates a static library named <lowercase>_effect from <ClassName>Effect.cpp/.h.
#   Links OpenCL + photoeditor_core + photoeditor_widgets + Qt6::Core/Gui.
#   Sets HAVE_OPENCL and AUTOMOC.
# ──────────────────────────────────────────────────────────────────────────────
function(add_effect_plugin CLASS_NAME)
    string(TOLOWER "${CLASS_NAME}" _target)
    set(_target "${_target}_effect")
    add_library(${_target} STATIC
        ${CLASS_NAME}Effect.cpp
        ${CLASS_NAME}Effect.h
    )
    find_package(OpenCL REQUIRED)
    target_link_libraries(${_target} PRIVATE
        OpenCL::OpenCL
        photoeditor_core
        photoeditor_widgets
        Qt6::Core
        Qt6::Gui
    )
    target_compile_definitions(${_target} PRIVATE HAVE_OPENCL)
    target_include_directories(${_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${_target} PROPERTIES AUTOMOC ON)
endfunction()

add_subdirectory(hotpixel)
add_subdirectory(exposure)
add_subdirectory(whitebalance)
add_subdirectory(brightness)
add_subdirectory(saturation)
add_subdirectory(blur)
add_subdirectory(grayscale)
add_subdirectory(unsharp)
add_subdirectory(denoise)
add_subdirectory(vignette)
add_subdirectory(filmgrain)
add_subdirectory(splittoning)
add_subdirectory(clarity)
add_subdirectory(colorbalance)
add_subdirectory(croprotate)

# Aggregator target: link this to get all effects + their include dirs
add_library(all_effects INTERFACE)
target_link_libraries(all_effects INTERFACE
    hotpixel_effect
    exposure_effect
    whitebalance_effect
    brightness_effect
    saturation_effect
    blur_effect
    grayscale_effect
    unsharp_effect
    denoise_effect
    vignette_effect
    filmgrain_effect
    splittoning_effect
    clarity_effect
    colorbalance_effect
    croprotate_effect
)
