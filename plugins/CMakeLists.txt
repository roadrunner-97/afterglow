add_subdirectory(hotpixel)
add_subdirectory(exposure)
add_subdirectory(brightness)
add_subdirectory(saturation)
add_subdirectory(blur)
add_subdirectory(grayscale)
add_subdirectory(unsharp)
add_subdirectory(denoise)

# Aggregator target: link this to get all effects + their include dirs
add_library(all_effects INTERFACE)
target_link_libraries(all_effects INTERFACE
    hotpixel_effect
    exposure_effect
    brightness_effect
    saturation_effect
    blur_effect
    grayscale_effect
    unsharp_effect
    denoise_effect
)
