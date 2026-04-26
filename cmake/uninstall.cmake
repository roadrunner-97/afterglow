# Driver for the `uninstall` custom target.  Reads the install_manifest.txt
# CMake writes during `install` and removes every listed path.  Invoked as:
#
#   cmake -DMANIFEST=<build>/install_manifest.txt -P uninstall.cmake
#
# (Run via sudo if the matching install was sudo.)

if(NOT DEFINED MANIFEST)
    message(FATAL_ERROR "MANIFEST not set; run via the `uninstall` target")
endif()

if(NOT EXISTS "${MANIFEST}")
    message(FATAL_ERROR
        "No install manifest at ${MANIFEST}.\n"
        "Run `cmake --install <build>` first — there is nothing to uninstall.")
endif()

file(STRINGS "${MANIFEST}" _files)
foreach(_file IN LISTS _files)
    if(EXISTS "${_file}" OR IS_SYMLINK "${_file}")
        message(STATUS "Removing: ${_file}")
        file(REMOVE "${_file}")
    else()
        message(STATUS "Already gone: ${_file}")
    endif()
endforeach()
