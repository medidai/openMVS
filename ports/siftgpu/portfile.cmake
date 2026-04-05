set(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/source")

if(NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR "Local source not found at ${SOURCE_PATH}")
endif()

set(ENABLE_CUDA OFF)
if("cuda" IN_LIST FEATURES)
    set(ENABLE_CUDA ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    -DCUDA_ENABLED=${ENABLE_CUDA}
)
vcpkg_cmake_build()
vcpkg_cmake_install()
vcpkg_copy_pdbs()

# Remove duplicate headers from debug directory
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_cmake_config_fixup(PACKAGE_NAME siftgpu CONFIG_PATH lib/cmake/siftgpu)
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
