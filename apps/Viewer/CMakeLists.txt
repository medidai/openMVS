if(NOT VIEWER_NAME)
	set(VIEWER_NAME "Viewer")
endif()

# Find required packages
FIND_PACKAGE(glad QUIET)
if(glad_FOUND)
	MESSAGE(STATUS "GLAD ${glad_VERSION} found")
else()
	MESSAGE("-- Can't find GLAD. Continuing without it.")
	RETURN()
endif()
FIND_PACKAGE(glfw3 QUIET)
if(glfw3_FOUND)
	MESSAGE(STATUS "GLFW3 ${glfw3_VERSION} found")
else()
	MESSAGE("-- Can't find GLFW3. Continuing without it.")
	RETURN()
endif()
FIND_PACKAGE(imgui QUIET)
if(imgui_FOUND)
	MESSAGE(STATUS "ImGUI ${imgui_VERSION} found")
else()
	MESSAGE("-- Can't find ImGUI. Continuing without it.")
	RETURN()
endif()
find_path(PORTABLE_FILE_DIALOGS_INCLUDE_DIRS "portable-file-dialogs.h")
if(PORTABLE_FILE_DIALOGS_INCLUDE_DIRS)
	include_directories(${PORTABLE_FILE_DIALOGS_INCLUDE_DIRS})
	MESSAGE(STATUS "portable-file-dialogs found")
else()
	MESSAGE("-- Can't find portable-file-dialogs. Using fallback implementation.")
	RETURN()
endif()

# List sources files
if(MSVC)
	create_rc_files(${VIEWER_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/Viewer.ico")
	FILE(GLOB LIBRARY_FILES_C "*.cpp" "${CMAKE_CURRENT_BINARY_DIR}/*.rc")
else()
	FILE(GLOB LIBRARY_FILES_C "*.cpp")
endif()
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")
FILE(GLOB SHADERS_LIBRARY_FILES "Shaders/*.*")
SOURCE_GROUP("Shaders" FILES ${SHADERS_LIBRARY_FILES})

cxx_executable_with_flags(${VIEWER_NAME} "Apps" "${cxx_default}" "MVS;glad::glad;${GLFW_STATIC_LIBRARIES};${glfw3_LIBRARY};${GLFW3_LIBRARY};glfw;imgui::imgui" ${LIBRARY_FILES_C} ${LIBRARY_FILES_H})

# Link portable-file-dialogs if available
if(portable-file-dialogs_FOUND)
	target_link_libraries(${VIEWER_NAME} PRIVATE portable-file-dialogs::portable-file-dialogs)
endif()

# Manually set Common.h as the precompiled header
IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16.0)
	TARGET_PRECOMPILE_HEADERS(${VIEWER_NAME} PRIVATE "Common.h")
endif()

# Install
INSTALL(TARGETS ${VIEWER_NAME}
	EXPORT OpenMVSTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
