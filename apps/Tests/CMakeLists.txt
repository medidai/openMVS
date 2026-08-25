if(OpenMVS_BUILD_SFM)
	if(MSVC)
		create_rc_files(Tests)
		FILE(GLOB LIBRARY_FILES_C "*.cpp" "${CMAKE_CURRENT_BINARY_DIR}/*.rc")
	else()
		FILE(GLOB LIBRARY_FILES_C "*.cpp")
	endif()
	FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")
	SET(TEST_LIBS "SFM;MVS")
else()
	SET(LIBRARY_FILES_C "TestsMVS.cpp" "TestsMVSOnly.cpp")
	SET(LIBRARY_FILES_H "TestsMVS.h")
	SET(TEST_LIBS "MVS")
endif()

ADD_DEFINITIONS(-D_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")

cxx_executable_with_flags(Tests "Apps" "${cxx_default}" "${TEST_LIBS}" ${LIBRARY_FILES_C} ${LIBRARY_FILES_H})

# Install
INSTALL(TARGETS Tests
	EXPORT OpenMVSTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
