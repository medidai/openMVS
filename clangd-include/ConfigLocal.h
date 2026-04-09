// Fallback for clangd / IDE parsing when no CMake build tree exists yet.
// A full configure generates ${CMAKE_BINARY_DIR}/ConfigLocal.h from
// build/Templates/ConfigLocal.h.in (see root CMakeLists.txt).

#define OpenMVS_MAJOR_VERSION 2
#define OpenMVS_MINOR_VERSION 3
#define OpenMVS_PATCH_VERSION 0

#define _HAS_EXCEPTIONS 1
#define _HAS_RTTI 1

#define _USE_OPENMP
#define _USE_BOOST
#define _USE_EIGEN

#define _USE_JPG
#define _USE_PNG
#define _USE_TIFF
