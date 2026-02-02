#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create tmp directory for building dependencies from source
mkdir -p tmp

echo "=== Installing system dependencies ==="
sudo apt-get update -yq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    build-essential \
    git \
    cmake \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-serialization-dev \
    libboost-thread-dev \
    libgmp-dev \
    libmpfr-dev \
    zlib1g-dev \
    python3-dev \
    libjxl-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libopenexr-dev

# Build Eigen 3.4 from source
echo "=== Building Eigen ==="
if [ ! -d "tmp/eigen" ]; then
    git clone https://gitlab.com/libeigen/eigen --branch 3.4 tmp/eigen
fi
mkdir -p tmp/eigen_build
cd tmp/eigen_build
cmake ../eigen
sudo make install
cd "$SCRIPT_DIR"

# Build OpenCV from source (need 4.12+ for IMWRITE_JPEGXL_QUALITY support)
echo "=== Building OpenCV ==="
if [ ! -d "tmp/opencv" ]; then
    git clone https://github.com/opencv/opencv --branch 4.12.0 --depth 1 tmp/opencv
fi
if [ ! -d "tmp/opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib --branch 4.12.0 --depth 1 tmp/opencv_contrib
fi
mkdir -p tmp/opencv_build
cd tmp/opencv_build
cmake ../opencv \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DWITH_TIFF=ON \
    -DWITH_OPENEXR=ON \
    -DWITH_JPEGXL=ON \
    -DWITH_EIGEN=ON \
    -DWITH_GTK=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF
make -j"$(nproc)"
sudo make install
sudo ldconfig
cd "$SCRIPT_DIR"

# Build CGAL from source
echo "=== Building CGAL ==="
if [ ! -d "tmp/cgal" ]; then
    git clone https://github.com/cgal/cgal --branch=v6.0.1 tmp/cgal
fi
mkdir -p tmp/cgal_build
cd tmp/cgal_build
cmake ../cgal
sudo make install
cd "$SCRIPT_DIR"

# Clone VCGLib
echo "=== Cloning VCGLib ==="
if [ ! -d "tmp/vcglib" ]; then
    git clone https://github.com/cdcseacave/VCG.git tmp/vcglib
fi

# Build OpenMVS
echo "=== Building OpenMVS ==="
mkdir -p openMVS_build
cd openMVS_build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DVCG_ROOT="$SCRIPT_DIR/tmp/vcglib" \
    -DOpenMVS_USE_CUDA=OFF \
    -DOpenMVS_USE_PYTHON=OFF \
    -DOpenMVS_USE_BREAKPAD=OFF

make -j"$(nproc)"

echo "=== Build complete ==="
echo "Binaries are in: $SCRIPT_DIR/openMVS_build/bin"
ls -la bin/ 2>/dev/null || echo "Note: binaries may be in a different location"
