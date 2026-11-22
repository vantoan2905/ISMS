#!/usr/bin/env bash
# Configure + build the BoT-SORT v2 tracker (library, standalone runner, Python module).
# Output goes to <repo>/build/botsort_native_v2.
#
# The Python module is built against PYTHON (default: the active env's python3),
# which must be the interpreter that runs the pipeline. BOTSORT_V2_PYTHON=OFF builds
# only the C++ library and botsort_v2_track.
#
# Camera motion compensation (CMC) needs OpenCV. A minimal *static* OpenCV
# (core, imgproc, video, calib3d and their dependencies, no I/O backends) is built
# once into <repo>/build/opencv_static and reused: it has to be static for the same
# reason yaml-cpp is (the module keeps its own libstdc++, see CMakeLists.txt).
# It lives outside build/botsort_native_v2 so wiping the tracker build does not
# rebuild OpenCV. BOTSORT_V2_CMC=OFF skips OpenCV (cmc_method must then be none);
# OPENCV_DIR=<dir with OpenCVConfig.cmake> uses an existing static build instead.
set -euo pipefail
cd "$(dirname "$0")"
REPO_BUILD="$(cd ../../../.. && pwd)/build"
BUILD_DIR="$REPO_BUILD/botsort_native_v2"

PYTHON="${PYTHON:-$(command -v python3)}"
WITH_PYTHON="${BOTSORT_V2_PYTHON:-ON}"
WITH_CMC="${BOTSORT_V2_CMC:-ON}"
OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
if [[ "$WITH_PYTHON" == "ON" ]]; then
    echo "Building against: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
fi

build_static_opencv() {
    local root="$REPO_BUILD/opencv_static"
    local prefix="$root/install-$OPENCV_VERSION"
    if [[ -f "$prefix/.complete" ]]; then
        OPENCV_DIR="$(dirname "$(find "$prefix" -name OpenCVConfig.cmake | head -1)")"
        return
    fi
    echo "Building static OpenCV $OPENCV_VERSION into $prefix (once)"
    local src="$root/opencv-$OPENCV_VERSION"
    if [[ ! -d "$src" ]]; then
        git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv.git "$src"
    fi
    cmake -S "$src" -B "$root/build-$OPENCV_VERSION" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DBUILD_SHARED_LIBS=OFF -DENABLE_PIC=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_LIST=core,imgproc,video,calib3d \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_java_bindings_generator=OFF \
        -DBUILD_ZLIB=ON -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_CUDA=OFF \
        -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF \
        -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_VA=OFF -DWITH_1394=OFF \
        -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF \
        -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF \
        -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF -DCV_TRACE=OFF -DOPENCV_GENERATE_PKGCONFIG=OFF
    cmake --build "$root/build-$OPENCV_VERSION" -j"$(nproc)"
    cmake --install "$root/build-$OPENCV_VERSION"
    touch "$prefix/.complete"
    OPENCV_DIR="$(dirname "$(find "$prefix" -name OpenCVConfig.cmake | head -1)")"
}

CMC_ARGS=(-DBOTSORT_V2_CMC="$WITH_CMC")
if [[ "$WITH_CMC" == "ON" ]]; then
    if [[ -z "${OPENCV_DIR:-}" ]]; then
        build_static_opencv
    fi
    CMC_ARGS+=(-DOpenCV_DIR="$OPENCV_DIR")
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PYTHON" -DBOTSORT_V2_PYTHON="$WITH_PYTHON" "${CMC_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Built:"
ls -1 "$BUILD_DIR"/botsort_v2_track "$BUILD_DIR"/botsort_native_v2*.so 2>/dev/null || true
