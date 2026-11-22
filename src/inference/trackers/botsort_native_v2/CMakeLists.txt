cmake_minimum_required(VERSION 3.18)
project(botsort_native_v2 LANGUAGES CXX)

# BoT-SORT tracker v2: the whole tracking stage in C++ (multi-source wrapper,
# lineage, mask compaction) with every parameter read from YAML.
#
# Targets:
#   botsort_core_v2     static library (include/botsort_v2/*.hpp), usable from C++
#   botsort_v2_track    standalone CSV-in / CSV-out runner, no Python
#   botsort_native_v2   Python module for src/inference/trackers/botsort_hungarian_v2.py
#                       (skip with -DBOTSORT_V2_PYTHON=OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BOTSORT_V2_PYTHON "Build the botsort_native_v2 Python module" ON)
# Camera motion compensation links OpenCV. It must be a *static* OpenCV built
# with PIC (build.sh builds a minimal one into <repo>/build/opencv_static and
# passes OpenCV_DIR): a shared OpenCV would bring the shared libstdc++ back into
# the Python module, which the link options below exist to avoid.
option(BOTSORT_V2_CMC "Build camera motion compensation (needs a static OpenCV)" ON)

include(FetchContent)

# Eigen: headers only, for the 8x8 Kalman math.
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

# yaml-cpp, always fetched and linked statically: the Python module must not
# share a C++ runtime or unwinder with the rest of the pipeline process (see the
# link options of botsort_native_v2 below), and a static build also means no apt
# package is needed in the image.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(yaml-cpp)
if(TARGET yaml-cpp::yaml-cpp)
  set(BOTSORT_V2_YAML yaml-cpp::yaml-cpp)
else()
  set(BOTSORT_V2_YAML yaml-cpp)
endif()

add_library(botsort_core_v2 STATIC
  src/settings.cpp
  src/multi_source_tracker.cpp
  src/gmc.cpp
)
target_include_directories(botsort_core_v2
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PUBLIC ${eigen_SOURCE_DIR}
)
target_link_libraries(botsort_core_v2 PUBLIC ${BOTSORT_V2_YAML})
if(BOTSORT_V2_CMC)
  set(OpenCV_STATIC ON)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d)
  if(NOT OpenCV_SHARED STREQUAL "OFF" AND OpenCV_SHARED)
    message(FATAL_ERROR "BOTSORT_V2_CMC needs a static OpenCV (found a shared one at "
                        "${OpenCV_DIR}); let build.sh build one or pass -DBOTSORT_V2_CMC=OFF")
  endif()
  # Only gmc.cpp includes OpenCV, so its headers stay private to the library.
  target_include_directories(botsort_core_v2 PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(botsort_core_v2 PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(botsort_core_v2 PRIVATE BOTSORT_V2_HAS_OPENCV)
  message(STATUS "botsort_v2: camera motion compensation with OpenCV ${OpenCV_VERSION}")
else()
  message(STATUS "botsort_v2: camera motion compensation disabled (BOTSORT_V2_CMC=OFF)")
endif()

add_executable(botsort_v2_track tools/botsort_v2_track.cpp)
target_link_libraries(botsort_v2_track PRIVATE botsort_core_v2)

if(BOTSORT_V2_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  # Use the selected Python interpreter for pybind11.
  set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(botsort_native_v2 src/bindings.cpp)
  target_link_libraries(botsort_native_v2 PRIVATE botsort_core_v2)
  # GStreamer pulls libunwind.so.8 into the pipeline process, and its global
  # _Unwind_RaiseException then shadows libgcc's: every C++ throw through the
  # shared libstdc++ aborts the process instead of raising. A private static libstdc++/libgcc with their symbols kept local
  # gives this module its own unwinder, so a bad option raises ValueError.
  target_link_options(botsort_native_v2 PRIVATE
    -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
endif()

set_target_properties(botsort_v2_track PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(BOTSORT_V2_PYTHON)
  set_target_properties(botsort_native_v2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
