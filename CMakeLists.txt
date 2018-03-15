cmake_minimum_required(VERSION 2.8.3)
project(affine_photometric_klt)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV 3.1.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(affine_photometric_klt main.cpp src/klt.cpp)
target_link_libraries(affine_photometric_klt ${OpenCV_LIBS})