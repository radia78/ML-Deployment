cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(obj_seg)

# REQUIRED PACKAGES
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

# Cmake flags??
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add the relevant cpp files
add_executable(obj_seg main.cpp person_segmentator/person_segmentator.cpp)

# include opencv headers
target_include_directories(obj_seg PRIVATE ${OpenCV_INCLUDE_DIRS})

# include libraries
target_link_libraries(obj_seg PUBLIC "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" TorchVision::TorchVision)

set_property(TARGET obj_seg PROPERTY CXX_STANDARD 17)