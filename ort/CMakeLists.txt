cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(obj_seg)

# REQUIRED PACKAGES
find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

# Add the executable scripts
add_executable(obj_seg main.cpp person_segmentator/person_segmentator.cpp)

# include opencv headers
target_include_directories(obj_seg PRIVATE ${OpenCV_INCLUDE_DIRS})

# include libraries
target_link_libraries(obj_seg PRIVATE "${OpenCV_LIBS}" onnxruntime::onnxruntime)

set_property(TARGET obj_seg PROPERTY CXX_STANDARD 17)