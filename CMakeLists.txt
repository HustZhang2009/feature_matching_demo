cmake_minimum_required(VERSION 2.8.3)
project(match_demo)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(match_demo src/main.cpp)
target_link_libraries(match_demo ${OpenCV_LIBRARIES})