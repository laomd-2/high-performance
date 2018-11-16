cmake_minimum_required(VERSION 3.12)
project(lab6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories("common")

add_executable(lab6 main.cpp)
target_link_libraries(lab6 ${MPI_LIBRARIES})