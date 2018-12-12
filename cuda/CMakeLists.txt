cmake_minimum_required(VERSION 3.12)
project(cuda CXX)

set(CMAKE_CXX_STANDARD 11)

include_directories("D:/Program Files/Cuda Toolkits/include")

add_executable(hello hello/main.cpp)
add_executable(memory memory/main.cpp)
add_executable(vector vector/main.cpp)
add_executable(matrix matrix/main.cpp)
add_executable(matrix_host matrix/host_s.cpp)
add_executable(matrix_vector matrix_vector/main.cpp)