cmake_minimum_required(VERSION 3.12)
project(cuda CXX)

set(CMAKE_CXX_STANDARD 11)

include_directories("D:/Program Files/Cuda Toolkits/include")

add_executable(memory memory/memory.cpp)
add_executable(vector vector/main.cpp)