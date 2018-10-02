cmake_minimum_required(VERSION 3.12)
project(matrix C)

set(CMAKE_C_STANDARD 99)
include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
include_directories("common")
link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(matrix dense/main.c)
target_link_libraries(matrix msmpi)