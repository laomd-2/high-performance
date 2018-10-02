cmake_minimum_required(VERSION 3.12)
project(matrix C)

set(CMAKE_C_STANDARD 99)
include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(matrix main.c mympi.h)
target_link_libraries(matrix msmpi)