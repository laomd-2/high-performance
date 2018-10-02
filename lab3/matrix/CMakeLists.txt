cmake_minimum_required(VERSION 3.12)
project(matrix C)

set(CMAKE_C_STANDARD 99)
include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(matrix main.c)
add_executable(test_ test.c)
target_link_libraries(matrix msmpi)
target_link_libraries(test_ msmpi)