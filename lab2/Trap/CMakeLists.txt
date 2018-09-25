cmake_minimum_required(VERSION 3.12)
project(Trap C)

set(CMAKE_C_STANDARD 99)
include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
add_executable(Trap main.c)
target_link_libraries(Trap msmpi)