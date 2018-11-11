cmake_minimum_required(VERSION 3.12)
project(lab6)

set(CMAKE_CXX_STANDARD 14)

include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
include_directories("common")
link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(lab6 main.cpp)
add_executable(get_data get_data.cpp)
target_link_libraries(lab6 msmpi)