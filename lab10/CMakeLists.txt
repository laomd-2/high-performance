cmake_minimum_required(VERSION 3.13)
project(lab10)

set(CMAKE_CXX_STANDARD  11)
include_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Include")
include_directories("D:/Program Files/CUDA/include")

link_directories("D:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(serial serial/main.cpp)
add_executable(cuda cuda/main.cpp)
add_executable(cannon mpi/cannon.cpp)
add_executable(dns mpi/dns.cpp)

target_link_libraries(cannon msmpi)
target_link_libraries(dns msmpi)
