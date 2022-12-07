cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project (WKE)

include_directories(MATAR/src)
add_subdirectory(MATAR)

include_directories(3wke)
add_subdirectory(3wke)
