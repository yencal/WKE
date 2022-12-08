cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project (3wke)

set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/MATAR/src/install-kokkos-cuda/kokkos/lib64/cmake/Kokkos ${Kokkos_DIR})
find_package(Kokkos REQUIRED)
add_definitions(-DHAVE_KOKKOS=1)
add_definitions(-DHAVE_CUDA=1)
add_executable(3wke 1d.cpp)
target_link_libraries(3wke ${CMAKE_SOURCE_DIR}/MATAR/build-kokkos-cuda/src/libmatar.a)
target_link_libraries(3wke Kokkos::kokkos)
