cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project (wke)

set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/MATAR/src/install-kokkos-pthreads/kokkos/lib64/cmake/Kokkos ${Kokkos_DIR})
find_package(Kokkos REQUIRED)
add_definitions(-DHAVE_KOKKOS=1)
#add_definitions(-DHAVE_CUDA=1)


add_executable(FVS 3wke_FVS.cpp)
target_link_libraries(FVS ${CMAKE_SOURCE_DIR}/MATAR/build-kokkos-pthreads/src/libmatar.a)
target_link_libraries(FVS Kokkos::kokkos)


add_executable(RD RD.cpp)
target_link_libraries(RD ${CMAKE_SOURCE_DIR}/MATAR/build-kokkos-pthreads/src/libmatar.a)
target_link_libraries(RD Kokkos::kokkos)
