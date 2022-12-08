# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /projects/opt/rhel7/ppc64le/cmake/3.19.2/bin/cmake

# The command to remove a file.
RM = /projects/opt/rhel7/ppc64le/cmake/3.19.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /vast/home/stevenw/WKE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /vast/home/stevenw/WKE

# Include any dependencies generated for this target.
include 3wke/CMakeFiles/RD.dir/depend.make

# Include the progress variables for this target.
include 3wke/CMakeFiles/RD.dir/progress.make

# Include the compile flags for this target's objects.
include 3wke/CMakeFiles/RD.dir/flags.make

3wke/CMakeFiles/RD.dir/RD.cpp.o: 3wke/CMakeFiles/RD.dir/flags.make
3wke/CMakeFiles/RD.dir/RD.cpp.o: 3wke/RD.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/vast/home/stevenw/WKE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3wke/CMakeFiles/RD.dir/RD.cpp.o"
	cd /vast/home/stevenw/WKE/3wke && /projects/opt/ppc64le/p9/gcc/9.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RD.dir/RD.cpp.o -c /vast/home/stevenw/WKE/3wke/RD.cpp

3wke/CMakeFiles/RD.dir/RD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RD.dir/RD.cpp.i"
	cd /vast/home/stevenw/WKE/3wke && /projects/opt/ppc64le/p9/gcc/9.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /vast/home/stevenw/WKE/3wke/RD.cpp > CMakeFiles/RD.dir/RD.cpp.i

3wke/CMakeFiles/RD.dir/RD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RD.dir/RD.cpp.s"
	cd /vast/home/stevenw/WKE/3wke && /projects/opt/ppc64le/p9/gcc/9.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /vast/home/stevenw/WKE/3wke/RD.cpp -o CMakeFiles/RD.dir/RD.cpp.s

# Object files for target RD
RD_OBJECTS = \
"CMakeFiles/RD.dir/RD.cpp.o"

# External object files for target RD
RD_EXTERNAL_OBJECTS =

3wke/RD: 3wke/CMakeFiles/RD.dir/RD.cpp.o
3wke/RD: 3wke/CMakeFiles/RD.dir/build.make
3wke/RD: MATAR/build-kokkos-pthreads/src/libmatar.a
3wke/RD: MATAR/src/install-kokkos-pthreads/kokkos/lib64/libkokkoscontainers.a
3wke/RD: MATAR/src/install-kokkos-pthreads/kokkos/lib64/libkokkoscore.a
3wke/RD: /usr/lib64/libpthread.so
3wke/RD: MATAR/src/install-kokkos-pthreads/kokkos/lib64/libkokkossimd.a
3wke/RD: 3wke/CMakeFiles/RD.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/vast/home/stevenw/WKE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RD"
	cd /vast/home/stevenw/WKE/3wke && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RD.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3wke/CMakeFiles/RD.dir/build: 3wke/RD

.PHONY : 3wke/CMakeFiles/RD.dir/build

3wke/CMakeFiles/RD.dir/clean:
	cd /vast/home/stevenw/WKE/3wke && $(CMAKE_COMMAND) -P CMakeFiles/RD.dir/cmake_clean.cmake
.PHONY : 3wke/CMakeFiles/RD.dir/clean

3wke/CMakeFiles/RD.dir/depend:
	cd /vast/home/stevenw/WKE && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /vast/home/stevenw/WKE /vast/home/stevenw/WKE/3wke /vast/home/stevenw/WKE /vast/home/stevenw/WKE/3wke /vast/home/stevenw/WKE/3wke/CMakeFiles/RD.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3wke/CMakeFiles/RD.dir/depend
