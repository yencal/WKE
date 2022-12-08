# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/projects/opt/rhel7/ppc64le/cmake/3.19.2/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/projects/opt/rhel7/ppc64le/cmake/3.19.2/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /vast/home/stevenw/WKE/CMakeFiles /vast/home/stevenw/WKE//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /vast/home/stevenw/WKE/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named distclean

# Build rule for target.
distclean: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 distclean
.PHONY : distclean

# fast build rule for target.
distclean/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/CMakeFiles/distclean.dir/build.make MATAR/CMakeFiles/distclean.dir/build
.PHONY : distclean/fast

#=============================================================================
# Target rules for targets named matar

# Build rule for target.
matar: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 matar
.PHONY : matar

# fast build rule for target.
matar/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/src/CMakeFiles/matar.dir/build.make MATAR/src/CMakeFiles/matar.dir/build
.PHONY : matar/fast

#=============================================================================
# Target rules for targets named test_shared_ptr

# Build rule for target.
test_shared_ptr: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_shared_ptr
.PHONY : test_shared_ptr

# fast build rule for target.
test_shared_ptr/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/CMakeFiles/test_shared_ptr.dir/build.make MATAR/examples/CMakeFiles/test_shared_ptr.dir/build
.PHONY : test_shared_ptr/fast

#=============================================================================
# Target rules for targets named test_for

# Build rule for target.
test_for: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_for
.PHONY : test_for

# fast build rule for target.
test_for/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/CMakeFiles/test_for.dir/build.make MATAR/examples/CMakeFiles/test_for.dir/build
.PHONY : test_for/fast

#=============================================================================
# Target rules for targets named mtest

# Build rule for target.
mtest: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 mtest
.PHONY : mtest

# fast build rule for target.
mtest/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/CMakeFiles/mtest.dir/build.make MATAR/examples/CMakeFiles/mtest.dir/build
.PHONY : mtest/fast

#=============================================================================
# Target rules for targets named farray_wrong

# Build rule for target.
farray_wrong: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 farray_wrong
.PHONY : farray_wrong

# fast build rule for target.
farray_wrong/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/laplace/CMakeFiles/farray_wrong.dir/build.make MATAR/examples/laplace/CMakeFiles/farray_wrong.dir/build
.PHONY : farray_wrong/fast

#=============================================================================
# Target rules for targets named farray_right

# Build rule for target.
farray_right: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 farray_right
.PHONY : farray_right

# fast build rule for target.
farray_right/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/laplace/CMakeFiles/farray_right.dir/build.make MATAR/examples/laplace/CMakeFiles/farray_right.dir/build
.PHONY : farray_right/fast

#=============================================================================
# Target rules for targets named carray_wrong

# Build rule for target.
carray_wrong: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 carray_wrong
.PHONY : carray_wrong

# fast build rule for target.
carray_wrong/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/laplace/CMakeFiles/carray_wrong.dir/build.make MATAR/examples/laplace/CMakeFiles/carray_wrong.dir/build
.PHONY : carray_wrong/fast

#=============================================================================
# Target rules for targets named carray_right

# Build rule for target.
carray_right: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 carray_right
.PHONY : carray_right

# fast build rule for target.
carray_right/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/laplace/CMakeFiles/carray_right.dir/build.make MATAR/examples/laplace/CMakeFiles/carray_right.dir/build
.PHONY : carray_right/fast

#=============================================================================
# Target rules for targets named test_floyd

# Build rule for target.
test_floyd: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_floyd
.PHONY : test_floyd

# fast build rule for target.
test_floyd/fast:
	$(MAKE) $(MAKESILENT) -f MATAR/examples/watt-graph/CMakeFiles/test_floyd.dir/build.make MATAR/examples/watt-graph/CMakeFiles/test_floyd.dir/build
.PHONY : test_floyd/fast

#=============================================================================
# Target rules for targets named RD

# Build rule for target.
RD: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 RD
.PHONY : RD

# fast build rule for target.
RD/fast:
	$(MAKE) $(MAKESILENT) -f 3wke/CMakeFiles/RD.dir/build.make 3wke/CMakeFiles/RD.dir/build
.PHONY : RD/fast

#=============================================================================
# Target rules for targets named FVS

# Build rule for target.
FVS: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 FVS
.PHONY : FVS

# fast build rule for target.
FVS/fast:
	$(MAKE) $(MAKESILENT) -f 3wke/CMakeFiles/FVS.dir/build.make 3wke/CMakeFiles/FVS.dir/build
.PHONY : FVS/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... distclean"
	@echo "... FVS"
	@echo "... RD"
	@echo "... carray_right"
	@echo "... carray_wrong"
	@echo "... farray_right"
	@echo "... farray_wrong"
	@echo "... matar"
	@echo "... mtest"
	@echo "... test_floyd"
	@echo "... test_for"
	@echo "... test_shared_ptr"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

