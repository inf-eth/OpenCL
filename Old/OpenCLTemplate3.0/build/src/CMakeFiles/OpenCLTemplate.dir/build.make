# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/root/source/repos/OpenCLTemplate3.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build

# Include any dependencies generated for this target.
include src/CMakeFiles/OpenCLTemplate.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/OpenCLTemplate.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/OpenCLTemplate.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/OpenCLTemplate.dir/flags.make

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/flags.make
src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o: /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate.cpp
src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o -MF CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o.d -o CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o -c /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate.cpp

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.i"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate.cpp > CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.i

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.s"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate.cpp -o CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.s

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/flags.make
src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o: /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplateMain.cpp
src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o -MF CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o.d -o CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o -c /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplateMain.cpp

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.i"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplateMain.cpp > CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.i

src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.s"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplateMain.cpp -o CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.s

src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/flags.make
src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o: /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/Timer.cpp
src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o: src/CMakeFiles/OpenCLTemplate.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o -MF CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o.d -o CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o -c /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/Timer.cpp

src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OpenCLTemplate.dir/Timer.cpp.i"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/Timer.cpp > CMakeFiles/OpenCLTemplate.dir/Timer.cpp.i

src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OpenCLTemplate.dir/Timer.cpp.s"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/Timer.cpp -o CMakeFiles/OpenCLTemplate.dir/Timer.cpp.s

# Object files for target OpenCLTemplate
OpenCLTemplate_OBJECTS = \
"CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o" \
"CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o" \
"CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o"

# External object files for target OpenCLTemplate
OpenCLTemplate_EXTERNAL_OBJECTS =

bin/x86_64/Release/OpenCLTemplate: src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplate.cpp.o
bin/x86_64/Release/OpenCLTemplate: src/CMakeFiles/OpenCLTemplate.dir/OpenCLTemplateMain.cpp.o
bin/x86_64/Release/OpenCLTemplate: src/CMakeFiles/OpenCLTemplate.dir/Timer.cpp.o
bin/x86_64/Release/OpenCLTemplate: src/CMakeFiles/OpenCLTemplate.dir/build.make
bin/x86_64/Release/OpenCLTemplate: /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/lib/libOpenCL.so
bin/x86_64/Release/OpenCLTemplate: src/CMakeFiles/OpenCLTemplate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/x86_64/Release/OpenCLTemplate"
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OpenCLTemplate.dir/link.txt --verbose=$(VERBOSE)
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/cmake -E copy_if_different /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate_Kernels.cl /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/bin/x86_64/Release/.
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && /usr/bin/cmake -E copy_if_different /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src/OpenCLTemplate_Kernels.cl ./

# Rule to build all files generated by this target.
src/CMakeFiles/OpenCLTemplate.dir/build: bin/x86_64/Release/OpenCLTemplate
.PHONY : src/CMakeFiles/OpenCLTemplate.dir/build

src/CMakeFiles/OpenCLTemplate.dir/clean:
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src && $(CMAKE_COMMAND) -P CMakeFiles/OpenCLTemplate.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/OpenCLTemplate.dir/clean

src/CMakeFiles/OpenCLTemplate.dir/depend:
	cd /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/root/source/repos/OpenCLTemplate3.0 /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/src /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src /mnt/c/Users/root/source/repos/OpenCLTemplate3.0/build/src/CMakeFiles/OpenCLTemplate.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/OpenCLTemplate.dir/depend

