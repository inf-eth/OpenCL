#cmake_minimum_required( VERSION 3.10.3...3.28.3 )
#set(CMAKE_SUPPRESS_REGENERATION TRUE)
project(OpenCL VERSION 1.0)
add_subdirectory(OpenCLWrapper)
add_subdirectory(OpenCLWrapperSimple)
add_subdirectory(OpenCLTemplate3.1)
add_subdirectory(OpenCLMatMult)