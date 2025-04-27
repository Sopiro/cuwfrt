cmake_minimum_required(VERSION 3.27)

project(cuwfrt LANGUAGES CXX CUDA)

# find cuda
# see: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

# default startup project for Visual Studio
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cuwfrt)
    set_property(TARGET cuwfrt PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
