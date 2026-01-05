cmake_minimum_required(VERSION 3.18)

# Handle GCC version compatibility with CUDA
if(UNIX AND NOT APPLE)
    # Check for compatible GCC versions for CUDA
    set(CUDA_COMPATIBLE_GCC_VERSIONS "gcc-14" "gcc-13" "gcc-12" "gcc-11")
    
    foreach(GCC_VERSION ${CUDA_COMPATIBLE_GCC_VERSIONS})
        find_program(COMPATIBLE_GCC_PATH ${GCC_VERSION})
        # Convert gcc-14 to g++-14
        string(REPLACE "gcc" "g++" GXX_VERSION ${GCC_VERSION})
        find_program(COMPATIBLE_GXX_PATH ${GXX_VERSION})
        if(COMPATIBLE_GCC_PATH AND COMPATIBLE_GXX_PATH)
            message(STATUS "Found compatible GCC for CUDA: ${COMPATIBLE_GCC_PATH}")
            message(STATUS "Found compatible G++ for CUDA: ${COMPATIBLE_GXX_PATH}")
            
            # Set both C and C++ compilers to use the compatible versions
            set(CMAKE_C_COMPILER ${COMPATIBLE_GCC_PATH})
            set(CMAKE_CXX_COMPILER ${COMPATIBLE_GXX_PATH})
            set(CMAKE_CUDA_HOST_COMPILER ${COMPATIBLE_GCC_PATH})
            break()
        endif()
    endforeach()
    
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        message(WARNING "No compatible GCC found for CUDA. Trying system GCC...")
        message(WARNING "If compilation fails, install gcc-14 or gcc-13: sudo pacman -S gcc14")
    endif()
endif()

project(GPUPathtracer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# Find SDL2 and GLEW (prefer system packages on Linux)
find_package(SDL2 QUIET)
find_package(GLEW QUIET)

# If not found, use the included versions
if(NOT SDL2_FOUND)
    set(SDL2_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include/SDL2)
    set(SDL2_LIBRARIES SDL2 SDL2main)
    link_directories(${CMAKE_SOURCE_DIR}/lib)
endif()

if(NOT GLEW_FOUND)
    set(GLEW_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include/GL)
    set(GLEW_LIBRARIES GLEW glew32s)
    link_directories(${CMAKE_SOURCE_DIR}/lib)
endif()

# Set include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/Src
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-delete-incomplete -Wno-type-limits -Wno-unused-parameter -Wno-unused-variable -Wno-pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

# Platform-specific definitions
if(WIN32)
    add_definitions(-DNOMINMAX)
else()
    # Linux-specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Collect source files
file(GLOB_RECURSE SOURCES 
    ${CMAKE_SOURCE_DIR}/Src/*.cpp
    ${CMAKE_SOURCE_DIR}/Src/*.c
)

# Add ImGui sources
file(GLOB IMGUI_SOURCES
    ${CMAKE_SOURCE_DIR}/include/Imgui/*.cpp
)
list(APPEND SOURCES ${IMGUI_SOURCES})

# Find system miniz package
find_package(PkgConfig REQUIRED)
pkg_check_modules(MINIZ REQUIRED miniz)

# Create executable
add_executable(pathtracer ${SOURCES})

# Add miniz include directories
target_include_directories(pathtracer PRIVATE ${MINIZ_INCLUDE_DIRS})
target_compile_options(pathtracer PRIVATE ${MINIZ_CFLAGS_OTHER})

# Set CUDA architectures (adjust based on your GPU)
set_property(TARGET pathtracer PROPERTY CUDA_ARCHITECTURES 60 61 70 75 80 86 90)

# Link libraries
target_link_libraries(pathtracer
    ${MINIZ_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_static_LIBRARY}
    ${CUDA_nvrtc_LIBRARY}
    /opt/cuda/targets/x86_64-linux/lib/libnvrtc.so
    cuda
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    m
)

# On Linux, we might need to link against specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(pathtracer
        GL
        X11
        Xxf86vm
        Xrandr
        pthread
        Xi
        Xinerama
        Xcursor
    )
endif()

# Copy any required runtime files
if(EXISTS ${CMAKE_SOURCE_DIR}/Data)
    file(COPY ${CMAKE_SOURCE_DIR}/Data DESTINATION ${CMAKE_BINARY_DIR})
endif()

# Set output directory
set_target_properties(pathtracer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CUDA version: ${CUDA_VERSION}")