cmake_minimum_required(VERSION 3.10)
project(SuperAPLLearningModel)

set(CMAKE_CXX_STANDARD 17)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CUDA_SOURCES src/cpp/qgemm_kernel.cu)
    message(STATUS "CUDA found. GPU support enabled.")
else()
    message(STATUS "CUDA not found. Building for CPU only.")
endif()

# Enable AVX2 instructions and Multi-Processor Compilation
if(MSVC)
    add_compile_options(/arch:AVX2 /MP)
else()
    add_compile_options(-mavx2 -mfma -fopenmp)
endif()

include_directories(include)

add_library(super_apl_engine SHARED
    src/cpp/qgemm_kernel.cpp
    src/cpp/engine_wrapper.cpp
    ${CUDA_SOURCES}
)

# Link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(super_apl_engine PUBLIC OpenMP::OpenMP_CXX)
endif()
