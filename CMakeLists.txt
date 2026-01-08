cmake_minimum_required(VERSION 3.15)
project(fast_hv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
# x86-64-v3: AVX2, FMA - common to Skylake and Zen2+
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=x86-64-v3 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# OpenMP
find_package(OpenMP REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Build the Python module
pybind11_add_module(_fast_hv
    src/fast_hv/pppa.cpp
    src/fast_hv/bindings.cpp
)

target_include_directories(_fast_hv PRIVATE src/fast_hv)
target_link_libraries(_fast_hv PRIVATE OpenMP::OpenMP_CXX)

# Install the module
install(TARGETS _fast_hv LIBRARY DESTINATION fast_hv)
