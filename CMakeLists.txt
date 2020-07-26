cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project (imgtest LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O1 -Wall")  # default
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O0 -g")     # debug    : -D CMAKE_BUILD_TYPE=Debug
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")        # release  : -D CMAKE_BUILD_TYPE=Release
                                                             # profile  : -D CMAKE_CXX_FLAGS="-pg"

set(CMAKE_CUDA_HOST_COMPILER "nvcc")

# timer lib
add_library (timer timer.cpp)

add_executable (${PROJECT_NAME}
    main.cu
)
set_property (TARGET ${PROJECT_NAME}
    APPEND PROPERTY
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries (${PROJECT_NAME}
    PRIVATE
        timer
)
