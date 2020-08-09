cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project (cuda-checker LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O1 -Wall")  # default
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O0 -g")     # debug    : -D CMAKE_BUILD_TYPE=Debug
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")        # release  : -D CMAKE_BUILD_TYPE=Release
                                                             # profile  : -D CMAKE_CXX_FLAGS="-pg"

set(CMAKE_CUDA_HOST_COMPILER "nvcc")

# directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# [output] library directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR})
# [output] executable directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# timer lib
add_library (timer ${SRC_DIR}/timer.cpp)
target_include_directories (timer
    PRIVATE
        ${INC_DIR}
)
# argtools lib
add_library (argtools ${SRC_DIR}/argtools.cpp)
target_include_directories (argtools
    PRIVATE
        ${INC_DIR}
)

add_executable (${PROJECT_NAME}
    ${SRC_DIR}/main.cu
)
set_property (TARGET ${PROJECT_NAME}
    APPEND PROPERTY
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories (${PROJECT_NAME}
    PRIVATE
        ${INC_DIR}
)
target_link_libraries (${PROJECT_NAME}
    PRIVATE
        timer
        argtools
)
