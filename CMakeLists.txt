cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tracknet)

set(CMAKE_LINK_WHAT_YOU_USE TRUE)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/include/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
