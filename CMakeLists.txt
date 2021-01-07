cmake_minimum_required(VERSION 3.15)
project(deepfool)

set(CMAKE_CXX_STANDARD 14)
# add_link_options(-fsanitize=undefined)
# add_compile_options(-fsanitize=undefined)
add_compile_options(-O3)

add_executable(deepfool main.cpp oracle.cpp calculator.cpp cfr.cpp game.cpp)