cmake_minimum_required(VERSION 3.5)
project(EMD)
set (CMAKE_CXX_STANDARD 11)
execute_process(COMMAND julia -e "using Conda; print(joinpath(Conda.LIBDIR, \"Libraries\"))" OUTPUT_VARIABLE EIGEN_INC)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")
message("${EIGEN_INC}")
include_directories(${EIGEN_INC})

add_executable(main EMD_wrapper.cpp test.cpp)