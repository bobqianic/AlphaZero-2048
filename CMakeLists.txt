cmake_minimum_required(VERSION 3.18)
project(rl2048 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "/root/libtorch")
find_package(Torch REQUIRED)

add_executable(rl2048
        main.cpp
        model.cpp
        generator.cpp
        trainer.cpp
        play.cpp
        util/cuda/encode2048_cuda.cu
)
target_link_libraries(rl2048 "${TORCH_LIBRARIES}")

# Needed on some systems
set_property(TARGET rl2048 PROPERTY CXX_STANDARD 17)
