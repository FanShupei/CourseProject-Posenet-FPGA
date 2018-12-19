cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(app)

find_package(OpenCV REQUIRED)

# add_compile_options(-Wdouble-promotion)

add_executable(app
    convlayer.cpp
    mobilenet.cpp
    pose_decoder.cpp)

target_link_libraries(app ${OpenCV_LIBS})
