cmake_minimum_required(VERSION 3.5)
project(imageObjects)

set(OpenCV_DIR "C:/Users/Samuel/Clibs/opencv/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/ImageObject.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})