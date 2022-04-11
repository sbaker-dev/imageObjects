cmake_minimum_required(VERSION 3.5)
project(imageObjects)

set(OpenCV_DIR "C:/Users/Samuel/Clibs/opencv/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/ImageObject.cpp src/ImageObject.h)

# The build executable, needs to be turned into a lib at some point
add_executable(UnitTesting UnitTests/UnitTest.cpp ${SOURCE_FILES})

target_link_libraries(UnitTesting ${OpenCV_LIBS})