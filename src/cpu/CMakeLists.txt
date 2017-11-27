cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
file(GLOB SRC
    "*.cpp")
set(CMAKE_CXX_STANDARD 11)
add_executable( main ${SRC} )
target_link_libraries( main ${OpenCV_LIBS} )
