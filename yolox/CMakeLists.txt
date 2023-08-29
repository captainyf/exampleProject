#cmake最低版本
cmake_minimum_required(VERSION 3.0)

#设置工程名字
project(YoloX)
message(STATUS ${PROJECT_NAME})

#指定C++版本
set(CMAKE_CXX_STANDARD 11)

#设置ncnn库的头文件、静态库的路径
set(NCNN_LIBS_PATH /home/captain/Installer/ncnn-20230517/build/install)
set(NCNN_INCLUDE_DIRS ${NCNN_LIBS_PATH}/include/ncnn/)
include_directories(${NCNN_INCLUDE_DIRS})
link_directories(${NCNN_LIBS_PATH}/lib)
link_libraries(libncnn.a)

#设置opencv头文件、库的路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#设置头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

#设置库文件输出路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#设置源文件、头文件路径
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

#生成动态库文件
add_library(YoloX SHARED ${SRC} ${HDR})

#链接opencv动态库
target_link_libraries(YoloX ${OpenCV_LIBS})
