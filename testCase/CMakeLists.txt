#cmake最低版本
cmake_minimum_required ( VERSION 3.0 )

#设置工程名字
project(demo)
message(STATUS ${PROJECT_NAME})

#指定C++版本
set(CMAKE_CXX_STANDARD 11)

#设置头文件路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(YOLOX_LIBS_PATH /home/captain/MyProjects/001_yolox/yolox)
include_directories(${YOLOX_LIBS_PATH}/include/)

#设置源文件路径
FILE(GLOB SRC ${PROJECT_SOURCE_DIR}/*.cpp)          #匹配CMakeList.txt路径下所有cpp文件

#设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#生成可执行文件
add_executable(${PROJECT_NAME} ${SRC})

#链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${YOLOX_LIBS_PATH}/lib/libYoloX.so)
