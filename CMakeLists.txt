cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS_DEBUG "-g3 -Og")
set(CMAKE_C_FLAGS_RELEASE "-g -O2")

# must be
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)


project(wdtagger)

find_package( OpenCV REQUIRED )
message(STATUS "Opencv_inc: ${OpenCV_INCLUDE_DIRS}")

set(INC_DIRS
  include/onnx
  include
  rc
)

set(SRC_FILES
  src/wdtagger.cpp
)


add_executable(${PROJECT_NAME} WIN32 ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIRS} ${OpenCV_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/onnxlib/onnxruntime-linux-x64-1.18.0/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} onnxruntime)
