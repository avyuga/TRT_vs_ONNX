cmake_minimum_required(VERSION 3.23)
project(TENSORRT_ONNX_DEMO)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)



# CUDA
if (NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2/")
endif()
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include
)
MESSAGE(STATUS "Found CUDA headers at ${CUDA_INCLUDE_DIR}")

# TensorRT
if (NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "~/workspace/cpp_libs/TensorRT-10.9.0.34")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found TensorRT NVinfer library at ${TENSORRT_LIBRARY_INFER}")


find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found INFER_PLUGIN libs at ${TENSORRT_LIBRARY_INFER_PLUGIN}")

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found NvONNXParser at ${TENSORRT_LIBRARY_ONNXPARSER}")



set(CUDART /usr/local/cuda/lib64/libcudart.so)

set(PNG_LIB "/opt/conda/lib/libpng16.so")
set(PNG_INCLUDE "/opt/conda/include/")
set(ONNX_DIR "~/workspace/cpp_libs/onnxruntime-linux-x64-gpu-1.20.0")

add_compile_options(-std=c++17 -Wno-deprecated-declarations -Dcimg_display=0)
set_property(GLOBAL PROPERTY LINKER_LANGUAGE CXX)

### TRT Specific
set(TARGET_TRT "Yolov10TRT")
add_executable(${TARGET_TRT} "${CMAKE_SOURCE_DIR}/src/Yolov10TRT.cpp" "${CMAKE_SOURCE_DIR}/src/Utility.cpp")

target_link_libraries(${TARGET_TRT} 
  ${CUDART}
  ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNXPARSER}
  ${PNG_LIB}
)

target_include_directories(${TARGET_TRT} 
    PUBLIC 
        ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${PNG_INCLUDE}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)


### ONNX Specific
set(TARGET_ONNX "Yolov10ONNX")
add_executable(${TARGET_ONNX} "${CMAKE_SOURCE_DIR}/src/Yolov10ONNX.cpp" "${CMAKE_SOURCE_DIR}/src/Utility.cpp")

target_link_libraries(${TARGET_ONNX} 
  "${ONNX_DIR}/lib/libonnxruntime.so"
  ${PNG_LIB}
)

target_include_directories(${TARGET_ONNX} 
    PUBLIC 
        "${ONNX_DIR}/include" ${PNG_INCLUDE}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)
