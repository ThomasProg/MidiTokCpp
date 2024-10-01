cmake_minimum_required(VERSION 3.10)

add_compile_definitions(WORKSPACE_PATH="${CMAKE_SOURCE_DIR}/")

# Project name
project(MidiTokCpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

# include tokenizer cpp as a sub directory
set(TOKENZIER_CPP_PATH ./thirdparty/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)




# Add the executable
file(GLOB SOURCES "src/*.cpp")
add_executable(MidiTokCpp ${SOURCES})
# add_executable(MidiTokCpp src/main.cpp src/midiTokenizer.cpp)

# Include directories
target_include_directories(MidiTokCpp PRIVATE include)





target_include_directories(MidiTokCpp PRIVATE ${TOKENZIER_CPP_PATH}/include)
# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary
target_link_libraries(MidiTokCpp PRIVATE tokenizers_cpp)


set(ONNXRUNTIME_ROOTDIR 
    "thirdparty/microsoft.ml.onnxruntime.1.19.2" 
)

# set(ONNXRUNTIME_LIBRARIES 
#     "onnxruntime.lib" 
# )

find_library(ONNX_RUNTIME onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
if(NOT ONNX_RUNTIME)
    message(FATAL_ERROR "Library onnx_runtime was not found!")
endif()

target_include_directories(MidiTokCpp PUBLIC
    "${ONNXRUNTIME_ROOTDIR}/build/native/include"                           # Pre-built package
)
target_link_directories(MidiTokCpp PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokCpp PUBLIC 
    ${ONNX_RUNTIME} 
)


# Specify a custom working directory when running the executable
add_custom_target(run
    COMMAND MidiTokCpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCpp
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokCpp>
)


