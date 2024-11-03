cmake_minimum_required(VERSION 3.10)

# 4 targets:
# - MidiTokCpp : a dynamic library
# - MidiTokCppStatic : a static library
# - MidiTokCppTest : a C++ exe linked to MidiTokCppStatic
# - MidiTokCppTestC : a C exe linked to MidiTokCppStatic

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
# file(GLOB SOURCES "src/*.cpp")
# list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.c")
file(GLOB SOURCES "src/main.cpp")
add_executable(MidiTokCppTest ${SOURCES})
target_compile_definitions(MidiTokCppTest PRIVATE BUILD_STATIC)

# Add the executable
# file(GLOB SOURCES2 "src/*.cpp" "src/*.cc")
# list(REMOVE_ITEM SOURCES2 "${CMAKE_SOURCE_DIR}/src/main.cpp")
file(GLOB SOURCES2 "src/main.cc")
add_executable(MidiTokCppTestC ${SOURCES2})
target_compile_definitions(MidiTokCppTestC PRIVATE BUILD_STATIC)

file(GLOB SOURCES3 "src/*.cpp")
list(REMOVE_ITEM SOURCES3 "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM SOURCES3 "${CMAKE_SOURCE_DIR}/src/main.cc")
add_library(MidiTokCpp SHARED ${SOURCES3})
target_compile_definitions(MidiTokCpp PRIVATE BUILD_DLL)

add_library(MidiTokCppStatic STATIC ${SOURCES3})
target_compile_definitions(MidiTokCppStatic PRIVATE BUILD_STATIC)


target_link_libraries(MidiTokCppTest PRIVATE MidiTokCppStatic)
target_link_libraries(MidiTokCppTestC PRIVATE MidiTokCppStatic)


# Include directories
target_include_directories(MidiTokCpp PRIVATE include)
target_include_directories(MidiTokCppStatic PRIVATE include)
target_include_directories(MidiTokCppTest PRIVATE include)
target_include_directories(MidiTokCppTestC PRIVATE include)





target_include_directories(MidiTokCpp PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(MidiTokCppStatic PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(MidiTokCppTest PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(MidiTokCppTestC PRIVATE ${TOKENZIER_CPP_PATH}/include)
# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary
target_link_libraries(MidiTokCpp PRIVATE tokenizers_cpp)
target_link_libraries(MidiTokCppStatic PRIVATE tokenizers_cpp)
target_link_libraries(MidiTokCppTest PRIVATE tokenizers_cpp)
target_link_libraries(MidiTokCppTestC PRIVATE tokenizers_cpp)


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
target_include_directories(MidiTokCppStatic PUBLIC
    "${ONNXRUNTIME_ROOTDIR}/build/native/include"                           # Pre-built package
)
target_include_directories(MidiTokCppTest PUBLIC
    "${ONNXRUNTIME_ROOTDIR}/build/native/include"                           # Pre-built package
)
target_include_directories(MidiTokCppTestC PUBLIC
    "${ONNXRUNTIME_ROOTDIR}/build/native/include"                           # Pre-built package
)
target_link_directories(MidiTokCpp PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokCpp PUBLIC 
    ${ONNX_RUNTIME} 
)

target_link_directories(MidiTokCppStatic PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokCppStatic PUBLIC 
    ${ONNX_RUNTIME} 
)

target_link_directories(MidiTokCppTest PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokCppTest PUBLIC 
    ${ONNX_RUNTIME} 
)
target_link_directories(MidiTokCppTestC PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokCppTestC PUBLIC 
    ${ONNX_RUNTIME} 
)


# Specify a custom working directory when running the executable
add_custom_target(run
    COMMAND MidiTokCpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCpp
)

# Specify a custom working directory when running the executable
add_custom_target(runStatic
    COMMAND MidiTokCppStatic
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCppStatic
)

# Specify a custom working directory when running the executable
add_custom_target(runTest
    COMMAND MidiTokCppTest
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCppTest
)

# Specify a custom working directory when running the executable
add_custom_target(runTestC
    COMMAND MidiTokCppTestC
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCppTestC
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokCpp>
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokCppStatic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokCppStatic>
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokCppTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokCppTest>
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokCppTestC POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokCppTestC>
)

