cmake_minimum_required(VERSION 3.10)

add_compile_definitions(WORKSPACE_PATH="${CMAKE_SOURCE_DIR}/")

# Project name
project(MidiTokCpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(MidiTokCpp src/main.cpp src/midiTokenizer.cpp)

# Include directories
target_include_directories(MidiTokCpp PRIVATE include)

# Specify a custom working directory when running the executable
add_custom_target(run
    COMMAND MidiTokCpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokCpp
)
