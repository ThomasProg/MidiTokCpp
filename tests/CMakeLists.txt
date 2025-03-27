include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

# Ensure correct std runtime linkage
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(CTEST_OUTPUT_LIMIT 10000)

# if(BUILD_TESTING)
    file(GLOB SOURCES "*.cpp")
    add_executable(MidiTokTests ${SOURCES})
    target_link_libraries(MidiTokTests PRIVATE gtest gtest_main)

    include(GoogleTest)
    gtest_discover_tests(MidiTokTests)
# endif()


target_compile_definitions(MidiTokTests PRIVATE BUILD_STATIC)
target_link_libraries(MidiTokTests PRIVATE MidiTokCppStatic)
target_include_directories(MidiTokTests PRIVATE ../include)
target_include_directories(MidiTokTests PRIVATE ../bindings/cpp)
target_include_directories(MidiTokTests PRIVATE ../bindings/c)
target_include_directories(MidiTokTests PRIVATE ../thirdparty/headeronly/)
target_include_directories(MidiTokTests PRIVATE ${TOKENZIER_CPP_PATH}/include)

target_link_libraries(MidiTokTests PRIVATE tokenizers_cpp)

target_include_directories(MidiTokTests PUBLIC
    "${ONNXRUNTIME_ROOTDIR}/build/native/include"                           # Pre-built package
)
target_link_directories(MidiTokTests PUBLIC "${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native")
target_link_libraries(MidiTokTests PUBLIC 
    ${ONNX_RUNTIME} 
)

add_custom_target(runTestA
    COMMAND MidiTokTests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS MidiTokTests
)

# Optionally, copy the .dll to the build directory
add_custom_command(TARGET MidiTokTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/${ONNXRUNTIME_ROOTDIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:MidiTokTests>
)


