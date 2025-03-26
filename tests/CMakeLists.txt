include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

# Ensure correct std runtime linkage
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# if(BUILD_TESTING)
    file(GLOB SOURCES "*.cpp")
    add_executable(MidiTokTests ${SOURCES})
    target_link_libraries(MidiTokTests PRIVATE gtest gtest_main)

    include(GoogleTest)
    gtest_discover_tests(MidiTokTests)
# endif()


