cmake_minimum_required(VERSION 3.24)
project(fastwave LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-deprecated -Wno-c++20-extensions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG -s -flto")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"-Wall -Wextra -Wno-deprecated -Wno-unknown-pragmas\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2811") # deal with boost μt warnings

option(FASTWAVE_BUILD_TESTS "Build a unit test executable" ON)

# -------------------- Dependencies --------------------
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cufft cublas)

include(FetchContent)

if(FASTWAVE_BUILD_TESTS)
    # We use boost μt instead of Catch2 because it's a much smaller header only library.
    #
    # Here are the things I tried:
    # * Using Catch2 in header-only mode: This makes nvcc use up all of my 32 gigs of RAM
    # * Compiling Catch2 with nvcc: Linker errors
    # * Compiling Catch2 with other compilers and then linking to it from my testmain .cu: Linker errors
    #
    # The main issue is that I want to be able to call kernels from the tests, even just host side, so I
    # have to keep the testmain as a .cu.
    #
    # For some reason it works when everything is in one target (see commit 7b782308fb), but I don't want
    # my test binary to be the same as my actual binary.
    FetchContent_Declare(
        ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG v1.1.9
    )
    FetchContent_GetProperties(ut)

    # The reason that we don't use make_available is because when the C++ compiler is set to
    # gcc, the flags set by the boost μt library break nvcc. It's easier to just skip the μt
    # CMakeLists.
    if(NOT ut_POPULATED)
        FetchContent_Populate(ut)
        add_library(ut INTERFACE)
        target_include_directories(ut INTERFACE ${ut_SOURCE_DIR}/include)
    endif()
endif()

FetchContent_Declare(
    npypp
    GIT_REPOSITORY https://github.com/pmontalb/NpyCpp.git
    GIT_TAG 76c056a41c1709607
)
FetchContent_GetProperties(npypp)

if(NOT npypp_POPULATED)
    FetchContent_Populate(npypp)
    add_library(npypp INTERFACE)
    target_include_directories(npypp INTERFACE ${npypp_SOURCE_DIR}/Npy++/ ${npypp_SOURCE_DIR}/MemoryMapping/)
endif()

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG f7320141120f720aecc4c32be25586e7da9eb978 # v0.8.0
)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    programoptions
    GIT_REPOSITORY https://github.com/Fytch/ProgramOptions.hxx.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(programoptions)


# -------------------- Targets --------------------
add_subdirectory(fwcuda)

# We keep fastwave as a library so that it can be included by the tests as
# well, even though we just end up using it as an executable and not as a
# library.
add_library(fastwave_lib src/fft.cpp src/simulation.cpp src/config_parsing.cpp)

target_include_directories(fastwave_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(fastwave_lib
    PUBLIC
    fwcuda
    yaml-cpp::yaml-cpp
    spdlog::spdlog
    PRIVATE
    CUDA::cufft
    CUDA::cublas
    npypp
)
target_compile_features(fastwave_lib PUBLIC cxx_std_17)

add_executable(fastwave main.cpp)
target_link_libraries(fastwave fastwave_lib fwcuda ProgramOptionsHxx)
target_compile_features(fastwave PUBLIC cxx_std_17)

if(FASTWAVE_BUILD_TESTS)
    include(CTest)

    add_executable(fwtest test/fwtest.cu)
    target_compile_features(fwtest PUBLIC cxx_std_20)
    # give fwtest access to private headers of fastwave_lib
    target_include_directories(fwtest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
    target_link_libraries(fwtest PRIVATE fastwave_lib fwcuda ut)
    add_test(NAME fwtest COMMAND fwtest)
endif()
