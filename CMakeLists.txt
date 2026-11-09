cmake_minimum_required(VERSION 3.20)
project(voxel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link the MSVC runtime statically (/MT, /MTd in Debug) so voxel.exe runs on
# PCs without the Visual C++ Redistributable. Set before the FetchContent
# dependencies below so their targets inherit it: mixing /MT and /MD doesn't
# link cleanly. cmake_minimum_required 3.20 makes CMP0091 NEW, which this needs.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Tracy profiler instrumentation. OFF by default: the profiler client is not
# fetched, nothing is linked, and every VOX_* macro in src/profiling.h expands
# to nothing, so release builds are unaffected.
option(VOXEL_TRACY "Enable Tracy profiler instrumentation" OFF)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.6
    SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)

# miniz: single-source zlib-compatible compression for the world cache file.
# miniz uses an old cmake_minimum_required so we relax the policy floor.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS  OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(AMALGAMATE_SOURCES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz.git
    GIT_TAG 3.0.2
)
FetchContent_MakeAvailable(miniz)

# miniaudio: single-header audio engine. The implementation is compiled in
# src/audio.cpp (MINIAUDIO_IMPLEMENTATION). The repo is header-only (no
# CMakeLists), so MakeAvailable just populates the source; we add its dir to
# the include path. On Windows miniaudio loads the audio backends at runtime,
# so no extra link libraries are needed.
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.21
)
FetchContent_MakeAvailable(miniaudio)

if(VOXEL_TRACY)
    # Listen on localhost only and skip the LAN discovery broadcast so the
    # profiled game never trips the Windows firewall prompt mid-capture.
    set(TRACY_ONLY_LOCALHOST ON CACHE BOOL "" FORCE)
    set(TRACY_NO_BROADCAST ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.13.1
    )
    FetchContent_MakeAvailable(tracy)
endif()

glad_add_library(glad_gl_core_43 STATIC API gl:core=4.3)

add_executable(voxel
    src/app.cpp
    src/commands.cpp
    src/world.cpp
    src/worldgen/worldgen.cpp
    src/worldgen/caves.cpp
    src/worldgen/flora.cpp
    src/worldgen/rings.cpp
    src/player.cpp
    src/path.cpp
    src/particles.cpp
    src/creatures.cpp
    src/materials.cpp
    src/forestry.cpp
    src/inventory.cpp
    src/items.cpp
    src/map.cpp
    src/water.cpp
    src/fire.cpp
    src/granular.cpp
    src/farm.cpp
    src/worldio.cpp
    src/console.cpp
    src/font8x8.cpp
    src/render.cpp
    src/audio.cpp
)
target_link_libraries(voxel PRIVATE glfw glm::glm glad_gl_core_43 miniz)
target_include_directories(voxel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${miniaudio_SOURCE_DIR})

if(VOXEL_TRACY)
    target_link_libraries(voxel PRIVATE Tracy::TracyClient)
    target_compile_definitions(voxel PRIVATE VOXEL_TRACY)
endif()

if(MSVC)
    target_compile_options(voxel PRIVATE /W3)
endif()

# game_state.h may only be included by app.cpp and commands.cpp (it is the
# app-layer aggregate; module code takes narrow parameters). Enforced before
# every build so the whitelist is mechanical, not just a header comment.
add_custom_command(TARGET voxel PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_game_state_includers.cmake
    COMMENT "Checking game_state.h includer whitelist")

# Asset tools. The voxelizer converts textured meshes (OBJ/GLB from the
# image-to-3D step) into MagicaVoxel .vox models for the creature pipeline.
# Header-only deps are vendored in tools/voxelizer/third_party.
option(VOXEL_TOOLS "Build asset pipeline tools (voxelizer)" ON)
if(VOXEL_TOOLS)
    add_executable(voxelizer tools/voxelizer/voxelizer.cpp)
    if(MSVC)
        target_compile_options(voxelizer PRIVATE /W3)
        target_compile_definitions(voxelizer PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()


# No automatic signing is configured. See README troubleshooting for optional
# local code-signing commands when Smart App Control is enabled.
