# game_state.h is the app-layer state aggregate (see its banner): only
# app.cpp and commands.cpp may include it. Module code must keep taking
# narrow parameters (World&, its own struct, PODs) -- the first stray
# includer would quietly end the forward-declaration discipline that keeps
# module headers independent, so the build enforces the whitelist here.
# Invoked as: cmake -DSRC_DIR=<src> -P check_game_state_includers.cmake
#
# Script mode carries no policy floor from the main CMakeLists, so declare
# one: without it, CMake 3.x defaults CMP0057 to OLD and the IN_LIST
# operator below is a hard error (CMake 4.x has it hard-wired NEW, which is
# why a local 4.x build passes while CI's 3.x fails).
cmake_minimum_required(VERSION 3.20)
set(ALLOWED "app.cpp" "commands.cpp")
file(GLOB_RECURSE CANDIDATES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.h")
foreach(f IN LISTS CANDIDATES)
    get_filename_component(name "${f}" NAME)
    if(name IN_LIST ALLOWED OR name STREQUAL "game_state.h")
        continue()
    endif()
    file(STRINGS "${f}" hits REGEX "#include[ \t]+\"game_state\\.h\"")
    if(hits)
        message(FATAL_ERROR
            "${f} includes game_state.h, which is app-layer-only "
            "(allowed: app.cpp, commands.cpp). Take narrow parameters "
            "instead -- see the banner in src/game_state.h.")
    endif()
endforeach()
