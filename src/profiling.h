#pragma once

// CPU-side Tracy instrumentation. Every macro expands to nothing unless the
// build sets VOXEL_TRACY (cmake -DVOXEL_TRACY=ON), so shipping builds carry
// zero profiling overhead. GPU zones live in profiling_gpu.h because the
// Tracy OpenGL header needs GL symbols in scope.
//
//   VOX_ZONE                 unnamed scoped zone (uses the function name)
//   VOX_ZONE_N("name")       named scoped zone; ends at enclosing scope end
//   VOX_ZONE_NV(var, "name") same, with an explicit variable name -- use for
//                            a second scoped zone in an already-zoned scope
//   VOX_ZONE_BEGIN(var, "name") / VOX_ZONE_END(var)
//                            explicit begin/end pair for flat sequential
//                            phases that share one scope
//   VOX_FRAME_MARK           frame boundary, place at buffer swap

#ifdef VOXEL_TRACY

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define VOX_ZONE ZoneScoped
#define VOX_ZONE_N(name) ZoneScopedN(name)
#define VOX_ZONE_NV(var, name) ZoneNamedN(var, name, true)
#define VOX_ZONE_BEGIN(var, name) TracyCZoneN(var, name, true)
#define VOX_ZONE_END(var) TracyCZoneEnd(var)
#define VOX_FRAME_MARK FrameMark

#else

#define VOX_ZONE
#define VOX_ZONE_N(name)
#define VOX_ZONE_NV(var, name)
#define VOX_ZONE_BEGIN(var, name)
#define VOX_ZONE_END(var)
#define VOX_FRAME_MARK

#endif
