#pragma once

// GPU-side Tracy instrumentation for the GL 4.3 renderer. Include AFTER the
// glad header: tracy/TracyOpenGL.hpp calls GL entry points directly. No-ops
// unless VOXEL_TRACY is set, same as profiling.h.
//
//   VOX_GPU_CONTEXT          one-time init on the GL thread, after GL loads
//   VOX_GPU_ZONE("name")     scoped GPU zone; ends at enclosing scope end
//   VOX_GPU_ZONE_BEGIN(var, "name") / VOX_GPU_ZONE_END(var)
//                            explicit begin/end pair for flat sequential
//                            passes that share one scope (renderMap)
//   VOX_GPU_COLLECT          per-frame timestamp collection, after swap

#ifdef VOXEL_TRACY

#include <tracy/TracyOpenGL.hpp>
#include <optional>

#define VOX_GPU_CONTEXT TracyGpuContext
#define VOX_GPU_ZONE(name) TracyGpuZone(name)
// Tracy only ships scope-bound GPU zones (tracy::GpuCtxScope); an optional
// gives them begin/end lifetime without introducing brace scopes.
#define VOX_GPU_ZONE_BEGIN(var, name)                                        \
    static constexpr tracy::SourceLocationData var##_srcloc{                 \
        name, TracyFunction, TracyFile, (uint32_t)TracyLine, 0 };            \
    std::optional<tracy::GpuCtxScope> var{ std::in_place, &var##_srcloc, true }
#define VOX_GPU_ZONE_END(var) (var).reset()
#define VOX_GPU_COLLECT TracyGpuCollect

#else

#define VOX_GPU_CONTEXT
#define VOX_GPU_ZONE(name)
#define VOX_GPU_ZONE_BEGIN(var, name)
#define VOX_GPU_ZONE_END(var)
#define VOX_GPU_COLLECT

#endif
