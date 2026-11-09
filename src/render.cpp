#include "render.h"
#include "world.h"
#include "particles.h"
#include "creatures.h"
#include "materials.h"
#include "inventory.h"
#include "common.h"
#include "console.h"
#include "farm.h"       // FT_PLANT + CROP_DEFS (the farm bar's seed row)
#include "font8x8.h"
#include "map.h"
#include "worldgen/worldgen.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "profiling.h"
#include "profiling_gpu.h"   // after <glad/gl.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>

namespace fs = std::filesystem;

// =============================================================================
// Shader I/O + safe compile/link helpers (no exit-on-failure: hot reload prints
// the error and keeps the previous program running).
// =============================================================================
namespace {

const char* SHADER_DIR = "src/shaders/";

std::string loadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[shader] cannot open %s\n", path.c_str());
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Recursively expand `#include "path"` directives into a single source string.
// Paths resolve relative to SHADER_DIR (so `#include "lib/noise.glsl"` ->
// src/shaders/lib/noise.glsl). `deps` collects every included file for hot
// reload; `seen` dedups so a lib pulled in twice expands once; `stack` guards
// against include cycles. A `#line` is emitted at each boundary so compiler
// error line numbers stay per-file correct. NOTE: only the top-level .vert/.frag
// carries `#version`; library files must not (it must stay the first line).
std::string preprocessShader(const std::string& path,
                             std::vector<std::string>& deps,
                             std::set<std::string>& seen,
                             std::vector<std::string>& stack) {
    std::stringstream in(loadFile(path));
    std::stringstream out;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        size_t h = line.find_first_not_of(" \t");
        if (h != std::string::npos && line[h] == '#') {
            size_t k = line.find_first_not_of(" \t", h + 1);
            if (k != std::string::npos && line.compare(k, 7, "include") == 0) {
                size_t q1 = line.find('"', k + 7);
                size_t q2 = (q1 == std::string::npos) ? std::string::npos
                                                      : line.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    std::string rel = line.substr(q1 + 1, q2 - q1 - 1);
                    std::string resolved =
                        (fs::path(SHADER_DIR) / rel).lexically_normal().string();
                    if (std::find(stack.begin(), stack.end(), resolved) != stack.end()) {
                        std::fprintf(stderr, "[shader] include cycle ignored: %s\n",
                                     resolved.c_str());
                    } else if (seen.insert(resolved).second) {
                        deps.push_back(resolved);
                        stack.push_back(resolved);
                        out << "#line 1\n"
                            << preprocessShader(resolved, deps, seen, stack)
                            << "#line " << (lineNo + 1) << "\n";
                        stack.pop_back();
                    }
                    continue;   // the #include directive line itself is dropped
                }
            }
        }
        out << line << "\n";
    }
    return out.str();
}

GLuint compileSafe(GLenum kind, const std::string& src, const char* tag) {
    if (src.empty()) return 0;
    GLuint s = glCreateShader(kind);
    const char* p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[shader %s] compile error:\n%s\n", tag, log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkSafe(GLuint vs, GLuint fs, const char* tag) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[shader %s] link error:\n%s\n", tag, log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// Build a fresh program from disk. Returns 0 on failure (compile or link).
// On success, frees any previous program ID held by `sp`.
GLuint compileFromFiles(ShaderProgram& sp, const char* tag) {
    std::vector<std::string> deps;
    // Include dedup is PER STAGE: each stage is its own GLSL translation
    // unit, so a lib included by both the .vert and the .frag (e.g.
    // lib/creature_trace.glsl) must expand in both. A shared `seen` silently
    // dropped the second expansion and the frag failed to compile. `deps`
    // stays shared — hot-reload tracking wants each file once.
    std::set<std::string> seenVs, seenFs;
    std::vector<std::string> stack;
    std::string vsSrc = preprocessShader(sp.vsPath, deps, seenVs, stack);
    std::string fsSrc = preprocessShader(sp.fsPath, deps, seenFs, stack);

    // Record the include dependencies + their mtimes now (even if compilation
    // fails below) so that fixing a broken lib triggers a reload next frame.
    sp.deps = std::move(deps);
    sp.depTimes.clear();
    sp.depTimes.reserve(sp.deps.size());
    {
        std::error_code ec;
        for (const auto& d : sp.deps)
            sp.depTimes.push_back(fs::last_write_time(d, ec));
    }

    GLuint vs = compileSafe(GL_VERTEX_SHADER, vsSrc, tag);
    GLuint fs = compileSafe(GL_FRAGMENT_SHADER, fsSrc, tag);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    GLuint p = linkSafe(vs, fs, tag);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

void initialLoad(ShaderProgram& sp, const char* vsRel, const char* fsRel, const char* tag) {
    sp.vsPath = std::string(SHADER_DIR) + vsRel;
    sp.fsPath = std::string(SHADER_DIR) + fsRel;
    GLuint p = compileFromFiles(sp, tag);
    if (!p) {
        std::fprintf(stderr, "[shader %s] FATAL: initial compile failed.\n", tag);
        std::exit(1);
    }
    sp.id = p;
    std::error_code ec;
    sp.vsTime = fs::last_write_time(sp.vsPath, ec);
    sp.fsTime = fs::last_write_time(sp.fsPath, ec);
}

// Returns true if the program was successfully reloaded (caller should
// re-fetch uniform locations). Returns false on no-change OR compile error.
bool reloadIfChanged(ShaderProgram& sp, const char* tag) {
    std::error_code ec;
    auto vt = fs::last_write_time(sp.vsPath, ec);
    auto ft = fs::last_write_time(sp.fsPath, ec);
    if (ec) return false;
    bool changed = (vt != sp.vsTime) || (ft != sp.fsTime);
    // Also recompile if any #include'd lib file changed on disk.
    for (size_t i = 0; !changed && i < sp.deps.size(); ++i) {
        std::error_code de;
        auto dt = fs::last_write_time(sp.deps[i], de);
        if (de) continue;
        if (i >= sp.depTimes.size() || dt != sp.depTimes[i]) changed = true;
    }
    if (!changed) return false;

    GLuint p = compileFromFiles(sp, tag);   // refreshes sp.deps/depTimes
    // Update mtimes regardless of success so we don't spin re-compiling
    // a broken file every frame.
    sp.vsTime = vt;
    sp.fsTime = ft;
    if (!p) return false;
    if (sp.id) glDeleteProgram(sp.id);
    sp.id = p;
    std::fprintf(stderr, "[shader %s] reloaded\n", tag);
    return true;
}

} // namespace

// Public (external linkage): compile a one-off program before the Renderer
// exists — used by the loading screen. Reuses the anon-namespace, #include-aware
// compileFromFiles path (same TU). Returns 0 on failure.
unsigned int loadStandaloneProgram(const char* vsRel, const char* fsRel, const char* tag) {
    ShaderProgram sp;
    sp.vsPath = std::string(SHADER_DIR) + vsRel;
    sp.fsPath = std::string(SHADER_DIR) + fsRel;
    return compileFromFiles(sp, tag);
}

// =============================================================================
// GPU pass profiler. GL_TIMESTAMP queries are placed at pass boundaries and
// read back two frames later, so the CPU never blocks waiting on the GPU.
// =============================================================================
void GpuProfiler::begin() {
    if (!created) {
        for (int r = 0; r < RING; ++r) glGenQueries(MARKS, q[r]);
        created = true;
    }
    cur = 0;
    glQueryCounter(q[ring][cur], GL_TIMESTAMP);   // t0 = frame start
    ringNames[ring][cur] = "start";
    ++cur;
}

void GpuProfiler::mark(const char* label) {
    if (!created || cur >= MARKS) return;
    glQueryCounter(q[ring][cur], GL_TIMESTAMP);
    ringNames[ring][cur] = label;
    ++cur;
}

void GpuProfiler::resolve() {
    ringMarks[ring] = cur;
    ++frames;
    int rb = (ring + 1) % RING;          // oldest slot; overwritten on next begin
    if (frames >= RING && ringMarks[rb] >= 2) {
        GLuint64 t0 = 0;
        glGetQueryObjectui64v(q[rb][0], GL_QUERY_RESULT, &t0);
        GLuint64 prev = t0;
        msCount = 0;
        msTotal = 0.0f;
        for (int i = 1; i < ringMarks[rb]; ++i) {
            GLuint64 ti = 0;
            glGetQueryObjectui64v(q[rb][i], GL_QUERY_RESULT, &ti);
            float dms = float(double(ti - prev) * 1e-6);
            ms[msCount]      = dms;
            msNames[msCount] = ringNames[rb][i];
            ++msCount;
            msTotal += dms;
            prev = ti;
        }
    }
    ring = rb;
}

// =============================================================================
// Per-program uniform refresh. Called on init and after each hot reload.
// =============================================================================
static void refreshMinimapUniforms(Renderer& R) {
    R.locMmViewport     = glGetUniformLocation(R.minimapProg.id, "viewport");
    R.locMmCamPos       = glGetUniformLocation(R.minimapProg.id, "camPos");
    R.locMmCamFwd       = glGetUniformLocation(R.minimapProg.id, "camFwd");
    R.locMmCamDist      = glGetUniformLocation(R.minimapProg.id, "camDist");
    R.locMmVoxPerPx     = glGetUniformLocation(R.minimapProg.id, "voxPerPx");
    R.locMmTanHalfH     = glGetUniformLocation(R.minimapProg.id, "tanHalfH");
    R.locMmPlayer       = glGetUniformLocation(R.minimapProg.id, "playerXZ");
    R.locMmPlayerYaw    = glGetUniformLocation(R.minimapProg.id, "playerYaw");
    R.locMmMove         = glGetUniformLocation(R.minimapProg.id, "moveTargetXZ");
    R.locMmMoveOn       = glGetUniformLocation(R.minimapProg.id, "moveOn");
    R.locMmFreeCam      = glGetUniformLocation(R.minimapProg.id, "freeCamOn");
    R.locMmDayLight     = glGetUniformLocation(R.minimapProg.id, "dayLight");
    R.locMmGridSize     = glGetUniformLocation(R.minimapProg.id, "gridSize");
    R.locMmBricks       = glGetUniformLocation(R.minimapProg.id, "bricksPerAxis");
    R.locMmWindowOrigin = glGetUniformLocation(R.minimapProg.id, "windowOriginB");
    R.locMmSurfaceY     = glGetUniformLocation(R.minimapProg.id, "surfaceY");
    R.locMmMaskOrigin   = glGetUniformLocation(R.minimapProg.id, "maskOrigin");
    R.locMmCreN         = glGetUniformLocation(R.minimapProg.id, "creN");
    R.locMmCre          = glGetUniformLocation(R.minimapProg.id, "creXZ");
    R.locMmHeightTex    = glGetUniformLocation(R.minimapProg.id, "heightTex");
    R.locMmExplored     = glGetUniformLocation(R.minimapProg.id, "exploredSoft");
}
static void refreshParticleUniforms(Renderer& R) {
    R.locPartViewProj = glGetUniformLocation(R.particle.id, "viewProj");
    R.locPartCamPos   = glGetUniformLocation(R.particle.id, "camPos");
    R.locPartViewport = glGetUniformLocation(R.particle.id, "viewport");
    R.locPartGPos     = glGetUniformLocation(R.particle.id, "gPosMat");
}
static void refreshHudUniforms(Renderer& R) {
    R.locHudViewport = glGetUniformLocation(R.hud.id, "viewport");
    R.locHudMat      = glGetUniformLocation(R.hud.id, "hudMat");
    R.locHudCount    = glGetUniformLocation(R.hud.id, "hudCount");
    R.locHudSelected = glGetUniformLocation(R.hud.id, "hudSelected");
    R.locHudTrayOn   = glGetUniformLocation(R.hud.id, "trayOn");
    R.locHudDevMode  = glGetUniformLocation(R.hud.id, "devMode");
    R.locHudName     = glGetUniformLocation(R.hud.id, "hudName");
    R.locHudNameFade = glGetUniformLocation(R.hud.id, "hudNameFade");
    R.locHudFont     = glGetUniformLocation(R.hud.id, "hudFont");
    R.locHudIcon     = glGetUniformLocation(R.hud.id, "hudIcon");
}
static void refreshPauseUniforms(Renderer& R) {
    R.locPauseViewport = glGetUniformLocation(R.pause.id, "viewport");
}
static void refreshConsoleUniforms(Renderer& R) {
    R.locConViewport = glGetUniformLocation(R.consoleProg.id, "viewport");
    R.locConChars    = glGetUniformLocation(R.consoleProg.id, "chars");
    R.locConFont     = glGetUniformLocation(R.consoleProg.id, "font");
}
static void refreshCtxUniforms(Renderer& R) {
    R.locCtxViewport = glGetUniformLocation(R.ctxProg.id, "viewport");
    R.locCtxMenuPos  = glGetUniformLocation(R.ctxProg.id, "menuPos");
    R.locCtxRowN     = glGetUniformLocation(R.ctxProg.id, "rowN");
    R.locCtxHover    = glGetUniformLocation(R.ctxProg.id, "hoverRow");
    R.locCtxChars    = glGetUniformLocation(R.ctxProg.id, "chars");
    R.locCtxFont     = glGetUniformLocation(R.ctxProg.id, "font");
    R.locCtxIcons    = glGetUniformLocation(R.ctxProg.id, "icons");
}
static void refreshDebugUniforms(Renderer& R) {
    R.locDbgViewport = glGetUniformLocation(R.debugProg.id, "viewport");
    R.locDbgChars    = glGetUniformLocation(R.debugProg.id, "chars");
    R.locDbgFont     = glGetUniformLocation(R.debugProg.id, "font");
}
static void refreshMapviewUniforms(Renderer& R) {
    R.locMvTex  = glGetUniformLocation(R.mapviewProg.id, "mapTex");
    R.locMvVeil = glGetUniformLocation(R.mapviewProg.id, "veil");
}
static void refreshGBufferUniforms(Renderer& R) {
    GLuint p = R.gbufferProg.id;
    R.locGBCamPos        = glGetUniformLocation(p, "camPos");
    R.locGBCamFwd        = glGetUniformLocation(p, "camFwd");
    R.locGBCamRight      = glGetUniformLocation(p, "camRight");
    R.locGBCamUp         = glGetUniformLocation(p, "camUp");
    R.locGBViewport      = glGetUniformLocation(p, "viewport");
    R.locGBTanHalfFov    = glGetUniformLocation(p, "tanHalfFov");
    R.locGBGridSize      = glGetUniformLocation(p, "gridSize");
    R.locGBBricks        = glGetUniformLocation(p, "bricksPerAxis");
    R.locGBSurfaceY      = glGetUniformLocation(p, "surfaceY");
    R.locGBWindowOrigin  = glGetUniformLocation(p, "windowOriginB");
    R.locGBTime          = glGetUniformLocation(p, "uTime");
    R.locGBHeightTex     = glGetUniformLocation(p, "heightTex");
    R.locGBJitter        = glGetUniformLocation(p, "jitterNdc");
    R.locGBRingOn          = glGetUniformLocation(p, "ringOn");
    R.locGBRingMask        = glGetUniformLocation(p, "ringReadyMask");
    R.locGBRingOrigins     = glGetUniformLocation(p, "ringOriginBs");
    R.locGBRingBricksY     = glGetUniformLocation(p, "ringBricksYs");
    R.locGBRingBmBases     = glGetUniformLocation(p, "ringBmBases");
    R.locGBRingCoarseBases = glGetUniformLocation(p, "ringCoarseBases");
    R.locGBChunkValid      = glGetUniformLocation(p, "chunkValidTex");
    R.locGBMapPass   = glGetUniformLocation(p, "mapPass");
    R.locGBMapFineVpp = glGetUniformLocation(p, "mapFineVpp");
    R.locGBClipLo    = glGetUniformLocation(p, "mapClipLo");
    R.locGBClipHi    = glGetUniformLocation(p, "mapClipHi");
    R.locGBExplored  = glGetUniformLocation(p, "mapExploredTex");
    R.locGBExploredY = glGetUniformLocation(p, "mapExploredYTex");
    R.locGBMaskOrigin = glGetUniformLocation(p, "mapMaskOrigin");
}
static void refreshMotionUniforms(Renderer& R) {
    GLuint p = R.motionProg.id;
    R.locMoPosMat         = glGetUniformLocation(p, "gPosMat");
    R.locMoNormal         = glGetUniformLocation(p, "gNormal");
    R.locMoCamPos         = glGetUniformLocation(p, "camPos");
    R.locMoCamFwd         = glGetUniformLocation(p, "camFwd");
    R.locMoCamRight       = glGetUniformLocation(p, "camRight");
    R.locMoCamUp          = glGetUniformLocation(p, "camUp");
    R.locMoTanHalfFov     = glGetUniformLocation(p, "tanHalfFov");
    R.locMoViewport       = glGetUniformLocation(p, "viewport");
    R.locMoPrevPos        = glGetUniformLocation(p, "prevPos");
    R.locMoPrevFwd        = glGetUniformLocation(p, "prevFwd");
    R.locMoPrevRight      = glGetUniformLocation(p, "prevRight");
    R.locMoPrevUp         = glGetUniformLocation(p, "prevUp");
    R.locMoPrevTanHalfFov = glGetUniformLocation(p, "prevTanHalfFov");
    R.locMoPrevAspect     = glGetUniformLocation(p, "prevAspect");
    R.locMoPrevValid      = glGetUniformLocation(p, "prevValid");
}
static void refreshCreatureUniforms(Renderer& R) {
    GLuint p = R.creatureProg.id;
    R.locCrCamPos     = glGetUniformLocation(p, "camPos");
    R.locCrCamFwd     = glGetUniformLocation(p, "camFwd");
    R.locCrCamRight   = glGetUniformLocation(p, "camRight");
    R.locCrCamUp      = glGetUniformLocation(p, "camUp");
    R.locCrViewport   = glGetUniformLocation(p, "viewport");
    R.locCrTanHalfFov = glGetUniformLocation(p, "tanHalfFov");
    R.locCrJitter     = glGetUniformLocation(p, "jitterNdc");
    R.locCrMapPass    = glGetUniformLocation(p, "mapPass");
    R.locCrClipLo     = glGetUniformLocation(p, "mapClipLo");
    R.locCrClipHi     = glGetUniformLocation(p, "mapClipHi");
}

static void refreshLightingUniforms(Renderer& R) {
    GLuint p = R.lightingProg.id;
    R.locLiCamPos        = glGetUniformLocation(p, "camPos");
    R.locLiCamFwd        = glGetUniformLocation(p, "camFwd");
    R.locLiCamRight      = glGetUniformLocation(p, "camRight");
    R.locLiCamUp         = glGetUniformLocation(p, "camUp");
    R.locLiViewport      = glGetUniformLocation(p, "viewport");
    R.locLiTanHalfFov    = glGetUniformLocation(p, "tanHalfFov");
    R.locLiGridSize      = glGetUniformLocation(p, "gridSize");
    R.locLiBricks        = glGetUniformLocation(p, "bricksPerAxis");
    R.locLiSunDir        = glGetUniformLocation(p, "sunDir");
    R.locLiDayLight      = glGetUniformLocation(p, "dayLight");
    R.locLiTime          = glGetUniformLocation(p, "uTime");
    R.locLiSurfaceY      = glGetUniformLocation(p, "surfaceY");
    R.locLiHeightTex     = glGetUniformLocation(p, "heightTex");
    R.locLiLightVol      = glGetUniformLocation(p, "lightVol");
    R.locLiFarHeightTex  = glGetUniformLocation(p, "farHeightTex");
    R.locLiFarOrigin     = glGetUniformLocation(p, "farOrigin");
    R.locLiFarStride     = glGetUniformLocation(p, "farStride");
    R.locLiFarTexels     = glGetUniformLocation(p, "farTexels");
    R.locLiFireN         = glGetUniformLocation(p, "uFireN");
    R.locLiFireLights    = glGetUniformLocation(p, "uFireLights");
    R.locLiCreN          = glGetUniformLocation(p, "uCreN");
    R.locLiCreBound      = glGetUniformLocation(p, "uCreBound");
    R.locLiCreRange      = glGetUniformLocation(p, "uCreRange");
    R.locLiAlbedoTex     = glGetUniformLocation(p, "gAlbedoTex");
    R.locLiPlayerAvatar  = glGetUniformLocation(p, "uPlayerAvatar");
    R.locLiCreBlobSkip   = glGetUniformLocation(p, "uCreBlobSkip");
    R.locLiPosMat        = glGetUniformLocation(p, "gPosMat");
    R.locLiNormal        = glGetUniformLocation(p, "gNormal");
    R.locLiMoonDir       = glGetUniformLocation(p, "moonDir");
    R.locLiMoonLight     = glGetUniformLocation(p, "moonLight");
    R.locLiWindowOrigin  = glGetUniformLocation(p, "windowOriginB");
    R.locLiJitter        = glGetUniformLocation(p, "jitterNdc");
    R.locLiColorMode     = glGetUniformLocation(p, "colorMode");
    R.locLiSlopeShade    = glGetUniformLocation(p, "slopeShade");
    R.locLiFogDensity    = glGetUniformLocation(p, "fogDensity");
    R.locLiTweak         = glGetUniformLocation(p, "uTweak");
    R.locLiMatTune       = glGetUniformLocation(p, "uMatTune");
    R.locLiFogFloor      = glGetUniformLocation(p, "fogFloor");
    R.locLiMapPass       = glGetUniformLocation(p, "mapPass");
    R.locLiHoverOn       = glGetUniformLocation(p, "uHoverOn");
    R.locLiHoverLo       = glGetUniformLocation(p, "uHoverLo");
    R.locLiHoverDims     = glGetUniformLocation(p, "uHoverDims");
    R.locLiHoverStrength = glGetUniformLocation(p, "uHoverStrength");
    R.locLiHoverMask     = glGetUniformLocation(p, "uHoverMask");
    R.locLiMapPlayer     = glGetUniformLocation(p, "mapPlayerPos");
    R.locLiMapCover      = glGetUniformLocation(p, "mapCover");
    R.locLiMapGhost      = glGetUniformLocation(p, "mapGhost");
    R.locLiMapStandoff   = glGetUniformLocation(p, "mapStandoff");
    R.locLiMapPath       = glGetUniformLocation(p, "mapPath");
    R.locLiMapPathN      = glGetUniformLocation(p, "mapPathN");
    R.locLiMapYaw        = glGetUniformLocation(p, "mapPlayerYaw");
    R.locLiMapWalk       = glGetUniformLocation(p, "mapWalkPhase");
    R.locLiMapMoveTarget = glGetUniformLocation(p, "mapMoveTarget");
    R.locLiMapMoveOn     = glGetUniformLocation(p, "mapMoveOn");
    R.locLiFarmOn        = glGetUniformLocation(p, "mapFarmOn");
    R.locLiFarmOk        = glGetUniformLocation(p, "mapFarmOk");
    R.locLiFarmMin       = glGetUniformLocation(p, "mapFarmMin");
    R.locLiFarmSize      = glGetUniformLocation(p, "mapFarmSize");
    R.locLiMapHoverOn    = glGetUniformLocation(p, "mapHoverOn");
    R.locLiMapHoverLo    = glGetUniformLocation(p, "mapHoverLo");
    R.locLiMapHoverHi    = glGetUniformLocation(p, "mapHoverHi");
    R.locLiRingOn        = glGetUniformLocation(p, "mapRingOn");
    R.locLiRingPos       = glGetUniformLocation(p, "mapRingPos");
    R.locLiRingT         = glGetUniformLocation(p, "mapRingT");
    R.locLiExplored      = glGetUniformLocation(p, "mapExploredTex");
    R.locLiExploredSoft  = glGetUniformLocation(p, "mapExploredSoftTex");
    R.locLiMaskOrigin    = glGetUniformLocation(p, "mapMaskOrigin");
}
static void refreshTaaUniforms(Renderer& R) {
    GLuint p = R.taaProg.id;
    R.locTaaCur       = glGetUniformLocation(p, "curTex");
    R.locTaaHist      = glGetUniformLocation(p, "histTex");
    R.locTaaMotion    = glGetUniformLocation(p, "motionTex");
    R.locTaaViewport  = glGetUniformLocation(p, "viewport");
    R.locTaaHistValid = glGetUniformLocation(p, "historyValid");
}

// =============================================================================
// Renderer init
// =============================================================================
void Renderer::init(const std::vector<float>& heightMap, const std::vector<float>& lightVol) {
    // Programs from disk, with hot-reload tracking.
    initialLoad(particle, "particle.vert",   "particle.frag", "particle");
    initialLoad(creatureProg, "creature.vert", "creature.frag", "creature");
    initialLoad(hud,      "fullscreen.vert", "hud.frag",      "hud");
    initialLoad(pause,    "fullscreen.vert", "pause.frag",    "pause");
    initialLoad(consoleProg, "fullscreen.vert", "console.frag", "console");
    initialLoad(ctxProg,     "fullscreen.vert", "ctxmenu.frag", "ctxmenu");
    refreshCtxUniforms(*this);
    initialLoad(tuneProg,    "fullscreen.vert", "tune.frag",    "tune");
    locTuneViewport = glGetUniformLocation(tuneProg.id, "viewport");
    locTuneChars    = glGetUniformLocation(tuneProg.id, "chars");
    locTuneFont     = glGetUniformLocation(tuneProg.id, "font");
    locTuneSel      = glGetUniformLocation(tuneProg.id, "selRow");
    locTuneRows     = glGetUniformLocation(tuneProg.id, "rows");
    initialLoad(minimapProg, "fullscreen.vert", "minimap.frag", "minimap");
    refreshMinimapUniforms(*this);
    initialLoad(farmBarProg, "fullscreen.vert", "farmbar.frag", "farmbar");
    locFbViewport = glGetUniformLocation(farmBarProg.id, "viewport");
    locFbChars    = glGetUniformLocation(farmBarProg.id, "chars");
    locFbFont     = glGetUniformLocation(farmBarProg.id, "font");
    locFbCols     = glGetUniformLocation(farmBarProg.id, "cols");
    locFbSlotW    = glGetUniformLocation(farmBarProg.id, "slotw");
    locFbSel      = glGetUniformLocation(farmBarProg.id, "sel");
    locFbRow      = glGetUniformLocation(farmBarProg.id, "rowIdx");
    initialLoad(bagProg, "fullscreen.vert", "bag.frag", "bag");
    locBagViewport = glGetUniformLocation(bagProg.id, "viewport");
    locBagItem     = glGetUniformLocation(bagProg.id, "bagItem");
    locBagCount    = glGetUniformLocation(bagProg.id, "bagCount");
    locBagSelected = glGetUniformLocation(bagProg.id, "selected");
    locBagHover    = glGetUniformLocation(bagProg.id, "hover");
    locBagTidyHot  = glGetUniformLocation(bagProg.id, "tidyHot");
    locBagFont     = glGetUniformLocation(bagProg.id, "font");
    locBagHdr      = glGetUniformLocation(bagProg.id, "hdr");
    locBagFtr      = glGetUniformLocation(bagProg.id, "ftr");
    locBagTdy      = glGetUniformLocation(bagProg.id, "tdy");
    locBagIcon     = glGetUniformLocation(bagProg.id, "bagIcon");
    initialLoad(debugProg, "fullscreen.vert", "debug.frag",    "debug");
    initialLoad(gbufferProg,  "fullscreen.vert", "gbuffer.frag",  "gbuffer");
    initialLoad(motionProg,   "fullscreen.vert", "motion.frag",   "motion");
    initialLoad(taaProg,      "fullscreen.vert", "taa.frag",      "taa");
    initialLoad(lightingProg, "fullscreen.vert", "lighting.frag", "lighting");
    initialLoad(mapviewProg, "fullscreen.vert", "mapview.frag", "mapview");
    refreshMapviewUniforms(*this);
    refreshParticleUniforms(*this);
    refreshHudUniforms(*this);
    refreshPauseUniforms(*this);
    refreshConsoleUniforms(*this);
    refreshDebugUniforms(*this);
    refreshGBufferUniforms(*this);
    refreshLightingUniforms(*this);
    refreshMotionUniforms(*this);
    refreshTaaUniforms(*this);
    refreshCreatureUniforms(*this);

    glGenVertexArrays(1, (GLuint*)&sceneVAO);
    // Creature boxes are generated from gl_VertexID — attribute-less VAO.
    glGenVertexArrays(1, (GLuint*)&creatureVAO);

    // Particle VBO.
    glGenVertexArrays(1, (GLuint*)&partVAO);
    glGenBuffers(1, (GLuint*)&partVBO);
    glBindVertexArray(partVAO);
    glBindBuffer(GL_ARRAY_BUFFER, partVBO);
    glBufferData(GL_ARRAY_BUFFER, ParticleSystem::MAX * 7 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glBindVertexArray(0);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Material UBO: matColors[256] (rgb albedo) followed by matProps[256]
    // (x=smoothness, y=metallic, z=emissive) — two std140 vec4 arrays, so the
    // props array begins at element MAT_COUNT in this flat buffer.
    glGenBuffers(1, (GLuint*)&matColorUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, matColorUBO);
    std::vector<glm::vec4> data(MAT_COUNT * 2);
    for (int i = 0; i < MAT_COUNT; ++i) {
        const MatProps& m = g_materials[i];
        data[i]             = glm::vec4(m.color, 0.0f);
        data[MAT_COUNT + i] = glm::vec4(m.smoothness, m.metallic, m.emissive, 0.0f);
    }
    glBufferData(GL_UNIFORM_BUFFER, data.size() * sizeof(glm::vec4),
                 data.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, matColorUBO);

    // Terrain height texture (per-column surface Y) for per-point cave depth.
    // REPEAT wrap: the shader samples worldPos/GRID, so wrapping IS the
    // toroidal window mapping — scrolling overwrites the entering strip's
    // texels in place (updateHeightRect).
    glGenTextures(1, (GLuint*)&heightTex);
    glBindTexture(GL_TEXTURE_2D, heightTex);
    if (!heightMap.empty()) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, GRID_XZ, GRID_XZ, 0,
                     GL_RED, GL_FLOAT, heightMap.data());
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Per-chunk-slot GPU residency (see render.h) — refreshed each frame in
    // fillGBuffer (1 KB), consulted per ray by the fine march.
    glGenTextures(1, (GLuint*)&chunkValidTex);
    glBindTexture(GL_TEXTURE_2D, chunkValidTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, CHUNKS_XZ, CHUNKS_XZ, 0,
                 GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Explored-world map masks (see map.h). Allocated zeroed up front so the
    // gbuffer program's map samplers always have correctly-typed textures on
    // their units, even before the map first opens; renderMap re-uploads them
    // lazily when the app's mask stamp changes.
    {
        constexpr int MCH = MapMasks::MASK_CH;
        std::vector<uint8_t> zero8(size_t(MCH) * MCH, 0);
        glGenTextures(1, (GLuint*)&exploredTex);
        glBindTexture(GL_TEXTURE_2D, exploredTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, MCH, MCH, 0,
                     GL_RED_INTEGER, GL_UNSIGNED_BYTE, zero8.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        std::vector<uint32_t> zero32(
            size_t(MCH) * MCH * MapMasks::YSLICE_U32, 0);
        glGenTextures(1, (GLuint*)&exploredYTex);
        glBindTexture(GL_TEXTURE_3D, exploredYTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI,
                     MCH, MapMasks::YSLICE_U32, MCH, 0,
                     GL_RED_INTEGER, GL_UNSIGNED_INT, zero32.data());
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Prefiltered fog-of-war ramp (phase A): R8 normalized + BILINEAR —
        // one hardware tap replaces the shader's 100-fetch box filter.
        // Border 0 = the darkness beyond the mask window.
        glGenTextures(1, (GLuint*)&exploredSoftTex);
        glBindTexture(GL_TEXTURE_2D, exploredSoftTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, MCH, MCH, 0,
                     GL_RED, GL_UNSIGNED_BYTE, zero8.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float darkBorder[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, darkBorder);
    }

    // Emissive light volume (3D, RGB16F). Trilinear filtering gives the soft
    // glow falloff. X(S)/Z(R) wrap GL_REPEAT so the volume is a toroidal
    // window that follows the origin (like heightTex); Y(T) clamps (the
    // volume is full world height, no vertical wrap). Built by
    // buildLightVolume() at the resident origin.
    glGenTextures(1, (GLuint*)&lightVolTex);
    glBindTexture(GL_TEXTURE_3D, lightVolTex);
    if (!lightVol.empty()) {
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F, LV_X, LV_Y, LV_Z, 0,
                     GL_RGB, GL_FLOAT, lightVol.data());
    }
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

}

void Renderer::updateLightVolume(const std::vector<float>& lightVol) {
    if (lightVol.empty() || !lightVolTex) return;
    glBindTexture(GL_TEXTURE_3D, lightVolTex);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, LV_X, LV_Y, LV_Z,
                    GL_RGB, GL_FLOAT, lightVol.data());
}

// Re-upload only a toroidal strip of the light volume (paired with
// buildLightVolumeStrip): xAxis -> an x-range [off,off+cnt) over full Y/Z;
// else a z-range over full X/Y. `cnt` may wrap the edge, split into two calls.
// ROW_LENGTH/IMAGE_HEIGHT let glTexSubImage3D pick the strided sub-columns
// straight out of the full-width CPU array — no repacking.
void Renderer::updateLightVolumeRect(const std::vector<float>& lv,
                                     bool xAxis, int off, int cnt) {
    if (lv.empty() || !lightVolTex || cnt <= 0) return;
    glBindTexture(GL_TEXTURE_3D, lightVolTex);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, LV_X);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, LV_Y);
    auto blit = [&](int o, int c) {
        if (xAxis)
            glTexSubImage3D(GL_TEXTURE_3D, 0, o, 0, 0, c, LV_Y, LV_Z,
                            GL_RGB, GL_FLOAT, lv.data() + size_t(o) * 3);
        else
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, o, LV_X, LV_Y, c,
                            GL_RGB, GL_FLOAT,
                            lv.data() + size_t(o) * LV_Y * LV_X * 3);
    };
    int dim = xAxis ? LV_X : LV_Z;
    if (off + cnt <= dim) blit(off, cnt);
    else { blit(off, dim - off); blit(0, off + cnt - dim); }
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
}

// Re-upload an arbitrary toroidal cell-box (full Y) given WORLD LV-cell ranges
// (paired with splatNewLights). Splits each axis at the toroidal seam (up to 4
// glTexSubImage3D calls); a box spanning a whole axis falls back to a full
// upload. ptr at (x,y=0,z) = ((z*LV_Y)*LV_X + x)*3.
void Renderer::updateLightVolumeBox(const std::vector<float>& lv,
                                    int wbx0, int wbx1, int wbz0, int wbz1) {
    if (lv.empty() || !lightVolTex) return;
    int wx = wbx1 - wbx0, wz = wbz1 - wbz0;
    if (wx <= 0 || wz <= 0) return;
    if (wx >= LV_X || wz >= LV_Z) { updateLightVolume(lv); return; }
    int xs[2], xw[2], nx;
    { int x0 = wbx0 & (LV_X - 1);
      if (x0 + wx <= LV_X) { xs[0] = x0; xw[0] = wx; nx = 1; }
      else { xs[0] = x0; xw[0] = LV_X - x0; xs[1] = 0; xw[1] = wx - (LV_X - x0); nx = 2; } }
    int zs[2], zw[2], nz;
    { int z0 = wbz0 & (LV_Z - 1);
      if (z0 + wz <= LV_Z) { zs[0] = z0; zw[0] = wz; nz = 1; }
      else { zs[0] = z0; zw[0] = LV_Z - z0; zs[1] = 0; zw[1] = wz - (LV_Z - z0); nz = 2; } }
    glBindTexture(GL_TEXTURE_3D, lightVolTex);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, LV_X);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, LV_Y);
    for (int ix = 0; ix < nx; ++ix)
    for (int iz = 0; iz < nz; ++iz)
        glTexSubImage3D(GL_TEXTURE_3D, 0, xs[ix], 0, zs[iz], xw[ix], LV_Y, zw[iz],
                        GL_RGB, GL_FLOAT,
                        lv.data() + (size_t(zs[iz]) * LV_Y * LV_X + xs[ix]) * 3);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
}

// =============================================================================
// Explored-world map (the frame): the orbit diorama. Two passes —
// gbufferProg with mapPass=1 into the map's OWN G-buffer (clip box + fog-of-war
// gating against the explored masks), then lightingProg with mapPass=1 straight
// to the default framebuffer (flat analytic shading via the ring path, paper
// background, player pin). Fixed studio-noon lighting keeps the map readable
// regardless of in-game time; live uTime keeps water shimmering.
// =============================================================================
void Renderer::renderMap(const World& w, const FrameView& fv) {
    VOX_ZONE_N("renderMap");
    const MapFrame& mf = fv.map;
    // Lazy explored-mask upload, keyed by the app's mask stamp (new marks or
    // a resident-window origin change).
    if (mf.masks && exploredUploaded != mf.masksVersion) {
        constexpr int MCH = MapMasks::MASK_CH;
        glBindTexture(GL_TEXTURE_2D, exploredTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MCH, MCH,
                        GL_RED_INTEGER, GL_UNSIGNED_BYTE, mf.masks->flags.data());
        glBindTexture(GL_TEXTURE_3D, exploredYTex);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                        MCH, MapMasks::YSLICE_U32, MCH,
                        GL_RED_INTEGER, GL_UNSIGNED_INT, mf.masks->ymask.data());
        glBindTexture(GL_TEXTURE_2D, exploredSoftTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MCH, MCH,
                        GL_RED, GL_UNSIGNED_BYTE, mf.masks->soft.data());
        exploredUploaded = mf.masksVersion;
    }

    // Zoom-driven LOD (phase A): when fine voxels sink below ~1.5 px, the
    // fine march hands the frame to ring0 (2x voxels, covers the whole
    // resident window) — TAA was averaging the detail away anyway. The
    // handoff is DITHERED in the shader (per-ray world-anchored coin whose
    // bias ramps over vpp 1.35..1.55), so the transition cross-fades under
    // TAA instead of popping the whole frame in one flip. Thresholds sit
    // HIGH: rings carry no player edits or small flora, so the swap must
    // hide inside sub-pixel mush. X-ray re-marches fine regardless.
    float mapVpp = 2.0f * MAP_TAN_HALF_FOV * mf.camDist / float(fv.height);
    if (!fv.ringsOn) mapVpp = 0.0f;   // no rings = nothing else to draw

    glm::vec3 f = glm::normalize(mf.camFwd);
    glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
    glm::vec3 u = glm::normalize(glm::cross(r, f));
    // The diorama lives under the REAL sky: live sun/moon direction and
    // day/night phase, live fire lights, live fog. (Fixed studio noon was
    // the map-as-overlay look.)
    // Map anti-aliasing: sub-pixel-jitter the map camera with the Halton
    // sequence and resolve against the map's own history below. curJitterNdc
    // is zero when /taa is off.
    bool mapTaa = fv.taa;

    // 1) G-buffer fill at window resolution into the map's own G-buffer.
    VOX_GPU_ZONE_BEGIN(zMapGbuf, "map gbuffer");
    ensureMapSurfaces(fv.width, fv.height);
    mapGbuf.ensure(fv.width, fv.height);
    mapGbuf.bindForWrite();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(gbufferProg.id);
    glUniform3fv(locGBCamPos,   1, &mf.camPos.x);
    glUniform3fv(locGBCamFwd,   1, &f.x);
    glUniform3fv(locGBCamRight, 1, &r.x);
    glUniform3fv(locGBCamUp,    1, &u.x);
    glUniform2f (locGBViewport, float(fv.width), float(fv.height));
    glUniform1f (locGBTanHalfFov, MAP_TAN_HALF_FOV);
    glUniform3i (locGBGridSize, GRID_XZ, GRID_Y, GRID_XZ);
    glUniform3i (locGBBricks,   BRICKS_XZ, BRICKS_Y, BRICKS_XZ);
    glUniform1f (locGBSurfaceY, float(SURFACE_Y));
    glUniform2i (locGBWindowOrigin, windowOriginB.x, windowOriginB.y);
    glUniform1f (locGBTime, fv.time);
    glUniform2f (locGBJitter, curJitterNdc.x, curJitterNdc.y);
    // The LOD rings carry the diorama beyond the resident window (fine voxels
    // where resident, ring cascade outside), so panning isn't fenced to the
    // window. Respects the console's ring toggle like the main view.
    glUniform1i (locGBRingOn, fv.ringsOn ? 1 : 0);
    // Ring windows/bases (mirrors fillGBuffer): this pass must own EVERY
    // uniform it reads — in single-renderer mode fillGBuffer never runs, so
    // inherited program state is zeros, not "whatever the main view set".
    {
        int mask = 0, originV[RING_COUNT * 2], bricksY[RING_COUNT];
        int bmBase[RING_COUNT], coarseBase[RING_COUNT];
        for (int ri = 0; ri < RING_COUNT; ++ri) {
            if (w.rings[ri].ready) mask |= 1 << ri;
            originV[ri * 2]     = w.rings[ri].originGPU.x;
            originV[ri * 2 + 1] = w.rings[ri].originGPU.y;
            bricksY[ri]    = ringBricksY(ri);
            bmBase[ri]     = int(World::ringBmOffset(ri));
            coarseBase[ri] = int(World::ringCoarseOffset(ri));
        }
        glUniform1i (locGBRingMask, mask);
        glUniform2iv(locGBRingOrigins, RING_COUNT, originV);
        glUniform1iv(locGBRingBricksY, RING_COUNT, bricksY);
        glUniform1iv(locGBRingBmBases, RING_COUNT, bmBase);
        glUniform1iv(locGBRingCoarseBases, RING_COUNT, coarseBase);
    }
    glUniform1i (locGBMapPass, 1);
    glUniform1f (locGBMapFineVpp, mapVpp);
    glUniform3fv(locGBClipLo, 1, &mf.clipLo.x);
    glUniform3fv(locGBClipHi, 1, &mf.clipHi.x);
    glUniform2i (locGBMaskOrigin, mf.maskOrigin.x, mf.maskOrigin.y);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w.brickmapSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w.brickpoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, w.coarseSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3,
                     w.ringPoolSSBO ? w.ringPoolSSBO : w.brickpoolSSBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, matColorUBO);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, heightTex); glUniform1i(locGBHeightTex, 0);
    // Chunk residency, refreshed like fillGBuffer's copy — streaming uploads
    // can still land while the map is open.
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, chunkValidTex);
    if (!w.gpuChunkValid.empty())
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, CHUNKS_XZ, CHUNKS_XZ,
                        GL_RED_INTEGER, GL_UNSIGNED_BYTE, w.gpuChunkValid.data());
    glUniform1i(locGBChunkValid, 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, exploredTex);  glUniform1i(locGBExplored, 2);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D, exploredYTex); glUniform1i(locGBExploredY, 3);
    // Depth + creature injection, mirroring fillGBuffer: the diorama is the
    // primary view, so wildlife must land in ITS G-buffer (the X-ray re-march
    // below deliberately excludes them — surface fauna has no cave layer).
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    drawCreatures(fv, mf.camPos, f, r, u, MAP_TAN_HALF_FOV);
    glDisable(GL_DEPTH_TEST);
    VOX_GPU_ZONE_END(zMapGbuf);

    // 2) Shade into the map's offscreen target with fixed map lighting
    //    (alpha = diorama coverage; the composite below reads it).
    VOX_GPU_ZONE_BEGIN(zMapShade, "map shade");
    glBindFramebuffer(GL_FRAMEBUFFER, mapSceneFBO);
    glViewport(0, 0, fv.width, fv.height);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(lightingProg.id);
    glUniform3fv(locLiCamPos,   1, &mf.camPos.x);
    glUniform3fv(locLiCamFwd,   1, &f.x);
    glUniform3fv(locLiCamRight, 1, &r.x);
    glUniform3fv(locLiCamUp,    1, &u.x);
    glUniform2f (locLiViewport, float(fv.width), float(fv.height));
    glUniform1f (locLiTanHalfFov, MAP_TAN_HALF_FOV);
    glUniform3i (locLiGridSize, GRID_XZ, GRID_Y, GRID_XZ);
    glUniform3i (locLiBricks,   BRICKS_XZ, BRICKS_Y, BRICKS_XZ);
    glUniform3fv(locLiSunDir, 1, &fv.sunDir.x);
    glUniform1f (locLiDayLight, fv.dayLight);
    glUniform3fv(locLiMoonDir, 1, &fv.moonDir.x);
    glUniform1f (locLiMoonLight, fv.moonLight);
    glUniform2i (locLiWindowOrigin, windowOriginB.x, windowOriginB.y);
    glUniform1f (locLiTime,      fv.time);
    glUniform1f (locLiSurfaceY,  float(SURFACE_Y));
    // Live fire glow (clustered CPU-side around the player every frame) —
    // campfires light the diorama at night.
    glUniform1i (locLiFireN, fv.fireLightN);
    if (fv.fireLightN > 0)
        glUniform4fv(locLiFireLights, fv.fireLightN, &fv.fireLights[0].x);
    // Creature shadows + palette albedo: per-creature bounds/ranges, the two
    // part SSBOs, and this pass's own G-buffer albedo texture.
    glUniform1i(locLiPlayerAvatar, fv.playerAvatarOn);
    glUniform1i(locLiCreN, fv.creaturePartN > 0 ? fv.creatureN : 0);
    glUniform2i(locLiCreBlobSkip, fv.creBlobSkip.x, fv.creBlobSkip.y);
    if (fv.creatureN > 0 && fv.creaturePartN > 0) {
        glUniform4fv(locLiCreBound, fv.creatureN, &fv.creatureBound[0].x);
        glUniform2iv(locLiCreRange, fv.creatureN, &fv.creatureRange[0].x);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,
                     creatureSSBO ? creatureSSBO : w.brickpoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5,
                     creaturePartSSBO ? creaturePartSSBO : w.brickpoolSSBO);
    glActiveTexture(GL_TEXTURE10);
    glBindTexture(GL_TEXTURE_2D, mapGbuf.albedo);
    glUniform1i(locLiAlbedoTex, 10);
    // applyAtmosphere reads the height-fog uniforms on the map path now.
    glUniform1f (locLiFogDensity, fv.fogDensity);
    glUniform1f (locLiFogFloor, fv.fogFloor);
    // Lighting/tonemap tuning: sun strength, ambient,
    // saturation/ACES, ring AO, albedo clumps all live in uTweak — without
    // this upload the whole diorama shades to black in single-renderer mode
    // (uninitialized uniforms are zero, and sun x0 + ambient x0 = nothing).
    glUniform1fv(locLiTweak, TWEAK_GPU, fv.tweak);
    glUniform1fv(locLiMatTune, MATTUNE_N, fv.matTune);
    glUniform1i (locLiColorMode, fv.colorMode);
    glUniform1i (locLiSlopeShade, fv.slopeShade);
    glUniform1i (locLiMapPass, 1);
    glUniform3fv(locLiMapPlayer, 1, &mf.playerPos.x);
    glUniform1f (locLiMapYaw,  mf.playerYaw);
    glUniform1f (locLiMapWalk, mf.walkPhase);
    glUniform3fv(locLiMapMoveTarget, 1, &mf.moveTarget.x);
    glUniform1i (locLiMapMoveOn, mf.moveActive);
    glUniform1i (locLiFarmOn,   mf.farmCurOn);
    glUniform1i (locLiFarmOk,   mf.farmOk);
    glUniform3fv(locLiFarmMin, 1, &mf.farmMin.x);
    glUniform1f (locLiFarmSize, mf.farmSize);
    glUniform1i (locLiMapHoverOn, mf.hoverOn);
    glUniform3fv(locLiMapHoverLo, 1, &mf.hoverLo.x);
    glUniform3fv(locLiMapHoverHi, 1, &mf.hoverHi.x);
    glUniform1i (locLiRingOn, mf.ringOn);
    glUniform3fv(locLiRingPos, 1, &mf.ringPos.x);
    glUniform1f (locLiRingT, mf.ringT);
    glUniform1i (locLiMapPathN, mf.pathN);
    if (mf.pathN > 0)
        glUniform4fv(locLiMapPath, mf.pathN, &mf.path[0].x);
    glUniform2i (locLiMaskOrigin, mf.maskOrigin.x, mf.maskOrigin.y);
    // The diorama is the whole frame (single renderer): coverage is always
    // full, and the X-ray ghosts the surface by BRIGHTNESS in the shader
    // instead of by alpha — there is no live scene behind it to see through to.
    glUniform1f (locLiMapCover, 1.0f);
    glUniform1f (locLiMapGhost, mf.xray ? 0.12f : 1.0f);
    glUniform1f (locLiMapStandoff, mf.camDist);
    glUniform2f (locLiJitter, curJitterNdc.x, curJitterNdc.y);
    // Far heightfield ON for the sun-shadow marches (farSunVis: terrain-cast
    // shadows + shadowed fog/haze under the live sun). No horizon can render
    // behind the diorama — the map miss branch returns sky first.
    glUniform1f (locLiFarTexels, float(FAR_TEX));
    // Slope shading normalizes far-height gradients even when farTexels is 0;
    // a zero farStride is normalize(vec3(0)) — undefined — on every
    // ring-cascade hit.
    glUniform2f (locLiFarOrigin, fv.farOrigin.x, fv.farOrigin.y);
    glUniform1f (locLiFarStride, float(FAR_STRIDE));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w.brickmapSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w.brickpoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, w.coarseSSBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, matColorUBO);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mapGbuf.posMat); glUniform1i(locLiPosMat, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, mapGbuf.normal); glUniform1i(locLiNormal, 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, heightTex);      glUniform1i(locLiHeightTex, 2);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D, lightVolTex);    glUniform1i(locLiLightVol, 3);
    glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, farHeightTex);   glUniform1i(locLiFarHeightTex, 6);
    glActiveTexture(GL_TEXTURE8); glBindTexture(GL_TEXTURE_2D, exploredTex);    glUniform1i(locLiExplored, 8);
    // Prefiltered ramp on its own unit (float sampler — the integer/float
    // unit-sharing rule only bites across TYPES).
    glActiveTexture(GL_TEXTURE9); glBindTexture(GL_TEXTURE_2D, exploredSoftTex); glUniform1i(locLiExploredSoft, 9);
    // Forestry hover highlight (this pass owns every uniform it reads). The
    // sampler parks on unit 3 (a valid 3D texture) when there's no mask —
    // a sampler3D defaulting to a unit holding a 2D texture is a draw-time
    // error on strict drivers even if never sampled.
    if (fv.hoverOn && hoverMaskTex) {
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_3D, hoverMaskTex);
        glUniform1i(locLiHoverMask, 11);
    } else {
        glUniform1i(locLiHoverMask, 3);
    }
    glUniform1i (locLiHoverOn, fv.hoverOn && hoverMaskTex ? 1 : 0);
    glUniform3fv(locLiHoverLo, 1, &fv.hoverLo.x);
    glUniform3fv(locLiHoverDims, 1, &fv.hoverDims.x);
    glUniform1f (locLiHoverStrength, fv.hoverStrength);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    VOX_GPU_ZONE_END(zMapShade);

    // 3) Motion vectors with the MAP camera, from the SURFACE layer's
    //    positions (full coverage — the X-ray pass below reuses mapGbuf, so
    //    this must run first). The main view has already consumed motionTex
    //    this frame; overwriting it here is safe.
    if (mapTaa) {
        VOX_GPU_ZONE("map motion");
        glBindFramebuffer(GL_FRAMEBUFFER, motionFBO);
        glViewport(0, 0, fv.width, fv.height);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(motionProg.id);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mapGbuf.posMat); glUniform1i(locMoPosMat, 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, mapGbuf.normal); glUniform1i(locMoNormal, 1);
        // Creature-part reprojection inputs. Without assets there are no
        // mat-38 pixels, so leaving the blocks unbound is never sampled.
        if (creaturePartSSBO) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, creaturePartSSBO);
        if (creatureSSBO)     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, creatureSSBO);
        glUniform3fv(locMoCamPos,   1, &mf.camPos.x);
        glUniform3fv(locMoCamFwd,   1, &f.x);
        glUniform3fv(locMoCamRight, 1, &r.x);
        glUniform3fv(locMoCamUp,    1, &u.x);
        glUniform1f (locMoTanHalfFov, MAP_TAN_HALF_FOV);
        glUniform2f (locMoViewport, float(fv.width), float(fv.height));
        glUniform3fv(locMoPrevPos,   1, &prevMapCamPos.x);
        glUniform3fv(locMoPrevFwd,   1, &prevMapCamFwd.x);
        glUniform3fv(locMoPrevRight, 1, &prevMapCamRight.x);
        glUniform3fv(locMoPrevUp,    1, &prevMapCamUp.x);
        glUniform1f (locMoPrevTanHalfFov, MAP_TAN_HALF_FOV);
        glUniform1f (locMoPrevAspect, float(fv.width) / float(fv.height));
        glUniform1i (locMoPrevValid, prevMapCamValid ? 1 : 0);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // 4) Cave X-ray layer: re-march the same G-buffer with mapPass=2
    //    (discovered underground SPACE is the solid), shade it as glow, and
    //    blend it over the ghosted surface in mapSceneTex. Every other
    //    uniform persists on both programs from the surface passes.
    if (mf.xray) {
        VOX_GPU_ZONE("map xray");
        mapGbuf.bindForWrite();
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(gbufferProg.id);
        glUniform1i(locGBMapPass, 2);
        // Veins need real voxel data — the X-ray stays fine-window-only.
        glUniform1i(locGBRingOn, 0);
        // Rebind this program's texture units: the surface LIGHTING pass has
        // since parked float textures on units 0-3, and the integer samplers
        // (chunk residency + explored masks) read undefined values off them —
        // which zeroes the fog gates and skips every chunk (no veins).
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, heightTex);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, chunkValidTex);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, exploredTex);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D, exploredYTex);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindFramebuffer(GL_FRAMEBUFFER, mapSceneFBO);
        glViewport(0, 0, fv.width, fv.height);
        glUseProgram(lightingProg.id);
        glUniform1i(locLiMapPass, 2);
        glUniform1i(locLiHoverOn, 0);                    // X-ray: no hover
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mapGbuf.posMat); glUniform1i(locLiPosMat, 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, mapGbuf.normal); glUniform1i(locLiNormal, 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, heightTex);      glUniform1i(locLiHeightTex, 2);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D, lightVolTex);    glUniform1i(locLiLightVol, 3);
        glEnable(GL_BLEND);
        // Straight-alpha over for colour; coverage-over for alpha, so the
        // final composite's veil math keeps working.
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_BLEND);
    }

    // 5) TAA resolve of the combined map image under the usual colour clamp.
    //    Alpha (coverage) rides along, so the slab's silhouette edges resolve
    //    soft against the scene.
    unsigned int mapResultTex = mapSceneTex;
    if (mapTaa) {
        VOX_GPU_ZONE("map TAA");
        int prev = 1 - mapHistIdx;
        glBindFramebuffer(GL_FRAMEBUFFER, mapHistFBO[mapHistIdx]);
        glViewport(0, 0, fv.width, fv.height);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(taaProg.id);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mapSceneTex);   glUniform1i(locTaaCur, 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, mapHist[prev]); glUniform1i(locTaaHist, 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, motionTex);     glUniform1i(locTaaMotion, 2);
        glUniform2f(locTaaViewport, float(fv.width), float(fv.height));
        glUniform1i(locTaaHistValid, mapHistValid ? 1 : 0);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        mapResultTex = mapHist[mapHistIdx];
        mapHistIdx = 1 - mapHistIdx;
        mapHistValid = true;
    } else {
        mapHistValid = false;
    }
    prevMapCamPos = mf.camPos; prevMapCamFwd = f;
    prevMapCamRight = r;       prevMapCamUp = u;
    prevMapCamValid = true;

    // 6) Composite over the live scene: coverage alpha + background veil
    //    (premultiplied — see mapview.frag).
    VOX_GPU_ZONE("map composite");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fv.width, fv.height);
    glUseProgram(mapviewProg.id);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mapResultTex);
    glUniform1i(locMvTex, 0);
    glUniform1f(locMvVeil, 0.42f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// Map overlay offscreen surfaces (scene target + TAA history ping-pong).
// Created lazily on open, released with the G-buffer on close.
void Renderer::ensureMapSurfaces(int width, int height) {
    if (mapSceneFBO && mapSurfW == width && mapSurfH == height) return;
    auto mk = [&](unsigned int& fbo, unsigned int& tex) {
        if (!fbo) glGenFramebuffers(1, (GLuint*)&fbo);
        if (!tex) glGenTextures(1, (GLuint*)&tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, tex, 0);
    };
    mk(mapSceneFBO, mapSceneTex);
    for (int i = 0; i < 2; ++i) mk(mapHistFBO[i], mapHist[i]);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    mapSurfW = width; mapSurfH = height;
    mapHistValid = false;      // stale after (re)create
    prevMapCamValid = false;
}

// =============================================================================
// GPU timer overlay (/gpu): a compact top-left text panel of per-pass GPU ms,
// drawn through debugProg with the same 8x8 bitmap-font packing as the console.
// Reads gpuProf's last resolved results (a couple of frames old).
// =============================================================================
void Renderer::drawGpuTimers(const FrameView& fv) {
    constexpr int COLS = 16, ROWS = 8;
    int grid[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; ++i) grid[i] = 32;   // spaces

    auto put = [&](int row, const char* s) {
        if (row < 0 || row >= ROWS) return;
        for (int c = 0; c < COLS && s[c]; ++c)
            grid[row * COLS + c] = (unsigned char)s[c];
    };

    put(0, "-- GPU (ms) --");
    if (gpuProf.msCount == 0) {
        put(1, "(warming up...)");
    } else {
        char buf[40];
        int rows = std::min(gpuProf.msCount, ROWS - 2);   // last row reserved for total
        for (int i = 0; i < rows; ++i) {
            std::snprintf(buf, sizeof(buf), "%-8.8s%6.2f",
                          gpuProf.msNames[i] ? gpuProf.msNames[i] : "?", gpuProf.ms[i]);
            put(1 + i, buf);
        }
        std::snprintf(buf, sizeof(buf), "%-8s%6.2f", "total", gpuProf.msTotal);
        put(ROWS - 1, buf);
    }

    // Pack the font (glyph rows 0-3 / 4-7) and the char grid (4 codes per int).
    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
        for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }
    constexpr int NINTS = (ROWS * COLS + 3) / 4;   // 48
    int charsPacked[NINTS];
    for (int i = 0; i < NINTS; ++i) charsPacked[i] = 0;
    for (int i = 0; i < ROWS * COLS; ++i)
        charsPacked[i >> 2] |= (grid[i] & 0xFF) << ((i & 3) * 8);

    glUseProgram(debugProg.id);
    glUniform2f (locDbgViewport, float(fv.width), float(fv.height));
    glUniform1iv(locDbgChars, NINTS, charsPacked);
    glUniform1iv(locDbgFont,  256, fontPacked);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// Camera-parameterized particle pass (see render.h). Uploads the sprite
// buffer and draws blended points soft-depth-tested against `posTex`.
void Renderer::drawParticlesPass(const ParticleSystem& ps, const FrameView& fv,
                                 glm::vec3 eye, glm::vec3 fwd, glm::vec3 up,
                                 float fovY, unsigned int posTex) {
    std::vector<float> buf;
    buf.reserve(ps.particles.size() * 7);
    for (const auto& q : ps.particles) {
        buf.push_back(q.pos.x); buf.push_back(q.pos.y); buf.push_back(q.pos.z);
        buf.push_back(q.color.r); buf.push_back(q.color.g); buf.push_back(q.color.b);
        buf.push_back(q.life / q.maxLife);
    }
    glBindBuffer(GL_ARRAY_BUFFER, partVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, buf.size() * sizeof(float), buf.data());

    glm::mat4 view = glm::lookAt(eye, eye + fwd, up);
    glm::mat4 proj = glm::perspective(fovY,
        float(fv.width) / float(fv.height), 0.1f, float(GRID_Y) * 2.0f);
    glm::mat4 vp = proj * view;

    glUseProgram(particle.id);
    glUniformMatrix4fv(locPartViewProj, 1, GL_FALSE, &vp[0][0]);
    glUniform3fv(locPartCamPos, 1, &eye.x);
    glUniform2f (locPartViewport, float(fv.width), float(fv.height));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, posTex);
    glUniform1i(locPartGPos, 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(partVAO);
    glDrawArrays(GL_POINTS, 0, GLsizei(ps.particles.size()));
    glDisable(GL_BLEND);
}

// =============================================================================
// Forestry context menu ("Choose Option"): panel geometry + icons live in
// ctxmenu.frag; here we pack the packed FONT8X8 + the 4x28 char grid.
// =============================================================================
void Renderer::drawCtxMenu(const FrameView& fv) {
    constexpr int COLS = FrameView::CTX_COLS, ROWS = 4;
    int grid[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; ++i) grid[i] = 32;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS && fv.ctxText[r][c]; ++c)
            grid[r * COLS + c] = (unsigned char)fv.ctxText[r][c];

    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
        for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }
    constexpr int NINTS = (ROWS * COLS + 3) / 4;
    int charsPacked[NINTS];
    for (int i = 0; i < NINTS; ++i) charsPacked[i] = 0;
    for (int i = 0; i < ROWS * COLS; ++i)
        charsPacked[i >> 2] |= (grid[i] & 0xFF) << ((i & 3) * 8);

    glUseProgram(ctxProg.id);
    glUniform2f (locCtxViewport, float(fv.width), float(fv.height));
    glUniform2f (locCtxMenuPos, fv.ctxPos.x, fv.ctxPos.y);
    glUniform1i (locCtxRowN, fv.ctxRowN);
    glUniform1i (locCtxHover, fv.ctxHover);
    glUniform1iv(locCtxChars, NINTS, charsPacked);
    glUniform1iv(locCtxFont, 256, fontPacked);
    glUniform1iv(locCtxIcons, 3, fv.ctxIcon);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// =============================================================================
// Dev console overlay: char grid + packed 8x8 font through consoleProg. Drawn
// over the scene each frame the console is open (and over the map view).
// =============================================================================
void Renderer::drawConsole(const FrameView& fv) {
    const Console& C = *fv.console;
    // Build the char grid: bottom row is the input line, the rows above
    // show the tail of the scrollback (newest at the bottom).
    int grid[Console::ROWS * Console::COLS];
    for (int i = 0; i < Console::ROWS * Console::COLS; ++i) grid[i] = 32;

    int visible = Console::VISIBLE_LINES;
    int sbCount = int(C.scrollback.size());
    int firstShown = (sbCount > visible) ? sbCount - visible : 0;
    for (int r = 0; r < visible; ++r) {
        int sbIdx = firstShown + r;
        if (sbIdx >= sbCount) break;
        const std::string& s = C.scrollback[sbIdx];
        for (int c = 0; c < Console::COLS && c < int(s.size()); ++c)
            grid[r * Console::COLS + c] = (unsigned char)s[c];
    }
    // Input line (last row): "> " + input + blinking cursor.
    {
        int r = Console::ROWS - 1;
        std::string line = "> " + C.input;
        for (int c = 0; c < Console::COLS && c < int(line.size()); ++c)
            grid[r * Console::COLS + c] = (unsigned char)line[c];
        int cursorCol = int(line.size());
        bool blink = (int(fv.time * 2.0f) & 1) == 0;
        if (blink && cursorCol < Console::COLS)
            grid[r * Console::COLS + cursorCol] = '_';
    }

    // Packed font: 2 ints per glyph (rows 0-3, rows 4-7).
    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int row = 0; row < 4; ++row) lo |= int(FONT8X8[g][row])     << (row * 8);
        for (int row = 0; row < 4; ++row) hi |= int(FONT8X8[g][row + 4]) << (row * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }

    // Pack the char grid 4 codes per int (1008 -> 252 ints) to stay
    // under the driver's indexable uniform-array size limit.
    const int NCELL = Console::ROWS * Console::COLS;
    int charsPacked[(NCELL + 3) / 4];
    for (int i = 0; i < (NCELL + 3) / 4; ++i) charsPacked[i] = 0;
    for (int i = 0; i < NCELL; ++i)
        charsPacked[i >> 2] |= (grid[i] & 0xFF) << ((i & 3) * 8);

    glUseProgram(consoleProg.id);
    glUniform2f (locConViewport, float(fv.width), float(fv.height));
    glUniform1iv(locConChars, (NCELL + 3) / 4, charsPacked);
    glUniform1iv(locConFont,  256, fontPacked);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// =============================================================================
// Deferred G-buffer MRT target (filled by gbuffer.frag).
// =============================================================================
void GBuffer::ensure(int width, int height) {
    if (fbo && w == width && h == height) return;
    if (!fbo) glGenFramebuffers(1, (GLuint*)&fbo);
    auto mk = [&](unsigned int& tex, GLint ifmt, GLenum fmt, GLenum type) {
        if (!tex) glGenTextures(1, (GLuint*)&tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, ifmt, width, height, 0, fmt, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    mk(posMat, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    mk(normal, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    mk(albedo, GL_RGBA8,   GL_RGBA, GL_UNSIGNED_BYTE);
    if (!depth) glGenRenderbuffers(1, (GLuint*)&depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, posMat, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, albedo, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    w = width; h = height;
}
void GBuffer::bindForWrite() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, bufs);
    glViewport(0, 0, w, h);
}
// =============================================================================
// Creature pass: rigid-part wildlife. Rest volumes + palettes bake offline
// (tools/creatures/bake_parts.py) into a static atlas SSBO; each frame the
// app poses the visible creatures (CreatureSystem::buildPartInstances) and
// the renderer streams the part records into the parts SSBO, rasterizing one
// oriented box per part (creature.vert) whose fragments DDA the rest volume
// (creature.frag via lib/creature_trace.glsl) and write the same G-buffer
// attributes as terrain — the deferred stack lights creatures unchanged, and
// because the volumes rotate instead of being re-sampled, animation cannot
// flicker.
// =============================================================================
void Renderer::uploadCreatureAssets(const std::vector<uint32_t>& atlas) {
    VOX_ZONE_N("uploadCreatureAssets");
    if (atlas.empty()) return;
    if (!creatureSSBO) glGenBuffers(1, (GLuint*)&creatureSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, creatureSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, atlas.size() * sizeof(uint32_t),
                 atlas.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    std::printf("[creatures] GPU part atlas: %.1f KB\n", double(atlas.size() * 4) / 1e3);
}

// (Re)upload the forestry hover mask: R8 0/1 over the hovered tree's bbox.
// Re-specified wholesale on hover change — a few hundred KB at hover rate.
void Renderer::updateHoverMask(glm::ivec3 dims, const uint8_t* data) {
    if (!hoverMaskTex) {
        glGenTextures(1, (GLuint*)&hoverMaskTex);
        glBindTexture(GL_TEXTURE_3D, hoverMaskTex);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_3D, hoverMaskTex);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, dims.x, dims.y, dims.z, 0,
                 GL_RED, GL_UNSIGNED_BYTE, data);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_3D, 0);
}

// Stream this frame's posed part records into the parts SSBO. Orphaned each
// frame (glBufferData) — the buffer is tiny (<= 24 creatures x parts x 128 B).
void Renderer::uploadCreatureParts(const FrameView& fv) {
    if (!fv.creatureParts || fv.creaturePartN <= 0) return;
    if (!creaturePartSSBO) glGenBuffers(1, (GLuint*)&creaturePartSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, creaturePartSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 size_t(fv.creaturePartN) * sizeof(CreaturePartInstance),
                 fv.creatureParts, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// Draw fv's creature parts into the currently bound G-buffer. The caller has
// GL_DEPTH_TEST enabled with the terrain fill's depth already written
// (gl_FragDepth = t/DEPTH_FAR); this pass switches to GL_LESS and restores
// GL_ALWAYS on exit. Culling/capping happened in buildPartInstances.
void Renderer::drawCreatures(const FrameView& fv, const glm::vec3& eye,
                             const glm::vec3& f, const glm::vec3& r,
                             const glm::vec3& u, float effTanHalfFov) {
    if (fv.creaturePartN <= 0 || !creatureSSBO || !creaturePartSSBO) return;
    VOX_GPU_ZONE("drawCreatures");
    glUseProgram(creatureProg.id);
    glUniform3fv(locCrCamPos,   1, &eye.x);
    glUniform3fv(locCrCamFwd,   1, &f.x);
    glUniform3fv(locCrCamRight, 1, &r.x);
    glUniform3fv(locCrCamUp,    1, &u.x);
    glUniform2f (locCrViewport, float(fv.width), float(fv.height));
    glUniform1f (locCrTanHalfFov, effTanHalfFov);
    glUniform2f (locCrJitter, curJitterNdc.x, curJitterNdc.y);
    // Diorama slab: creature voxels clip against it exactly like terrain
    // does, so the Q/E slice cuts through bodies instead of leaving them
    // hovering over the carved-away cut face.
    glUniform1i (locCrMapPass, 1);
    glUniform3fv(locCrClipLo, 1, &fv.map.clipLo.x);
    glUniform3fv(locCrClipHi, 1, &fv.map.clipHi.x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, creatureSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, creaturePartSSBO);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);   // rasterize BACK faces: works with the camera inside the box
    glBindVertexArray(creatureVAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, fv.creaturePartN);
    glBindVertexArray(0);
    glDisable(GL_CULL_FACE);
    glDepthFunc(GL_ALWAYS);
}


// =============================================================================
// Per-frame
// =============================================================================
void Renderer::updateHeightRect(int x0, int z0, int rw, int rh, const float* data) {
    glBindTexture(GL_TEXTURE_2D, heightTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, x0, z0, rw, rh, GL_RED, GL_FLOAT, data);
}

void Renderer::uploadFarHeightmap(const std::vector<float>& farH, const std::vector<uint32_t>& farC) {
    if (farH.empty()) return;
    // NEAREST on both so the far field reads as CHUNKY flat cells (voxel look),
    // not a smooth interpolated mesh; CLAMP outside the built region.
    if (farHeightTex == 0) glGenTextures(1, (GLuint*)&farHeightTex);
    glBindTexture(GL_TEXTURE_2D, farHeightTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, FAR_TEX, FAR_TEX, 0, GL_RED, GL_FLOAT, farH.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (!farC.empty()) {
        if (farColorTex == 0) glGenTextures(1, (GLuint*)&farColorTex);
        glBindTexture(GL_TEXTURE_2D, farColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, FAR_TEX, FAR_TEX, 0, GL_RGBA, GL_UNSIGNED_BYTE, farC.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
}

// Halton low-discrepancy sequence (base 2/3): drives the TAA sub-pixel jitter.
static float halton(int i, int base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

void Renderer::renderFrame(const World& w,
                           const ParticleSystem& ps, const FrameView& fv) {
    VOX_ZONE_N("renderFrame");
    // This frame's posed creature parts: one upload, consumed by the creature
    // G-buffer pass(es) and the lighting shadow loop alike.
    uploadCreatureParts(fv);
    // Hot reload: any shader file changed on disk -> recompile + relink.
    if (reloadIfChanged(particle, "particle")) refreshParticleUniforms(*this);
    if (reloadIfChanged(creatureProg, "creature")) refreshCreatureUniforms(*this);
    if (reloadIfChanged(hud,      "hud"))      refreshHudUniforms(*this);
    if (reloadIfChanged(ctxProg,  "ctxmenu"))  refreshCtxUniforms(*this);
    if (reloadIfChanged(pause,    "pause"))    refreshPauseUniforms(*this);
    if (reloadIfChanged(consoleProg, "console")) refreshConsoleUniforms(*this);
    if (reloadIfChanged(tuneProg, "tune")) {
        locTuneViewport = glGetUniformLocation(tuneProg.id, "viewport");
        locTuneChars    = glGetUniformLocation(tuneProg.id, "chars");
        locTuneFont     = glGetUniformLocation(tuneProg.id, "font");
        locTuneSel      = glGetUniformLocation(tuneProg.id, "selRow");
        locTuneRows     = glGetUniformLocation(tuneProg.id, "rows");
    }
    if (reloadIfChanged(minimapProg, "minimap")) refreshMinimapUniforms(*this);
    if (reloadIfChanged(farmBarProg, "farmbar")) {
        locFbViewport = glGetUniformLocation(farmBarProg.id, "viewport");
        locFbChars    = glGetUniformLocation(farmBarProg.id, "chars");
        locFbFont     = glGetUniformLocation(farmBarProg.id, "font");
        locFbCols     = glGetUniformLocation(farmBarProg.id, "cols");
        locFbSlotW    = glGetUniformLocation(farmBarProg.id, "slotw");
        locFbSel      = glGetUniformLocation(farmBarProg.id, "sel");
        locFbRow      = glGetUniformLocation(farmBarProg.id, "rowIdx");
    }
    if (reloadIfChanged(bagProg, "bag")) {
        locBagViewport = glGetUniformLocation(bagProg.id, "viewport");
        locBagItem     = glGetUniformLocation(bagProg.id, "bagItem");
        locBagCount    = glGetUniformLocation(bagProg.id, "bagCount");
        locBagSelected = glGetUniformLocation(bagProg.id, "selected");
        locBagHover    = glGetUniformLocation(bagProg.id, "hover");
        locBagTidyHot  = glGetUniformLocation(bagProg.id, "tidyHot");
        locBagFont     = glGetUniformLocation(bagProg.id, "font");
        locBagHdr      = glGetUniformLocation(bagProg.id, "hdr");
        locBagFtr      = glGetUniformLocation(bagProg.id, "ftr");
        locBagTdy      = glGetUniformLocation(bagProg.id, "tdy");
    locBagIcon     = glGetUniformLocation(bagProg.id, "bagIcon");
    }
    if (reloadIfChanged(debugProg, "debug")) refreshDebugUniforms(*this);
    if (reloadIfChanged(gbufferProg,  "gbuffer"))  refreshGBufferUniforms(*this);
    if (reloadIfChanged(lightingProg, "lighting")) refreshLightingUniforms(*this);
    if (reloadIfChanged(motionProg,   "motion"))   refreshMotionUniforms(*this);
    if (reloadIfChanged(taaProg,      "taa"))      refreshTaaUniforms(*this);
    if (reloadIfChanged(mapviewProg,  "mapview"))  refreshMapviewUniforms(*this);

    if (fv.showGpuTimers) gpuProf.begin();

    // -- Motion target (RG16F) --------------------------------------------
    // renderMap's map-camera motion pass writes it; the map TAA resolve
    // samples it. Recreated when the window size changes.
    if (motionFBO == 0) glGenFramebuffers(1, (GLuint*)&motionFBO);
    if (motionTex == 0 || fboW != fv.width || fboH != fv.height) {
        if (motionTex == 0) glGenTextures(1, (GLuint*)&motionTex);
        glBindTexture(GL_TEXTURE_2D, motionTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, fv.width, fv.height, 0,
                     GL_RG, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, motionFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, motionTex, 0);
        fboW = fv.width; fboH = fv.height;
    }

    // TAA sub-pixel jitter (Halton 2,3): renderMap jitters the map camera
    // with the shared sequence and resolves against its own history.
    if (fv.taa) {
        int n = int(frameCount % 8u) + 1;
        glm::vec2 j(halton(n, 2) - 0.5f, halton(n, 3) - 0.5f);
        curJitterNdc = j * glm::vec2(2.0f / float(fv.width), 2.0f / float(fv.height));
    } else {
        curJitterNdc = glm::vec2(0.0f);
    }
    ++frameCount;   // advances the jitter sequence for the map TAA

    // Clear the backbuffer: post-swap contents are undefined, and the map
    // composite below blends against it — nothing may blend against garbage.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fv.width, fv.height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // -- The diorama IS the frame ----------------------------------------------
    renderMap(w, fv);
    // Particles re-camera onto the map view after renderMap (chop dust, leaf
    // bursts and pickup pops must be visible in the diorama). X-ray skips
    // them: the map G-buffer holds the cave re-march by then, and dust would
    // depth-test against the wrong layer.
    if (!ps.particles.empty() && !fv.map.xray) {
        glm::vec3 mf = glm::normalize(fv.map.camFwd);
        glm::vec3 mr = glm::normalize(glm::cross(mf, glm::vec3(0, 1, 0)));
        glm::vec3 mu = glm::cross(mr, mf);
        drawParticlesPass(ps, fv, fv.map.camPos, mf, mu,
                          2.0f * std::atan(MAP_TAN_HALF_FOV),
                          mapGbuf.posMat);
    }
    if (fv.showGpuTimers) gpuProf.mark("map");
    // /hud hides the passive minimap; the farm bar and context menu are
    // summoned UI (G / a click) and stay usable during a recording.
    if (!fv.hideHud)
        drawMinimap(w, fv);   // RuneScape-style minimap + free-cam orb (top right)
    if (fv.map.farmOn) drawFarmBar(fv);   // Farm Mode toolbar (bottom)
    if (fv.ctxOpen) drawCtxMenu(fv);      // forestry "Choose Option"

    // -- Hotbar HUD + chrome (DEV badge) ----------------------------------------
    // The tray is always visible during play (incl. Farm Mode — the farm tool
    // bar docks ABOVE it), so a harvest deposit always has a tray to land on.
    // Two exceptions: the open satchel hides the tray body (its own bottom
    // row stands in) while the shader still runs so the badge never
    // blinks out; /hud (clean recording) skips the whole block, badge and all.
    bool showHotbar = fv.hotbar && !fv.invOpen && !fv.hideHud;
    if (showHotbar || (!fv.hideHud && fv.devMode)) {
        int mats[Hotbar::SLOTS] = {};
        int counts[Hotbar::SLOTS] = {};
        int hicons[Hotbar::SLOTS] = {};
        if (showHotbar)
            for (int i = 0; i < Hotbar::SLOTS; ++i) {
                const ItemStack& s = fv.hotbar->bag.slots[i];
                mats[i]   = s.mat();                      // item -> material tint
                counts[i] = int(s.count);
                hicons[i] = int(itemDef(s.item).icon) | (int(s.data) << 2);
            }
        int nameChars[8] = {};
        if (showHotbar && fv.hotbarName)
            for (int i = 0; fv.hotbarName[i] && (i >> 2) < 8; ++i)
                nameChars[i >> 2] |= (int((unsigned char)fv.hotbarName[i]) & 0xFF) << ((i & 3) * 8);
        int fontPacked[256];
        for (int g = 0; g < 128; ++g) {
            int lo = 0, hi = 0;
            for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
            for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
            fontPacked[g * 2 + 0] = lo;
            fontPacked[g * 2 + 1] = hi;
        }
        glUseProgram(hud.id);
        glUniform2f (locHudViewport, float(fv.width), float(fv.height));
        glUniform1iv(locHudMat,      Hotbar::SLOTS, mats);
        glUniform1iv(locHudCount,    Hotbar::SLOTS, counts);
        glUniform1iv(locHudIcon,     Hotbar::SLOTS, hicons);
        glUniform1i (locHudSelected, showHotbar ? fv.hotbar->selected : -1);
        // The shader draws the wooden tray for ANY fragment in its rect —
        // without this gate a badge-only pass painted a fully visible
        // empty tray under the open satchel.
        glUniform1i (locHudTrayOn,   showHotbar ? 1 : 0);
        glUniform1i (locHudDevMode,  fv.devMode ? 1 : 0);
        glUniform1iv(locHudName, 8, nameChars);
        glUniform1f (locHudNameFade, fv.hotbarNameFade);
        glUniform1iv(locHudFont, 256, fontPacked);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_BLEND);
    }

    // -- Inventory satchel (E) — drawn over the diorama, above the hotbar -------
    if (fv.invOpen && fv.hotbar) drawBag(fv);

    // -- Pause overlay --------------------------------------------------------
    if (fv.paused) {
        glUseProgram(pause.id);
        glUniform2f(locPauseViewport, float(fv.width), float(fv.height));
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(sceneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_BLEND);
    }

    // -- Dev console ----------------------------------------------------------
    if (fv.console && fv.console->open) drawConsole(fv);

    if (fv.showGpuTimers) {
        gpuProf.mark("overlay");
        gpuProf.resolve();
        drawGpuTimers(fv);
    }
    if (fv.tuneOpen) drawTunePanel(fv);
}

// =============================================================================
// RuneScape-style minimap: a live top-down voxel render in the top-right of
// the diorama, framed in a stone ring, with the player arrow, the orbit
// camera's frustum wedge, click-to-move flag, creature dots, and the free-cam
// orb button. The fragment casts one short vertical ray per disc pixel via
// the shared brick DDA, so the pass needs the scene SSBOs + heightTex; fog of
// war reuses exploredSoftTex (uploaded by renderMap, which always runs first
// while the map is open). Geometry contract with app.cpp's minimapHit lives
// in common.h (MM_*).
// =============================================================================
void Renderer::drawMinimap(const World& w, const FrameView& fv) {
    VOX_GPU_ZONE("minimap");
    glUseProgram(minimapProg.id);
    glUniform2f (locMmViewport, float(fv.width), float(fv.height));
    glUniform3fv(locMmCamPos, 1, &fv.map.camPos.x);
    glUniform3fv(locMmCamFwd, 1, &fv.map.camFwd.x);
    glUniform1f (locMmCamDist, fv.map.camDist);
    glUniform1f (locMmVoxPerPx, fv.map.mmVoxPerPx);
    glUniform1f (locMmTanHalfH,
                 MAP_TAN_HALF_FOV * float(fv.width) / float(std::max(fv.height, 1)));
    glUniform2f (locMmPlayer, fv.map.playerPos.x, fv.map.playerPos.z);
    glUniform1f (locMmPlayerYaw, fv.map.playerYaw);
    glUniform2f (locMmMove, fv.map.moveTarget.x, fv.map.moveTarget.z);
    glUniform1i (locMmMoveOn, fv.map.moveActive);
    glUniform1i (locMmFreeCam, fv.map.freeCam);
    glUniform1f (locMmDayLight, fv.dayLight);
    glUniform3i (locMmGridSize, GRID_XZ, GRID_Y, GRID_XZ);
    glUniform3i (locMmBricks, BRICKS_XZ, BRICKS_Y, BRICKS_XZ);
    glUniform2i (locMmWindowOrigin, windowOriginB.x, windowOriginB.y);
    glUniform1f (locMmSurfaceY, float(SURFACE_Y));
    glUniform2i (locMmMaskOrigin, fv.map.maskOrigin.x, fv.map.maskOrigin.y);
    float cre[FrameView::MAX_CREATURES * 2];
    int   total    = std::min(fv.creatureN, FrameView::MAX_CREATURES);
    int   creCount = 0;
    for (int i = 0; i < total; ++i) {
        // Falling trees ride the creature-bound array for traced shadows —
        // the creBlobSkip range exempts them here too, or every fell paints
        // a transient wildlife dot at the toppling trunk.
        if (i >= fv.creBlobSkip.x && i < fv.creBlobSkip.x + fv.creBlobSkip.y)
            continue;
        cre[creCount * 2 + 0] = fv.creatureBound[i].x;
        cre[creCount * 2 + 1] = fv.creatureBound[i].z;
        ++creCount;
    }
    glUniform1i (locMmCreN, creCount);
    if (creCount) glUniform2fv(locMmCre, creCount, cre);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w.brickmapSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w.brickpoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, w.coarseSSBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, matColorUBO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, heightTex);
    glUniform1i(locMmHeightTex, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, exploredSoftTex);
    glUniform1i(locMmExplored, 1);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// =============================================================================
// Farm Mode toolbar (bottom-centre): three tool slots, active one tinted.
// While the Plant tool is armed a SEED bar (one slot per crop species) floats
// above it. Fixed-width slots so slot = col/slotw in the shader lines up with
// the labels; geometry mirrors farmbar.frag + farmBarHit (app.cpp).
// =============================================================================
void Renderer::drawFarmBar(const FrameView& fv) {
    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
        for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }
    glUseProgram(farmBarProg.id);
    glUniform2f (locFbViewport, float(fv.width), float(fv.height));
    glUniform1iv(locFbFont,  256, fontPacked);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    auto drawRow = [&](const char* label, int cols, int slotw, int sel, int row) {
        int charsPacked[16] = {};   // shader cap: 64 columns
        for (int i = 0; i < cols && label[i]; ++i)
            charsPacked[i >> 2] |= (label[i] & 0xFF) << ((i & 3) * 8);
        glUniform1iv(locFbChars, 16, charsPacked);
        glUniform1i (locFbCols,  cols);
        glUniform1i (locFbSlotW, slotw);
        glUniform1i (locFbSel,   sel);
        glUniform1i (locFbRow,   row);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    };
    drawRow("   1 TILL     2 PLANT    3 HARVEST  ", 36, 12,
            fv.map.farmTool, 0);
    if (fv.map.farmTool == FT_PLANT) {
        // One 8-col slot per species: " NAME   " (names cap at 7 chars).
        char seeds[CROP_SPECIES_N * 8 + 1];
        std::memset(seeds, ' ', sizeof(seeds) - 1);
        seeds[sizeof(seeds) - 1] = '\0';
        for (int sp = 0; sp < CROP_SPECIES_N; ++sp) {
            const char* nm = CROP_DEFS[sp].name;
            for (int k = 0; nm[k] && k < 7; ++k) {
                char ch = nm[k];
                seeds[sp * 8 + 1 + k] =
                    (ch >= 'a' && ch <= 'z') ? char(ch - 'a' + 'A') : ch;
            }
        }
        drawRow(seeds, CROP_SPECIES_N * 8, 8, fv.map.farmSeed, 1);
    }
    glDisable(GL_BLEND);
}

// =============================================================================
// Inventory "satchel" panel: a warm bag card floating over a dimmed world.
// Row 0 (slots 0..7) is the hotbar row; icons reuse the iso-cube look. Geometry
// lives in bag.frag; hit-testing mirrors it in app.cpp (bagCellRect/bagHitTest).
// =============================================================================
void Renderer::drawBag(const FrameView& fv) {
    constexpr int BAG = Hotbar::BAG;
    int items[BAG] = {};
    int counts[BAG] = {};
    int icons[BAG] = {};
    for (int i = 0; i < BAG; ++i) {
        const ItemStack& s = fv.hotbar->bag.slots[i];
        items[i]  = s.mat();            // material tint id for the icon
        counts[i] = int(s.count);
        icons[i]  = int(itemDef(s.item).icon) | (int(s.data) << 2);  // mode | grade<<2
    }
    // pack a string into an int[] (4 chars/int, LSB first)
    auto pack = [](const char* s, int* out, int nInts) {
        for (int i = 0; i < nInts; ++i) out[i] = 0;
        for (int i = 0; s[i] && (i >> 2) < nInts; ++i)
            out[i >> 2] |= (int((unsigned char)s[i]) & 0xFF) << ((i & 3) * 8);
    };
    int hdr[9], ftr[12], tdy[2];
    pack("SATCHEL", hdr, 9);
    pack("TIDY", tdy, 2);
    // Footer: hovered item's name + count, else a gentle hint.
    char foot[48] = {};
    if (fv.invHover >= 0 && fv.invHover < BAG &&
        !fv.hotbar->bag.slots[fv.invHover].isEmpty()) {
        const ItemStack& s = fv.hotbar->bag.slots[fv.invHover];
        std::snprintf(foot, sizeof(foot), "%s  x%d", itemDisplayName(s.item, s.data), int(s.count));
    } else {
        std::snprintf(foot, sizeof(foot), "click a stack to bring it to hand");
    }
    pack(foot, ftr, 12);

    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
        for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }
    glUseProgram(bagProg.id);
    glUniform2f (locBagViewport, float(fv.width), float(fv.height));
    glUniform1iv(locBagItem,     BAG, items);
    glUniform1iv(locBagCount,    BAG, counts);
    glUniform1iv(locBagIcon,     BAG, icons);
    glUniform1i (locBagSelected, fv.hotbar->selected);
    glUniform1i (locBagHover,    fv.invHover);
    glUniform1i (locBagTidyHot,  fv.invTidyHot);
    glUniform1iv(locBagFont, 256, fontPacked);
    glUniform1iv(locBagHdr, 9,  hdr);
    glUniform1iv(locBagFtr, 12, ftr);
    glUniform1iv(locBagTdy, 2,  tdy);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

// =============================================================================
// /tune: the live visual-tuning slider panel (tweaks.h is the table; the
// values ride to the shaders in uTweak[]). Same 8x8 bitmap-font packing as
// the console/GPU-timer overlays; keyboard-driven (arrows), so no mouse UI.
// =============================================================================
void Renderer::drawTunePanel(const FrameView& fv) {
    const int COLS = TUNE_COLS;
    int rows;
    std::vector<int> grid;
    auto mkgrid = [&](int r) { rows = r; grid.assign(size_t(rows) * COLS, 32); };
    auto put = [&](int row, const char* s) {
        for (int c = 0; c < COLS && s[c]; ++c)
            grid[size_t(row) * COLS + c] = (unsigned char)s[c];
    };
    auto barRow = [&](int gridRow, bool selected, const char* name,
                      float v, float mn, float mx) {
        float t = (v - mn) / (mx - mn);
        char bar[13];
        for (int b = 0; b < 12; ++b) bar[b] = (float(b) + 0.5f < t * 12.0f) ? '=' : '-';
        bar[12] = 0;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%c%-14.14s%s%7.2f",
                      selected ? '>' : ' ', name, bar, v);
        put(gridRow, buf);
    };

    if (fv.tunePage == 0) {                         // ---- LIGHTING ----
        mkgrid(TWEAK_N + 2);
        put(0, "-- TUNE: LIGHTING  tab->mats  --");
        for (int i = 0; i < TWEAK_N; ++i)
            barRow(1 + i, i == fv.tuneSel, TWEAKS[i].name,
                   fv.tweak[i], TWEAKS[i].mn, TWEAKS[i].mx);
        put(rows - 1, "drag/arrows  R reset  save");
    } else if (fv.tunePage == 1) {                  // ---- MATERIALS ----
        mkgrid(MP_COUNT + 3);
        put(0, "-- TUNE: MATERIALS tab->fauna --");
        const MatEntry& me = MAT_LIST[fv.tuneMatIdx];
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%c material: < %-12.12s >",
                      (fv.tuneSel == 0) ? '>' : ' ', me.name);
        put(1, buf);
        for (int p = 0; p < MP_COUNT; ++p)
            barRow(2 + p, fv.tuneSel == 1 + p, MAT_PARAMS[p].name,
                   fv.matTune[me.id * MP_COUNT + p], MAT_PARAMS[p].mn, MAT_PARAMS[p].mx);
        put(rows - 1, "<>=material  drag/arrows  save");
    } else {                                        // ---- CREATURES ----
        static_assert(CREATURE_TUNE_N <= 16, "grow FrameView::tuneCreature");
        mkgrid(CREATURE_TUNE_N + 3);
        put(0, "-- TUNE: CREATURES tab->light --");
        if (!fv.tuneSpecies[0]) {
            put(1, "  (no species loaded)");
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%c species:  < %-12.12s >",
                          (fv.tuneSel == 0) ? '>' : ' ', fv.tuneSpecies);
            put(1, buf);
            for (int i = 0; i < CREATURE_TUNE_N; ++i)
                barRow(2 + i, fv.tuneSel == 1 + i, CREATURE_TUNE[i].label,
                       fv.tuneCreature[i], CREATURE_TUNE[i].mn, CREATURE_TUNE[i].mx);
        }
        put(rows - 1, "<>=species  drag/arrows  save");
    }

    int fontPacked[256];
    for (int g = 0; g < 128; ++g) {
        int lo = 0, hi = 0;
        for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
        for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
        fontPacked[g * 2 + 0] = lo;
        fontPacked[g * 2 + 1] = hi;
    }
    const int nInts = (rows * COLS + 3) / 4;
    std::vector<int> charsPacked(nInts, 0);
    for (int i = 0; i < rows * COLS; ++i)
        charsPacked[i >> 2] |= (grid[i] & 0xFF) << ((i & 3) * 8);

    glUseProgram(tuneProg.id);
    glUniform2f (locTuneViewport, float(fv.width), float(fv.height));
    glUniform1iv(locTuneChars, nInts, charsPacked.data());
    glUniform1iv(locTuneFont,  256, fontPacked);
    glUniform1i (locTuneSel, fv.tuneSel + 1);   // +1: header row
    glUniform1i (locTuneRows, rows);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(sceneVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}
