#pragma once
#include "tweaks.h"
#include <glm/glm.hpp>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct World;
struct ParticleSystem;
struct Hotbar;
struct Console;
struct MapMasks;
struct CreaturePartInstance;

// Explored-world map view: what the renderer needs to draw the orbit
// diorama (the frame). The masks pointer (owned by the app, see map.h)
// carries a target-centred window into the global exploration record;
// masksVersion changes whenever it needs re-uploading (new marks or a
// mask-origin move).
struct MapFrame {
    bool      xray = false;          // cave X-ray view (ghost surface + veins)
    glm::vec3 camPos{0.0f};
    glm::vec3 camFwd{0.0f, -1.0f, 0.0f};
    glm::vec3 clipLo{0.0f};          // diorama clip box, world voxels
    glm::vec3 clipHi{0.0f};
    glm::vec3 playerPos{0.0f};       // analytic voxel character (feet)
    float     playerYaw = 0.0f;      // facing (fwd = cos/sin yaw)
    float     camDist = 0.0f;        // orbit camera-to-target standoff (voxels)
    float     walkPhase = -1.0f;     // stride phase; negative = idle pose
    glm::vec3 moveTarget{0.0f};      // click-to-move destination marker
    int       moveActive = 0;
    // Route trail for the shader: (x, z, y, cumulative XZ length). Entry 0
    // is the character; the rest are the remaining waypoints, decimated.
    static constexpr int PATH_MAX = 32;
    int       pathN = 0;
    glm::vec4 path[PATH_MAX];
    int       freeCam = 0;           // minimap orb state (icon only)
    float     mmVoxPerPx = 3.0f;     // minimap zoom: world voxels per pixel
    // Farm Mode (G): the toolbar + cursor footprint overlay.
    int        farmOn   = 0;         // Farm Mode active -> draw toolbar
    int        farmTool = 0;         // selected tool (FarmTool enum)
    int        farmSeed = 0;         // selected seed (CropSpecies; Plant tool's
                                     // seed bar shows + highlights it)
    int        farmCurOn = 0;        // a cell is under the cursor -> draw footprint
    int        farmOk   = 0;         // active tool valid on that cell (green/red)
    glm::vec3  farmMin{0.0f};        // footprint cell min corner (x, topY, z)
    float      farmSize = 0.0f;      // cell side (voxels)
    // God-harvest: hovered ripe-crop glow (voxel bbox) + press-ring "divine touch".
    int        hoverOn = 0;
    glm::vec3  hoverLo{0.0f}, hoverHi{0.0f};   // world-voxel bbox of the hovered crop
    int        ringOn = 0;
    glm::vec3  ringPos{0.0f};
    float      ringT   = 0.0f;       // 1 -> 0
    const MapMasks* masks = nullptr;
    glm::ivec2 maskOrigin{0};        // mask texel (0,0) in world chunk coords
    uint64_t  masksVersion = 0;
};

// Bundled per-frame state passed from app to renderer.
struct FrameView {
    int        width  = 0;
    int        height = 0;
    glm::vec3  sunDir{0, 1, 0};
    float      dayLight = 1.0f;
    glm::vec3  moonDir{0, -1, 0};      // opposite the sun; up at night
    float      moonLight = 0.0f;       // 0..1 cool-light strength (peaks midnight)
    bool       paused = false;
    bool       devMode = false;       // dev god-mode -> show DEV badge
    bool       hideHud = false;       // /hud: skip passive chrome (recording)
    // Dynamic fire point lights (warm pools cast by burning clusters), built on
    // the CPU each frame from the fire automaton's active bricks.
    static constexpr int MAX_FIRE_LIGHTS = 24;
    int        fireLightN = 0;
    glm::vec4  fireLights[MAX_FIRE_LIGHTS] = {};   // xyz = world pos, w = strength
    bool       showGpuTimers = false; // /gpu: per-pass GPU timing overlay
    bool       taa = true;            // /taa: temporal anti-aliasing
    float      time = 0.0f;            // seconds since startup, drives sky drift
    glm::vec2  farOrigin{0.0f, 0.0f};  // world-voxel min corner of the far-field heightfield region
    bool       ringsOn = true;         // /rings: far-field LOD ring 1 (voxel-clipmap C0)
    int        colorMode = 0;          // /srgb: 0 legacy output, 1 sRGB-encode experiment
    int        slopeShade = 0;         // /slopeshade: macro terrain slope normals (diffuse, opt-in)
    float      fogDensity = 0.0f;      // height-fog extinction at the floor (0 = clear)
    float      tweak[TWEAK_GPU] = {};  // /tune lighting values (indices = tweaks.h)
    float      matTune[MATTUNE_N] = {};// /tune materials: per-material cell params
    int        tuneOpen = 0;           // /tune panel visible
    int        tunePage = 0;           // 0 = lighting, 1 = materials, 2 = creatures
    int        tuneSel  = 0;           // selected row within the page
    int        tuneMatIdx = 0;         // selected material (index into MAT_LIST)
    // Creatures page: selected species name + its CREATURE_TUNE row values
    // (creatures.h), copied per frame so the renderer stays decoupled.
    char       tuneSpecies[24] = {};
    float      tuneCreature[16] = {};
    float      fogFloor   = 0.0f;      // world Y of peak fog density
    const Hotbar* hotbar = nullptr;   // optional HUD; null skips
    // Held-item name label above the hotbar, fades in on a selection change.
    const char*   hotbarName = nullptr;   // selected item's name (null = blank)
    float         hotbarNameFade = 0.0f;  // 0..1 opacity
    // Inventory "satchel" panel (E): floats over the dimmed diorama.
    int           invOpen = 0;        // draw the bag panel
    int           invHover = -1;      // hovered bag slot 0..BAG-1 (footer + highlight)
    int           invTidyHot = 0;     // cursor over the Tidy button
    const Console* console = nullptr; // optional dev console; null/closed skips
    MapFrame map;                     // explored-world map — the frame's sole view

    // Posed creature parts (rigid-part wildlife, creatures.h). The instance
    // records live in an app-owned vector (pointer, MapMasks-style); the
    // renderer streams them into the parts SSBO each frame and rasterizes
    // one box per part in renderMap's map-pass G-buffer fill. Bounds +
    // ranges feed the lighting pass's creature shadow loop and contact blob.
    // Render cap (nearest first past it). Keep comfortably above what herds
    // + debug spawns produce on screen: a LIVE creature silently not drawing
    // reads as a despawn-in-view. Must match uCreBound/uCreRange in
    // lighting.frag.
    static constexpr int MAX_CREATURES = 32;
    const CreaturePartInstance* creatureParts = nullptr;
    int        creaturePartN = 0;
    int        creatureN = 0;
    int        playerAvatarOn = 0;   // 1 = voxel avatar rendered; analytic character off
    glm::vec4  creatureBound[MAX_CREATURES] = {};   // xyz world centre, w radius
    glm::ivec2 creatureRange[MAX_CREATURES] = {};   // first part instance, count
    // Bounds in [x, x+y) skip the contact-blob loop (falling trees: their
    // huge bound would paint a giant dark disc; they keep traced shadows).
    glm::ivec2 creBlobSkip{0, 0};

    // Forestry context menu (RuneScape "Choose Option"). Window coords from
    // the top-left; row 0 of ctxText is the header, rows 1.. the entries.
    // Geometry mirrors ctxmenu.frag: CTX_W wide, CTX_HDR header, CTX_ROW rows.
    static constexpr float CTX_W = 300.0f, CTX_HDR = 26.0f, CTX_ROW = 30.0f;
    static constexpr int CTX_COLS = 28;
    bool       ctxOpen = false;
    glm::vec2  ctxPos{0.0f};
    int        ctxRowN = 0;          // entry rows (excluding header)
    int        ctxHover = -1;
    char       ctxText[4][CTX_COLS + 1] = {};   // header + up to 3 entries
    int        ctxIcon[3] = {};      // per entry: 0 none, 1 axe, 2 spade, 3 pick

    // Forestry hover highlight (diorama): a 0/1 mask over the hovered tree's
    // bbox lives in the renderer's 3D texture (Renderer::updateHoverMask);
    // the lighting pass rims + lifts masked wood/leaf pixels.
    int        hoverOn = 0;
    glm::vec3  hoverLo{0.0f};
    glm::vec3  hoverDims{0.0f};
    float      hoverStrength = 0.0f;   // eased 0..1 (breathes in the shader)
};

// A compiled GL program plus the source file paths used to build it.
// Hot reload: each frame we stat the files; if either has a newer mtime,
// recompile + relink and swap the program in place.
struct ShaderProgram {
    unsigned int id = 0;
    std::string  vsPath, fsPath;
    std::filesystem::file_time_type vsTime{};
    std::filesystem::file_time_type fsTime{};
    // Resolved #include dependency paths (across both stages) and their mtimes,
    // tracked so hot reload also recompiles when an included lib/*.glsl changes.
    std::vector<std::string> deps;
    std::vector<std::filesystem::file_time_type> depTimes;
};

// Compile a standalone GL program from shader files (paths relative to the
// shader dir, e.g. "fullscreen.vert"), with the same #include resolution the
// renderer uses. Returns the program id, or 0 on failure. Used for the loading
// screen, which must draw BEFORE Renderer::init exists. Caller owns the id.
unsigned int loadStandaloneProgram(const char* vsRel, const char* fsRel, const char* tag);

// Per-pass GPU timing via GL_TIMESTAMP queries. Triple-buffered so a frame's
// results are read back two frames later (when they're guaranteed ready),
// never stalling the pipeline waiting on the GPU. Toggled with /gpu.
struct GpuProfiler {
    static constexpr int RING  = 3;       // frames of read-back latency + 1
    static constexpr int MARKS = 8;       // max timestamps placed per frame

    unsigned int q[RING][MARKS] = {};     // GL_TIMESTAMP query objects
    int          ringMarks[RING] = {};    // timestamps placed in each ring slot
    const char*  ringNames[RING][MARKS] = {};
    bool         created = false;
    int          ring    = 0;             // ring slot being recorded this frame
    int          frames  = 0;             // frames recorded since enabled
    int          cur     = 0;             // marks placed in the current frame

    // Resolved results for display (one entry per timed interval).
    float        ms[MARKS]        = {};
    const char*  msNames[MARKS]   = {};
    int          msCount          = 0;
    float        msTotal          = 0.0f;

    void begin();                 // start a frame: place the t0 timestamp
    void mark(const char* label); // place a timestamp ending the interval `label`
    void resolve();               // read back the oldest ready frame into ms[]
};

// Deferred G-buffer: the MRT target gbuffer.frag fills with primary-hit
// attributes (world-pos+material, normal+distance, albedo).
struct GBuffer {
    unsigned int fbo    = 0;
    unsigned int posMat = 0;   // RGBA32F: xyz world pos, w material id (-1 miss)
    unsigned int normal = 0;   // RGBA16F: xyz normal, w ray distance
    unsigned int albedo = 0;   // RGBA8:   rgb albedo
    // Depth renderbuffer: gbuffer.frag writes gl_FragDepth = t/DEPTH_FAR so
    // the creature OBB pass can hardware-depth-test against the raymarched
    // terrain (a texture read of the bound posMat would be GL 4.3 feedback UB).
    unsigned int depth  = 0;
    int w = 0, h = 0;
    void ensure(int width, int height);   // (re)create attachments on size change
    void bindForWrite();                  // bind FBO + draw buffers + viewport
};

struct Renderer {
    // Toroidal-window origin in brick coords; (0,0) until streaming lands.
    // Set from the world's streaming window (app.cpp); consumed by the deferred
    // G-buffer / lighting passes for correct wrapped brick addressing.
    glm::ivec2 windowOriginB{0, 0};

    // Particle program (point sprites): runs with the map camera, occluding
    // against the map G-buffer.
    ShaderProgram particle;
    int locPartViewProj = -1;
    int locPartCamPos = -1, locPartViewport = -1, locPartGPos = -1;  // scene occlusion
    unsigned int partVAO = 0, partVBO = 0;
    void drawParticlesPass(const ParticleSystem& ps, const FrameView& fv,
                           glm::vec3 eye, glm::vec3 fwd, glm::vec3 up,
                           float fovY, unsigned int posTex);

    // Creature pass: rigid-part wildlife rasterized as one oriented box per
    // posed bone part; fragments DDA the part's rest volume via the shared
    // lib/creature_trace.glsl. Static volumes + palettes live in the atlas
    // SSBO (binding 4, uploaded once); per-frame posed instances stream into
    // the parts SSBO (binding 5).
    ShaderProgram creatureProg;
    int locCrCamPos = -1, locCrCamFwd = -1, locCrCamRight = -1, locCrCamUp = -1;
    int locCrViewport = -1, locCrTanHalfFov = -1, locCrJitter = -1;
    int locCrMapPass = -1, locCrClipLo = -1, locCrClipHi = -1;
    unsigned int creatureVAO = 0, creatureSSBO = 0, creaturePartSSBO = 0;
    // Upload the static volume/palette atlas (CreatureSystem::buildGpuAtlas).
    void uploadCreatureAssets(const std::vector<uint32_t>& atlas);
    // Stream fv's posed part instances into the parts SSBO (once per frame,
    // shared by both G-buffer fills and the lighting shadow loop).
    void uploadCreatureParts(const FrameView& fv);
    // Draw fv's creature parts into the CURRENTLY BOUND G-buffer (depth test
    // GL_LESS against the terrain fill). Called from renderMap's map-pass
    // fill.
    void drawCreatures(const FrameView& fv, const glm::vec3& eye,
                       const glm::vec3& f, const glm::vec3& r,
                       const glm::vec3& u, float effTanHalfFov);
    // Lighting-pass creature inputs: traced cast/self shadows (the shared
    // part trace), the palette-albedo substitution (gAlbedoTex), and the
    // contact blob — driven by per-creature bounds + part ranges.
    int locLiCreN = -1, locLiCreBound = -1, locLiCreRange = -1;
    int locLiAlbedoTex = -1, locLiPlayerAvatar = -1, locLiCreBlobSkip = -1;

    // Pause overlay program.
    ShaderProgram pause;
    int locPauseViewport = -1;

    // HUD program (hotbar tray + DEV badge).
    ShaderProgram hud;
    int locHudViewport = -1, locHudMat = -1, locHudCount = -1, locHudSelected = -1;
    int locHudTrayOn = -1;   // 0 hides the tray body (satchel open) but the
                             // pass still draws the badge
    int locHudDevMode = -1;
    int locHudName = -1, locHudNameFade = -1, locHudFont = -1;   // held-item name label
    int locHudIcon = -1;                                          // per-slot icon mode/grade

    // Dev console program.
    ShaderProgram consoleProg;
    int locConViewport = -1, locConChars = -1, locConFont = -1;
    void drawConsole(const FrameView& fv);

    // Forestry context menu program (RuneScape "Choose Option").
    ShaderProgram ctxProg;
    int locCtxViewport = -1, locCtxMenuPos = -1, locCtxRowN = -1;
    int locCtxHover = -1, locCtxChars = -1, locCtxFont = -1, locCtxIcons = -1;
    void drawCtxMenu(const FrameView& fv);

    // GPU pass-timer overlay (/gpu): a compact top-left text panel.
    ShaderProgram debugProg;
    int locDbgViewport = -1, locDbgChars = -1, locDbgFont = -1;
    GpuProfiler gpuProf;
    void drawGpuTimers(const FrameView& fv);

    // Deferred G-buffer fill (raymarch -> MRT G-buffer) into mapGbuf.
    ShaderProgram gbufferProg;
    int locGBCamPos = -1, locGBCamFwd = -1, locGBCamRight = -1, locGBCamUp = -1;
    int locGBViewport = -1, locGBTanHalfFov = -1, locGBGridSize = -1, locGBBricks = -1;
    int locGBSurfaceY = -1;
    int locGBWindowOrigin = -1, locGBTime = -1, locGBHeightTex = -1;  // streaming + biome surfaceColor
    int locGBJitter = -1;  // TAA sub-pixel jitter
    // Far-field LOD rings (voxel clipmap C1): primary-ray cascade uniforms.
    int locGBRingOn = -1, locGBRingMask = -1, locGBRingOrigins = -1;
    int locGBRingBricksY = -1, locGBRingBmBases = -1, locGBRingCoarseBases = -1;
    int locGBChunkValid = -1;

    // Deferred lighting pass: reads mapGbuf and shades it (the frame's
    // primary shading).
    ShaderProgram lightingProg;
    int locLiCamPos = -1, locLiCamFwd = -1, locLiCamRight = -1, locLiCamUp = -1;
    int locLiViewport = -1, locLiTanHalfFov = -1, locLiGridSize = -1, locLiBricks = -1;
    int locLiSunDir = -1, locLiDayLight = -1;
    int locLiTime = -1, locLiSurfaceY = -1;
    int locLiHeightTex = -1, locLiLightVol = -1;
    int locLiPosMat = -1, locLiNormal = -1;
    int locLiMoonDir = -1, locLiMoonLight = -1, locLiWindowOrigin = -1;  // day/night + streaming
    int locLiFireN = -1, locLiFireLights = -1;   // dynamic fire point lights
    int locLiJitter = -1;  // TAA sub-pixel jitter
    int locLiFarHeightTex = -1, locLiFarOrigin = -1, locLiFarStride = -1, locLiFarTexels = -1;  // far-field sun-shadow marches
    int locLiColorMode = -1;// /srgb output-encode A/B
    int locLiSlopeShade = -1;// /slopeshade macro terrain diffuse normals
    int locLiFogDensity = -1, locLiFogFloor = -1;   // height fog (atmosphere.glsl)
    int locLiTweak = -1;                            // /tune uniform array
    int locLiMatTune = -1;                          // /tune materials uniform
    ShaderProgram tuneProg;                          // /tune slider panel overlay
    int locTuneViewport = -1, locTuneChars = -1, locTuneFont = -1, locTuneSel = -1, locTuneRows = -1;
    void drawTunePanel(const FrameView& fv);
    // RuneScape-style minimap (diorama HUD, top right): live top-down voxel
    // render + player arrow + camera wedge + free-cam orb. Geometry contract
    // with app.cpp's minimapHit lives in common.h (MM_*).
    ShaderProgram minimapProg;
    int locMmViewport = -1, locMmCamPos = -1, locMmCamFwd = -1, locMmCamDist = -1;
    int locMmVoxPerPx = -1, locMmTanHalfH = -1, locMmPlayer = -1, locMmPlayerYaw = -1;
    int locMmMove = -1, locMmMoveOn = -1, locMmFreeCam = -1, locMmDayLight = -1;
    int locMmGridSize = -1, locMmBricks = -1, locMmWindowOrigin = -1, locMmSurfaceY = -1;
    int locMmMaskOrigin = -1, locMmCreN = -1, locMmCre = -1;
    int locMmHeightTex = -1, locMmExplored = -1;
    void drawMinimap(const World& w, const FrameView& fv);
    ShaderProgram farmBarProg;                       // Farm Mode toolbar (bottom)
    int locFbViewport = -1, locFbChars = -1, locFbFont = -1, locFbCols = -1,
        locFbSlotW = -1, locFbSel = -1, locFbRow = -1;
    void drawFarmBar(const FrameView& fv);
    ShaderProgram bagProg;                           // inventory "satchel" panel
    int locBagViewport = -1, locBagItem = -1, locBagCount = -1, locBagSelected = -1,
        locBagHover = -1, locBagTidyHot = -1, locBagFont = -1,
        locBagHdr = -1, locBagFtr = -1, locBagTdy = -1, locBagIcon = -1;
    void drawBag(const FrameView& fv);

    // Motion vectors: per-pixel screen-space motion from map-camera
    // reprojection, feeding the map TAA resolve.
    ShaderProgram motionProg;
    unsigned int motionFBO = 0, motionTex = 0;   // RG16F, window-sized
    int fboW = 0, fboH = 0;                      // motionTex resize key
    int locMoPosMat = -1, locMoNormal = -1;
    int locMoCamPos = -1, locMoCamFwd = -1, locMoCamRight = -1, locMoCamUp = -1;
    int locMoTanHalfFov = -1, locMoViewport = -1;
    int locMoPrevPos = -1, locMoPrevFwd = -1, locMoPrevRight = -1, locMoPrevUp = -1;
    int locMoPrevTanHalfFov = -1, locMoPrevAspect = -1, locMoPrevValid = -1;

    // Temporal AA: jitter the camera sub-pixel each frame, then resolve the
    // current frame against the motion-reprojected history under a colour clamp.
    ShaderProgram taaProg;
    int locTaaCur = -1, locTaaHist = -1, locTaaMotion = -1, locTaaViewport = -1, locTaaHistValid = -1;
    unsigned int frameCount = 0;             // drives the Halton jitter sequence
    glm::vec2 curJitterNdc{0.0f};            // this frame's jitter (zero when TAA off)

    // Explored-world map (the frame): a two-pass orbit-diorama render —
    // gbufferProg with mapPass=1 (clip box + explored gating) into mapGbuf,
    // then lightingProg with mapPass=1 straight to the default framebuffer.
    // Explored masks upload lazily, keyed by fv.map.masksVersion.
    GBuffer mapGbuf;
    unsigned int exploredTex  = 0;   // R8UI  MASK_CH^2: bit0 surface seen, bit1 any
    unsigned int exploredYTex = 0;   // R32UI MASK_CH x 8 x MASK_CH: visited-Y-slice words
    unsigned int exploredSoftTex = 0; // R8 bilinear: prefiltered fog-of-war ramp
    uint64_t exploredUploaded = 0;   // masksVersion currently on the GPU
    int locGBMapPass = -1, locGBClipLo = -1, locGBClipHi = -1;
    int locGBMapFineVpp = -1;   // zoom LOD: dithered fine->ring0 handoff
    int locGBExplored = -1, locGBExploredY = -1, locGBMaskOrigin = -1;
    int locLiMapPass = -1, locLiMapPlayer = -1, locLiMapCover = -1;
    int locLiMapGhost = -1;   // X-ray surface dim (brightness, not alpha)
    int locLiMapStandoff = -1; // orbit standoff: zoom-independent atmosphere
    int locLiMapPath = -1, locLiMapPathN = -1;   // click-to-move trail
    int locLiMapYaw = -1, locLiMapWalk = -1;          // character facing + stride
    int locLiMapMoveTarget = -1, locLiMapMoveOn = -1; // destination marker
    // Farm Mode footprint overlay (lighting.frag).
    int locLiFarmOn = -1, locLiFarmOk = -1, locLiFarmMin = -1, locLiFarmSize = -1;
    // God-harvest crop hover + press ring (mapHover*/mapRing* uniforms) —
    // distinct from the forestry tree-hover locs (uHover*) further down.
    int locLiMapHoverOn = -1, locLiMapHoverLo = -1, locLiMapHoverHi = -1;
    int locLiRingOn = -1, locLiRingPos = -1, locLiRingT = -1;
    int locLiExplored = -1;   // fog-of-war chunk flags (unit 8 in EVERY pass)
    int locLiExploredSoft = -1;   // prefiltered ramp (float sampler, unit 9)
    int locLiMaskOrigin = -1;
    // The diorama shades into mapSceneTex, gets its own TAA resolve (the
    // shared motionProg/taaProg run with the MAP camera into the mapHist
    // ping-pong), and mapviewProg composites the result onto the backbuffer
    // (coverage alpha + background veil).
    ShaderProgram mapviewProg;
    int locMvTex = -1, locMvVeil = -1;
    unsigned int mapSceneFBO = 0, mapSceneTex = 0;
    unsigned int mapHistFBO[2] = {0, 0}, mapHist[2] = {0, 0};
    int  mapSurfW = 0, mapSurfH = 0;
    int  mapHistIdx = 0;
    bool mapHistValid = false;
    glm::vec3 prevMapCamPos{0.0f}, prevMapCamFwd{0.0f};
    glm::vec3 prevMapCamRight{0.0f}, prevMapCamUp{0.0f};
    bool prevMapCamValid = false;
    void renderMap(const World& w, const FrameView& fv);
    void ensureMapSurfaces(int width, int height);

    // Forestry hover highlight: R8 3D mask over the hovered tree's bbox
    // (re-specified per hover change; a few hundred KB, hover-rate).
    unsigned int hoverMaskTex = 0;
    void updateHoverMask(glm::ivec3 dims, const uint8_t* data);
    int locLiHoverOn = -1, locLiHoverLo = -1, locLiHoverDims = -1;
    int locLiHoverStrength = -1, locLiHoverMask = -1;

    // std140 UBO of per-material colors, indexed by material ID.
    unsigned int matColorUBO = 0;

    // Terrain surface-height texture (R32F, GRID_XZ^2) for per-point depth.
    unsigned int heightTex = 0;

    // Per-chunk-slot GPU residency (R8UI, CHUNKS_XZ^2; mirrors
    // World::gpuChunkValid every frame). The fine primary march consults it
    // PER RAY: entering a blank (not-yet-uploaded) chunk hands off to the
    // ring cascade at that exact t — a DIRECTIONAL reveal, so one missing
    // chunk never coarsens the rest of the view (the old global-scalar cap
    // painted coarse slabs over valid terrain in unrelated directions).
    unsigned int chunkValidTex = 0;

    // Far-field LoD heightfield (R32F) + paired surface colour (RGBA8), FAR_TEX^2,
    // CLAMP + NEAREST (chunky voxel look), marched in the lighting pass as the
    // horizon background. Recentred at build/teleport.
    unsigned int farHeightTex = 0;
    unsigned int farColorTex  = 0;

    // Coarse 3D emissive light volume (RGB16F) — glow crystals etc. baked at
    // worldgen, trilinearly sampled in the shader to light deep caverns.
    unsigned int lightVolTex = 0;

    // Fullscreen-triangle VAO reused for the main scene + pause overlay.
    unsigned int sceneVAO = 0;

    // heightMap (GRID_XZ^2 R32F) and lightVol (LV_X*LV_Y*LV_Z RGB) come from
    // the world's GenContext — the renderer holds GPU copies only.
    void init(const std::vector<float>& heightMap, const std::vector<float>& lightVol);
    // Overwrite a rect of the height texture (toroidal window scroll: the
    // entering strip's texels are the leaving strip's, REPEAT-wrapped).
    // `data` is row-major (w floats per row, h rows).
    void updateHeightRect(int x0, int z0, int w, int h, const float* data);
    // (Re)upload the far-field heightfield + surface colour after a rebuild.
    void uploadFarHeightmap(const std::vector<float>& farH, const std::vector<uint32_t>& farC);
    void renderFrame(const World& w,
                     const ParticleSystem& ps, const FrameView& fv);
    // Re-upload the light volume after it's rebuilt (e.g. a crystal edit).
    void updateLightVolume(const std::vector<float>& lightVol);
    // Partial re-upload of a toroidal strip (paired with buildLightVolumeStrip).
    void updateLightVolumeRect(const std::vector<float>& lv,
                               bool xAxis, int off, int cnt);
    // Partial re-upload of a toroidal cell-box (paired with splatNewLights).
    void updateLightVolumeBox(const std::vector<float>& lv,
                              int wbx0, int wbx1, int wbz0, int wbz1);
};
