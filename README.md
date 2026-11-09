# tiny-voxel

A diorama voxel sandbox with 10 cm voxels, procedural terrain, recursive
trees you can chop down RuneScape-style, day/night cycle, drifting clouds,
water simulation, wildlife, and a shader hot-reload dev loop.

## Build

Requires **CMake 3.20+**, a **C++17** compiler (MSVC / gcc / clang), and
**Python 3.x** (used at configure time by `glad` to generate the GL loader).
All other dependencies (GLFW, GLM, glad) are pulled automatically by CMake's
`FetchContent`.

```sh
cmake -S . -B build
cmake --build build --config Release
.\build\Release\voxel.exe   # run from the project root so it can find src/shaders/
```

On Linux/Mac swap the `\` for `/` and drop `--config Release`.

## Controls

- **RMB** click/hold: walk there · **LMB drag** orbit · **wheel** zoom
- **LMB click a tree or cactus**: "Choose Option" menu — chop it down; the
  debris magnetizes to the player and lands in the bag
- **1–8** hotbar slot · **E** satchel · **G** Farm Mode
- **Q/E** diorama slice · **Tab** cave X-ray · **Space** jump · **Ctrl** sprint
- **Shift** crouch · **F** fly (dev) · **F3** dev/playtest toggle ·
  **F11** fullscreen · **Esc** pause
- **`/`** dev console · **Tab** completes · **↑/↓** recall history (`/help`
  lists commands; `/time` sets time of day)

## Hot reload

Edit any file in `src/shaders/*` while the game is running and save — the
renderer detects the change and recompiles within a frame. On compile error
the previous program stays active; fix the file and save again.

## Troubleshooting

### Windows Smart App Control blocks `voxel.exe`

If SAC is **on**, Windows can block unsigned local builds. SAC off is the
dev-friendly choice. If you want to keep it on, sign the build with a local
self-signed development certificate.

Run this once from an elevated PowerShell in the project root:

```powershell
$subject = "CN=VoxelDevSelfSigned"
$cert = Get-ChildItem Cert:\LocalMachine\My |
    Where-Object { $_.Subject -eq $subject } |
    Select-Object -First 1

if (-not $cert) {
    $cert = New-SelfSignedCertificate `
        -Subject $subject `
        -CertStoreLocation Cert:\LocalMachine\My `
        -Type CodeSigningCert `
        -KeyUsage DigitalSignature `
        -KeyAlgorithm RSA `
        -KeyLength 2048 `
        -KeyExportPolicy Exportable `
        -NotAfter (Get-Date).AddYears(10)
}

foreach ($storeName in @("Root", "TrustedPublisher")) {
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new(
        $storeName, "LocalMachine")
    $store.Open("ReadWrite")
    $trusted = $store.Certificates |
        Where-Object { $_.Thumbprint -eq $cert.Thumbprint }
    if (-not $trusted) {
        $store.Add($cert)
    }
    $store.Close()
}

$cert.Thumbprint | Set-Content .cert-thumbprint.txt
```

If normal PowerShell cannot use the certificate, copy it to your user
certificate store once:

```powershell
$thumb = (Get-Content .cert-thumbprint.txt).Trim()
$pfxPath = Join-Path $env:TEMP "voxeldev-$thumb.pfx"
$pwd = ConvertTo-SecureString -String "voxeldev" -Force -AsPlainText

Export-PfxCertificate -Cert "Cert:\LocalMachine\My\$thumb" `
    -FilePath $pfxPath -Password $pwd | Out-Null
Import-PfxCertificate -FilePath $pfxPath `
    -CertStoreLocation Cert:\CurrentUser\My -Password $pwd | Out-Null
Remove-Item $pfxPath -Force
```

After each Release build, sign the executable:

```powershell
$thumb = (Get-Content .cert-thumbprint.txt).Trim()
$exe = ".\build\Release\voxel.exe"
$signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" |
    Sort-Object FullName -Descending |
    Select-Object -First 1

if (-not (Test-Path $exe)) { throw "$exe not found. Build first." }
if (-not $signtool) { throw "signtool.exe not found. Install the Windows SDK." }

& $signtool.FullName sign /sha1 $thumb /fd SHA256 $exe
```

If you skipped the user-store copy and are signing from the machine store in
an elevated shell, add `/sm` before `/sha1`.

## Layout

```
src/
  app.cpp              top-level loop, input callbacks, gameplay glue
  game_state.h         the GameState aggregate (app.cpp + commands.cpp only)
  commands.{h,cpp}     dev-console command registry, handlers, /help, tab
  console.{h,cpp}      console text model: input line, scrollback, history
  world.{h,cpp}        sparse voxel data + GPU sync + raycast
  worldgen/            terrain, caves, flora, LOD rings, streaming
  worldio.{h,cpp}      world edits save/load + chunk cache
  player.{h,cpp}       player struct + collision
  path.{h,cpp}         A* click-to-move routing
  map.{h,cpp}          explored-map state + diorama view
  farm.{h,cpp}         Farm Mode: toolbar, crops, plot jobs
  creatures.{h,cpp}    wildlife: defs, herds, animation, ragdoll-free AI
  particles.{h,cpp}    dust + ambient particle system
  forestry.{h,cpp}     tree chopping: flood-fill identity, felling, debris
  water.{h,cpp}        cellular-automaton fluid sim
  fire.{h,cpp}         fire spread automaton
  granular.{h,cpp}     falling-sand granular sim
  materials.{h,cpp}    one-row-per-material data table
  tweaks.h             /tune tables: shipped lighting/material defaults
  audio.{h,cpp}        music + ambience playback
  render.{h,cpp}       all GL state, draw passes, hot-reload
  font8x8.{h,cpp}      bitmap font for overlays/console
  shaders/             editable .glsl files (hot-reloaded)
  common.h             world constants and material IDs
```
