<#
.SYNOPSIS
  Capture a Tracy profile of tiny-voxel and export zone CSVs.

.DESCRIPTION
  Launches the Tracy-enabled build (cmake -DVOXEL_TRACY=ON), waits for the
  world to finish loading by polling game.log.txt for "entering main loop"
  (stdout is freopen'd into that file -- launch-to-live is ~25-40 s, and the
  process's real stdout carries nothing), records N seconds with
  tracy-capture, then exports zone statistics and per-occurrence CSVs with
  tracy-csvexport and prints the top-20 zones by max duration.

  The first run builds the game and/or the two Tracy command-line tools if
  they are missing (the tools come from the same FetchContent checkout the
  game builds against). The game's Tracy client is pinned to -Port via the
  TRACY_PORT environment variable, so the capture only ever attaches to the
  instance this script launched.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\profile.ps1 -Seconds 45 -KeepGame
#>
[CmdletBinding()]
param(
    [int]$Seconds = 30,             # capture duration once the world is live
    [string]$BuildDir = "build-tracy",
    [string]$OutDir = "",           # default: profiles\<timestamp> under repo root
    [int]$Port = 8086,              # Tracy client data port (TRACY_PORT)
    [switch]$KeepGame,              # leave the game running after the capture
    [int]$StartupTimeoutSec = 120
)

$ErrorActionPreference = 'Stop'
$repo  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo $BuildDir

function Find-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $vs = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path $vs) { return $vs }
    throw "cmake not found on PATH or at the VS 2026 bundled location"
}
$cmake = Find-CMake

function Invoke-CMake {
    param([string[]]$CMakeArgs, [string]$What)
    & $cmake @CMakeArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake failed ($What)" }
}

# --- game build (configure + build on first run) ----------------------------
$gameExe = Join-Path $build "Release\voxel.exe"
if (-not (Test-Path $gameExe)) {
    Write-Host "[profile] building Tracy-enabled game into $BuildDir ..."
    Invoke-CMake @("-S", $repo, "-B", $build, "-DVOXEL_TRACY=ON") "configure $BuildDir"
    Invoke-CMake @("--build", $build, "--config", "Release", "--parallel") "build $BuildDir"
}

# --- tracy-capture / tracy-csvexport (built from the FetchContent checkout) --
function Ensure-TracyTool {
    param([string]$Name)   # "capture" or "csvexport"
    $exe = Join-Path $build "tracy-tools\$Name\Release\tracy-$Name.exe"
    if (Test-Path $exe) { return $exe }
    $src = Join-Path $build "_deps\tracy-src\$Name"
    if (-not (Test-Path $src)) { throw "Tracy sources missing at $src -- configure $BuildDir with -DVOXEL_TRACY=ON first" }
    Write-Host "[profile] building tracy-$Name (one-time) ..."
    # The tools CPM-fetch capstone, whose repo has paths past MAX_PATH; feed
    # git core.longpaths through the environment so nothing global changes.
    $env:GIT_CONFIG_COUNT   = "1"
    $env:GIT_CONFIG_KEY_0   = "core.longpaths"
    $env:GIT_CONFIG_VALUE_0 = "true"
    $bld = Join-Path $build "tracy-tools\$Name"
    Invoke-CMake @("-S", $src, "-B", $bld, "-DNO_ISA_EXTENSIONS=ON") "configure tracy-$Name"
    Invoke-CMake @("--build", $bld, "--config", "Release", "--parallel") "build tracy-$Name"
    if (-not (Test-Path $exe)) { throw "tracy-$Name build produced no exe at $exe" }
    return $exe
}
$captureExe   = Ensure-TracyTool "capture"
$csvexportExe = Ensure-TracyTool "csvexport"

# --- output location ---------------------------------------------------------
if (-not $OutDir) { $OutDir = Join-Path $repo ("profiles\" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$trace = Join-Path $OutDir "capture.tracy"

# --- launch the game from the repo root (src/shaders/ resolves relative) -----
$log = Join-Path $repo "game.log.txt"
try { Remove-Item $log -Force -ErrorAction Stop } catch {}
$env:TRACY_PORT = "$Port"
$game = Start-Process -FilePath $gameExe -WorkingDirectory $repo -PassThru
Write-Host "[profile] launched voxel.exe (pid $($game.Id), Tracy port $Port); waiting for 'entering main loop' ..."

$ready = $false
$deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
try {
    while ((Get-Date) -lt $deadline) {
        if ($game.HasExited) { throw "voxel.exe exited during startup (code $($game.ExitCode))" }
        if ((Test-Path $log) -and
            ((Get-Item $log).LastWriteTime -gt $game.StartTime) -and
            (Select-String -Path $log -Pattern "entering main loop" -Quiet)) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "gave up after ${StartupTimeoutSec}s waiting for 'entering main loop' in $log" }
    Write-Host "[profile] world is live; capturing $Seconds s to $trace ..."

    & $captureExe -o $trace -a 127.0.0.1 -p $Port -s $Seconds -f
    if ($LASTEXITCODE -ne 0) { throw "tracy-capture failed (exit $LASTEXITCODE)" }
} finally {
    if (-not $KeepGame -and -not $game.HasExited) {
        try { Stop-Process -Id $game.Id -Force -Confirm:$false } catch {}
    }
}

# --- CSV exports --------------------------------------------------------------
function Export-Csv-Mode {
    param([string[]]$ToolArgs, [string]$Dest, [string]$What)
    $lines = & $csvexportExe @ToolArgs
    if ($LASTEXITCODE -ne 0) { throw "tracy-csvexport failed ($What)" }
    [IO.File]::WriteAllLines($Dest, [string[]]$lines)
    Write-Host "[profile] wrote $What -> $Dest"
}
$statsCsv  = Join-Path $OutDir "zones_stats.csv"
$unwrapCsv = Join-Path $OutDir "zones_unwrapped.csv"
$gpuCsv    = Join-Path $OutDir "gpu_zones.csv"
Export-Csv-Mode @($trace)        $statsCsv  "zone statistics"
Export-Csv-Mode @("-u", $trace)  $unwrapCsv "per-occurrence zone events"
try { Export-Csv-Mode @("-g", $trace) $gpuCsv "per-occurrence GPU zone events" }
catch { Write-Warning "GPU zone export failed: $_" }

# --- top-20 zones by max duration ---------------------------------------------
$rows = Import-Csv $statsCsv
$top = $rows | Sort-Object { [double]$_.max_ns } -Descending | Select-Object -First 20
$fmt = $top | Format-Table -AutoSize `
    @{ n = 'zone';     e = { $_.name } },
    @{ n = 'count';    e = { $_.counts } },
    @{ n = 'mean_ms';  e = { '{0:N3}' -f ([double]$_.mean_ns  / 1e6) } },
    @{ n = 'max_ms';   e = { '{0:N3}' -f ([double]$_.max_ns   / 1e6) } },
    @{ n = 'total_ms'; e = { '{0:N3}' -f ([double]$_.total_ns / 1e6) } }
Write-Host ""
Write-Host "=== top 20 zones by max duration ==="
$fmt | Out-String -Width 200 | Write-Host
$top | Select-Object name, counts, mean_ns, max_ns, total_ns |
    Export-Csv -NoTypeInformation -Path (Join-Path $OutDir "top_zones.csv")
Write-Host "[profile] trace + CSVs in $OutDir"
