<#
 One-shot setup for a fresh Windows machine (tested design target: RTX PRO 6000 Blackwell, also fine for Ada/Ampere).
 Run from the repo root:   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
 Add -CheckOnly to only report what is installed / missing (changes nothing).

 Needs beforehand: git, Miniconda/Anaconda, NVIDIA driver, CUDA Toolkit (nvcc), Visual Studio 2019/2022 C++ Build Tools.
 The script tells you exactly what is missing and how to install it.
#>
param([string]$EnvName = "splatlab", [switch]$CheckOnly)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$problems = @()

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  OK  $m" -ForegroundColor Green }
function Bad($m, $fix) { Write-Host "  MISSING  $m" -ForegroundColor Red; Write-Host "           -> $fix" -ForegroundColor Yellow; $script:problems += $m }
function Exec($exe) {   # usage: Exec exe arg1 arg2 ... ; stops the script on non-zero exit
    $a = $args
    & $exe @a
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $exe $a" -ForegroundColor Red; exit 1 }
}

# ---------------------------------------------------------------- prerequisites
Step "Checking prerequisites"
if (Get-Command git -ErrorAction SilentlyContinue) { Ok "git" } else { Bad "git" "winget install Git.Git" }

$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) { Ok "conda" } else { Bad "conda" "winget install Anaconda.Miniconda3   (then open a NEW terminal, run 'conda init powershell' once)" }

$cap = $null; $gpuName = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
    $cap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
    Ok "GPU: $gpuName (compute capability $cap)"
} else { Bad "NVIDIA driver / nvidia-smi" "install the current NVIDIA driver from nvidia.com" }

$nvccMajor = 0; $nvccMinor = 0
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $m = [regex]::Match((nvcc --version | Out-String), "release (\d+)\.(\d+)")
    $nvccMajor = [int]$m.Groups[1].Value; $nvccMinor = [int]$m.Groups[2].Value
    Ok "CUDA toolkit nvcc $nvccMajor.$nvccMinor"
} else { Bad "CUDA Toolkit (nvcc)" "winget install Nvidia.CUDA   (admin; then open a NEW terminal). Blackwell needs CUDA 12.8 or newer." }

$blackwell = $false
if ($cap) {
    $capMajor = [int]($cap.Split(".")[0])
    $blackwell = $capMajor -ge 12
    if ($blackwell -and ($nvccMajor -lt 12 -or ($nvccMajor -eq 12 -and $nvccMinor -lt 8)) -and $nvccMajor -ne 0) {
        Bad "CUDA Toolkit >= 12.8 (found $nvccMajor.$nvccMinor; Blackwell GPUs need 12.8+)" "winget install Nvidia.CUDA   then put the new CUDA bin folder first on PATH"
    }
}

# Visual Studio C++ tools: prefer 2022 (17.x), then 2019; VS 2026 (18.x) is too new for current CUDA.
$vsw = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = $null
if (Test-Path $vsw) {
    $vsPath = & $vsw -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version "[16.0,18.0)" -latest -property installationPath
    if ($vsPath) {
        $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        Ok "Visual Studio C++ tools: $vsPath"
    }
}
if (-not $vcvars) {
    Bad "Visual Studio 2019/2022 C++ Build Tools" 'winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
}

if ($problems.Count -gt 0) {
    Write-Host "`nFix the MISSING items above, open a NEW terminal, and re-run this script." -ForegroundColor Red
    exit 1
}
if ($CheckOnly) { Write-Host "`nAll prerequisites present. (CheckOnly: nothing installed.)" -ForegroundColor Green; exit 0 }

# ---------------------------------------------------------------- tools
New-Item -ItemType Directory -Force "tools", "inputs", "logs" | Out-Null

Step "COLMAP 4.2.1 (CUDA build)"
if (-not (Test-Path "tools\colmap\bin\colmap.exe")) {
    Exec curl.exe -L --fail -o tools\colmap.zip https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip
    Expand-Archive tools\colmap.zip tools\colmap -Force
    Remove-Item tools\colmap.zip
}
Ok "tools\colmap"

Step "Real-ESRGAN ncnn-vulkan"
if (-not (Test-Path "tools\realesrgan\realesrgan-ncnn-vulkan.exe")) {
    Exec curl.exe -L --fail -o tools\resr.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
    Expand-Archive tools\resr.zip tools\realesrgan -Force
    Remove-Item tools\resr.zip
}
Ok "tools\realesrgan"

# ---------------------------------------------------------------- python env
Step "Python environment '$EnvName'"
$base = (conda info --base).Trim()
$py = Join-Path $base "envs\$EnvName\python.exe"
if (-not (Test-Path $py)) {
    Exec conda create -y -n $EnvName -c conda-forge --override-channels python=3.10
}
Ok $py

if ($blackwell) { $torchArgs = @("torch==2.7.0", "torchvision==0.22.0", "--index-url", "https://download.pytorch.org/whl/cu128") }
else            { $torchArgs = @("torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121") }
Write-Host "Installing PyTorch: $($torchArgs -join ' ')"
Exec $py -m pip install @torchArgs
Exec $py -m pip install "numpy<2" "setuptools<81" ninja opencv-python pillow tqdm pyyaml scipy scikit-image lpips pycolmap==4.2.1 imageio imageio-ffmpeg plyfile torchmetrics streamlit pandas
Exec $py -m pip install gsplat==1.5.3
Exec $py scripts\patch_gsplat_windows.py

# ---------------------------------------------------------------- machine-specific build env wrapper
Step "Writing tools\env.local.cmd"
$envCmd = @"
@echo off
rem Generated by setup_windows.ps1 for this machine. Usage: tools\env.local.cmd python scripts\...
call "$vcvars" >nul
set TORCH_CUDA_ARCH_LIST=$cap
set MAX_JOBS=8
set PATH=$base\envs\$EnvName;$base\envs\$EnvName\Scripts;%PATH%
rem windows.h defines the macro "small", which breaks some CUB headers
set CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX
%*
"@
Set-Content -Path tools\env.local.cmd -Value $envCmd -Encoding ASCII
Ok "tools\env.local.cmd (arch $cap)"

# ---------------------------------------------------------------- self-test
Step "GPU self-test (first run compiles gsplat's CUDA code: expect 5-15 minutes)"
cmd /c "tools\env.local.cmd python scripts\selftest.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSelf-test FAILED. Send the output above (or logs) back to the assistant." -ForegroundColor Red
    exit 1
}
Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next (CLI):  tools\env.local.cmd python scripts\prepare_data.py"
Write-Host "Next (Web UI): .\launch_ui.cmd"
