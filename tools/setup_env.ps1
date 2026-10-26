$ErrorActionPreference="Continue"
$r="D:\GeminiSandbox\3D_Reconstruction_Testing"
Write-Host "== COLMAP"
if(!(Test-Path "$r\tools\colmap")){
  Invoke-WebRequest https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip -OutFile "$r\tools\colmap.zip"
  Expand-Archive "$r\tools\colmap.zip" "$r\tools\colmap" -Force; Remove-Item "$r\tools\colmap.zip"
}
Write-Host "== conda env"
conda create -y -n splatlab python=3.10
$py="C:\Users\Jack\miniconda3\envs\splatlab\python.exe"
& $py -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
& $py -m pip install "numpy<2" opencv-python pillow tqdm pyyaml scikit-image lpips pycolmap imageio imageio-ffmpeg plyfile torchmetrics ninja
Write-Host "== DONE"
