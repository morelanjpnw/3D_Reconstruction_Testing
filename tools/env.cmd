@echo off
rem Build env for gsplat CUDA JIT (CUDA 11.8 + VS2019 BuildTools) -- usage: tools\env.cmd python scripts\...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set TORCH_CUDA_ARCH_LIST=8.6
set MAX_JOBS=4
set PATH=C:\Users\Jack\miniconda3\envs\splatlab;C:\Users\Jack\miniconda3\envs\splatlab\Scripts;%PATH%
rem windows.h defines the macro "small", which breaks CUDA 11.8's CUB headers
set CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX
%*
