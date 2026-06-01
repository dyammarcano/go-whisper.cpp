@echo off
REM Build whisper.cpp with CUDA as MSVC shared DLLs (whisper.dll + ggml*.dll incl.
REM ggml-cuda.dll) into repo-root build-cuda\bin\. CUDA/MSVC stays sealed in the DLLs;
REM the MinGW cgo host links only whisper.dll's C API. Needs VS2022 + CUDA toolkit (nvcc).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=%USERPROFILE%\scoop\shims;%PATH%"
set "CUDA=%USERPROFILE%\scoop\apps\cuda\current"
cd /d "%~dp0.." || exit /b 1
rmdir /s /q build-cuda 2>nul
cmake -S whisper.cpp -B build-cuda -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor" ^
  -DCMAKE_C_FLAGS="/Zc:preprocessor" -DCMAKE_CXX_FLAGS="/Zc:preprocessor" ^
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF ^
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF || exit /b 1
REM Cap parallel jobs: the heavy fattn-vec hs256 CUDA template instances can fail
REM (nvcc subprocess exit 1, no diagnostic) under full 8-way Ninja parallelism on
REM CUDA 13 + arch 75. -j 4 builds them reliably.
cmake --build build-cuda --config Release --target whisper -j 4 || exit /b 1
echo === CUDA DLLs ===
dir /s /b build-cuda\bin\*.dll
