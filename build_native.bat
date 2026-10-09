@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

python "%~dp0build_native.py"
exit /b %errorlevel%
