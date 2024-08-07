@echo off
setlocal

glslangValidator -V image_shader.comp -o image_shader.spv


REM Check if the compilation was successful
if %ERRORLEVEL% neq 0 (
    echo Shader compilation failed.
    exit /b %ERRORLEVEL%
)

echo Shader compiled successfully: 
endlocal