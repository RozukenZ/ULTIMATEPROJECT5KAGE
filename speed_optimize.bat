@echo off
REM Ollama Speed Optimization Script for Windows

REM Set environment variables for maximum performance
set OLLAMA_NUM_PARALLEL=4
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_MAX_QUEUE=512
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KEEP_ALIVE=24h
set OLLAMA_HOST=127.0.0.1:11434

echo [SPEED] Ollama Speed Optimization Applied!
echo Model will stay loaded for 24 hours for instant responses

REM Pre-load the model for instant access
echo [FAST] Pre-loading model for zero-delay responses...
start /B ollama run umm-informatics-speed --keepalive 24h

echo [OK] Speed optimization complete!
echo Usage: ollama run umm-informatics-speed
pause
