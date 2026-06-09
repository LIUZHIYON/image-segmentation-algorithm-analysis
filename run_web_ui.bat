@echo off
REM ========================================
REM  图像分割算法分析系统 启动脚本
REM  直接使用 openmmlab 环境的 Python
REM ========================================

set PYTHON_EXE=E:\anaconda3\envs\openmmlab\python.exe

cd /d E:\bishe_custom_data

set KMP_DUPLICATE_LIB_OK=TRUE
set YOLO_VERBOSE=False

echo ============================================
echo   启动图像分割算法分析系统
echo   Python: %PYTHON_EXE%
echo   地址: http://localhost:7860
echo ============================================

%PYTHON_EXE% web_ui.py

pause
