@echo off
REM =================================================================
REM  使用 WatchDog 启动 run_baseline.py 的批处理脚本
REM =================================================================

REM --- 1. 配置参数 ---

REM !! 在这里设置您的 Anaconda 环境名称 !!
set CONDA_ENV_NAME=OpenEQA

REM 要监控的脚本
set SCRIPT_TO_RUN=run_baseline.py

REM 要测试的环境 (可以写多个，用空格隔开)
set ENVS="ModularNeighborhood" "Map_ChemicalPlant_1" "Pyramid" "Greek_Island" "SuburbNeighborhood_Day" "LV_Bazaar" "DowntownWest" "PlanetOutDoor" "RussianWinterTownDemo01" "AsianMedivalCity" "Medieval_Castle" "SnowMap" "Real_Landscape" "Demonstration_Castle" "Venice"

REM 要测试的问题类型 (可以写多个，用空格隔开)
set QUESTION_TYPES=counting relative_location relative_distance state

REM 使用的模型
set MODEL=gemini_pro

REM 是否从断点恢复 (如果要恢复，取消下一行的注释)
set RESUME_FLAG=--resume

REM WatchDog 的日志目录
set LOG_DIR=watchdog_logs

REM offscreen (如果要恢复，取消下一行的注释)
REM set OFFSCREEN_FLAG=--offscreen

REM WatchDog 监听 Unreal Engine 进程 PID 的端口
set PID_PORT=50007

REM --- 2. 激活 Anaconda 虚拟环境 ---
echo Activating Anaconda environment: %CONDA_ENV_NAME%...
call conda activate %CONDA_ENV_NAME%

if %errorlevel% neq 0 (
    echo Failed to activate Conda environment. Please check the environment name.
    pause
    exit /b
)

REM --- 3. 执行 WatchDog ---
echo Starting WatchDog to monitor %SCRIPT_TO_RUN%...

python watch_dog.py ^
    %SCRIPT_TO_RUN% ^
    --envs %ENVS% ^
    --question_types %QUESTION_TYPES% ^
    --model %MODEL% ^
    --log-dir %LOG_DIR% ^
    --pid-port %PID_PORT% ^
    %RESUME_FLAG%
    %OFFSCREEN_FLAG%

echo.
echo WatchDog script has finished.
pause