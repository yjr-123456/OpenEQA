import subprocess
import time
import sys
import os
import psutil
from datetime import datetime

def kill_unreal_processes():
    target_processes = ["UnrealZoo_UE5_5.exe", "UnrealZoo_UE5_5.app"]
    killed_processes = []

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            proc_name = proc.info['name']
            if proc_name and any(target in proc_name for target in target_processes):
                proc.kill()
                killed_processes.append(f"{proc_name}({proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if os.name == "nt":
        for target in target_processes:
            subprocess.run(["taskkill", "/IM", target, "/F"], capture_output=True)

    return len(killed_processes)

def cleanup_and_wait():
    killed_count = kill_unreal_processes()
    if killed_count > 0:
        time.sleep(3)
        kill_unreal_processes()

def monitor_log_file(log_path, max_silence):
    """监控日志文件是否长时间没有更新"""
    if not os.path.exists(log_path):
        return 0

    last_mod_time = os.path.getmtime(log_path)
    silence_duration = time.time() - last_mod_time
    return silence_duration

def run_with_file_logging():
    restart_count = 0
    max_restarts = 200
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    print("WatchDog 启动中 (文件日志模式)")
    not_done = True
    while restart_count < max_restarts and not_done:
        restart_count += 1
        print(f"\n{'='*60}")
        print(f"第 {restart_count} 次启动 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if restart_count > 1:
            cleanup_and_wait()
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"./watch_dog_logger/{time_stamp}/run_{restart_count}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"日志文件: {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable, "depth_test.py", "--resume", "--model", "openai_gpt4o_2024_11_20"],
                stdout=log_file,
                stderr=log_file
            )

        print(f"程序已启动，PID: {process.pid}")
        max_silence = 180  # 3分钟
        check_interval = 30

        while True:
            if process.poll() is not None:
                print("进程已退出")
                not_done = False
                break

            silence_duration = monitor_log_file(log_path, max_silence)
            if silence_duration > max_silence:
                print(f"日志文件 {silence_duration:.0f}s 未更新，认为程序卡死")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                break

            print(f"监控中... 静默 {silence_duration:.0f}s / {max_silence}s")
            time.sleep(check_interval)

        if process.returncode == 0:
            print("程序正常结束")
            break
        else:
            print("准备重启...")

    print("WatchDog 结束")
    cleanup_and_wait()

if __name__ == "__main__":
    run_with_file_logging()
