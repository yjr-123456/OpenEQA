#!/usr/bin/env python3
import subprocess
import time
import sys
import os
import psutil
import signal
import argparse
import tempfile
import json
import platform
import uuid
from datetime import datetime
from pathlib import Path
import threading
import socket
current_dir = os.path.dirname(os.path.abspath(__file__))


class WatchDog:
    def __init__(self, script_path, script_args=None, log_dir="logs", instance_id=None, 
                 max_silence=180, check_interval=30, max_restarts=200, pid_port=50007):
        self.script_path = script_path
        self.script_args = script_args or []
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.log_base_dir = Path(log_dir) / f"watch_dog_{self.instance_id}"
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        self.max_silence = max_silence
        self.check_interval = check_interval
        self.max_restarts = max_restarts
        self.pid_port = pid_port

        self.managed_pids = set()
        self.current_process = None

        self.is_mac = platform.system() == "Darwin"
        self.is_windows = platform.system() == "Windows"

        self.state_dir = Path(self.log_base_dir) / "watchdog_states"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / f"watchdog_{self.instance_id}.json"

        print(f"WatchDog 实例 [{self.instance_id}] 已初始化")
        print(f"监控脚本: {self.script_path} {' '.join(self.script_args)}")
        print(f"监听端口: {self.pid_port}")
        self.start_pid_server(port=self.pid_port)
        self.process_check_interval = 10

    def check_managed_processes(self):
        missing_pids = []
        for pid in list(self.managed_pids):
            if not psutil.pid_exists(pid):
                missing_pids.append(pid)
                self.managed_pids.discard(pid)
        if missing_pids:
            print(f"检测到托管进程已退出: {missing_pids}")
            self.save_state()
        return missing_pids

    def save_state(self):
        state = {
            "instance_id": self.instance_id,
            "managed_pids": list(self.managed_pids),
            "script_path": self.script_path,
            "last_update": time.time()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.managed_pids = set(state.get("managed_pids", []))
                print(f"已加载状态: 管理 {len(self.managed_pids)} 个进程")
                return True
            except Exception as e:
                print(f"加载状态失败: {e}")
        return False

    def kill_managed_processes(self):
        killed_count = 0
        pids_to_check = self.managed_pids.copy()
        for pid in pids_to_check:
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
                print(f"终止进程: {proc_name} (PID: {pid})")
                if self.is_windows:
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.managed_pids.discard(pid)
        if killed_count > 0:
            time.sleep(3)
            for pid in self.managed_pids.copy():
                try:
                    proc = psutil.Process(pid)
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.managed_pids.discard(pid)
        self.save_state()
        return killed_count

    def monitor_log_file(self, log_path):
        if not os.path.exists(log_path):
            return 0
        last_mod_time = os.path.getmtime(log_path)
        silence_duration = time.time() - last_mod_time
        return silence_duration

    def run(self):
        restart_count = 0
        log_base_dir = self.log_base_dir
        log_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"WatchDog [{self.instance_id}] 启动中")
        self.load_state()
        if self.managed_pids:
            print(f"清理之前的进程 ({len(self.managed_pids)} 个)")
            self.kill_managed_processes()
        not_done = True
        while restart_count < self.max_restarts and not_done:
            restart_count += 1
            print(f"\n{'='*60}")
            print(f"第 {restart_count} 次启动 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_base_dir / f"run_{restart_count}_{time_stamp}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"日志文件: {log_path}")

            with open(log_path, "w", encoding="utf-8") as log_file:
                cmd = [sys.executable, "-u", self.script_path] + self.script_args
                os.environ["PYTHONIOENCODING"] = "utf-8"
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    bufsize=1,
                    text=True
                )
            print(f"程序已启动，PID: {self.current_process.pid}")
            self.save_state()
            time.sleep(10)
            last_pid_check = time.time()
            while True:
                if self.current_process.poll() is not None:
                    print(f"进程已退出，返回码: {self.current_process.returncode}")
                    if self.current_process.returncode == 0:
                        not_done = False
                    break
                silence_duration = self.monitor_log_file(log_path)
                if silence_duration > self.max_silence:
                    print(f"日志文件 {silence_duration:.0f}s 未更新，认为程序卡死")
                    self.current_process.terminate()
                    time.sleep(5)
                    if self.current_process.poll() is None:
                        self.current_process.kill()
                    break
                now = time.time()
                if now - last_pid_check >= self.process_check_interval:
                    missing_pids = self.check_managed_processes()
                    last_pid_check = now
                    if missing_pids:
                        print("关键托管进程缺失，终止并准备重启主程序")
                        self.current_process.terminate()
                        time.sleep(5)
                        if self.current_process.poll() is None:
                            self.current_process.kill()
                        break
                self.save_state()
                print(f"监控中... [{self.instance_id}] 静默 {silence_duration:.0f}s / {self.max_silence}s, 管理 {len(self.managed_pids)} 个进程")
                time.sleep(self.check_interval)
            print("清理UnrealEngine进程...")
            self.kill_managed_processes()
            if not not_done:
                print("程序正常结束")
                break
            else:
                print("准备重启...")
        print(f"WatchDog [{self.instance_id}] 结束运行")
        if self.state_file.exists():
            self.state_file.unlink()

    def start_pid_server(self, host='127.0.0.1', port=50007):
        def server():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                s.listen(5)
                print(f"PID监听服务器已启动: {host}:{port}")
                while True:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data:
                            try:
                                pid = int(data.decode())
                                print(f"收到来自run_baseline的PID: {pid}")
                                self.managed_pids.add(pid)
                            except Exception as e:
                                print(f"PID解析失败: {e}")
        t = threading.Thread(target=server, daemon=True)
        t.start()

def main():
    parser = argparse.ArgumentParser(description='高级WatchDog - 支持多实例和跨平台')
    parser.add_argument('script', help='要监控的脚本路径')
    # parser.add_argument('--envs', nargs='+', required=True, help='环境列表，如 ModularSciFiVillage Cabin_Lake')
    # parser.add_argument('--question_types', nargs='+', required=True, help='问题类型列表，如 counting state')
    parser.add_argument('--id', help='WatchDog实例ID (默认自动生成)')
    parser.add_argument('--log-dir', default='logs', help='日志目录')
    parser.add_argument('--silence', type=int, default=300, help='最大日志静默时间(秒)')
    parser.add_argument('--interval', type=int, default=10, help='检查间隔(秒)')
    parser.add_argument('--max-restarts', type=int, default=200, help='最大重启次数')
    parser.add_argument('--pid-port', type=int, default=50007, help='监听PID的端口')
    parser.add_argument('--model', type=str, default='gemini_pro', help='模型名称')
    parser.add_argument("--camera_height", dest="camera_height", type=int, default=1200, help="camera height from the ground")
    parser.add_argument('--ts', type=int,default=0, help='时间戳参数')
    args = parser.parse_args()
    script_args = []
    script_args += ['--model', args.model]
    script_args += ['--camera_height', str(args.camera_height)]
    # script_args += ['-e', env_name]
    script_args += ["--use_pid", "True"]
    script_args += ['--pid_port', str(args.pid_port)]
    # script_args += ['--floor_height', str(floor_height)]
    # script_args += ['--min_total', str(min_total)]
    # script_args += ['--max_total', str(max_total)]
    # script_args += ['--agent_categories', json.dumps(agent_categories)]
    # script_args += ['--type_ranges', json.dumps(type_ranges)]
    # if args.other_args:
    #     # 按空格分割并加入
    #     script_args += args.other_args.split()
    if args.ts and args.ts > 0:
        time.sleep(args.ts)
    watchdog = WatchDog(
        script_path=args.script,
        script_args=script_args,
        log_dir=args.log_dir,
        instance_id=args.id,
        max_silence=args.silence,
        check_interval=args.interval,
        max_restarts=args.max_restarts,
        pid_port=args.pid_port
    )
    try:
        watchdog.run()
    except KeyboardInterrupt:
        print("\nWatchDog被用户中断")
        watchdog.kill_managed_processes()
    except Exception as e:
        print(f"WatchDog发生错误: {e}")
        import traceback
        traceback.print_exc()
        watchdog.kill_managed_processes()

if __name__ == "__main__":
    main()