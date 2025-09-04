#!/usr/bin/env python3
# filepath: e:\EQA\unrealzoo_gym\example\watch_dog.py
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

class WatchDog:
    def __init__(self, script_path, script_args=None, log_dir="logs", instance_id=None, 
                 max_silence=180, check_interval=30, max_restarts=200):
        """初始化WatchDog实例
        
        Args:
            script_path: 要监控的脚本路径
            script_args: 脚本参数列表
            log_dir: 日志存放目录
            instance_id: 实例ID（如果为None则自动生成）
            max_silence: 最大日志静默时间（秒）
            check_interval: 检查间隔（秒）
            max_restarts: 最大重启次数
        """
        self.script_path = script_path
        self.script_args = script_args or []
        self.log_dir = log_dir
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.max_silence = max_silence
        self.check_interval = check_interval
        self.max_restarts = max_restarts
        
        # 子进程相关
        self.managed_pids = set()  # 由此实例管理的所有进程ID
        self.current_process = None
        
        # 系统相关
        self.is_mac = platform.system() == "Darwin"
        self.is_windows = platform.system() == "Windows"
        
        # 状态跟踪文件
        self.state_dir = Path(tempfile.gettempdir()) / "watchdog_states"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / f"watchdog_{self.instance_id}.json"
        
        print(f"WatchDog 实例 [{self.instance_id}] 已初始化")
        print(f"监控脚本: {self.script_path} {' '.join(self.script_args)}")
    
    def save_state(self):
        """保存当前状态到文件"""
        state = {
            "instance_id": self.instance_id,
            "managed_pids": list(self.managed_pids),
            "script_path": self.script_path,
            "last_update": time.time()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def load_state(self):
        """从文件加载状态"""
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
    
    def find_child_processes(self, parent_pid):
        """递归查找所有子进程"""
        try:
            parent = psutil.Process(parent_pid)
            children = parent.children(recursive=True)
            return [child.pid for child in children]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
    
    
    def kill_managed_processes(self):
        """仅终止由该WatchDog实例管理的进程"""
        killed_count = 0
        
        # 复制集合，因为我们会在迭代过程中修改它
        pids_to_check = self.managed_pids.copy()
        
        for pid in pids_to_check:
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
                print(f"终止进程: {proc_name} (PID: {pid})")
                
                # 尝试优雅终止
                if self.is_windows:
                    proc.terminate()
                else:
                    # Mac/Linux上，SIGTERM更适合
                    proc.send_signal(signal.SIGTERM)
                
                killed_count += 1
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # 进程已不存在，从列表中移除
                self.managed_pids.discard(pid)
        
        # 如果有进程被终止，等待一会儿再强制终止
        if killed_count > 0:
            time.sleep(3)
            
            # 强制终止仍然存在的进程
            for pid in self.managed_pids.copy():
                try:
                    proc = psutil.Process(pid)
                    proc.kill()  # 强制终止
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.managed_pids.discard(pid)
        
        # 保存状态
        self.save_state()
        return killed_count
    
    def monitor_log_file(self, log_path):
        """监控日志文件是否长时间没有更新"""
        if not os.path.exists(log_path):
            return 0

        last_mod_time = os.path.getmtime(log_path)
        silence_duration = time.time() - last_mod_time
        return silence_duration
    
    def run(self):
        """运行WatchDog监控循环"""
        restart_count = 0
        log_base_dir = Path(self.log_dir) / f"watch_dog_{self.instance_id}"
        log_base_dir.mkdir(parents=True, exist_ok=True)

        print(f"WatchDog [{self.instance_id}] 启动中")
        
        # 尝试加载之前的状态
        self.load_state()
        
        # 如果有之前的进程，先清理
        if self.managed_pids:
            print(f"清理之前的进程 ({len(self.managed_pids)} 个)")
            self.kill_managed_processes()
        
        not_done = True
        while restart_count < self.max_restarts and not_done:
            restart_count += 1
            print(f"\n{'='*60}")
            print(f"第 {restart_count} 次启动 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            # 创建新的日志目录和文件
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_base_dir / f"run_{restart_count}_{time_stamp}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"日志文件: {log_path}")

            with open(log_path, "w", encoding="utf-8") as log_file:
                # 启动目标脚本
                cmd = [sys.executable, self.script_path] + self.script_args
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file
                )

            print(f"程序已启动，PID: {self.current_process.pid}")
            last_position = 0 
            # 等待一段时间让UnrealEngine进程启动
            time.sleep(10)
            
            last_position = self.parse_logs_for_pid(log_path, last_position=last_position)
            self.save_state()
            # 监控循环
            while True:
                # 检查主进程是否还在运行
                if self.current_process.poll() is not None:
                    print(f"进程已退出，返回码: {self.current_process.returncode}")
                    if self.current_process.returncode == 0:
                        not_done = False  # 正常退出，不需要重启
                    break
                
                # 检查日志文件更新
                silence_duration = self.monitor_log_file(log_path)
                if silence_duration > self.max_silence:
                    print(f"日志文件 {silence_duration:.0f}s 未更新，认为程序卡死")
                    self.current_process.terminate()
                    time.sleep(5)
                    if self.current_process.poll() is None:
                        self.current_process.kill()
                    break
                last_position = self.parse_logs_for_pid(log_path, last_position=last_position)
                self.save_state()

                print(f"监控中... [{self.instance_id}] 静默 {silence_duration:.0f}s / {self.max_silence}s, 管理 {len(self.managed_pids)} 个进程")
                time.sleep(self.check_interval)

            # 清理当前运行的UnrealEngine进程
            print("清理UnrealEngine进程...")
            self.kill_managed_processes()
            
            if not not_done:
                print("程序正常结束")
                break
            else:
                print("准备重启...")

        print(f"WatchDog [{self.instance_id}] 结束运行")
        
        # 清理状态文件
        if self.state_file.exists():
            self.state_file.unlink()

    def parse_logs_for_pid(self, log_path, last_position=0 ):
        """从日志文件中解析所有PID信息"""
        import re
        try:
            if not os.path.exists(log_path):
                return last_position
                
            found_pids = set()  # 用于跟踪此次发现的PID
            file_size = os.path.getsize(log_path)
            if file_size < last_position:
                last_position = 0
            if file_size == last_position:
                return last_position

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_position)
                content = f.read()
                
                # 查找常见的PID模式
                pid_patterns = [
                    r'Running docker-free env, pid:(\d+)',  # 您的日志中的格式
                    r'pid[:\s=]+(\d+)',                     # 其他可能的格式
                    r'process id[:\s=]+(\d+)',              # 其他可能的格式
                    r'started process (\d+)',               # 启动进程的格式
                    r'launched with pid (\d+)',             # 另一种常见格式
                    r'PID: (\d+)'                           # 简单PID标记
                ]
                
                for pattern in pid_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            pid = int(match)
                            if pid in self.managed_pids:
                                continue  # 跳过已经管理的PID
                                
                            print(f"从日志文件中发现进程PID: {pid}")
                            
                            # 验证进程是否存在
                            try:
                                proc = psutil.Process(pid)
                                proc_name = proc.name()
                                print(f"已确认进程存在: {proc_name} (PID: {pid})")
                                self.managed_pids.add(pid)
                                found_pids.add(pid)
                                
                                # 查找该进程的子进程
                                child_pids = self.find_child_processes(pid)
                                for child_pid in child_pids:
                                    if child_pid in self.managed_pids:
                                        continue  # 跳过已经管理的子PID
                                        
                                    try:
                                        child = psutil.Process(child_pid)
                                        print(f"添加子进程: {child.name()} (PID: {child_pid})")
                                        self.managed_pids.add(child_pid)
                                        found_pids.add(child_pid)
                                    except:
                                        pass
                                        
                            except:
                                print(f"PID {pid} 不存在或无法访问")
                        except ValueError:
                            continue
        
            return file_size
    
        except Exception as e:
            print(f"解析日志文件时出错: {e}")
            return last_position



def main():
    parser = argparse.ArgumentParser(description='高级WatchDog - 支持多实例和跨平台')
    parser.add_argument('script', help='要监控的脚本路径')
    # 改为接受单个字符串参数，而非多个值
    parser.add_argument('--args', type=str, help='传递给脚本的参数，JSON列表格式，如: \'["--model", "gemini_pro"]\'')
    parser.add_argument('--id', help='WatchDog实例ID (默认自动生成)')
    parser.add_argument('--log-dir', default='logs', help='日志目录')
    parser.add_argument('--silence', type=int, default=180, help='最大日志静默时间(秒)')
    parser.add_argument('--interval', type=int, default=30, help='检查间隔(秒)')
    parser.add_argument('--max-restarts', type=int, default=200, help='最大重启次数')
    
    args = parser.parse_args()
    
    # 解析JSON格式的脚本参数
    script_args = []
    if args.args:
        try:
            import json
            script_args = json.loads(args.args)
            if not isinstance(script_args, list):
                print("警告: --args参数必须是JSON列表格式，例如: '[\"--model\", \"gemini_pro\"]'")
                script_args = []
        except json.JSONDecodeError:
            print(f"警告: 无法解析JSON参数: {args.args}")
            print("应使用格式如: --args '[\"--model\", \"gemini_pro\"]'")
    
    watchdog = WatchDog(
        script_path=args.script,
        script_args=script_args,
        log_dir=args.log_dir,
        instance_id=args.id,
        max_silence=args.silence,
        check_interval=args.interval,
        max_restarts=args.max_restarts
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