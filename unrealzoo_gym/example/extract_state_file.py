import os
import shutil
import argparse
import fnmatch

def extract_specific_files(source_base_path, dest_base_path, filename_pattern):
    """
    遍历源路径，查找匹配特定模式（支持通配符）的文件，并将其连同目录结构一起复制到目标路径。

    Args:
        source_base_path (str): 要搜索的源根目录。
        dest_base_path (str): 要将文件复制到的目标根目录。
        filename_pattern (str): 要查找和复制的特定文件名模式 (例如 'status_recorder_*.json')。
    """
    print(f"开始从 '{source_base_path}' 提取匹配 '{filename_pattern}' 的文件...")
    
    os.makedirs(dest_base_path, exist_ok=True)
    
    copied_count = 0
    for root, dirs, files in os.walk(source_base_path):
        # fnmatch.filter 支持通配符匹配
        for filename in fnmatch.filter(files, filename_pattern):
            source_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, source_base_path)
            dest_dir_path = os.path.join(dest_base_path, relative_path)
            dest_file_path = os.path.join(dest_dir_path, filename)
            
            os.makedirs(dest_dir_path, exist_ok=True)
            
            shutil.copy2(source_file_path, dest_file_path)
            print(f"已复制: {dest_file_path}")
            copied_count += 1
            
    print(f"\n处理完成！总共复制了 {copied_count} 个文件到 '{dest_base_path}'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从一个或多个环境中提取特定文件。")
    parser.add_argument('--env', type=str, nargs='+', required=True, help="要处理的环境名称。可提供多个值 (例如 'AsianMedivalCity' 'Venice')。使用 'all' 处理所有环境。")
    parser.add_argument('--target_filename', type=str, default="status_recorder_doubao.json", help="要提取的特定文件名。可使用通配符 (例如 'status_recorder_*.json' 或 '*')。")
    
    args = parser.parse_args()

    # --- 动态路径配置 ---
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 源目录应该是脚本所在目录下的 'QA_data_sub'
    source_root = "C:/QA_data_sub"
    
    # 创建一个统一的输出根目录
    output_root = os.path.join(script_dir, "QA_data_sub")
    os.makedirs(output_root, exist_ok=True)
    
    environments_to_process = []
    if 'all' in args.env:
        print(f"检测到 'all' 参数，将处理 '{source_root}' 下的所有环境...")
        try:
            environments_to_process = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
        except FileNotFoundError:
            print(f"错误：源根目录 '{source_root}' 不存在。请检查路径。")
            exit(1)
    else:
        environments_to_process = args.env

    print(f"计划处理的环境: {', '.join(environments_to_process)}")

    for env_name in environments_to_process:
        source_directory = os.path.join(source_root, env_name)

        if not os.path.isdir(source_directory):
            print(f"\n--- 警告：环境目录 '{source_directory}' 不存在，已跳过。 ---")
            continue

        # 在统一的输出根目录下，为每个环境创建子目录
        destination_directory = os.path.join(output_root, env_name)
        
        print(f"\n--- 开始处理环境: {env_name} ---")
        # --- 执行 ---
        extract_specific_files(source_directory, destination_directory, args.target_filename)

    print("\n所有指定环境的任务已完成。")