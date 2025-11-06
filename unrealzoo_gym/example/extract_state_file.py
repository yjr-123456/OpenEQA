import os
import shutil

def extract_specific_files(source_base_path, dest_base_path, filename_to_find):
    """
    遍历源路径，查找特定文件名，并将其连同目录结构一起复制到目标路径。

    Args:
        source_base_path (str): 要搜索的源根目录。
        dest_base_path (str): 要将文件复制到的目标根目录。
        filename_to_find (str): 要查找和复制的特定文件名。
    """
    print(f"开始从 '{source_base_path}' 提取 '{filename_to_find}' 文件...")
    
    # 确保目标根目录存在
    os.makedirs(dest_base_path, exist_ok=True)
    
    copied_count = 0
    # os.walk会遍历所有子目录
    for root, dirs, files in os.walk(source_base_path):
        if filename_to_find in files:
            # 1. 构建完整的文件源路径
            source_file_path = os.path.join(root, filename_to_find)
            
            # 2. 计算相对于源根目录的路径，以保持结构
            relative_path = os.path.relpath(root, source_base_path)
            
            # 3. 构建目标目录和文件路径
            dest_dir_path = os.path.join(dest_base_path, relative_path)
            dest_file_path = os.path.join(dest_dir_path, filename_to_find)
            
            # 4. 创建目标子目录（如果尚不存在）
            os.makedirs(dest_dir_path, exist_ok=True)
            
            # 5. 复制文件
            shutil.copy2(source_file_path, dest_file_path)
            print(f"已复制: {dest_file_path}")
            copied_count += 1
            
    print(f"\n处理完成！总共复制了 {copied_count} 个文件到 '{dest_base_path}'。")

if __name__ == "__main__":
    # --- 配置 ---
    # 源文件夹
    source_directory = "E:/EQA/unrealzoo_gym/example/QA_data_sub"
    
    # 新建一个文件夹来存放提取出的文件
    destination_directory = "E:/EQA/unrealzoo_gym/example/QA_data_status_recorder"
    
    # 您想要提取的特定文件名
    target_filename = "status_recorder_gemini-2.5-flash.json"
    
    # --- 执行 ---
    extract_specific_files(source_directory, destination_directory, target_filename)