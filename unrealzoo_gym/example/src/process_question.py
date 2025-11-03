import json
import os
from datetime import datetime
from .llm_eval import calculate_accuracy


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 在断点续传场景中，文件不存在是正常情况，静默处理
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def save_results_to_file(results, correct_answers, total_questions, env_name=None, question_type=None, filename_prefix="./experiment_results"):
    """
    保存结果到JSON文件 (新版：适配字典格式的results)
    """

    
    os.makedirs(filename_prefix, exist_ok=True)
    file_path = os.path.join(filename_prefix, f"results.json")
    
    # 准备保存的数据
    save_data = {
        "summary": {
            "env_name": env_name,
            "question_type": question_type,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": calculate_accuracy(correct_answers, total_questions),
        },
        "detailed_results": results  # 直接使用新的results列表
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

def append_results_to_file(*args, **kwargs):
    """
    废弃函数：在新流程中，所有结果一次性保存，不再需要追加。
    """
    print("Warning: append_results_to_file is deprecated and should not be used.")
    # 为了兼容旧的调用，直接调用保存函数
    return save_results_to_file(*args, **kwargs)


def load_or_create_state_file(state_file_path):
    """
    加载或创建状态文件。如果文件不存在或为空，返回空字典。
    """
    state_data = load_json_file(state_file_path)
    return state_data if state_data is not None else {}

def update_state_file(file_path, question_type, question_id, status, **kwargs):
    """
    更新状态文件中的单个问题状态 (新版)
    """
    try:
        # 先加载现有数据
        state_data = load_or_create_state_file(file_path)
        
        # 确保问题类型键存在
        if question_type not in state_data:
            state_data[question_type] = {}
        
        # 确保问题ID键存在
        if question_id not in state_data[question_type]:
            state_data[question_type][question_id] = {}

        # 更新状态和附加信息
        state_data[question_type][question_id]['status'] = status
        state_data[question_type][question_id]['timestamp'] = datetime.now().isoformat()
        for key, value in kwargs.items():
            state_data[question_type][question_id][key] = value

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error updating state file {file_path}: {e}")


def is_question_completed(state_data, question_type, question_id):
    """
    检查单个问题是否已完成 (新版)
    """
    if question_type in state_data and question_id in state_data[question_type]:
        return state_data[question_type][question_id].get("status") == "completed"
    return False


def load_completed_results_from_state(state_data):
    """
    从状态文件中加载已完成问题的统计信息 (新版)
    
    Returns:
        tuple: (total_completed_questions, correct_answers, results_list)
    """
    total_completed = 0
    correct_answers = 0
    results = []
    
    # 遍历问题类型
    for q_type, questions in state_data.items():
        if not isinstance(questions, dict):
            continue
        # 遍历该类型下的所有问题
        for q_id, q_info in questions.items():
            if isinstance(q_info, dict) and q_info.get("status") == "completed":
                total_completed += 1
                is_correct = q_info.get("is_correct", False)
                if is_correct:
                    correct_answers += 1
                
                # 结果格式与 run_baseline.py 保持一致
                results.append({
                    "scenario": "unknown", # 状态文件本身不包含场景名，需要从外部获取
                    "q_type": q_type,
                    "q_id": q_id,
                    "agent_answer": q_info.get("agent_answer"),
                    "ground_truth": q_info.get("ground_truth"),
                    "is_correct": is_correct
                })
    
    return total_completed, correct_answers, results

def get_completed_stats_for_scenario(*args, **kwargs):
    """
    废弃函数：在新流程中，状态文件是每个场景一个，此函数不再需要。
    """
    print("Warning: get_completed_stats_for_scenario is deprecated.")
    return 0, 0, []
