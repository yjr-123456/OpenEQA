import json
import os
from datetime import datetime
from .llm_eval import calculate_accuracy


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def save_results_to_file(results, correct_answers, total_questions, env_name=None, question_type=None, filename_prefix="./experiment_results"):
    """
    保存结果到JSON文件
    """
    # 根据参数生成文件名
    if env_name and question_type:
        filename = f"{env_name}_{question_type}.json"
    else:
        # 回退到时间戳命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.json"
    
    os.makedirs(filename_prefix, exist_ok=True)
    file_path = os.path.join(filename_prefix, filename)
    
    # 准备保存的数据
    save_data = {
        "env_name": env_name,
        "question_type": question_type,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": calculate_accuracy(correct_answers, total_questions),
        "detailed_results": [
            {
                "question_id": i + 1,
                "scenario_name": result[2] if len(result) > 2 else "unknown",  # scenario信息
                "agent_answer": result[0],
                "ground_truth": result[1]
            }
            for i, result in enumerate(results)
        ]
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f" Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f" Error saving results: {e}")
        return None

def append_results_to_file(results, correct_answers, total_questions, filename, env_name=None, question_type=None, filename_prefix="./experiment_results"):
    """
    追加结果到现有文件
    """
    file_path = os.path.join(filename_prefix, filename)

    try:
        # 读取现有数据
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = {
                "env_name": env_name,
                "question_type": question_type,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "detailed_results": []
            }
        
        # 更新数据
        existing_data["total_questions"] = total_questions
        existing_data["correct_answers"] = correct_answers
        existing_data["accuracy"] = calculate_accuracy(correct_answers, total_questions)
        existing_data["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加新的详细结果
        start_idx = len(existing_data["detailed_results"])
        for i, result in enumerate(results[start_idx:], start=start_idx):
            existing_data["detailed_results"].append({
                "question_id": i + 1,
                "scenario_name": result[2] if len(result) > 2 else "unknown",  # scenario信息
                "agent_answer": result[0],
                "ground_truth": result[1]
            })
        
        # 保存更新后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f" Results updated in: {filename}")
        return True
    except Exception as e:
        print(f" Error updating results: {e}")
        return False


def load_or_create_state_file(state_file_path):
    """
    加载或创建状态文件
    """
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                state_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state_data = {}
    else:
        state_data = {}
    
    return state_data

def update_state_file(state_file_path, scenario_name, question_id, status="completed", 
                     agent_answer=None, ground_truth=None, is_correct=None):
    """
    更新状态文件，包含答案和正确性信息
    """
    state_data = load_or_create_state_file(state_file_path)
    
    if scenario_name not in state_data:
        state_data[scenario_name] = {}
    
    state_data[scenario_name][question_id] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "agent_answer": agent_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct
    }
    
    try:
        with open(state_file_path, 'w') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error updating state file: {e}")
        return False

def is_question_completed(state_data, scenario_name, question_id):
    """
    检查问题是否已完成
    """
    if scenario_name not in state_data:
        return False
    
    if question_id not in state_data[scenario_name]:
        return False
    
    return state_data[scenario_name][question_id].get("status") == "completed"

def load_completed_results_from_state(state_data):
    """
    从状态文件中加载已完成问题的统计信息
    
    Returns:
        tuple: (total_completed_questions, correct_answers, results_list)
    """
    total_completed = 0
    correct_answers = 0
    results = []
    
    for scenario_name, scenario_data in state_data.items():
        if isinstance(scenario_data, dict):
            for question_id, question_info in scenario_data.items():
                if isinstance(question_info, dict) and question_info.get("status") == "completed":
                    total_completed += 1
                    
                    agent_answer = question_info.get("agent_answer")
                    ground_truth = question_info.get("ground_truth")
                    is_correct = question_info.get("is_correct", False)
                    
                    if is_correct:
                        correct_answers += 1
                    
                    # 添加scenario信息到结果列表中
                    results.append((agent_answer, ground_truth, scenario_name))
    
    return total_completed, correct_answers, results

def get_completed_stats_for_scenario(state_data, scenario_name, qa_dict):
    """
    获取特定场景的完成统计信息
    """
    if scenario_name not in state_data:
        return 0, 0, []
    
    scenario_data = state_data[scenario_name]
    if not isinstance(scenario_data, dict):
        return 0, 0, []
    
    completed_count = 0
    correct_count = 0
    scenario_results = []
    
    for question_id in qa_dict.keys():
        if question_id in scenario_data:
            question_info = scenario_data[question_id]
            if isinstance(question_info, dict) and question_info.get("status") == "completed":
                completed_count += 1
                
                agent_answer = question_info.get("agent_answer")
                ground_truth = question_info.get("ground_truth")
                is_correct = question_info.get("is_correct", False)
                
                if is_correct:
                    correct_count += 1
                
                scenario_results.append((agent_answer, ground_truth, scenario_name))
    
    return completed_count, correct_count, scenario_results
