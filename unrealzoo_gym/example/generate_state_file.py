#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
断点续传状态文件生成器
用于为run_baseline.py生成初始的状态文件
"""

import os
import json
import argparse
from datetime import datetime

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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

def generate_state_file(qa_path, env_list, question_type_folders, model,mode="empty"):
    """
    生成状态文件
    
    Args:
        qa_path: QA数据路径
        env_list: 环境列表
        question_type_folders: 问题类型文件夹列表
        mode: 生成模式
            - "empty": 生成空状态文件（所有问题标记为未完成）
            - "template": 生成模板文件（包含结构但无状态）
            - "reset": 重置现有状态文件（将所有问题标记为未完成）
    """
    
    total_environments = 0
    total_scenarios = 0
    total_questions = 0
    
    for env_name in env_list:
        print(f"\n=== 处理环境: {env_name} ===")
        total_environments += 1
        
        for q_type_folder_name in question_type_folders:
            print(f"  处理问题类型: {q_type_folder_name}")
            
            type_specific_folder_dir = os.path.join(qa_path, env_name, q_type_folder_name)
            if not os.path.isdir(type_specific_folder_dir):
                print(f"    警告: 文件夹不存在 {type_specific_folder_dir}, 跳过")
                continue
            
            # 状态文件路径
            state_file_path = os.path.join(type_specific_folder_dir, f"status_recorder_{model}.json")

            # 获取所有场景文件夹
            scenario_folder_names = [d for d in os.listdir(type_specific_folder_dir) 
                                   if os.path.isdir(os.path.join(type_specific_folder_dir, d))]
            
            if not scenario_folder_names:
                print(f"    警告: 没有找到场景文件夹在 {type_specific_folder_dir}")
                continue
            
            # 初始化状态数据
            state_data = {}
            current_scenarios = 0
            current_questions = 0
            
            for scenario_folder_name in scenario_folder_names:
                current_scenarios += 1
                id_folder_path = os.path.join(type_specific_folder_dir, scenario_folder_name)
                file_path = os.path.join(id_folder_path, f"{q_type_folder_name}.json")
                
                if not os.path.isfile(file_path):
                    print(f"    警告: JSON文件不存在 {file_path}, 跳过")
                    continue
                
                # 加载QA数据
                qa_data = load_json_file(file_path)
                if qa_data is None:
                    continue
                
                qa_dict = qa_data.get("Questions", {})
                if not qa_dict:
                    print(f"    警告: 没有找到问题在 {file_path}")
                    continue
                
                # 为每个问题创建状态记录
                scenario_state = {}
                for question_id in qa_dict.keys():
                    if mode == "empty" or mode == "reset":
                        # 标记为未完成，包含完整的字段结构
                        scenario_state[question_id] = {
                            "status": "pending",
                            "timestamp": datetime.now().isoformat(),
                            "created_by": "state_generator",
                            "agent_answer": None,
                            "ground_truth": None,
                            "is_correct": None
                        }
                    elif mode == "template":
                        # 只创建结构，不设置状态
                        scenario_state[question_id] = {
                            "status": None,
                            "timestamp": None,
                            "created_by": "state_generator",
                            "agent_answer": None,
                            "ground_truth": None,
                            "is_correct": None
                        }
                    
                    current_questions += 1
                
                state_data[scenario_folder_name] = scenario_state
                print(f"    场景 {scenario_folder_name}: {len(qa_dict)} 个问题")
            
            # 保存状态文件
            try:
                # 如果是重置模式，先备份现有文件
                if mode == "reset" and os.path.exists(state_file_path):
                    backup_path = state_file_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(state_file_path, backup_path)
                    print(f"    现有状态文件已备份到: {backup_path}")
                
                with open(state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                
                print(f"    ✓ 状态文件已生成: {state_file_path}")
                print(f"    包含 {current_scenarios} 个场景, {current_questions} 个问题")
                
                total_scenarios += current_scenarios
                total_questions += current_questions
                
            except Exception as e:
                print(f"    ✗ 保存状态文件失败: {e}")
    
    print(f"\n=== 生成完成 ===")
    print(f"总计处理: {total_environments} 个环境, {total_scenarios} 个场景, {total_questions} 个问题")
    return True

def scan_qa_structure(qa_path):
    """扫描QA数据结构"""
    print("=== 扫描QA数据结构 ===")
    
    if not os.path.exists(qa_path):
        print(f"错误: QA路径不存在: {qa_path}")
        return None, None
    
    env_list = []
    question_types = set()
    
    for item in os.listdir(qa_path):
        env_path = os.path.join(qa_path, item)
        if os.path.isdir(env_path):
            env_list.append(item)
            print(f"发现环境: {item}")
            
            # 扫描问题类型
            for sub_item in os.listdir(env_path):
                sub_path = os.path.join(env_path, sub_item)
                if os.path.isdir(sub_path):
                    question_types.add(sub_item)
    
    question_type_list = list(question_types)
    print(f"发现 {len(env_list)} 个环境: {env_list}")
    print(f"发现 {len(question_type_list)} 种问题类型: {question_type_list}")
    
    return env_list, question_type_list

def check_existing_states(qa_path, env_list, model, question_type_folders):
    """检查现有状态文件"""
    print("\n=== 检查现有状态文件 ===")
    
    existing_files = []
    for env_name in env_list:
        for q_type in question_type_folders:
            type_specific_folder_dir = os.path.join(qa_path, env_name, q_type)
            state_file_path = os.path.join(type_specific_folder_dir, f"status_recorder_{model}.json")

            if os.path.exists(state_file_path):
                existing_files.append(state_file_path)
                
                # 分析现有文件
                try:
                    with open(state_file_path, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)
                    
                    total_scenarios = len(state_data)
                    total_questions = sum(len(scenario_data) for scenario_data in state_data.values())
                    completed_questions = 0
                    correct_answers = 0
                    
                    for scenario_data in state_data.values():
                        if isinstance(scenario_data, dict):
                            for question_id, question_info in scenario_data.items():
                                if isinstance(question_info, dict) and question_info.get("status") == "completed":
                                    completed_questions += 1
                                    if question_info.get("is_correct", False):
                                        correct_answers += 1
                    
                    accuracy = (correct_answers / completed_questions * 100) if completed_questions > 0 else 0
                    
                    print(f"  {state_file_path}")
                    print(f"    场景: {total_scenarios}, 问题: {total_questions}")
                    print(f"    已完成: {completed_questions}, 正确: {correct_answers}, 准确率: {accuracy:.1f}%")
                    
                except Exception as e:
                    print(f"  {state_file_path} (读取失败: {e})")
    
    if not existing_files:
        print("  没有发现现有状态文件")
    
    return existing_files

def analyze_state_file(state_file_path):
    """分析单个状态文件的详细信息"""
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        print(f"\n=== 分析状态文件: {state_file_path} ===")
        
        total_scenarios = len(state_data)
        total_questions = 0
        completed_questions = 0
        correct_answers = 0
        pending_questions = 0
        
        for scenario_name, scenario_data in state_data.items():
            if isinstance(scenario_data, dict):
                scenario_total = len(scenario_data)
                scenario_completed = 0
                scenario_correct = 0
                scenario_pending = 0
                
                for question_id, question_info in scenario_data.items():
                    total_questions += 1
                    if isinstance(question_info, dict):
                        status = question_info.get("status")
                        if status == "completed":
                            completed_questions += 1
                            scenario_completed += 1
                            if question_info.get("is_correct", False):
                                correct_answers += 1
                                scenario_correct += 1
                        elif status == "pending":
                            pending_questions += 1
                            scenario_pending += 1
                
                accuracy = (scenario_correct / scenario_completed * 100) if scenario_completed > 0 else 0
                print(f"  场景 {scenario_name}: {scenario_total}题, 完成{scenario_completed}, 正确{scenario_correct}, 待处理{scenario_pending}, 准确率{accuracy:.1f}%")
        
        overall_accuracy = (correct_answers / completed_questions * 100) if completed_questions > 0 else 0
        
        print(f"\n总计: {total_scenarios}场景, {total_questions}问题")
        print(f"已完成: {completed_questions}, 正确: {correct_answers}, 待处理: {pending_questions}")
        print(f"整体准确率: {overall_accuracy:.2f}%")
        
        return {
            "total_scenarios": total_scenarios,
            "total_questions": total_questions,
            "completed_questions": completed_questions,
            "correct_answers": correct_answers,
            "pending_questions": pending_questions,
            "accuracy": overall_accuracy
        }
        
    except Exception as e:
        print(f"分析失败: {e}")
        return None


def reset_by_question_type(qa_path, env_list, target_question_types, model, reset_mode="pending"):
    """
    按问题类别重置状态文件
    
    Args:
        qa_path: QA数据路径
        env_list: 环境列表
        target_question_types: 要重置的问题类型列表
        reset_mode: 重置模式
            - "pending": 重置为待处理状态
            - "remove": 从状态文件中移除相关记录
            - "backup": 重置前备份
    """
    
    total_reset_questions = 0
    total_reset_scenarios = 0
    
    for env_name in env_list:
        print(f"\n=== 处理环境: {env_name} ===")
        
        for q_type_folder_name in target_question_types:
            print(f"  重置问题类型: {q_type_folder_name}")
            
            type_specific_folder_dir = os.path.join(qa_path, env_name, q_type_folder_name)
            if not os.path.isdir(type_specific_folder_dir):
                print(f"    警告: 文件夹不存在 {type_specific_folder_dir}, 跳过")
                continue
            
            # 状态文件路径
            state_file_path = os.path.join(type_specific_folder_dir, f"status_recorder_{model}.json")

            if not os.path.exists(state_file_path):
                print(f"    状态文件不存在: {state_file_path}, 跳过")
                continue
            
            try:
                # 读取现有状态文件
                with open(state_file_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # 备份原文件
                if reset_mode == "backup":
                    backup_path = state_file_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, indent=2, ensure_ascii=False)
                    print(f"    原状态文件已备份到: {backup_path}")
                
                # 统计重置前的状态
                original_completed = 0
                original_correct = 0
                reset_scenarios = 0
                reset_questions = 0
                
                for scenario_name, scenario_data in state_data.items():
                    if isinstance(scenario_data, dict):
                        scenario_has_completed = False
                        scenario_reset_count = 0
                        
                        for question_id, question_info in scenario_data.items():
                            if isinstance(question_info, dict):
                                if question_info.get("status") == "completed":
                                    original_completed += 1
                                    scenario_has_completed = True
                                    if question_info.get("is_correct", False):
                                        original_correct += 1
                                
                                # 执行重置
                                if reset_mode == "remove":
                                    # 移除模式：删除completed状态的记录
                                    if question_info.get("status") == "completed":
                                        del scenario_data[question_id]
                                        reset_questions += 1
                                        scenario_reset_count += 1
                                else:
                                    # pending模式：重置为待处理状态
                                    if question_info.get("status") == "completed":
                                        question_info["status"] = "pending"
                                        question_info["timestamp"] = datetime.now().isoformat()
                                        question_info["reset_by"] = "type_reset"
                                        question_info["agent_answer"] = None
                                        question_info["ground_truth"] = None
                                        question_info["is_correct"] = None
                                        reset_questions += 1
                                        scenario_reset_count += 1
                        
                        if scenario_has_completed and scenario_reset_count > 0:
                            reset_scenarios += 1
                
                # 保存修改后的状态文件
                with open(state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                
                print(f"    ✓ 重置完成: {state_file_path}")
                print(f"    重置问题: {reset_questions}, 涉及场景: {reset_scenarios}")
                print(f"    原状态: 完成{original_completed}题, 正确{original_correct}题")
                
                total_reset_questions += reset_questions
                total_reset_scenarios += reset_scenarios
                
            except Exception as e:
                print(f"    ✗ 重置失败: {e}")
    
    print(f"\n=== 重置完成 ===")
    print(f"总计重置: {total_reset_questions} 个问题, 涉及 {total_reset_scenarios} 个场景")
    return True

def reset_specific_scenarios(qa_path, env_name, question_type, scenario_names, model, reset_mode="pending"):
    """
    重置特定场景的状态
    
    Args:
        qa_path: QA数据路径
        env_name: 环境名称
        question_type: 问题类型
        scenario_names: 要重置的场景名称列表
        reset_mode: 重置模式
    """
    
    type_specific_folder_dir = os.path.join(qa_path, env_name, question_type)
    state_file_path = os.path.join(type_specific_folder_dir, f"status_recorder_{model}.json")

    if not os.path.exists(state_file_path):
        print(f"状态文件不存在: {state_file_path}")
        return False
    
    try:
        # 读取状态文件
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # 备份
        backup_path = state_file_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        print(f"原状态文件已备份到: {backup_path}")
        
        reset_count = 0
        
        for scenario_name in scenario_names:
            if scenario_name in state_data:
                scenario_data = state_data[scenario_name]
                if isinstance(scenario_data, dict):
                    for question_id, question_info in scenario_data.items():
                        if isinstance(question_info, dict) and question_info.get("status") == "completed":
                            if reset_mode == "remove":
                                del scenario_data[question_id]
                            else:
                                question_info["status"] = "pending"
                                question_info["timestamp"] = datetime.now().isoformat()
                                question_info["reset_by"] = "scenario_reset"
                                question_info["agent_answer"] = None
                                question_info["ground_truth"] = None
                                question_info["is_correct"] = None
                            reset_count += 1
                    print(f"场景 {scenario_name}: 重置 {reset_count} 个问题")
            else:
                print(f"场景 {scenario_name} 不存在于状态文件中")
        
        # 保存修改
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        print(f"总计重置 {reset_count} 个问题")
        return True
        
    except Exception as e:
        print(f"重置失败: {e}")
        return False

def list_question_types_with_status(qa_path, env_list):
    """
    列出所有问题类型及其完成状态
    """
    print("\n=== 问题类型完成状态统计 ===")
    
    all_types = {}
    
    for env_name in env_list:
        env_path = os.path.join(qa_path, env_name)
        if not os.path.isdir(env_path):
            continue
        
        for q_type in os.listdir(env_path):
            q_type_path = os.path.join(env_path, q_type)
            if not os.path.isdir(q_type_path):
                continue
            
            if q_type not in all_types:
                all_types[q_type] = {
                    "environments": 0,
                    "total_questions": 0,
                    "completed_questions": 0,
                    "correct_answers": 0,
                    "envs_detail": {}
                }
            
            state_file_path = os.path.join(q_type_path, "status_recorder.json")
            
            if os.path.exists(state_file_path):
                try:
                    with open(state_file_path, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)
                    
                    env_total = 0
                    env_completed = 0
                    env_correct = 0
                    
                    for scenario_data in state_data.values():
                        if isinstance(scenario_data, dict):
                            for question_info in scenario_data.values():
                                if isinstance(question_info, dict):
                                    env_total += 1
                                    if question_info.get("status") == "completed":
                                        env_completed += 1
                                        if question_info.get("is_correct", False):
                                            env_correct += 1
                    
                    all_types[q_type]["environments"] += 1
                    all_types[q_type]["total_questions"] += env_total
                    all_types[q_type]["completed_questions"] += env_completed
                    all_types[q_type]["correct_answers"] += env_correct
                    all_types[q_type]["envs_detail"][env_name] = {
                        "total": env_total,
                        "completed": env_completed,
                        "correct": env_correct,
                        "accuracy": (env_correct / env_completed * 100) if env_completed > 0 else 0
                    }
                    
                except Exception as e:
                    print(f"读取状态文件失败 {state_file_path}: {e}")
    
    # 打印统计结果
    for q_type, stats in all_types.items():
        total = stats["total_questions"]
        completed = stats["completed_questions"]
        correct = stats["correct_answers"]
        accuracy = (correct / completed * 100) if completed > 0 else 0
        progress = (completed / total * 100) if total > 0 else 0
        
        print(f"\n问题类型: {q_type}")
        print(f"  涉及环境: {stats['environments']}")
        print(f"  总问题数: {total}")
        print(f"  已完成: {completed} ({progress:.1f}%)")
        print(f"  正确答案: {correct}")
        print(f"  准确率: {accuracy:.1f}%")
        
        # 显示每个环境的详细情况
        for env_name, env_stats in stats["envs_detail"].items():
            print(f"    {env_name}: {env_stats['completed']}/{env_stats['total']} "
                  f"({env_stats['accuracy']:.1f}%)")
    
    return all_types


def main():
    parser = argparse.ArgumentParser(description="生成断点续传状态文件")
    parser.add_argument("-p", "--qa_path", 
                       default=os.path.join(os.path.dirname(__file__), 'QA_Data'),
                       help="QA数据路径")
    parser.add_argument("-m", "--mode", 
                       choices=["empty", "template", "reset", "scan", "analyze", "reset_type", "reset_scenario", "list_types"],
                       default="scan",
                       help="运行模式: empty(生成空状态), template(生成模板), reset(重置现有), "
                            "scan(仅扫描), analyze(详细分析), reset_type(按问题类型重置), "
                            "reset_scenario(重置特定场景), list_types(列出问题类型状态)")
    parser.add_argument("-e", "--envs", 
                       nargs="+",
                       help="指定环境列表(默认使用所有发现的环境)")
    parser.add_argument("-t", "--types", 
                       nargs="+", 
                       help="指定问题类型列表(默认使用所有发现的类型)")
    parser.add_argument("-f", "--force", 
                       action="store_true",
                       help="强制覆盖现有文件")
    parser.add_argument("--state-file", 
                       help="指定要分析的状态文件路径(用于analyze模式)")
    parser.add_argument("--reset-mode", 
                       choices=["pending", "remove", "backup"],
                       default="pending",
                       help="重置模式: pending(重置为待处理), remove(移除记录), backup(备份后重置)")
    parser.add_argument("--scenarios", 
                       nargs="+",
                       help="指定要重置的场景名称列表(用于reset_scenario模式)")
    parser.add_argument("--env", 
                       help="指定单个环境名称(用于reset_scenario模式)")
    parser.add_argument("--type", 
                       help="指定单个问题类型(用于reset_scenario模式)")
    parser.add_argument("--model", type=str, default="doubao", required=True, help="模型名称")
    args = parser.parse_args()
    
    print("=== 断点续传状态文件生成器 ===")
    print(f"QA数据路径: {args.qa_path}")
    print(f"运行模式: {args.mode}")
    
    # 如果是分析模式且指定了文件
    if args.mode == "analyze" and args.state_file:
        if os.path.exists(args.state_file):
            analyze_state_file(args.state_file)
        else:
            print(f"错误: 状态文件不存在: {args.state_file}")
        return 0
    
    # 扫描QA数据结构
    discovered_envs, discovered_types = scan_qa_structure(args.qa_path)
    if discovered_envs is None:
        return 1
    
    # 确定要使用的环境和类型
    env_list = args.envs if args.envs else discovered_envs
    question_type_folders = args.types if args.types else discovered_types
    
    print(f"\n将处理的环境: {env_list}")
    print(f"将处理的问题类型: {question_type_folders}")
    
    # 新增模式处理
    if args.mode == "reset_type":
        if not args.types:
            print("错误: reset_type模式需要指定 --types 参数")
            return 1
        
        print(f"\n将重置问题类型: {args.types}")
        print(f"重置模式: {args.reset_mode}")
        
        response = input("确认要重置这些问题类型的状态吗? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return 0

        success = reset_by_question_type(args.qa_path, env_list, args.types, args.model, args.reset_mode)
        if success:
            print("\n✓ 按问题类型重置成功!")
        return 0
    
    elif args.mode == "reset_scenario":
        if not args.env or not args.type or not args.scenarios:
            print("错误: reset_scenario模式需要指定 --env, --type 和 --scenarios 参数")
            return 1
        
        print(f"\n将重置环境 {args.env} 中问题类型 {args.type} 的场景: {args.scenarios}")
        print(f"重置模式: {args.reset_mode}")
        
        response = input("确认要重置这些场景的状态吗? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return 0

        success = reset_specific_scenarios(args.qa_path, args.env, args.type, args.scenarios, args.model, args.reset_mode)
        if success:
            print("\n✓ 场景重置成功!")
        return 0
    
    elif args.mode == "list_types":
        list_question_types_with_status(args.qa_path, env_list)
        return 0
    
    # 检查现有状态文件
    existing_files = check_existing_states(args.qa_path, env_list, args.model, question_type_folders)

    if args.mode == "scan":
        print("\n=== 扫描完成 ===")
        return 0
    
    if args.mode == "analyze":
        print("\n=== 详细分析所有状态文件 ===")
        for state_file in existing_files:
            analyze_state_file(state_file)
        return 0
    
    # 如果存在文件且未强制覆盖，询问用户
    if existing_files and not args.force and args.mode != "reset":
        print(f"\n发现 {len(existing_files)} 个现有状态文件")
        response = input("是否继续? 这将覆盖现有文件 (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return 0
    
    # 生成状态文件
    success = generate_state_file(args.qa_path, env_list, question_type_folders, args.model, args.mode)

    if success:
        print("\n✓ 状态文件生成成功!")
        print("现在可以使用 'python run_baseline.py --resume' 进行断点续传")
        return 0
    else:
        print("\n✗ 状态文件生成失败!")
        return 1

if __name__ == "__main__":
    main()