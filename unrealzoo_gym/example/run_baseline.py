import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import gymnasium as gym
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, configUE, augmentation
from example.solution.baseline.VLM.agent_predict import agent

from src.load_env_config import process_in_vehicle_players, obs_transform
from src.llm_eval import compare_answers_with_api, calculate_accuracy
from src.process_question import *
os.environ["UnrealEnv"] = "/Volumes/KINGSTON/UnrealEnv"

# def send_pid_to_watchdog(pid, host='127.0.0.1', port=50007):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((host, port))
#         s.sendall(str(pid).encode())


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--envs", nargs='+', default=["track_train"])
        parser.add_argument("-u", "--task_type", default="UnrealCvEQA_general")
        parser.add_argument("-o", "--action_obs", default="DiscreteRgbd")
        parser.add_argument("-v", "--v_type", default="v0")
        parser.add_argument("-s", "--seed", type=int, default=0)
        parser.add_argument("-t", "--time-dilation", type=int, default=-1)
        parser.add_argument("-d", "--early-done", type=int, default=-1)
        parser.add_argument("-q", "--QA_path", default=os.path.join(current_dir, 'QA_Data'))
        parser.add_argument("-p", "--pid_port", type=int, default=50007, help="UnrealCV watchdog pid")
        parser.add_argument("--use_pid", type=bool,default=False, help="Whether to use pid watchdog to monitor the UE process")
        parser.add_argument("--question_types", nargs='+', default=["counting"], help="List of question types to evaluate")
        parser.add_argument("--resume", action='store_true', help="Resume from previous progress")
        parser.add_argument("--model", default="doubao", help="choose evaluation models")
        parser.add_argument("--config_path", default=os.path.join(current_dir, "solution"), help="configuration file path")
        parser.add_argument("--ue_log_dir", default=os.path.join(current_dir, "unreal_log_path"), help="unreal engine logging directory")
        args = parser.parse_args()
        for env_name in args.envs:
            # init agent
            AG = agent(model = args.model, config_path=args.config_path)
            obs_name = "BP_Character_C_1"
            print("Initializing UnrealCV Gym environment...")
            env_id = f'{args.task_type}-{env_name}-{args.action_obs}-{args.v_type}'
            question_type = None
            save_dir = None
            total_questions = 0
            correct_answers = 0
            results = []
            results_filename = None
            base_save_dir = os.path.join("experiment_results", "baseline", args.model)
            for q_type_folder_name in args.question_types:
                question_type = q_type_folder_name
                type_specific_folder_dir = os.path.join(args.QA_path, env_name, q_type_folder_name)
                if not os.path.isdir(type_specific_folder_dir):
                    print(f"Warning: Type folder not found {type_specific_folder_dir}, skipping.")
                    continue
                
                scenario_folder_names = [d for d in os.listdir(type_specific_folder_dir) 
                                        if os.path.isdir(os.path.join(type_specific_folder_dir, d))]
                
                # 状态文件
                state_file_path = os.path.join(type_specific_folder_dir, f"status_recorder_{args.model}.json")
                state_data = load_or_create_state_file(state_file_path)
                
                # 初始化统计变量
                total_questions = 0
                correct_answers = 0
                results = []
                results_filename = None
                save_dir = os.path.join(base_save_dir, env_name, q_type_folder_name)
                os.makedirs(save_dir, exist_ok=True)
                
                if args.resume:
                    completed_questions, completed_correct, completed_results = load_completed_results_from_state(state_data)
                    total_questions = completed_questions
                    correct_answers = completed_correct
                    results = completed_results
                    print(f"Resume mode: Found {completed_questions} completed questions")
                    print(f"Resume mode: Previous correct answers: {completed_correct}")
                    print(f"Resume mode: Previous accuracy: {calculate_accuracy(completed_correct, completed_questions):.2f}%")
                # 初始化环境
                env = gym.make(env_id)
                if args.time_dilation > 0:
                    env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
                if args.early_done > 0:
                    env = early_done.EarlyDoneWrapper(env, args.early_done)

                for scenario_folder_name in scenario_folder_names:
                    id_folder_path = os.path.join(type_specific_folder_dir, scenario_folder_name)
                    file_path = os.path.join(id_folder_path, f"{q_type_folder_name}.json")
                    
                    if not os.path.isfile(file_path):
                        print(f"Warning: JSON file not found in {id_folder_path}, skipping.")
                        continue
                    
                    # 加载QA数据
                    QA_data_loaded = load_json_file(file_path)
                    if QA_data_loaded is None:
                        continue
                    
                    # 处理in_vehicle状态的player位置
                    temp_config_data = {"target_configs": QA_data_loaded.get("target_configs", {})}
                    processed_config_data = process_in_vehicle_players(temp_config_data)
                    QA_data_loaded["target_configs"] = processed_config_data["target_configs"]
                    
                    QA_dict = QA_data_loaded.get("Questions", {})
                    if not QA_dict:
                        print(f"Warning: No questions found in {file_path}, skipping.")
                        continue
                    
                    # 检查该场景是否已完全完成
                    if args.resume and scenario_folder_name in state_data:
                        scenario_data = state_data[scenario_folder_name]
                        if isinstance(scenario_data, dict):
                            all_completed = all(is_question_completed(state_data, scenario_folder_name, qid) 
                                                for qid in QA_dict.keys())
                            if all_completed:
                                print(f"Scenario {scenario_folder_name} already completed, skipping.")
                                continue
                    
                    print(f"\n--- Processing File for Interaction: {file_path} ---")
                    
                    # 环境设置代码
                    target_configs = QA_data_loaded.get("target_configs", {})
                    safe_start_config = QA_data_loaded.get("safe_start")
                    if len(safe_start_config) > 1:
                        safe_start_config = [safe_start_config]
                    refer_agents_category_config = QA_data_loaded.get("refer_agents_category")
                    agent_num = sum(len(target_configs.get(t, {}).get("name", [])) for t in target_configs)
                    start_pose = QA_data_loaded.get("safe_start", [])
                    if len(start_pose) == 1:
                        start_pose = start_pose[0]
                    assert len(start_pose) == 6

                    unwrapped_env = env.unwrapped
                    unwrapped_env.safe_start = safe_start_config
                    unwrapped_env.refer_agents_category = refer_agents_category_config
                    unwrapped_env.target_configs = target_configs
                    unwrapped_env.is_eval = True
                    # set log dir
                    if args.ue_log_dir:
                        os.makedirs(args.ue_log_dir, exist_ok=True)
                        unwrapped_env.ue_log_path = args.ue_log_dir
                    # pid config
                    if args.use_pid:
                        env.unwrapped.send_pid = True
                        env.unwrapped.watchdog_port = args.pid_port

                    env = augmentation.RandomPopulationWrapper(env, num_min=agent_num + 1, num_max=agent_num + 1, height_bias=100)
                    env = configUE.ConfigUEWrapper(env, resolution=(512,512), offscreen=False)

                    print(f"Resetting environment for file: {os.path.basename(file_path)}")
                    states, info = env.reset(seed=args.seed)
                    
                    obs_rgb, obs_depth = obs_transform(states)
                    for question_id, question_data in QA_dict.items():
                        # 检查单个问题是否已完成
                        if args.resume and is_question_completed(state_data, scenario_folder_name, question_id):
                            print(f"Question {question_id} in {scenario_folder_name} already completed, skipping.")
                            continue
                        
                        total_questions += 1
                        # 设置玩家位置
                        loca = start_pose[0:3]
                        rota = start_pose[3:]
                        env.unwrapped.unrealcv.set_obj_location(obs_name, loca)
                        env.unwrapped.unrealcv.set_obj_rotation(obs_name, rota)

                        question_stem = question_data.get("question", "")
                        question_options = question_data.get("options", [])
                        question_answer = question_data.get("answer", None)
                        
                        if not question_stem or not question_answer:
                            print(f"Warning: Question stem is not complete in {file_path}, skipping.")
                            continue
                        
                        AG.reset(question=question_stem, obs_rgb=obs_rgb, obs_depth=obs_depth, target_type=refer_agents_category_config,
                                question_type=question_type, answer_list=question_options, batch_id = scenario_folder_name,
                                question_answer=question_answer, env_name=env_name,logger_base_dir=current_dir)

                        max_step = AG.max_step
                        answer = None
                        cur_step = 0
                        
                        for cur_step in range(0, max_step+1):
                            time_1 = time.time()
                            action = AG.predict(obs_rgb, obs_depth,info)
                            time_2 = time.time()
                            print(f"======================Step {cur_step} Action: {action} Time: {time_2 - time_1:.2f}s======================")
                            actions = action + [-1]*agent_num
                            print(actions)
                            obs, reward, termination, truncation, info = env.step(actions)
                            obs_rgb, obs_depth = obs_transform(obs)

                            if AG.termination:
                                answer = AG.final_answer
                                break
                            if AG.truncation:
                                answer = AG.final_answer
                                print(f"Episode truncated after {cur_step} steps.")
                                break
                        
                        # 判断答案正确性
                        is_correct = compare_answers_with_api(
                            agent_answer=answer, 
                            ground_truth=question_answer,
                            question_stem=question_stem,
                            question_type=question_type
                        )
                        
                        # 更新统计 - 添加scenario信息到results中
                        results.append((answer, question_answer, scenario_folder_name))  # 添加scenario_folder_name
                        if is_correct:
                            correct_answers += 1
                            print(f" Correct answer for question: {question_stem}")
                            print(f"  Expected: {question_answer}, Got: {answer}")
                        else:
                            print(f" Incorrect answer for question: {question_stem}")
                            print(f"  Expected: {question_answer}, Got: {answer}")
                        
                        # 更新状态文件，包含答案和正确性信息
                        update_state_file(
                            state_file_path, 
                            scenario_folder_name, 
                            question_id, 
                            status="completed",
                            agent_answer=answer,
                            ground_truth=question_answer,
                            is_correct=is_correct
                        )
                        
                        # 定期保存结果
                        if total_questions % 10 == 0:
                            current_accuracy = calculate_accuracy(correct_answers, total_questions)
                            print(f"\n === Progress Update ===")
                            print(f"Processed {total_questions} questions so far.")
                            print(f"Correct answers: {correct_answers}")
                            print(f"Current accuracy: {current_accuracy:.2f}%")
                            print("=" * 30)
                        
                        # 保存结果到文件
                        if results_filename is None:
                            results_filename = save_results_to_file(
                                results, correct_answers, total_questions,
                                env_name=env_name, question_type=question_type,
                                filename_prefix=save_dir
                            )
                        else:
                            append_results_to_file(
                                results, correct_answers, total_questions, results_filename,
                                env_name=env_name, question_type=question_type,
                                filename_prefix=save_dir
                            )
                        print(f"Question {total_questions} | Accuracy: {calculate_accuracy(correct_answers, total_questions):.1f}%")
                        print(f"Completed scenario: {scenario_folder_name} in environment: {env_name}")
                env.close()
                time.sleep(10)
    except KeyboardInterrupt:        
        print("\n=== 程序被中断 ===")
        print(f"已处理 {total_questions} 个问题")
        print(f"正确答案: {correct_answers}")
        if total_questions > 0:
            print(f"当前准确率: {calculate_accuracy(correct_answers, total_questions):.2f}%")
        print("进度已保存，可以使用 --resume 参数继续")
        if 'env' in locals():
            try:
                env.close()
                print("Environment closed.")
            except Exception as close_e:
                print(f"Error closing environment after interrupt: {close_e}")

    except Exception as e:
        print(f"An error occurred: {e}")    
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            try:
                env.close()
            except Exception as close_e:
                print(f"Error closing environment after exception: {close_e}")

    finally:
        if total_questions > 0 and question_type and save_dir:
            final_accuracy = calculate_accuracy(correct_answers, total_questions)
            print(f"\n === FINAL RESULTS ===")
            print(f"Total Questions: {total_questions}")
            print(f"Correct Answers: {correct_answers}")
            print(f"Final Accuracy: {final_accuracy:.2f}%")
            print("=" * 30)
                
            if results_filename is None:
                results_filename = save_results_to_file(
                        results, correct_answers, total_questions,
                        env_name=env_name, question_type=question_type,
                        filename_prefix=save_dir
                    )
            else:
                append_results_to_file(
                        results, correct_answers, total_questions, results_filename,
                        env_name=env_name, question_type=question_type,
                        filename_prefix=save_dir
                    )
                
            print(f" All results saved to: {results_filename}")
        
        if 'env' in locals():
            try:
                env.close()
                print("Environment closed.")
            except Exception as close_e:
                print(f"Error closing environment: {close_e}")



