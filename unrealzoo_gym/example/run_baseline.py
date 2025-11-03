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
# os.environ["UnrealEnv"] = "/Volumes/KINGSTON/UnrealEnv"

# def send_pid_to_watchdog(pid, host='127.0.0.1', port=50007):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((host, port))
#         s.sendall(str(pid).encode())


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--envs", nargs='+', default=["track_train"])
    parser.add_argument("-u", "--task_type", default="UnrealCvEQA_general")
    parser.add_argument("-o", "--action_obs", default="DiscreteRgbd")
    parser.add_argument("-v", "--v_type", default="v0")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--time-dilation", type=int, default=-1)
    parser.add_argument("-d", "--early-done", type=int, default=-1)
    parser.add_argument("-q", "--QA_path", default=os.path.join(current_dir, 'QA_data_sub'))
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
        base_save_dir = os.path.join("experiment_results", "baseline", args.model)
        log_base_dir = f"{base_save_dir}/logs/{env_name}"
        env_dir = os.path.join(args.QA_path, env_name)
        if not os.path.isdir(env_dir):
            print(f"Warning: Environment folder not found {env_dir}, skipping.")
            continue

        scenario_folder_names = [d for d in os.listdir(env_dir) 
                                    if os.path.isdir(os.path.join(env_dir, d))]

        for scenario_folder_name in scenario_folder_names:
            # init env
            env = gym.make(env_id)
            if args.time_dilation > 0:
                env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
            if args.early_done > 0:
                env = early_done.EarlyDoneWrapper(env, args.early_done)
            scenario_dir = os.path.join(env_dir, scenario_folder_name)
            qa_file_path = os.path.join(scenario_dir, "qa_data.json")
            
            if not os.path.isfile(qa_file_path):
                print(f"Warning: qa_data.json not found in {scenario_dir}, skipping.")
                continue

            state_file_path = os.path.join(scenario_dir, f"status_recorder_{args.model}.json")
            state_data = load_or_create_state_file(state_file_path) # 假设这个函数能处理文件不存在的情况

            # 加载QA数据
            QA_data_loaded = load_json_file(qa_file_path)
            if QA_data_loaded is None or "Questions" not in QA_data_loaded:
                continue

            # 处理in_vehicle状态的player位置
            temp_config_data = {"target_configs": QA_data_loaded.get("target_configs", {})}
            processed_config_data = process_in_vehicle_players(temp_config_data)
            QA_data_loaded["target_configs"] = processed_config_data["target_configs"]
            
            all_questions_in_scene = QA_data_loaded.get("Questions", {})
            if not all_questions_in_scene:
                print(f"Warning: No questions found in {qa_file_path}, skipping.")
                continue

            # load configs
            target_configs = QA_data_loaded.get("target_configs", {})
            refer_agents_category_config = QA_data_loaded.get("refer_agents_category")
            agent_num = sum(len(target_configs.get(t, {}).get("name", [])) for t in target_configs)
            start_pose = QA_data_loaded.get("safe_start", [])
            if len(start_pose) != 6:
                start_pose = start_pose[0]
            assert len(start_pose) == 6
            # set configs to env
            unwrapped_env = env.unwrapped
            unwrapped_env.safe_start = [start_pose]
            unwrapped_env.refer_agents_category = refer_agents_category_config
            unwrapped_env.target_configs = target_configs
            unwrapped_env.is_eval = True
            if args.ue_log_dir:
                os.makedirs(args.ue_log_dir, exist_ok=True)
                unwrapped_env.ue_log_path = args.ue_log_dir
            if args.use_pid:
                unwrapped_env.send_pid = True
                unwrapped_env.watchdog_port = args.pid_port

            env = augmentation.RandomPopulationWrapper(env, num_min=agent_num + 1, num_max=agent_num + 1, height_bias=100)
            env = configUE.ConfigUEWrapper(env, resolution=(512,512), offscreen=False)

            print(f"\n--- Processing Scenario: {scenario_folder_name} in Env: {env_name} ---")
            states, info = env.reset(seed=args.seed)
            obs_rgb, obs_depth = obs_transform(states)
            try:
                for q_type in args.question_types:
                    if q_type not in all_questions_in_scene:
                        continue 
                    QA_dict = all_questions_in_scene[q_type]
                    # initialize metrics
                    total_questions = 0
                    correct_answers = 0
                    results = []
                    results_filename = os.path.join(base_save_dir, env_name, scenario_folder_name, q_type)
                    if os.path.exists(os.path.join(results_filename, "results.json")):
                        loaded_file = load_json_file(os.path.join(results_filename, "results.json"))
                        results = loaded_file.get("detailed_results", [])
                        total_questions = loaded_file["summary"]["total_questions"]
                        correct_answers = loaded_file["summary"]["correct_answers"]
                        print(f"Resuming from existing results: {len(results)} entries loaded.")
                    for question_id, question_data in QA_dict.items():
                        if args.resume and is_question_completed(state_data, q_type, question_id):
                            print(f"Question {question_id} ({q_type}) in {scenario_folder_name} already completed, skipping.")
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
                            print(f"Warning: Question {question_id} is not complete, skipping.")
                            continue
                        question_log_dir = os.path.join(log_base_dir, scenario_folder_name, f"{q_type}_{question_id}")
                        os.makedirs(question_log_dir, exist_ok=True)
                        AG.reset(question=question_stem, obs_rgb=obs_rgb, obs_depth=obs_depth, target_type=refer_agents_category_config,
                                question_type=q_type, answer_list=question_options, batch_id = scenario_folder_name,
                                question_answer=question_answer, env_name=env_name,logger_base_dir=question_log_dir)

                        max_step = AG.max_step
                        answer = None
                        action_list = []
                        for cur_step in range(0, max_step+1):
                            action = AG.predict(obs_rgb, obs_depth,info)
                            actions = action + [-1]*agent_num
                            action_list.append(action)
                            obs, reward, termination, truncation, info = env.step(actions)
                            obs_rgb, obs_depth = obs_transform(obs)

                            if AG.termination:
                                answer = AG.final_answer
                                break
                            if AG.truncation:
                                answer = AG.final_answer
                                print(f"Episode truncated after {cur_step} steps.")
                                break
                        
                        is_correct = compare_answers_with_api(
                            agent_answer=answer, 
                            ground_truth=question_answer,
                            question_stem=question_stem,
                            question_type=q_type
                        )
                        
                        results.append({
                            "scenario": scenario_folder_name,
                            "q_type": q_type,
                            "q_id": question_id,
                            "agent_answer": answer,
                            "ground_truth": question_answer,
                            "is_correct": is_correct
                        })
                        if is_correct:
                            correct_answers += 1
                            print(f" Correct answer for question: {question_stem}")
                        else:
                            print(f" Incorrect answer for question: {question_stem}")
                        print(f"  Expected: {question_answer}, Got: {answer}")
                        
                        update_state_file(
                            state_file_path, 
                            q_type, 
                            question_id, 
                            status="completed",
                            agent_answer=answer,
                            ground_truth=question_answer,
                            is_correct=is_correct,
                            action_list=action_list
                        )
                        
                        if total_questions % 10 == 0:
                            current_accuracy = calculate_accuracy(correct_answers, total_questions)
                            print(f"\n === Progress Update: {env_name} ===")
                            print(f"Processed {total_questions} questions so far.")
                            print(f"Current accuracy: {current_accuracy:.2f}%")
                            print("=" * 30)
        
                    save_results_to_file(
                        results, correct_answers, total_questions,
                        env_name=env_name, question_type=q_type,
                        filename_prefix=os.path.join(base_save_dir, env_name, scenario_folder_name, q_type)
                    )
            
            except KeyboardInterrupt:        
                print("\n=== 程序被中断 ===")
                print(f"已处理 {total_questions} 个问题")
                print(f"正确答案: {correct_answers}")
                if total_questions > 0:
                    print(f"当前场景、问题类型准确率: {calculate_accuracy(correct_answers, total_questions):.2f}%")
                    save_results_to_file(
                                results, correct_answers, total_questions,
                                env_name=env_name, question_type=q_type,
                                filename_prefix=os.path.join(base_save_dir, env_name, scenario_folder_name, q_type)
                            )
                print("进度已保存，可以使用 --resume 参数继续")
                if 'env' in locals():
                    try:
                        env.close()
                        print("Environment closed.")
                    except Exception as close_e:
                        print(f"Error closing environment after interrupt: {close_e}")
                exit(-1)
            except Exception as e:
                print(f"An error occurred: {e}")    
                if total_questions > 0:
                    print(f"已处理 {total_questions} 个问题")
                    print(f"正确答案: {correct_answers}")
                    print(f"当前场景、问题类型准确率: {calculate_accuracy(correct_answers, total_questions):.2f}%")
                    save_results_to_file(
                                results, correct_answers, total_questions,
                                env_name=env_name, question_type=q_type,
                                filename_prefix=os.path.join(base_save_dir, env_name, scenario_folder_name, q_type)
                            )
                import traceback
                traceback.print_exc()
                if 'env' in locals():
                    try:
                        env.close()
                    except Exception as close_e:
                        print(f"Error closing environment after exception: {close_e}")
                exit(-1)
            finally:
                pass
            
            env.close()
        print(f"\n--- Environment {env_name} processing complete. ---")
        time.sleep(5) 
    print("\n=== All environments processed. Exiting. ===")
    exit(-1)




