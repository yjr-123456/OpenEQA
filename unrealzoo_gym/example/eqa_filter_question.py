import argparse
import gym
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import numpy as np
import cv2
import time
import threading
import tkinter as tk
from tkinter import messagebox # For dialogs
from PIL import Image, ImageTk

from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE, show_question

# === Global Variables ===
goal_to_show = np.zeros((480, 640, 3), dtype=np.uint8) # Default image size
lock = threading.Lock()
current_action = None # Stores the action from GUI sim controls
action_event = threading.Event() # Signals when a new action is available
active_qa_data_from_main = None # Main thread's currently loaded full QA data (dict)
active_qa_file_path_from_main = None # Main thread's currently loaded QA file path (str)
global_question_type = None # Global variable to store the question type for the current interaction
terminate_file_interaction_event = threading.Event()
state_file_path = None # Path to the state file for tracking visited scenarios
# === GUI ===
def run_tkinter_gui():
    editor_root = tk.Tk()
    editor_root.title("UnrealCV Gym Controller & QA Editor")

    # --- Top Frame for Image Display and Sim Controls ---
    top_frame = tk.Frame(editor_root)
    top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    image_label = tk.Label(top_frame) # For displaying env image
    image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    sim_controls_frame = tk.Frame(top_frame, padx=10, pady=10)
    sim_controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    tk.Label(sim_controls_frame, text="Simulation Controls:", font=("Arial", 10, "bold")).pack(pady=(0,5))
    sim_btn_frame = tk.Frame(sim_controls_frame)
    sim_btn_frame.pack(pady=5)

    def set_action_from_gui(a):
        global current_action
        with lock:
            current_action = [a] 
            action_event.set()
    def signal_terminate_interaction():
        terminate_file_interaction_event.set()
        print("GUI: Terminate current file interaction signal sent.")
    tk.Button(sim_btn_frame, text='Forward', width=10, command=lambda: set_action_from_gui(0)).grid(row=0, column=1, pady=2)
    tk.Button(sim_btn_frame, text='Backward', width=10, command=lambda: set_action_from_gui(1)).grid(row=2, column=1, pady=2) # Moved backward down
    
    tk.Button(sim_btn_frame, text='Turn Left', width=10, command=lambda: set_action_from_gui(4)).grid(row=1, column=0, padx=5, pady=2) # Was left
    tk.Button(sim_btn_frame, text='Turn Right', width=10, command=lambda: set_action_from_gui(5)).grid(row=1, column=2, padx=5, pady=2) # Was right
    tk.Button(sim_controls_frame, text='Terminate File Interaction', command=signal_terminate_interaction, bg="orange", fg="white").pack(pady=10)
    def refresh_image_display():
        with lock:
            img_display_data = goal_to_show.copy()
        
        if img_display_data.ndim == 3 and img_display_data.shape[2] == 3:
            img_display_data = cv2.cvtColor(img_display_data, cv2.COLOR_BGR2RGB)
        elif img_display_data.ndim == 2:
            img_display_data = cv2.cvtColor(img_display_data, cv2.COLOR_GRAY2RGB)
        
        img_pil = Image.fromarray(img_display_data)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        image_label.configure(image=img_tk)
        image_label.image = img_tk 
        editor_root.after(100, refresh_image_display)

    editor_root.after(100, refresh_image_display)

    # --- Bottom Frame for QA Editor ---
    qa_editor_frame = tk.Frame(editor_root, bd=2, relief=tk.SUNKEN)
    qa_editor_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    tk.Label(qa_editor_frame, text="QA Editor", font=("Arial", 14, "bold")).pack(pady=5)

    gui_qa_data_cache = {} 
    gui_current_file_path_cache = None

    file_op_frame = tk.Frame(qa_editor_frame)
    file_op_frame.pack(fill=tk.X, padx=5, pady=2)

    list_edit_frame = tk.Frame(qa_editor_frame)
    list_edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

    list_frame = tk.Frame(list_edit_frame)
    list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
    tk.Label(list_frame, text="Questions (QID):").pack(anchor=tk.W)
    qa_listbox = tk.Listbox(list_frame, width=35, height=15, exportselection=False)
    qa_listbox.pack(side=tk.LEFT, fill=tk.Y)
    qa_scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=qa_listbox.yview)
    qa_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    qa_listbox.config(yscrollcommand=qa_scrollbar.set)

    edit_frame = tk.Frame(list_edit_frame)
    edit_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    tk.Label(edit_frame, text="Question ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
    qid_var = tk.StringVar()
    qid_entry = tk.Entry(edit_frame, textvariable=qid_var, width=50)
    qid_entry.grid(row=0, column=1, sticky=tk.EW, pady=2)

    tk.Label(edit_frame, text="Question Stem:").grid(row=1, column=0, sticky=tk.NW, pady=2)
    stem_text = tk.Text(edit_frame, width=50, height=5, wrap=tk.WORD)
    stem_text.grid(row=1, column=1, sticky=tk.EW, pady=2)

    tk.Label(edit_frame, text="Options (one per line):").grid(row=2, column=0, sticky=tk.NW, pady=2)
    options_text = tk.Text(edit_frame, width=50, height=6, wrap=tk.WORD)
    options_text.grid(row=2, column=1, sticky=tk.EW, pady=2)
    
    tk.Label(edit_frame, text="Answer:").grid(row=3, column=0, sticky=tk.NW, pady=2)
    answer_text = tk.Text(edit_frame, width=50, height=3, wrap=tk.WORD)
    answer_text.grid(row=3, column=1, sticky=tk.EW, pady=2)
    
    edit_frame.columnconfigure(1, weight=1)

    qa_action_buttons_frame = tk.Frame(edit_frame)
    qa_action_buttons_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.EW) # Changed row to 4

    def populate_qa_listbox():
        nonlocal gui_qa_data_cache, gui_current_file_path_cache
        qa_listbox.delete(0, tk.END)
        
        temp_active_data_snapshot = None
        with lock:
            if active_qa_data_from_main is not None:
                original_questions = active_qa_data_from_main.get("Questions", {})
                copied_questions = {}
                if isinstance(original_questions, dict):
                    for qid, q_data in original_questions.items():
                        if isinstance(q_data, dict):
                            copied_questions[qid] = {
                                "question": q_data.get("question", ""),
                                "options": list(q_data.get("options", [])) if isinstance(q_data.get("options"), list) else [],
                                "answer": q_data.get("answer", "") 
                            }
                gui_qa_data_cache = copied_questions
            else:
                gui_qa_data_cache = {}
            gui_current_file_path_cache = active_qa_file_path_from_main
            temp_active_data_snapshot = active_qa_data_from_main

        if temp_active_data_snapshot and gui_qa_data_cache:
            for qid_key in sorted(gui_qa_data_cache.keys()):
                qa_listbox.insert(tk.END, qid_key)
            status_label.config(text=f"Loaded: {os.path.basename(gui_current_file_path_cache or 'Unknown')}")
            if qa_listbox.size() > 0:
                qa_listbox.selection_set(0)
                display_selected_question()
            else:
                clear_qa_fields()
        else:
            gui_qa_data_cache = {} 
            status_label.config(text="No QA data or 'Questions' missing. Click 'Load/Refresh'.")
            clear_qa_fields()

    def display_selected_question(event=None):
        try:
            selected_index_tuple = qa_listbox.curselection()
            if not selected_index_tuple:
                return
            selected_qid = qa_listbox.get(selected_index_tuple[0])
            
            if selected_qid in gui_qa_data_cache:
                question_data = gui_qa_data_cache[selected_qid]
                qid_var.set(selected_qid)
                qid_entry.config(state=tk.DISABLED)
                stem_text.delete("1.0", tk.END)
                stem_text.insert("1.0", question_data.get("question", ""))
                options_text.delete("1.0", tk.END)
                options_list = question_data.get("options", [])
                if isinstance(options_list, list):
                     options_text.insert("1.0", "\n".join(options_list))
                elif isinstance(options_list, str): 
                     options_text.insert("1.0", options_list)
                
                answer_text.delete("1.0", tk.END)
                answer_text.insert("1.0", question_data.get("answer", ""))
                add_btn.config(state=tk.DISABLED)
                update_btn.config(state=tk.NORMAL)
                # add_update_btn.config(text="Update Selected", command=update_selected_question)
            else:
                clear_qa_fields()
        except Exception as e:
            status_label.config(text=f"Error displaying: {e}")

    qa_listbox.bind("<<ListboxSelect>>", display_selected_question)

    def clear_qa_fields():
        qid_var.set("")
        qid_entry.config(state=tk.NORMAL)
        stem_text.delete("1.0", tk.END)
        options_text.delete("1.0", tk.END)
        answer_text.delete("1.0", tk.END)
        add_btn.config(state=tk.NORMAL)
        update_btn.config(state=tk.DISABLED)
        # add_update_btn.config(text="Add New", command=add_new_question)
        if qa_listbox.size() > 0:
            qa_listbox.selection_clear(0, tk.END)

    def add_new_question():
        new_qid = qid_var.get().strip()
        new_stem = stem_text.get("1.0", tk.END).strip()
        new_options_str = options_text.get("1.0", tk.END).strip()
        new_options = [opt.strip() for opt in new_options_str.split("\n") if opt.strip()]
        new_answer = answer_text.get("1.0", tk.END).strip()

        if not new_qid:
            status_label.config(text="Error: Question ID cannot be empty.")
            return
        if new_qid in gui_qa_data_cache:
            status_label.config(text=f"Error: QID '{new_qid}' already exists.")
            return
        if not new_stem:
            status_label.config(text="Error: Question Stem cannot be empty.")
            return
        if new_options:
            gui_qa_data_cache[new_qid] = {
                "question": new_stem, 
                "options": new_options,
                "answer": new_answer
            }
        else:
            gui_qa_data_cache[new_qid] = {
                "question": new_stem,
                "answer": new_answer
            }
        
        qa_listbox.insert(tk.END, new_qid)
        try:
            current_list_items = list(qa_listbox.get(0, tk.END))
            idx = current_list_items.index(new_qid)
            qa_listbox.selection_clear(0, tk.END)
            qa_listbox.selection_set(idx)
            qa_listbox.see(idx)
            display_selected_question() 
        except ValueError:
             populate_qa_listbox() 
        status_label.config(text=f"Added: {new_qid}")

    def update_selected_question():
        selected_qid = qid_var.get() 
        if not selected_qid or selected_qid not in gui_qa_data_cache:
            status_label.config(text="Error: No question selected or QID invalid.")
            return

        updated_stem = stem_text.get("1.0", tk.END).strip()
        updated_options_str = options_text.get("1.0", tk.END).strip()
        updated_options = [opt.strip() for opt in updated_options_str.split("\n") if opt.strip()]
        updated_answer = answer_text.get("1.0", tk.END).strip()
        
        if not updated_stem:
            status_label.config(text="Error: Question Stem cannot be empty.")
            return

        gui_qa_data_cache[selected_qid]["question"] = updated_stem
        gui_qa_data_cache[selected_qid]["options"] = updated_options
        gui_qa_data_cache[selected_qid]["answer"] = updated_answer
        status_label.config(text=f"Updated: {selected_qid}")

    def copy_as_new_question():
        qid_var.set("")
        qid_entry.config(state=tk.NORMAL)
        
        add_btn.config(state=tk.NORMAL)
        update_btn.config(state=tk.DISABLED)
        
        if qa_listbox.size() > 0:
            qa_listbox.selection_clear(0, tk.END)
        
        status_label.config(text="context copied,please input new question ID and click 'Add New' button")

    def delete_selected_question():
        try:
            selected_index_tuple = qa_listbox.curselection()
            if not selected_index_tuple:
                status_label.config(text="No question selected to delete.")
                return
            selected_index = selected_index_tuple[0]
            selected_qid = qa_listbox.get(selected_index)

            if selected_qid in gui_qa_data_cache:
                del gui_qa_data_cache[selected_qid]
                qa_listbox.delete(selected_index)
                clear_qa_fields() 
                status_label.config(text=f"Deleted: {selected_qid}")
            else:
                status_label.config(text="Error: QID not found in cache.")
        except Exception as e:
            status_label.config(text=f"Error deleting: {e}")

    def save_qa_data_to_file():
        nonlocal gui_current_file_path_cache
        if not gui_current_file_path_cache:
            status_label.config(text="Error: No file path. Load QA data first.")
            return
        
        data_to_save_final = None
        with lock: 
            if active_qa_data_from_main is not None:
                import copy
                data_to_save_final = copy.deepcopy(active_qa_data_from_main)
            else:
                data_to_save_final = {}

        questions_for_saving = {}
        for qid, data in gui_qa_data_cache.items():
            if "options" not in data or not data["options"]:
                questions_for_saving[qid] = {
                    "question": data.get("question", ""),
                    "answer": data.get("answer", "")
                }
            else:
                questions_for_saving[qid] = {
                    "question": data.get("question", ""),
                    "options": data.get("options", []),
                    "answer": data.get("answer", "")
                }
        data_to_save_final["Questions"] = questions_for_saving
        
        if "target_configs" not in data_to_save_final:
            data_to_save_final["target_configs"] = {} 
        if "safe_start" not in data_to_save_final:
            data_to_save_final["safe_start"] = None
        else:
            if len(data_to_save_final["safe_start"]) > 1:
                data_to_save_final["safe_start"] = [data_to_save_final["safe_start"]]
            data_to_save_final["safe_start"] = data_to_save_final["safe_start"]
        if not questions_for_saving and not messagebox.askyesno("Confirm Save", "No questions in editor. Save empty 'Questions' field?"):
            status_label.config(text="Save cancelled by user.")
            return

        try:
            with open(gui_current_file_path_cache, 'w') as f:
                json.dump(data_to_save_final, f, indent=4, sort_keys=True)
            status_label.config(text=f"Saved to: {os.path.basename(gui_current_file_path_cache)}")
            print(f"Saved to: {os.path.basename(gui_current_file_path_cache)}")
        except Exception as e:
            status_label.config(text=f"Error saving file: {e}")

    load_refresh_btn = tk.Button(file_op_frame, text="Load/Refresh QA", command=populate_qa_listbox)
    load_refresh_btn.pack(side=tk.LEFT, padx=5, pady=2)
    save_btn = tk.Button(file_op_frame, text="Save QA to JSON", command=save_qa_data_to_file)
    save_btn.pack(side=tk.LEFT, padx=5, pady=2)
    status_label = tk.Label(file_op_frame, text="Status: Load a QA file.", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)

    clear_new_btn = tk.Button(qa_action_buttons_frame, text="Clear / New", command=clear_qa_fields, width=12)
    clear_new_btn.pack(side=tk.LEFT, padx=5)
    # add_update_btn = tk.Button(qa_action_buttons_frame, text="Add New", command=add_new_question, width=1)
    # add_update_btn.pack(side=tk.LEFT, padx=5)

    copy_as_new_btn = tk.Button(qa_action_buttons_frame, text="Copy as New Question", command=copy_as_new_question, width=15)
    copy_as_new_btn.pack(side=tk.LEFT, padx=5)

    add_btn = tk.Button(qa_action_buttons_frame, text="Add New", command=add_new_question, width=12)
    add_btn.pack(side=tk.LEFT, padx=5)

    update_btn = tk.Button(qa_action_buttons_frame, text="Update Selected", command=update_selected_question, width=12)
    update_btn.pack(side=tk.LEFT, padx=5)

    delete_btn = tk.Button(qa_action_buttons_frame, text="Delete Selected", command=delete_selected_question, width=12)
    delete_btn.pack(side=tk.LEFT, padx=5)

    # init buttons' state
    update_btn.config(state=tk.DISABLED) # Initially disabled until a question is selected
    add_btn.config(state=tk.NORMAL) # Always enabled to add new questions
    clear_qa_fields()
    populate_qa_listbox()
    editor_root.mainloop()

def build_gui():
    gui_thread = threading.Thread(target=run_tkinter_gui, daemon=True)
    gui_thread.start()

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

if __name__ == '__main__':
    env_list = [
    "Map_ChemicalPlant_1",
    "ModularNeighborhood",
    "ModularSciFiVillage",
    "RuralAustralia_Example_01",
    "ModularVictorianCity",
    "Cabin_Lake",
    "Pyramid"
    # "ModularGothic_Day",
    # "Greek_Island"
    ] 
    question_type_folders = ["counting"]
    for env_name in env_list:
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--env_id", default=f'UnrealCvEQA_general-{env_name}-DiscreteColorMask-v0')
        parser.add_argument("-s", "--seed", type=int, default=0)
        parser.add_argument("-t", "--time-dilation", type=int, default=-1)
        parser.add_argument("-d", "--early-done", type=int, default=-1)
        parser.add_argument("-p", "--QA_path", default=os.path.join(os.path.dirname(__file__), 'QA_Data'))
        args = parser.parse_args()

        build_gui()

        base_env = None
        try:
            print("Initializing UnrealCV Gym environment...")
            base_env = gym.make(args.env_id)
            base_env = base_env 

            if args.time_dilation > 0:
                base_env = time_dilation.TimeDilationWrapper(base_env, args.time_dilation)
            if args.early_done > 0:
                base_env = early_done.EarlyDoneWrapper(base_env, args.early_done)

            for q_type_folder_name in question_type_folders:
                global_question_type = q_type_folder_name
                type_specific_folder_dir = os.path.join(args.QA_path, env_name, q_type_folder_name)
                if not os.path.isdir(type_specific_folder_dir):
                    print(f"Warning: Type folder not found {type_specific_folder_dir}, skipping.")
                    continue
                
                scenario_folder_names = [d for d in os.listdir(type_specific_folder_dir) if os.path.isdir(os.path.join(type_specific_folder_dir, d))]
                # state files
                state_file_path = os.path.join(type_specific_folder_dir, "status_recorder.json")
                with open(state_file_path, 'r') as f:
                    state_data = json.load(f)
                
                for scenario_folder_name in scenario_folder_names:
                    if state_data.get(scenario_folder_name, True):
                        continue  # Skip folders marked as visited

                    terminate_file_interaction_event.clear()
                    id_folder_path = os.path.join(type_specific_folder_dir, scenario_folder_name)
                    file_path = os.path.join(id_folder_path, f"{q_type_folder_name}.json")
                    
                    if not os.path.isfile(file_path):
                        print(f"Warning: JSON file (tried {q_type_folder_name}.json and qa_data.json) not found in {id_folder_path}, skipping.")
                        continue
                    
                    print(f"\n--- Processing File for Interaction: {file_path} ---")
                    QA_data_loaded = load_json_file(file_path)
                    if QA_data_loaded is None:
                        continue

                    with lock:
                        active_qa_data_from_main = QA_data_loaded
                        active_qa_file_path_from_main = file_path
                    
                    target_configs = QA_data_loaded.get("target_configs", {})
                    QA_dict = QA_data_loaded.get("Questions", {}) 
                    safe_start_config = QA_data_loaded.get("safe_start")
                    if len(safe_start_config) > 1:
                        safe_start_config = [safe_start_config]
                    refer_agents_category_config = QA_data_loaded.get("refer_agents_category")
                    agent_num = sum(len(target_configs.get(t, {}).get("name", [])) for t in target_configs)

                    current_env = base_env 
                    unwrapped_env = current_env.unwrapped
                    unwrapped_env.safe_start = safe_start_config
                    unwrapped_env.refer_agents_category = refer_agents_category_config

                    display_question_stem = f"Interacting with: {os.path.basename(file_path)}"
                    display_question_options = []
                    display_question_answer = None
                    if QA_dict: 
                        try:
                            first_qid = sorted(list(QA_dict.keys()))[0]
                            display_question_stem = QA_dict[first_qid].get("question", display_question_stem)
                            display_question_options = QA_dict[first_qid].get("options", [])
                            display_question_answer = QA_dict[first_qid].get("answer", None)
                        except IndexError: 
                            pass 
                    current_env = show_question.ShowQuestionWrapper(current_env, display_question_stem, display_question_options,display_question_answer)

                    current_env = augmentation.RandomPopulationWrapper(current_env, target_configs, num_min=agent_num + 1, num_max=agent_num + 1)
                    current_env = configUE.ConfigUEWrapper(current_env, offscreen=False)

                    print(f"Resetting environment for file: {os.path.basename(file_path)}")
                    states, info = current_env.reset()
                    with lock:
                        goal_to_show[:] = info.get("img_show", np.zeros_like(goal_to_show))

                    cnt_step_for_file_interaction = 0
                    max_interaction_steps = 50
                    # idle_timeouts = 0
                    # max_idle_timeouts = 30 
                    # act in env
                    while cnt_step_for_file_interaction < max_interaction_steps:
                        print(f"File: {os.path.basename(file_path)} - Step: {cnt_step_for_file_interaction}. Waiting for action...")
                        
                        if terminate_file_interaction_event.is_set():
                            print(f"File interaction for {os.path.basename(file_path)} terminated by GUI.")
                            break
                        action_event.wait(timeout=5) # Wait for action from GUI, timeout after 5 seconds
                        if not action_event.is_set():
                            # print(f"Timeout waiting for action for file {os.path.basename(file_path)}. Idle count: {idle_timeouts + 1}")
                            continue
                        if terminate_file_interaction_event.is_set(): # Check again after wait
                            print(f"Interaction for file {os.path.basename(file_path)} terminated by user during wait.")
                            break


                        
                        # idle_timeouts = 0 
                        action_from_gui_list = None
                        with lock:
                            action_from_gui_list = current_action
                            current_action = None 
                        action_event.clear()

                        if action_from_gui_list is None:
                            continue
                        
                        actions_for_env = action_from_gui_list + [-1] * agent_num
                        
                        print(f"Conducting action: {actions_for_env} for file: {os.path.basename(file_path)}")
                        obs, reward, done, truncated, info = current_env.step(actions_for_env)
                        
                        with lock:
                            goal_to_show[:] = info.get("img_show", np.zeros_like(goal_to_show))

                        print(f"File: {os.path.basename(file_path)} - Step {cnt_step_for_file_interaction}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

                        if done or truncated:
                            print(f"Environment episode ended for {os.path.basename(file_path)}. Resetting for continued interaction.")
                            states, info = current_env.reset() 
                            with lock:
                                goal_to_show[:] = info.get("img_show", np.zeros_like(goal_to_show))
                        
                        cnt_step_for_file_interaction += 1
                    
                    print(f"--- Finished interaction period for {file_path}. ---")
                    state_data[scenario_folder_name] = True
                    if terminate_file_interaction_event.is_set():
                        print(f"Proceeding to next file due to user termination for {os.path.basename(file_path)}.")
                        # save state data
                        try:
                            with open(state_file_path, 'w') as f:
                                json.dump(state_data, f, indent=4)
                            print(f"State data updated in {state_file_path}")
                        except Exception as e:
                            print(f"Error saving state data: {e}")
                    # elif idle_timeouts >= max_idle_timeouts:
                    #     print(f"Proceeding to next file due to max idle timeouts for {os.path.basename(file_path)}.")
                    elif cnt_step_for_file_interaction >= max_interaction_steps:
                        print(f"Proceeding to next file due to max interaction steps for {os.path.basename(file_path)}.")
                    print("Waiting a few seconds for GUI interaction (e.g., save) before next file...")
                    time.sleep(3)


        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            try:
                with open(state_file_path, 'w') as f:
                    json.dump(state_data, f, indent=4)
                print(f"State data updated in {state_file_path}")
            except Exception as e:
                print(f"Error saving state data: {e}")
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            try:
                with open(state_file_path, 'w') as f:
                    json.dump(state_data, f, indent=4)
                print(f"State data updated in {state_file_path}")
            except Exception as e:
                print(f"Error saving state data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            if base_env is not None and hasattr(base_env, 'close'):
                try:
                    base_env.close()
                    print("Environment closed.")
                except Exception as e_close:
                    print(f"Error closing environment: {e_close}")
            cv2.destroyAllWindows()
            print("Exited.")
