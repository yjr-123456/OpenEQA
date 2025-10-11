import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to sys.path
import gymnasium as gym
from gymnasium import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import json
import random
import math
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
print("Current Directory:", os.getcwd())
# os.environ['UnrealEnv']='D:\\UnrealEnv\\'

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

def load_model_config(config_path="model_config.json"):
    """加载模型配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"模型配置文件 {config_path} 不存在，使用默认配置")
        return None
    except json.JSONDecodeError:
        print(f"模型配置文件 {config_path} 格式错误")
        return None


def create_client(model_name="doubao", config_path="model_config.json"):
    """根据模型名称创建OpenAI客户端"""
    config = load_model_config(config_path)
    
    if config is None:
        raise ValueError(f"无法加载模型配置文件 {config_path}")
    
    model_config = config["models"].get(model_name)
    if model_config is None:
        print(f"模型 {model_name} 配置不存在，使用默认模型")
        model_name = config["default_model"]
        model_config = config["models"][model_name]
    key_name = model_config["api_key_env"]
    # 从环境变量获取API密钥
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"环境变量 {key_name} 未设置")

    client = OpenAI(
        base_url=model_config["base_url"],
        api_key=api_key
    )
    
    return client, model_config["model_name"], model_config

# 全局客户端和模型配置
client = None
current_model_name = None
current_model_config = None

def initialize_model(model_name="doubao",model_config_path="model_config.json"):
    """初始化指定的模型"""
    global client, current_model_name, current_model_config
    client, current_model_name, current_model_config = create_client(model_name,model_config_path)
    print(f"已初始化模型: {model_name} ({current_model_name})")

system_prompt = """
    You are a classification expert.I will give you a list of object names, please give the category they belong to.
    We will probably provide you with some existing categories, which you have defined before, if you find that the object names belong to these categories, please classify them into these categories again, otherwise, please create new categories for them.
    note:
    1. I will not give you any category names, you need to summarize the category names by yourself.
    2. Classify it more precisely.
    3. you can create new categories if necessary if there are no categories or object do not belong to existing categories.
    4. category name format: general category name/specific category name.
    output format:
    Please use the JSON format to output the category names and the object names they belong to.
    Just an Example:
    
    {
    "categories": [
        {"category_name": "CategoryA", "objects": ["object1", "object2"]},
        {"category_name": "CategoryB", "objects": ["object3", "object4"]}
    ]
    }

"""

def obs_transform(obs, agent_id = 0):
    # obs_rgb = cv2.cvtColor(obs[agent_id][..., :3], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    obs_rgb = obs[agent_id][..., :3]
    obs_mask = obs[agent_id][..., 3:]
    return obs_rgb, obs_mask

def caculate_target_mask_ratio(obs_mask, bias=15):
    # obs_mask: (H, W, 3)
    # bias: 容差，允许像素值 >= 255-bias 都算白色
    white_pixels = np.all(np.asarray(obs_mask) >= (255 - bias), axis=-1)
    white_area = np.sum(white_pixels)
    print("==========obs shape",obs_mask.shape,"============")
    cv2.imwrite("debug_mask.png", obs_mask)
    total_area = obs_mask.shape[0] * obs_mask.shape[1]
    ratio = white_area / total_area
    return ratio

def adjust_target_scale(env, obj_name, size_scale=[1,1,1], min_ratio= 0.4, max_ratio=0.9, still_action=[[-1]]):
    # min_ratio, max_ratio = 0.5, 0.8
    # size_scale = 1.0
    max_iters = 20

    for i in range(max_iters):
        # 设置物体尺寸
        env.unwrapped.unrealcv.set_obj_scale(obj_name, size_scale)
        # 采集 mask
        obs, _, _, _, _ = env.step(still_action)
        obs_rgb, obs_mask = obs_transform(obs)
        ratio = caculate_target_mask_ratio(obs_mask)
        
        # 判断物体是否完整可见（简单判断：mask四边是否有白色像素）
        mask_margin = 5  # 边缘宽度
        mask_h, mask_w = obs_mask.shape[:2]
        border_pixels = np.concatenate([
            obs_mask[:mask_margin, :, :].reshape(-1, 3),
            obs_mask[-mask_margin:, :, :].reshape(-1, 3),
            obs_mask[:, :mask_margin, :].reshape(-1, 3),
            obs_mask[:, -mask_margin:, :].reshape(-1, 3)
        ])

        if np.any(np.all(border_pixels == 255, axis=-1)) and ratio > min_ratio + 0.1:
            size_scale = [x * 0.9 for x in size_scale]
            print(f"物体部分超出视野，减小尺寸，当前尺寸: {size_scale}, 当前占比: {ratio:.2f}")
            continue

        # 根据占比调整尺寸
        if ratio < min_ratio:
            size_scale = [x * 1.1 for x in size_scale]
            print(f"物体尺寸过小，增大尺寸，当前尺寸: {size_scale}, 当前占比: {ratio:.2f}")
        elif ratio > max_ratio:
            size_scale = [x * 0.9 for x in size_scale]
            print(f"物体尺寸过大，减小尺寸，当前尺寸: {size_scale}, 当前占比: {ratio:.2f}")
        else:
            print(f"物体尺寸调整完成，当前尺寸: {size_scale}, 当前占比: {ratio:.2f}")
            break
    return env, size_scale

def extract_agent_from_image(color_img, mask_img, crop_to_bbox=True, color_lower=np.array([250,250,250]), color_upper=np.array([255,255,255])):
    import numpy as np
    import cv2

    # 1. 用RGB值分割白色区域
    rgb_mask = mask_img  # 假设mask_img为RGB或BGR格式
    # 如果mask_img是BGR格式，先转为RGB
    if rgb_mask.shape[2] == 3 and np.any(rgb_mask[..., 0] != rgb_mask[..., 2]):
        rgb_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    # 生成二值掩码
    binary_mask = cv2.inRange(rgb_mask, color_lower, color_upper)  # 单通道 HxW 掩码

    # 2. 创建完整的 RGBA 图像和背景移除的图像
    rgba_output_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2BGRA)
    rgba_output_full[:,:,3] = binary_mask

    bg_removed_full = color_img.copy()
    bg_removed_full[binary_mask == 0] = [255, 255, 255]

    if not crop_to_bbox:
        return rgba_output_full, bg_removed_full

    # 3. 计算边界框并裁剪
    pixel_points = cv2.findNonZero(binary_mask)
    if pixel_points is not None and len(pixel_points) > 0:
        x_coordinates = pixel_points[:, :, 0]
        y_coordinates = pixel_points[:, :, 1]
        x_min = np.min(x_coordinates)
        x_max = np.max(x_coordinates)
        y_min = np.min(y_coordinates)
        y_max = np.max(y_coordinates)
        if x_max >= x_min and y_max >= y_min:
            cropped_rgba_output = rgba_output_full[y_min : y_max + 1, x_min : x_max + 1]
            cropped_bg_removed = bg_removed_full[y_min : y_max + 1, x_min : x_max + 1]
            return cropped_rgba_output, cropped_bg_removed
        else:
            return rgba_output_full, bg_removed_full
    else:
        print("警告: 在binary_mask中未找到用于裁剪的代理，返回完整图像。")
        return rgba_output_full, bg_removed_full

def calculate_ideal_camera_pose_lhs_custom(
    object_max_size_w,     
    image_width_pix,       
    fx,                    
    target_coverage=0.8,   
    target_point_w=np.array([0.0, 0.0, 0.0])
):
    
    # --- 1. Calculate ideal camera depth Zc (geometry calculation) ---
    target_size_pix = image_width_pix * target_coverage
    if target_size_pix <= 0:
        raise ValueError("Target pixel size must be positive.")
        
    Z_c = (object_max_size_w * fx) / target_size_pix
    
    # --- 2. Construct rotation matrix R_w2c for LHS to RHS projection ---
    # R_x180: Rotate 180 degrees around X axis.
    # Effect: [X, Y, Z] -> [X, -Y, -Z]
    # Purpose: Convert LHS Z axis (depth) to RHS Z axis (depth), and flip Y axis to match image.
    R_w2c = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    
    # --- 3. Determine camera translation vector tvec ---
    # tvec = P_c - R_w2c * P_w 
    
    # Expected object center position in camera coordinate system (centered, depth Zc)
    # Z_c must be positive, meaning object is in front of camera
    target_point_c = np.array([0.0, 0.0, Z_c])
    
    tvec = target_point_c - R_w2c @ target_point_w
    
    # --- 4. Convert to OpenCV format rvec ---
    rvec = cv2.Rodrigues(R_w2c)[0]
    
    print(f"--- LHS (custom) camera pose calculation result ---")
    print(f"Calculated camera depth (Z_c): {Z_c:.3f} meters")
    print(f"LHS to RHS projection rotation matrix R_w2c:\n{R_w2c}")
    print(f"Ideal rotation vector (rvec):\n{rvec.T}")
    print(f"Ideal translation vector (tvec):\n{tvec.T}")
    
    return rvec, tvec


if __name__ == '__main__':
    env_name = "SuburbNeighborhood_Day"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvObjectInfoCollection-{env_name}-ContinuousColorMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=1, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=30, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("--obj_path", default=os.path.join(os.path.dirname(__file__), 'Obj_info'),help="path to save object informations")
    parser.add_argument("--model", default="doubao", help="choose evaluation models")
    parser.add_argument("--config_path", default=os.path.join(current_dir, "solution"), help="configuration file path")
    args = parser.parse_args()
    print(args.env_id)
    env = gym.make(args.env_id)

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(1080,1080))
    # env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    # env = agents.NavAgents(env, mask_agent=True)
    # episode_count = 50
    # rewards = 0
    # done = False
    # Total_rewards = 0

    with open(f"{args.obj_path}/{env_name}_obj_info.json", 'r', encoding='utf-8') as f:
        classified_objects = json.load(f)
    # set img path
    img_path = f"{args.obj_path}/{env_name}_cropped_imgs"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    object_dict = classified_objects["categories"]
    still_action = [[-1]]
    cam_id = 0
    # AG = ground_agent(model=args.model, config_path=args.config_path)
    # AG.reset()
    try:
        states, info = env.reset(seed=int(args.seed))
        objects = env.unwrapped.unrealcv.get_objects()
        env.unwrapped.unrealcv.set_hide_objects(objects)  # hide all objects
        for category_dict in object_dict:
            objects = category_dict["objects"]
            for obj in objects:
                if obj.get('cropped_img_path', None) is not None:
                    print(f"Object {obj['name']} already has cropped images, skipping.")
                    continue
                obj_name = obj['name']
                env.unwrapped.unrealcv.set_show_obj(obj_name)  # show target object
                obj_location, obj_rotation = obj['pose'][:3], obj['pose'][3:]
                obj_size_box = env.unwrapped.unrealcv.get_obj_size(obj_name)
                # set mask to white
                obj_color = env.unwrapped.unrealcv.get_obj_color(obj_name) # store original color
                # print(obj_color)
                assert len(obj_color) == 3
                env.unwrapped.unrealcv.set_obj_color(obj_name, [255, 255, 255])

                obs, reward, termination, truncation, info = env.step(still_action)
                obs_rgb, obs_mask = obs_transform(obs)  # (H, W, 3), (H, W, 3)
                ratio = caculate_target_mask_ratio(obs_mask)
                scale = [1,1,1]
                while ratio <= 0.05:
                    # small object, move closer
                    if ratio <= 0.01:
                        scale = [s * 1.5 for s in scale]
                        env.unwrapped.unrealcv.set_obj_scale(obj_name, scale)
                    else:
                        target_pose_0 -= 200
                    env.unwrapped.unrealcv.set_obj_location(obj_name, [target_pose_0, target_pos[1], target_pos[2]])
                    color_mask = env.unwrapped.unrealcv.read_image(1, "object_mask")
                    ratio = caculate_target_mask_ratio(color_mask)
                    print(f"物体太小，移动更近一些, 当前占比: {ratio:.2f}, 位置: {target_pose_0}")

                # restore color
                env.unwrapped.unrealcv.set_obj_color(obj_name, obj_color)
                color = env.unwrapped.unrealcv.get_obj_color(obj_name)
                print("===========restore color",color,"============")
                # set to original scale
                env.unwrapped.unrealcv.set_obj_scale(obj_name, ordinary_scale)
                env.unwrapped.unrealcv.set_obj_location(obj_name, obj_location)
                env.unwrapped.unrealcv.set_obj_rotation(obj_name, obj_rotation)
                # img_list = []
                obj_img_path = f"{img_path}/{obj_name}"
                if not os.path.exists(obj_img_path):
                    os.makedirs(obj_img_path)
                for i in range(1,4):
                    obs_rgb = locals()[f"obs_rgb_{i}"]
                    obs_mask = locals()[f"obs_mask_{i}"]
                    rgba_img, bg_removed_img = extract_agent_from_image(obs_rgb, obs_mask, crop_to_bbox=True)
                    # img_list.append(bg_removed_img)
                    cv2.imwrite(f"{obj_img_path}/view_{i}.png", bg_removed_img)
                # save abs img path to json
                img_paths = [f"{obj_img_path}/view_{i}.png" for i in range(1, 4)]
                obj['cropped_img_path'] = img_paths
                env.unrwrapped.unrealcv.set_hide_obj(obj_name)  # hide target object
                with open(f"{args.obj_path}/{env_name}_obj_info.json", 'w', encoding='utf-8') as f:
                    json.dump(classified_objects, f, ensure_ascii=False, indent=2)



    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        # 保存最新数据
        with open(f"{args.obj_path}/{env_name}_obj_info.json", 'w', encoding='utf-8') as f:
            json.dump(classified_objects, f, indent=2, ensure_ascii=False)
        env.close()
    finally:
        # 再保存一次，确保数据完整
        with open(f"{args.obj_path}/{env_name}_obj_info.json", 'w', encoding='utf-8') as f:
            json.dump(classified_objects, f, indent=2, ensure_ascii=False)
        env.close()
