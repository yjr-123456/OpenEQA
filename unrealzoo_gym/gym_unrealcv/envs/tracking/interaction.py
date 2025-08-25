from gym_unrealcv.envs.navigation.interaction import Navigation
import numpy as np
import math
import time
import json
import re
class Tracking(Navigation):
    def __init__(self, port=9000, ip='127.0.0.1', resolution=(160, 120), comm_mode='tcp'):
        super(Tracking, self).__init__(port=port, ip=ip, resolution=resolution, comm_mode=comm_mode)
        self.obstacles = []

    def random_texture(self, backgrounds, img_dirs, num=5):
        if num < 0:
            sample_index = range(len(backgrounds))
        else:
            sample_index = np.random.choice(len(backgrounds), num, replace=False)
        for id in sample_index:
            target = backgrounds[id]
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(target, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            time.sleep(0.03)

    def random_player_texture(self, player, img_dirs, num):
        sample_index = np.random.choice(5, num)
        for id in sample_index:
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(player, (1, 1, 1), np.random.uniform(0, 1, 3),
                             img_dir, np.random.randint(2, 6), id)
            time.sleep(0.03)

    def random_character(self, target):  # appearance, speed, acceleration
        self.set_speed(target, np.random.randint(40, 100))
        self.set_acceleration(target, np.random.randint(100, 300))
        self.set_maxdis2goal(target, np.random.randint(200, 3000))

    # functions for character setting
    def set_speed(self, target, speed):
        cmd = 'vbp {target} set_speed {speed}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, speed=speed))
        return speed

    def set_acceleration(self, target, acc):
        cmd = 'vbp {target} set_acc {acc}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, acc=acc))
        return acc

    def set_maxdis2goal(self, target, dis):
        cmd = 'vbp {target} set_maxrange {dis}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, dis=dis))
        return dis

    def set_appearance(self, target, id, spline=False):
        if spline:
            cmd = 'vbp {target} set_app {id}'
        else:
            cmd = 'vbp {target} set_mixamoapp {id}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, id=id), -1)
        return id

    def start_walking(self, target):
        cmd = 'vbp {target} start'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target))
        if 'true' in res:
            return True
        if 'false' in res:
            return False

    def get_pose(self, cam_id, mode='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if mode == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if mode == 'hard':
            self.cam[cam_id]['location'] = self.get_cam_location(cam_id)
            self.cam[cam_id]['rotation'] = self.get_cam_rotation(cam_id)
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
            return pose

    def move_2d(self, cam_id, angle, length):
        yaw_exp = (self.cam[cam_id]['rotation'][1] + angle) % 360
        delt_x = length * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = length * math.sin(yaw_exp / 180.0 * math.pi)

        location_now = self.cam[cam_id]['location']
        location_exp = [location_now[0] + delt_x, location_now[1]+delt_y,location_now[2]]

        self.moveto(cam_id, location_exp)
        if angle != 0:
            self.set_rotation(cam_id, [0, yaw_exp, self.pitch])

        location_now = self.get_location(cam_id)
        error = self.get_distance(location_now, location_exp)

        if error < 10:
            return False
        else:
            return True

    def get_startpoint(self, target_pos=[], distance=None, reset_area=[], exp_height=200, direction=None):
        for i in range(5):  # searching a safe point
            if direction == None:
                direction = 2 * np.pi * np.random.sample(1)
            else:
                direction = direction % (2 * np.pi)
            if distance == None:
                x = np.random.randint(reset_area[0], reset_area[1])
                y = np.random.randint(reset_area[2], reset_area[3])
            else:
                dx = float(distance * np.cos(direction))
                dy = float(distance * np.sin(direction))
                x = dx + target_pos[0]
                y = dy + target_pos[1]
            cam_pos_exp = [x, y, exp_height]
            yaw = float(direction / np.pi * 180 - 180)
            if reset_area[0] < x < reset_area[1] and reset_area[2] < y < reset_area[3]:
                cam_pos_exp[0] = x
                cam_pos_exp[1] = y
                return [cam_pos_exp, yaw]
        return []

    def reset_target(self, target):
        cmd = 'vbp {target} reset'
        res=None
        while res is None:
            res = self.client.request(cmd.format(target=target))

    def set_phy(self, obj, state):
        cmd = f'vbp {obj} set_phy {state}'
        res=None
        while res is None:
            res = self.client.request(cmd, -1)

    def simulate_physics(self, objects):
        for obj in objects:
            self.set_phy(obj, 1)

    def set_move(self, target, angle, velocity):
        cmd = 'vbp {target} set_move {angle} {velocity}'.format(target=target, angle=angle, velocity=velocity)
        res = None
        while res is None:
            res = self.client.request(cmd, -1)
    def set_move_new(self, target, param_list):
        angle =param_list[0]
        velocity = param_list[1]
        cmd = 'vbp {target} set_move {angle} {velocity}'.format(target=target, angle=angle, velocity=velocity)
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_move_bp(self, target, params, return_cmd=False):
        '''
        new move function, can adapt to different number of params
        2 params: [v_angle, v_linear], used for agents moving in plane, e.g. human, car, animal
        4 params: [v_ x, v_y, v_z, v_yaw], used for agents moving in 3D space, e.g. drone
        '''
        params_str = ' '.join([str(param) for param in params])
        cmd = f'vbp {target} set_move {params_str}'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_move_batch(self, objs_list, action_list):
        cmd = 'vbp {obj} set_move {angle} {velocity}'
        cmd_list = []
        for i in range(len(objs_list)):
            cmd_list.append(cmd.format(obj=objs_list[i], angle=action_list[i][1], velocity=action_list[i][0]))
        self.client.request(cmd_list, -1) # -1 means async request

    def set_move_with_cam_batch(self, objs_list, action_list, cam_ids, cam_rots):
        cmd_move = 'vbp {obj} set_move {angle} {velocity}'
        cmd_rot_cam = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        cmd_list = []
        for i in range(len(objs_list)):
            cmd_list.append(cmd_move.format(obj=objs_list[i], angle=action_list[i][1], velocity=action_list[i][0]))
        for i in range(len(cam_ids)):
            rot = cam_rots[i]
            cam_id = cam_ids[i]
            self.client.request(cmd_rot_cam.format(cam_id=cam_id, roll=rot[0], yaw=rot[1], pitch=rot[2]))
            self.cam[cam_id]['rotation'] = rot
        res = self.client.request(cmd_list, -1) # -1 means async request

    def get_hit(self, target):
        cmd = f'vbp {target} get_hit'
        res = None
        while res is None:
            res = self.client.request(cmd)
        if 'true' in res:
            return True
        if 'false' in res:
            return False

    def random_lit(self, light_list):
        for lit in light_list:
            if 'sky' in lit:
                self.set_skylight(lit, [1, 1, 1], np.random.uniform(1, 10))
            else:
                lit_direction = np.random.uniform(-1, 1, 3)
                if 'directional' in lit:
                    lit_direction[0] = lit_direction[0] * 60
                    lit_direction[1] = lit_direction[1] * 80
                    lit_direction[2] = lit_direction[2] * 60
                else:
                    lit_direction *= 180
                self.set_light(lit, lit_direction, np.random.uniform(1, 5), np.random.uniform(0.3, 1, 3))

    def set_random(self, target, value=1):
        cmd = 'vbp {target} set_random {value}'.format(target=target, value=value)
        res=None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_interval(self, interval, target=None):
        if target is None:
            cmd = 'vbp set_interval {value}'.format(value=interval)
        else:
            cmd = 'vbp {target} set_interval {value}'.format(target=target, value=interval)
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def init_objects(self, objects):
        self.objects_dict = dict()
        for obj in objects:
            print (obj)
            self.objects_dict[obj] = self.get_obj_location(obj)
        return self.objects_dict

    def set_obj_scale(self, obj, scale):
        cmd = 'vset /object/{obj}/scale {x} {y} {z}'.format(obj=obj, x=scale[0], y=scale[1], z=scale[2])
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def random_obstacles(self, objects, img_dirs, num, area, start_area, texture=False):
        sample_index = np.random.choice(len(objects), num, replace=False)
        for id in sample_index:
            obstacle = objects[id]
            self.obstacles.append(obstacle)
            # texture
            if texture:
                img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
                self.set_texture(obstacle, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            # scale
            self.set_obj_scale(obstacle, np.random.uniform(0.3, 3, 3))
            # location
            obstacle_loc = [start_area[0], start_area[2], 0]
            while start_area[0] <= obstacle_loc[0] <= start_area[1] and start_area[2] <= obstacle_loc[1] <= start_area[3]:
                obstacle_loc[0] = np.random.uniform(area[0]+100, area[1]-100)
                obstacle_loc[1] = np.random.uniform(area[2]+100, area[3]-100)
                obstacle_loc[2] = np.random.uniform(area[4], area[5])-50
            self.set_obj_location(obstacle, obstacle_loc)
            time.sleep(0.01)

    def clean_obstacles(self):
        for obj in self.obstacles:
            self.set_obj_location(obj, self.objects_dict[obj])
        self.obstacles = []

    def new_obj(self, obj_class_name, obj_name, loc, rot=[0, 0, 0]):
        # spawn, set obj pose, enable physics
        [x, y, z] = loc
        [pitch, yaw, roll] = rot
        cmd = [f'vset /objects/spawn {obj_class_name} {obj_name}',
               f'vset /object/{obj_name}/location {x} {y} {z}',
               f'vset /object/{obj_name}/rotation {pitch} {yaw} {roll}',
               f'vbp {obj_name} set_phy 0'
               ]
        self.client.request(cmd, -1)
        return obj_name

    def move_goal(self, obj, goal):
        cmd = 'vbp {obj} move_to_goal {x} {y}'.format(obj=obj, x=goal[0] , y=goal[1])
        res = None
        while res is None:
            res = self.client.request(cmd)

    def get_cam_flag(self, observation_type, use_color=False, use_mask=False, use_depth=False, use_cam_pose=False):
        # get flag for camera
        # observation_type: 'color', 'depth', 'mask', 'cam_pose'
        flag = [False, False, False, False]
        flag[0] = use_cam_pose
        flag[1] = observation_type == 'Color' or observation_type == 'Rgbd' or use_color
        flag[2] = observation_type == 'Mask' or use_mask
        flag[3] = observation_type == 'Depth' or observation_type == 'Rgbd' or use_depth
        print('cam_flag:', flag)
        return flag

    def set_cam(self, obj, loc=[0, 30, 70], rot=[0, 0, 0]):
        # set the camera pose relative to a actor
        x, y, z = loc
        roll, pitch, yaw = rot
        cmd = f'vbp {obj} set_cam {x} {y} {z} {roll} {pitch} {yaw}'
        res = self.client.request(cmd, -1)
        return res

    def adjust_fov(self, cam_id, delta_fov, min_max=[45, 135]):  # increase/decrease fov
        return self.set_fov(cam_id, np.clip(self.cam[cam_id]['fov']+delta_fov, min_max[0], min_max[1]))

    def set_obj_rotation_bp(self, obj, rot):
        [roll, yaw, pitch] = rot
        cmd = f'vbp {obj} set_rotation {roll} {pitch} {yaw}'  # roll, pitch, yaw
        res = self.client.request(cmd)
        return res

    def stop_car(self, obj):
        cmd = f'vbp {obj} set_stop'
        res = self.client.request(cmd, -1)
        return res

    def move_to(self, obj, loc): # navigate the car to a goal location
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res
    def nav_to_goal_bypath(self, obj, loc): # navigate the agent to a goal location
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        # The goal should be reachable in the environment.
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res
    def set_max_nav_speed(self, obj, max_vel): # set the maximum navigation speed of the car
        cmd = f'vbp {obj} set_nav_speed {max_vel}'
        res = self.client.request(cmd, -1)
        return res
    def set_standup(self,obj):
        cmd = f'vbp {obj} set_standup {obj}'
        res = self.client.request(cmd, -1)
        return res
    def enter_exit_car(self, obj, player_index):
        # enter or exit the car for a player.
        # If the player is already in the car, it will exit the car. Otherwise, it will enter the car.
        cmd = f'vbp {obj} enter_exit_car {player_index}'
        res = self.client.request(cmd, -1)
        return res

    def get_pose_img_batch(self, objs_list, cam_ids, img_flag=[False, True, False, False]):
        # get pose and image of objects in objs_list from cameras in cam_ids
        cmd_list = []
        decoder_list = []
        [use_cam_pose, use_color, use_mask, use_depth] = img_flag
        for obj in objs_list:
            cmd_list.extend([self.get_obj_location(obj, True),
                             self.get_obj_rotation(obj, True)])

        for cam_id in cam_ids:
            if cam_id < 0:
                continue
            if use_cam_pose:
                cmd_list.extend([self.get_cam_location(cam_id, return_cmd=True),
                                 self.get_cam_rotation(cam_id, return_cmd=True)])
            if use_color:
                cmd_list.append(self.get_image(cam_id, 'lit', 'png', return_cmd=True))
            if use_mask:
                cmd_list.append(self.get_image(cam_id, 'lit', 'png', return_cmd=True))
            if use_depth:
                cmd_list.append(f'vget /camera/{cam_id}/depth npy')
        decoders = [self.decoder.decode_map[self.decoder.cmd2key(cmd)] for cmd in cmd_list]
        res_list = self.batch_cmd(cmd_list, decoders)
        obj_pose_list = []
        cam_pose_list = []
        img_list = []
        mask_list = []
        depth_list = []
        # start to read results
        start_point = 0
        for i, obj in enumerate(objs_list):
            obj_pose_list.append(res_list[start_point] + res_list[start_point+1])
            start_point += 2
        for i, cam_id in enumerate(cam_ids):
            # print(cam_id)
            if cam_id < 0:
                img_list.append(np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8))
                continue
            if use_cam_pose:
                cam_pose_list.append(res_list[start_point] + res_list[start_point+1])
                start_point += 2
            if use_color:
                # image = self.decoder.decode_bmp(res_list[start_point])
                img_list.append(res_list[start_point])
                start_point += 1
            if use_mask:
                # image = self.decoder.decode_bmp(res_list[start_point])
                mask_list.append(res_list[start_point])
                start_point += 1
            if use_depth:
                # image = 1 / self.decoder.decode_depth(res_list[start_point])
                depth_list.append(res_list[start_point])  # 500 is the default max depth of most depth cameras
                start_point += 1

        return obj_pose_list, cam_pose_list, img_list, mask_list, depth_list
    def generate_nav_goal(self, player, radius_max,radius_min):  # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {player} generate_nav_goal {radius_max} {radius_min} '
        res = self.client.request(cmd)
        answer_dict = json.loads(res)
        try:
            loc = answer_dict["nav_goal"]
        except:
            loc = answer_dict["Nav_goal"]
        coordinates = re.findall(r"[-+]?\d*\.\d+|\d+", loc)
        # Convert the numbers to floats and store them in an array
        coordinates = [float(coordinate) for coordinate in coordinates]
        return coordinates[0],coordinates[1],coordinates[2]

