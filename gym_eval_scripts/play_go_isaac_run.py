import sys
# sys.path.append('/dingpengxiang/Pengxiang/walk-these-ways/quart_scripts')

# SCRIPT_PATH='/dingpengxiang/Pengxiang/Quart++'
# sys.path.append(SCRIPT_PATH)

# ROOT_PATH = '/dingpengxiang/Pengxiang/Quart++/isaacSim/videos'
# # os.path.join(MINI_GYM_ROOT_DIR[:-16], 'quadruped_rt1')
# sys.path.append(ROOT_PATH)

# ENV_PATH='/dingpengxiang/Pengxiang/walk-these-ways'
# sys.path.append(ENV_PATH)

# ORI_PATH='/dingpengxiang/Pengxiang/walk-these-ways/test_scripts'
# sys.path.append(ORI_PATH)
import os
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(f"{PROJECT_PATH}/walk-these-ways-quart")
from path_plan import *
from datetime import datetime
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H%M")

import multiprocessing as mp
import isaacgym

import torch
import numpy as np
import glob
import pickle as pkl
from ml_logger import logger

from pathlib import Path
# import pdb; pdb.set_trace()
from go1_gym_quart import MINI_GYM_ROOT_DIR
import glob
import os
import cv2


from go1_gym_quart.envs import *
from go1_gym_quart.envs.base.legged_robot_config import Cfg
from go1_gym_quart.envs.go1.go1_config import config_go1
from go1_gym_quart.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_quart.utils.terrain import Terrain
from tqdm import tqdm
import os
import cv2
from matplotlib import colors
import csv
import threading
from isaacgym import gymtorch, gymapi, gymutil


import sys
from go1_gym_quart import MINI_GYM_ROOT_DIR


from torchvision import transforms
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

resize = 300
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

'''
variable: traget obj, obstacle obj, obj_appearance
obj: cube, ball, cabinet, robot, mug, brick, meat can, banana
appearance: color, size, texture, alphabet

task: 
1. go to obj/place (static obj)         {obj_num >= 1}
2. stop obj (moving obj)                {obj_num >= 1}
3. go to obj and avoid obstacle obj     {obj_num >= 1, obs_num >=1}
4. go to obj/place and unload loads

'''


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def tranj_gen(st,ed,ob,sv):
    dxy = 0.05
    pts = np.array([st,ed])

    if ob.shape[0]==0:
        ob = st+[1.5,0.]
        obs = np.array([ob-[0.2,0.7],ob+[0.2,0.7]]).reshape([pts.shape[0],pts.shape[1],-1,pts.shape[2]])
    else:
        obs = np.array([ob-[0.5,1.],ob+[0.5,1.]]).reshape([pts.shape[0],pts.shape[1],-1,pts.shape[2]])
    xy_max = np.max([pts.max(0),obs.max(0).max(1)],0)+0.3
    xy_min = np.min([pts.min(0),obs.min(0).min(1)],0)-0.3
    tar_pos = []
    tar_vel = []

    for i in range(len(xy_max)):
        a_star = AStarPlanner(xy_max[i],xy_min[i],obs[:,i,:,:], dxy, 0.3)
        path = a_star.planning(st[i], ed[i])
        if path.shape[0]<20:
            print(path)
            print(st[i], ed[i])
            print(obs[:,i,:,:])
        print(i,path.shape[0])

        last_len = int(path.shape[0]/10)
        vel = np.ones(path.shape[0])*sv[i]
        vel[-last_len:]= np.arange(last_len-1,-1,-1)*sv[i]/(last_len-1)
        tar_pos.append(path)
        tar_vel.append(vel)
    return tar_pos,tar_vel


def tranj_gen_crawl(st,ed,sv):
    dxy = 0.05
    pts = np.array([st,ed]) 
    # obs = np.array([ob-0.3,ob+0.3]).reshape([pts.shape[0],pts.shape[1],-1,pts.shape[2]]) 

    xy_max = np.max([pts.max(0)],0)
    xy_min = np.min([pts.min(0)],0)

    xy_size = ((xy_max - xy_min)/dxy).astype(int)+1
    pts_int = ((pts - xy_min)/dxy).astype(int)
    # obs_int = ((obs - np.expand_dims(xy_min,1))/dxy).astype(int)
    tar_pos = [] 
    tar_vel = [] 

    for i in range(len(xy_size)):
        grid = np.zeros(xy_size[i])

        start = Node(pts_int[0,i])
        path = d_star_algorithm(start, pts_int[1,i], grid) 
        path = xy_min[i]+path*dxy  
        last_len = int(path.shape[0]/10)
        vel = np.ones(path.shape[0])*sv[i]
        vel[-last_len:]= np.arange(last_len-1,-1,-1)*sv[i]/(last_len-1)
        tar_pos.append(path)
        tar_vel.append(vel)
    return tar_pos,tar_vel



def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, args, task_cfg):

    headless = args.headless
    dirs = glob.glob(f"{PROJECT_PATH}/walk-these-ways-quart/runs/{label}/*")  #['runs/gait-conditioned-agility/pretrain-v0/train/025417.456545']
    # import pdb; pdb.set_trace()
    logdir = sorted(dirs)[0]   #'runs/gait-conditioned-agility/pretrain-v0/train/025417.456545'


    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file) 
        cfg = pkl_cfg["Cfg"]

        for key, value in cfg.items():
            # print("current key:",key)
            if key == 'command_ranges':
                continue
            elif hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.commands.resampling_time = 100

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = int(args.env_num)
    Cfg.env.num_envs = int(args.env_num)

    Cfg.terrain.num_rows = 9
    Cfg.terrain.num_cols = 9
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    Cfg.task = task_cfg['action']

    if task_cfg['action'] == 'unload':
        Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/wr2/wr2/wr2_load.urdf'
    else:
        Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/wr2/wr2/wr2.urdf'
    # Cfg.control.control_type = "actuator_net"


    # all_color = ['beige','black','blue','brown','burlywood','cyan','darkcyan','darkgray','darkgreen','darkorange','darkred','deeppink','deepskyblue','gold','gray','green','lightblue','lightcyan','lightgray','lightgreen','lightpink','lightskyblue','lightyellow','mediumblue','mediumpurple','navy','orange','pink','purple','red','silver','skyblue','tan','white','yellow',]
    # all_color = ['beige','black','blue','brown','cyan','gold','gray','green','navy','orange',
    #          'pink','purple','red','silver','tan','white','yellow']

    if args.test_type == 'seen':
        all_color = ['blue','green','red','yellow']
        objs = {'trashcan':['texture'], 'wardrobe':['texture'], 'drawers':['texture'], 
        'chair':['texture'], 'ball':all_color, 'fence':['texture'], 'fan':['texture'], 
        'vase':['texture'], 'bookshelf':['texture'], 'bench':['texture'], 'oven':['texture'], 
        'cooker':['texture'], 'table':['texture'], 'barrel':['texture'], 'sofa':['texture'], 
        'piano':['texture'], 'cube':all_color}
    else:
        all_color = ['blue','green','red','yellow','gold','orange','pink','purple']
        objs = {'ball':all_color,'cube':all_color,
            'trashcan': ['texture'],'trashcan1': ['texture'],
            'wardrobe': ['texture'],'wardrobe1': ['texture'],
            'drawers': ['texture'],'drawers1': ['texture'],
            'chair': ['texture'],'chair1': ['texture'],
            'fence': ['texture'],'fence1': ['texture'],
            'fan': ['texture'],'fan1': ['texture'],
            'vase': ['texture'],'vase1': ['texture'],
            'bookshelf': ['texture'],'bookshelf1': ['texture'],
            'bench': ['texture'],'bench1': ['texture'],
            'oven': ['texture'],'oven1': ['texture'],
            'cooker': ['texture'],'cooker1': ['texture'],
            'table': ['texture'],'table1': ['texture'],
            'barrel': ['texture'],'barrel1': ['texture'],
            'sofa': ['texture'],'sofa1': ['texture'],
            'piano': ['texture'],'piano1': ['texture'],
            'pillow1': ['texture'],'pillow2': ['texture']
        }
    import re
    # import pdb; pdb.set_trace()
    if task_cfg['action'] != 'crawl':
        for ob in list(objs.keys()): 
            if re.sub(r'[0-9]+', '', task_cfg['object']) == re.sub(r'[0-9]+', '', ob): 
                objs.pop(ob)


    if task_cfg['color'] in all_color: all_color.remove(task_cfg['color']) #
    obj_avoid = [[]]*Cfg.env.num_envs
    obj_avoid_color = [[]]*Cfg.env.num_envs
    obj_load = [['ball']]*Cfg.env.num_envs
    obj_load_color = [[]]*Cfg.env.num_envs

    if task_cfg['action'] == 'stop' or task_cfg['action'] == 'push':
        objs = {'ball':all_color}
    elif task_cfg['action'] == 'distinguish': 
        objs = {'a':['texture'], 'b':['texture'], 'c':['texture'],'d':['texture'],'e':['texture'],'f':['texture']} 
        objs.pop(task_cfg['object'])
    elif task_cfg['action'] == 'unload':
        obj_load_color = np.array([np.random.choice(all_color,Cfg.env.num_envs)]).reshape(Cfg.env.num_envs,-1)
        if task_cfg['color'] in all_color: all_color.remove(task_cfg['color'])
        objs = {'traybox':all_color}
    elif task_cfg['action'] == 'go_through':
        obj_stop = [['ball']]*Cfg.env.num_envs
        obj_stop_color = [['red']]*Cfg.env.num_envs
        if task_cfg['color'] in all_color: all_color.remove(task_cfg['color'])
        objs = {'rectangle tunnel':all_color,
                'triangle tunnel':all_color}
    elif task_cfg['action'] == 'go_avoid':
        for k in ['piano','cooker','bookshelf','trashcan','drawer','oven','bench']:
            if k in objs.keys(): objs.pop(k)

        obj_avoid = np.random.choice(list(objs.keys()),Cfg.env.num_envs)
        obj_avoid_color = np.array([np.random.choice(objs[n]) for n in obj_avoid]).reshape(Cfg.env.num_envs,-1)
        obj_avoid = obj_avoid.reshape(Cfg.env.num_envs,-1)
    elif task_cfg['action'] == 'crawl':
        objs = {'gate3':all_color,'gate1':all_color,'gate2':all_color}
        

    obs = np.random.choice(list(objs.keys()),Cfg.env.num_envs)  #<class 'numpy.ndarray'>, array(['wardrobe','bench',...,'trashcan']),len=25
    obs_color = np.array([np.random.choice(objs[n]) for n in obs]).reshape(Cfg.env.num_envs,-1) #[texture]*25 
    obs = obs.reshape(Cfg.env.num_envs,-1)

    if task_cfg['action'] == 'crawl':
        obs = [[]]*Cfg.env.num_envs
        obj_avoid = [[]]*Cfg.env.num_envs
        obj_avoid_color = [[]]*Cfg.env.num_envs
        obj_load = [[]]*Cfg.env.num_envs
        obj_load_color = [[]]*Cfg.env.num_envs
    
    if task_cfg['action'] != 'go_through':
        obj_stop = [[]]*Cfg.env.num_envs   #n empty list
        obj_stop_color = [[]]*Cfg.env.num_envs   

    Cfg.obj_name = {'target':[task_cfg['object']],  #task object name
                    'load':obj_load,
                    'obstacle':obs,
                    'stop':obj_stop,
                    'avoid':obj_avoid}

    Cfg.obj_color = {'target':[task_cfg['color']],  #task object color
                     'load': obj_load_color,
                     'obstacle':obs_color,
                     'stop':obj_stop_color,
                     'avoid': obj_avoid_color}
    
    Cfg.obj_size = {'target':['small'], 'obstacle':[]}
    Cfg.obj_place = {'target':['front'], 'obstacle':[]}

    # print("Loading obj_name: ",Cfg.obj_name)
    # print("Loading obj_color: ",Cfg.obj_color)

    from go1_gym_quart.envs.wrappers.history_wrapper import HistoryWrapper

    # env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)  
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=args.headless, cfg=Cfg)  #physic engine，load urdf
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)  #<function load_policy.<locals>.policy at 0x7f9c7344ea60>

    return env, policy



def is_target_in_view(info, threshold_angle):
    body_yaw = info['_body_yaw']  #  (3,)
    body_xy = info['_body_xy']     #  (num_envs, 2)
    target_xy = info['_target_xy']  


    dt = target_xy - body_xy  # (3, 2)

    target_angle = np.arctan2(dt[:, 1], dt[:, 0])
    robot_yaw = body_yaw.flatten() 

    offset_angle = (target_angle-robot_yaw) % (2 * np.pi)
    # offset_angle = (robot_yaw+np.pi/2 - target_angle) % (2 * np.pi)
    offset_angle[offset_angle > np.pi] -= 2 * np.pi  # set to [-pi, pi]
    # print('offset_angle: ', offset_angle) #

    rad_threshold = np.deg2rad(threshold_angle)

    # if target in view
    in_view = np.abs(offset_angle) <= (rad_threshold)
    return in_view

def play_go1_plus(model, args, task_cfg):  

    print("current path:",os.getcwd())
    task_output_path=f"{args.save_folder}/{formatted_datetime}/{task_cfg['name']}"

    use_cuda = torch.cuda.is_available()  # True
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_type = args.dataset_type  #Full
    range_dir = './datasets/{}'.format(dataset_type)   #'./Quart/datasets/Full'
    range_name = 'merged_quadruped_data_info'
    range_path = os.path.join(range_dir,range_name)  #'./Quart/datasets/Full/merged_quadruped_data_info'
    


    #/dingpengxiang/Pengxiang/walk-these-ways/Quart/test.py
    print("ckpt_path:",args.ckpt_path)
    # agent = Quart_plus(exp_id=exp_id, range_path=range_path, ckpt_path=ckpt_path, vocab_path="/dingpengxiang/Pengxiang/Quart++/vocabs/vocab_fuyu.json")
    # import pdb; pdb.set_trace()

    ob_keys = ["joint_pos","joint_vel","joint_pos_target","joint_vel_target",
        "body_linear_vel","body_angular_vel","body_linear_vel_cmd",
        "body_angular_vel_cmd","contact_states","foot_positions",
        "body_pos","torques","body_quat"]

    # set_true='true'
    # headless = set_true == 'true'

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, args, task_cfg)
    # env, policy = load_env(label, argv)

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    gait_duration = 0.2
    frequency = 0.02
    stance_width_cmd = 0.25

    obs = env.reset()

    env.commands[:, 0] = x_vel_cmd
    env.commands[:, 1] = y_vel_cmd
    env.commands[:, 2] = yaw_vel_cmd
    env.commands[:, 3] = body_height_cmd
    env.commands[:, 4] = step_frequency_cmd
    env.commands[:, 5:8] = gait 
    env.commands[:, 8] = 0.2
    env.commands[:, 9] = footswing_height_cmd
    env.commands[:, 10] = pitch_cmd
    env.commands[:, 11] = roll_cmd
    env.commands[:, 12] = stance_width_cmd

    task = task_cfg['action'] #go_to

    if task_cfg['color'] == 'texture':
        object_instruction = task_cfg['object']
    else:
        object_instruction = f"{task_cfg['color']} {task_cfg['object']}"
        # object_instruction = task_cfg['color']+' '+task_cfg['object']
    gait_instruction = 'normally with a trotting gait'
    
    

    if task == 'go_to':
        task_instruction = f'go to the {object_instruction}' 
    elif task == 'go_avoid':
        task_instruction = f'go to the {object_instruction} and avoid the obstacle'
    elif task == 'stop':
        task_instruction = f'stop the {object_instruction}'
    elif task == 'crawl':
        task_instruction = f'crawl through the bar'
    elif task == 'distinguish':
        task_instruction = f'distinguish letter {object_instruction}'
    elif task == 'go_through':
        task_instruction = f'go through the {object_instruction}'
    elif task == 'unload':
        task_instruction = f'unload the ball into the {object_instruction}'
    else:
        raise NotImplementedError

    print('Instruction is:', task_instruction)  # go to the xxx + normally with a trotting gait
    
    print('prepare')
    # import pdb;pdb.set_trace()
    for i in range(100):
        # print('prepare:', i)
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        # print('Info body_xyz is:', info['_body_xyz'])
        # print('Info load_xyz is:', info['_load_xyz'])
        # print('Info yaw is:', info['_body_yaw'])
        # print('Info _target_xy is:', info['_target_xy'])
    print('start')


    success_len = np.zeros(Cfg.env.num_envs)
    success = np.zeros(Cfg.env.num_envs, dtype=bool)
    if task == 'unload' or task == 'go_through':
        sub_success = np.zeros(Cfg.env.num_envs)
        
    network_state = None

# # ###################goto
    selected_env = 0
    save_type='multi'  #save one or multi environment videos
    if Cfg.env.num_envs>=50:
        save_envImg_num=int(Cfg.env.num_envs/4)  #the number of envs to save videos
    else:
        save_envImg_num=int(Cfg.env.num_envs/2)  #the number of envs to save videos

    if save_type=='one':
        selected_env = 0
        frames = []
    else:
        frames = {}
        for i in range(save_envImg_num):
            frames[i]=[]

    set_velocity = 1.0*np.random.randint(1,4,Cfg.env.num_envs)/3+0.1*np.random.random(Cfg.env.num_envs)
    set_step_freq = np.random.randint(2,5,Cfg.env.num_envs)
    set_body_height = np.random.random(Cfg.env.num_envs)/5-0.1
    set_footswing_height = np.random.random(Cfg.env.num_envs)/10+0.08
    set_stance_width = np.random.random(Cfg.env.num_envs)/5+0.1
    set_pitch = np.zeros_like(set_velocity)


    T = 20.0  # max simulation time
    if task == 'push':
        info['_target_xy']+=[0.5,0]
        T = 15
    elif task == 'stop':
        info['_target_xy']+=[-0.5,0]
    elif task == 'unload':
        info['_target_xy']+=[0.2,0]
        T = 15
    elif task == 'go_through':
        info['_target_xy']+=[2,0]
        T = 15
    elif task == 'crawl':
        info['_target_xy']+=[2.5, 0]
        T = 20

    time = 0.0
    step_cnt = 0


    if task == 'stop':
        forces = torch.zeros((Cfg.env.num_envs*(17+int(args.num)), 3), device=env.device, dtype=torch.float)
        torques = torch.zeros((Cfg.env.num_envs*(17+int(args.num)), 3), device=env.device, dtype=torch.float)
        tar_force_idx = np.arange(17,Cfg.env.num_envs*(17+int(args.num)),17+int(args.num))
        torques[tar_force_idx, 1] = -20
        env.env.gym.apply_rigid_body_force_tensors(env.env.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    terminates = torch.zeros(Cfg.env.num_envs, dtype=torch.bool, device=device)  
    terminates_pre = torch.zeros(Cfg.env.num_envs, dtype=torch.bool, device=device)  
    add_infer_time = torch.zeros(Cfg.env.num_envs, dtype=torch.float16, device=device)  
    fin_infer_time = torch.zeros(Cfg.env.num_envs, dtype=torch.float16, device=device)  

    init_pos = env.root_states.clone() #3*25
    bx, by, bz = init_pos[selected_env, 0], init_pos[selected_env, 1], init_pos[selected_env, 2]
    
    terminate_flag = 0  # terminate flag for all environments
    terminate_t = 0.0   # terminate time for all environments
    frequency = 0.02
    # output_actions_5 = [None] * args.env_num
    output_actions_10 = [None] * args.env_num

    while terminate_t < 1.0 and time < T:  #check every env.dt=0.02
        cmd = {'x_vel_cmd' : env.commands.cpu().numpy()[:, 0],
                'y_vel_cmd' : env.commands.cpu().numpy()[:, 1],
                'yaw_vel_cmd' : env.commands.cpu().numpy()[:, 2],
                'body_height_cmd' : env.commands.cpu().numpy()[:, 3],
                'step_frequency_cmd' : env.commands.cpu().numpy()[:, 4],
                'gait ' : env.commands.cpu().numpy()[:, 5:8],
                'footswing_height_cmd' : env.commands.cpu().numpy()[:, 9],
                'pitch_cmd' : env.commands.cpu().numpy()[:, 10],
                'roll_cmd' : env.commands.cpu().numpy()[:, 11],
                'stance_width_cmd' : env.commands.cpu().numpy()[:, 12],}


        with torch.no_grad():
            actions = policy(obs) #obs is a dict


        if round(time*1000)%round(frequency*1000) == 0:  #manully stable frequency 
            # import pdb; pdb.set_trace()
            step_cnt+=1
            task_instructions = "What action should the legged robot take to {}?".format(task_instruction)
            print("========= New Step ==========")
            print('instructions: ', task_instructions)


            for i in range(Cfg.env.num_envs):
                if bool(terminates[i]):
                    env.commands[i, 0:3] = 0. #stop moving
                    print(".............")
                    print('Step count:',step_cnt,", Current env:",i,', current time: ', round(time,3),', finished processing! Result:',success[i])


                else:
                    print(".............")
                    print('Step count:',step_cnt, ", Current env:",i,", still processing...",', current time: ', round(time,3),'Result:',success[i])

                    images=info['imgs'][i,:,:,::-1]  #BGR to RGB

                    images = np.asarray(Image.fromarray(images).resize((240, 240)))
                    if i == 0 and step_cnt==1:
                        save_image = Image.fromarray(images)
                        img_path=f"{task_output_path}/{i}.jpg"
                        folder_path = os.path.dirname(img_path)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        save_image.save(img_path)
                        print("check the enviroment img at: ",img_path)

                    contact_states = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) > 1.0
                    proprioceptions = torch.cat((env.projected_gravity, env.dof_pos, env.dof_vel, contact_states), dim=-1).to(device)

                    task_instructions = task_instructions + '\n'
                    # import pdb; pdb.set_trace()


                    #####
                    # model output 10 action steps per inference, here dilute the action steps to 50 steps to match dataset frequency and avoid unfinished actions.
                    seq_sig = step_cnt%50  

                    if seq_sig==1:  # inference every n step
                        with torch.no_grad():
                            try:
                                output_actions_all10, output_time = model.inference_one_time_step(images, task_instructions, proprioceptions, network_state)
                                output_actions_10[i] = output_actions_all10['commands']
                                # print("Failed output_actions:", output_actions_all10)
                            except:
                                pass
                    try:
                        commands = output_actions_10[i][(seq_sig-1)//5][1:].unsqueeze(0)

                    except:
                        # import pdb; pdb.set_trace()
                        pass

                    # import pdb; pdb.set_trace()
                    terminates[i] = output_actions_10[i][(seq_sig-1)//5][0].bool().unsqueeze(0)


                    env.commands[i, 0:3] = commands[:, 0:3]  # xy yaw
                    env.commands[i, 3] = commands[:, 3]      #body_height
                    env.commands[i, 4] = commands[:, 4]      #stepfrequency
                    env.commands[i, 5:8] = commands[:, 5:8]  #gait
                    env.commands[i, 8] = gait_duration  #gait_duration
                    env.commands[i, 9:11] = commands[:, 8:10]   # footswing_height, pitch
                    env.commands[i, 11] = 0.  #roll_cmd, roll abandoned
                    env.commands[i, 12] = commands[:, 10]  #stance_width
                    terminate_flag = torch.all(terminates) or torch.all(torch.tensor(success))
                    

        obs, rew, done, info = env.step(actions)

        if time == 0.0:
            ar = info['_body_xy']
            obs_s=info['_obstacle_xy']

        dt = info['_body_xy']-info['_target_xy'] #the (x,y) difference between current position and target position 
        st = info['_body_xy']-ar # the difference of start and end position of go


        if task == 'crawl':
            dt[:, 0] -= 1.5

        if task == 'push' or task == 'stop' or task == 'crawl':
            success = np.logical_or(success, np.hypot(dt[:,0],dt[:,1])<1.0)
        elif task == 'distinguish':
            success1 = np.logical_or(success, abs(dt[:,1])<0.2)  #y axis distance <0.2m
            success2 = np.logical_or(success, is_target_in_view(info, 5)) #angle angle <5 degree
            success = np.logical_or(success1,success2)
        elif task == 'go_through':
            dt_through = info['_body_xy']-info['_target_xy']+[-1,0]
            sub_success = np.logical_or(sub_success, np.hypot(dt[:,0],dt[:,1])<0.8)
            success = np.logical_or(success, np.hypot(dt_through[:,0],dt_through[:,1])<1.0)
        elif task == 'unload':
            dt_load = info['_load_xy']-info['_target_xy'] #distance between ball and plate
            sub_success = np.logical_or(sub_success, np.hypot(dt[:,0],dt[:,1])<0.8) #sub success，distance between robot and plate
            success = np.logical_or(success, np.hypot(dt_load[:,0],dt_load[:,1])<.10) 
        else: #go_to, go_avoid
            success = np.logical_or(success, np.hypot(dt[:,0],dt[:,1])<1.0)

        success_len[np.logical_and(success_len<1,success>0)] = step_cnt
        # print(success_len)
        
        time+=env.dt 
        if bool(terminate_flag):  #while if all env ended, wait for 1 second to end all process
            terminate_t += env.dt

        if save_type=='one':
            bx, by, bz = init_pos[selected_env, 0], init_pos[selected_env, 1], init_pos[selected_env, 2]
            env.gym.set_camera_location(env.rendering_camera, env.envs[0], gymapi.Vec3(bx - 2.0, by - 0.0, bz + 3.0),
                                            gymapi.Vec3(bx, by, bz))
            frame = env.gym.get_camera_image(env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR)
            frame = frame.reshape((env.camera_props.height, env.camera_props.width, 4))
            img = info['imgs'][selected_env, :, :, ::-1]  # info['imgs'] (25,480,640,3)，convert last dim 
            img = np.concatenate((frame[:, :, :-1], img), axis=1)
            # img = frame
            frames.append(img)
        elif save_type=='multi':  #the case of saving multi env videos
            for i in range(save_envImg_num):
                if not bool(terminates_pre[i]): 
                    if i==0:
                        env.gym.set_camera_location(env.rendering_camera, env.envs[0], gymapi.Vec3(bx - 2.0, by - 0.0, bz + 3.0),
                        gymapi.Vec3(bx, by, bz))
                        frame = env.gym.get_camera_image(env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR)
                        frame = frame.reshape((env.camera_props.height, env.camera_props.width, 4)) 
                        img = info['imgs'][selected_env, :, :, ::-1]  # info['imgs'] (25,480,640,4)，RGBA to RGB
                        # img = info['imgs'][selected_env, :, :, ::-1]  # info['imgs'] (25,480,640,3)，convert last dim 
                        img = np.concatenate((frame[:, :, :-1], img), axis=1)  #将图像从 RGBA 转换为 RGB
                    else:
                        img = info['imgs'][i,:,:,::-1]
                    frames[i].append(img)      
        terminates_pre = terminates
        # import pdb; pdb.set_trace()

    if save_type=='one':
        logger.save_video(frames, f"{task_output_path}/videos/{result}", fps=1 / env.dt)

    elif save_type=='multi':
        for i in range(save_envImg_num):
            # if not os.path.exists(video_path):
            #     os.makedirs(video_path)
            result=success[i]
            try:
                logger.save_video(frames[i], f"{task_output_path}/videos/{task_cfg['name']}_env{i}_{result}", fps=1 / env.dt)
            except:  
                pass
    print("Saved to: ", args.save_folder)
            
    print("=============")
    print("CURRENT TASK: ",task_cfg['name'])
    print('success:', success)
    
    # count the total num and succes rate
    total_count = success.size
    true_count = np.sum(success)  
    success_percentage = (true_count / total_count) * 100

    print("Total:{}, Success num (True):{}, Success Rate: {:.2f}%".format(total_count, true_count, success_percentage))
    print("=============")


    # import pdb; pdb.set_trace()
    save_and_write_info(step_cnt,args, task_cfg, {key: value for key, value in info.items() if key in ob_keys},
            cmd,info['imgs'],info['depths'],success,success_len,
            set_velocity,set_step_freq,env.cfg.obj_name['obstacle'],
            env.cfg.obj_color['obstacle'])

    print("------closing env------")
    env.close()

    return task_output_path
    
  
task_names = {
    'gf':'go_forward',
    'gb':'go_backward',
    'gl':'go_left',
    'gr':'go_right',
    'tr':'turn_right',
    'tl':'turn_left',
    'go_to': 'go_to_the_OBJ',
    'go_through': 'go_through_the_OBJ',
    'unload': 'unload_to_the_OBJ',
    'distinguish': 'distinguish_the_OBJ',
    'stop': 'stop_the_OBJ',
    'push': 'push_the_OBJ',
    'go_avoid': 'go_to_the_OBJ_and_avoid_obstacle'
}


def save_and_write_info(step_cnt, args, task_cfg, robot_states, remote_cmd, imgs, depths, success, success_len, vel, freq, name, color):
    speed_level = ['slowly', 'normally', 'quickly']
    # FilePath = '/wangdonglin/sim_quadruped_data_v1/' + f'{task_cfg['action']}_{task_cfg['color']}_{task_cfg['object']}'
    if args.save_folder!=None:
        FilePath = '{}/{}/{}/'.format(args.save_folder, formatted_datetime, task_cfg['name'])
    else:
        print("Need a save folder!")
    print("Results saved to :", FilePath)
    os.makedirs(FilePath,exist_ok=True)

    def save(step_cnt, idx, track):
        os.makedirs(f"{FilePath}/{track}/command", exist_ok=True)
        os.makedirs(f"{FilePath}/{track}/action", exist_ok=True)
        os.makedirs(f"{FilePath}/{track}/image", exist_ok=True)
        os.makedirs(f"{FilePath}/{track}/depth", exist_ok=True)
        np.save(f"{FilePath}/{str(track)}/command/{step_cnt:03d}.npy", {k: v[idx] for k, v in remote_cmd.items()})
        np.save(f"{FilePath}/{str(track)}/action/{step_cnt:03d}.npy", {k: v[idx] for k, v in robot_states.items()})
        cv2.imwrite(f"{FilePath}/{str(track)}/image/{step_cnt:03d}.png", imgs[idx])
        depths[idx].save(f"{FilePath}/{str(track)}/depth/{step_cnt:03d}.png")

        with open(f"{FilePath}/{track}/info.csv", 'w', newline='') as fn:
            writer = csv.writer(fn)
            # print("remote_cmd",remote_cmd)
            cmd_bh = str(remote_cmd['body_height_cmd'][idx])
            # cmd_ga = gait[idx]
            cmd_ga = str(remote_cmd['gait '][idx])
            cmd_fh = str(remote_cmd['footswing_height_cmd'][idx])
            cmd_p = str(remote_cmd['pitch_cmd'][idx])
            cmd_r = str(remote_cmd['roll_cmd'][idx])
            cmd_sw = str(remote_cmd['stance_width_cmd'][idx])
            cmd_v = f'{vel[idx]:.3f}'
            cmd_sf = str(freq[idx])
            success_info = str(success[idx])
            obj_num = task_cfg['num']
            tar_obj = ' '.join(task_cfg['color']+task_cfg['object'])
            other_obj = [f'{c} {o}' for c, o in zip(color[idx], name[idx])]
            if task_cfg['action'] == 'crawl':
                instruction = task_cfg['action'] + f', {speed_level[round(vel[idx]*2)-1]}'
            else:
                instruction = task_names[task_cfg['action']].replace('_', ' ').replace('OBJ', tar_obj) + f', {speed_level[round(vel[idx]*2)-1]}'

            header = ['urdf', 'instruction', 'traj_len',
                      'body_height_cmd', 'gait', 'step freq', 'footswing_height_cmd', 'pitch_cmd', 'roll_cmd', 'stance_width_cmd',
                      'speed', 'obj_num', 'tar_obj', 'other_obj','is_success']
            data = ['wr2'] + [instruction, str(success_len[idx]),
                              cmd_bh, cmd_ga, cmd_sf, cmd_fh, cmd_p, cmd_r, cmd_sw,
                              cmd_v, obj_num, tar_obj,other_obj] +[success_info]
            # import pdb; pdb.set_trace()
            writer.writerows([header, data])

    ps = []
    print('success:',success)
    for idx in range(len(success)):
        files = os.listdir(FilePath)
        if step_cnt == 0:
            track = f'{len(files):06d}'
            os.mkdir(f"{FilePath}/{track}")
        else:
            # track = f'{len(files) - len(success) + idx:06d}'
            track=f'{idx:06d}'
            
        p1 = threading.Thread(target=save, args=(step_cnt, idx, track))
        p1.start()
        ps.append(p1)

    for pi in ps:
        p1.join()
    

if __name__ == '__main__':
    import sys
    play_go1(sys.argv)