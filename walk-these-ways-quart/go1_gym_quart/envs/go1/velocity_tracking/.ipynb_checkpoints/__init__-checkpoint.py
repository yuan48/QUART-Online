from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta
from typing import Union
import numpy as np

from go1_gym_quart.envs.base.legged_robot import LeggedRobot
from go1_gym_quart.envs.base.legged_robot_config import Cfg


def euler_from_quaternioneul(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = t2.clip(-1.,1.)
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x,pitch_y,yaw_z # in radians

class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)


    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras,self.imgs,self.depths = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        ###wangzt
        # quat = gymapi.Quat()
        # quat.x,quat.y,quat.z,quat.w = self.root_states[self.ids_sim['robot'], 3:7].detach().cpu().numpy()[0]
        # rpy = quat.to_euler_zyx()
        x,y,z,w = self.root_states[self.ids_sim['robot'], 3:7].detach().cpu().numpy().T
        roll,_,yaw = euler_from_quaternioneul(x,y,z,w)
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros_like(self.dof_vel).cpu().numpy(),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[self.ids_sim['robot'], 0:3].detach().cpu().numpy(),
            "body_quat": self.root_states[self.ids_sim['robot'], 3:7].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy(),
            "imgs": self.imgs,
            "depths": self.depths,
            "_target_xy": self.root_states[self.ids_sim['target'], 0:2].detach().cpu().numpy(),
            "_load_xy": self.root_states[self.ids_sim['load'], 0:2].detach().cpu().numpy(),
            "_obstacle_xy": self.root_states[self.ids_sim['obstacle'], 0:2].detach().cpu().numpy(),
            "_body_xy": self.root_states[self.ids_sim['robot'], 0:2].detach().cpu().numpy(),
            "_body_yaw": yaw,
            "_body_roll": roll,
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

