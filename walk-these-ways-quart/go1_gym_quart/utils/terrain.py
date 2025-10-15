# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

from go1_gym_quart.envs.base.legged_robot_config import Cfg


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        if self.eval_cfg is None:
            return self.cfg.row_indices, self.cfg.col_indices, [], []
        else:
            self._load_cfg(self.eval_cfg)
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
            self.eval_cfg.x_offset = self.cfg.tot_rows
            self.eval_cfg.rows_offset = self.cfg.num_rows
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if cfg.curriculum:
            self.curriculum(cfg)
        elif cfg.selected:
            self.selected_terrain(cfg)
        else:
            self.randomized_terrain(cfg)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            # if i == 1 and j == 1:
            #     step_height *= -1
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice < proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            pass
        elif choice < proportions[7]:
            pass
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0
        elif choice < proportions[10]:
            step_height *= -0.8
            self.pyramid_stairs_terrain_with_platform(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            # terrain_utils.stairs_terrain(terrain, step_width=0.31, step_height=step_height)
            # for k in range(cfg.num_sub_terrains):
            # # Env coordinates in the world
            #     (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            # # while(i != Cfg.terrain.num_rows and j != Cfg.terrain.num_rows):
            #     while i != Cfg.terrain.num_rows:
            #         i =+ 1
            #         while j != Cfg.terrain.num_rows:
            #             j =+ 1
            #             if (i == 1 and j == 1):
            #                 step_height *= -1
            #                 terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=1.)
            #             pass
                # step_height *= -1
                # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        return terrain
    
    def pyramid_stairs_terrain_with_platform(self, terrain, step_width, step_height, platform_size=1.0):
        """
        Generate stairs

        Parameters:
            terrain (terrain): the terrain
            step_width (float):  the width of the step [meters]
            step_height (float): the step_height [meters]
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        step_width = int(step_width / terrain.horizontal_scale)
        step_height = int(step_height / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        height = 0
        start_x = 10
        stop_x = terrain.width - 10
        start_y = 10
        stop_y = terrain.length - 10
        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            start_x += step_width
            stop_x -= step_width
            start_y += step_width
            stop_y -= step_width
            height += step_height
            terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
        return terrain
    
    def _step_height(self, cfg):
        for k in range(cfg.num_sub_terrains):
        # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
        # while(i != Cfg.terrain.num_rows and j != Cfg.terrain.num_rows):
            while i != Cfg.terrain.num_rows:
                i =+ 1
                while j != Cfg.terrain.num_rows:
                    j =+ 1
                    if (i == 1 and j == 1):
                        x = -1
                    else:
                        x = 0
        return x 
    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

