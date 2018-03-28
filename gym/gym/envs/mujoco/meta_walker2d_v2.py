import numpy as np

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import ipdb

possible_x = np.array([3, 8, 20, 25, 35, 42])

def random_obstacles(curbs_x, ceils_x):
    idx = np.random.permutation(np.arange(6))
    curbs_x = possible_x[idx[:4]] + np.random.rand(4)*2
    ceils_x = possible_x[idx[4:]] + np.random.rand(2)*2
    curbs_x = np.sort(curbs_x)
    ceils_x = np.sort(ceils_x)
    return curbs_x, ceils_x

class MetaWalker2d_v2_Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "random_steps": 0,
                "obs_randomness": 0,
                "x_vel_weight": 1,
                "alive_reward": 0.1,
                "ctrl_weight": 1e-3,
                "time_penalty": 0,
                "pass_bonus": 100,
                "falldown_penalty": 100,
                "collision_penalty": 10
            }
        self._obstacles = None
        self._curb_stage = 0
        self._num_curbs = 0
        self._ceil_stage = 0
        self._num_ceils = 0
        self.pass_state = [False] * (self._num_curbs + self._num_ceils)
        mujoco_env.MujocoEnv.__init__(self, "meta_walker2d_v2.xml", 4)
        utils.EzPickle.__init__(self)

        # get sub_policy observation space
        #sub_policy_obs = self.get_sub_policy_observation()
        sub_policy_obs_dim = self.observation_space.shape[0] - 4
        high = np.inf*np.ones(sub_policy_obs_dim)
        low = -high
        self.sub_policy_observation_space = spaces.Box(low, high)
        self.reward_type = ["alive_reward", "ctrl_reward", "x_vel_reward",
                            "pass_obstacles_bonus", "falldown_penalty", "collision_penalty"]

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs:
            # only consider the current observation
            if len(obs.shape) > 1:
                n = obs.shape[0]
                temp = np.reshape(obs, [n, self.stacked_obs, -1])
                sub_obs = temp[:, -1, :-4]
            else:
                temp = np.reshape(obs, [self.stacked_obs, -1])
                sub_obs = temp[-1, :-4]
            return sub_obs
        else:
            return obs[:-4]

    def pad_obs_tf(self, obs, pad):
        import tensorflow as tf
        if self.stacked_obs:
            # only consider the current observation
            temp = tf.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[:, :-4]
            padding = tf.ones_like(temp[:, -4:]) * pad
            sub_obs = tf.concat([sub_obs, padding], -1)
            sub_obs = tf.reshape(sub_obs, [-1])
            return sub_obs
        else:
            return obs[:-4]

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _get_obstacles_observation(self):
        if self._obstacles is None:
            self._put_obstacles()
        x_agent = self.get_body_com("torso")[0]
        ## Curbs
        self._curb_stage = 0
        # where is the next curb
        for (pos, size) in zip(self._obstacles['curbs']['pos'], self._obstacles['curbs']['size']):
            if x_agent > pos[0] + size[0] + 1.5:
                self._curb_stage += 1
            else:
                break
        # return the next curb
        if self._curb_stage >= self._num_curbs:
            curb_obs = (40, 40)
        else:
            curb_start = self._obstacles['curbs']['pos'][self._curb_stage][0] - \
                            self._obstacles['curbs']['size'][self._curb_stage][0]
            curb_end = self._obstacles['curbs']['pos'][self._curb_stage][0] + \
                            self._obstacles['curbs']['size'][self._curb_stage][0]
            curb_obs = (curb_start - x_agent, curb_end - x_agent)

        ## Ceilings
        self._ceil_stage = 0
        # where is the next ceil
        for (pos, size) in zip(self._obstacles['ceils']['pos'], self._obstacles['ceils']['size']):
            if x_agent > pos[0] + size[0]:
                self._ceil_stage += 1
            else:
                break
        # return the next ceil
        if self._ceil_stage >= self._num_ceils:
            ceil_obs = (40, 40)
        else:
            ceil_start = self._obstacles['ceils']['pos'][self._ceil_stage][0] - \
                            self._obstacles['ceils']['size'][self._ceil_stage][0]
            ceil_end = self._obstacles['ceils']['pos'][self._ceil_stage][0] + \
                            self._obstacles['ceils']['size'][self._ceil_stage][0]
            ceil_obs = (ceil_start - x_agent, ceil_end - x_agent)
        return curb_obs + ceil_obs

    def _step(self, a):

        posbefore = self.model.data.qpos[0, 0]
        stagebefore = self._curb_stage + self._ceil_stage

        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        ob = self._get_obs()
        done = not(height > 0.3)
        stageafter = self._curb_stage + self._ceil_stage

        x_vel_reward = self._config["x_vel_weight"] * (posafter - posbefore) / self.dt
        alive_reward = self._config["alive_reward"]
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()
        falldown_penalty = 0
        if done:
            falldown_penalty = -1 * self._config["falldown_penalty"]


        pass_obstacles_bonus = 0
        if (stagebefore != stageafter) and (stagebefore < (self._num_curbs + self._num_ceils)) and \
                (not self.pass_state[stagebefore]):
            pass_obstacles_bonus = self._config["pass_bonus"]
            self.pass_state[stagebefore] = True

        is_collision = (self.collision_detection('curb') or self.collision_detection('ceil'))

        collision_penalty = 0
        if is_collision:
            collision_penalty = -1 * self._config["collision_penalty"]

        reward = alive_reward + ctrl_reward + x_vel_reward - \
                    self._config["time_penalty"] + pass_obstacles_bonus + falldown_penalty + collision_penalty

        if self.viewer is not None:
            self.viewer_setup()

        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "x_vel_reward": x_vel_reward,
                "pass_obstacles_bonus": pass_obstacles_bonus,
                "falldown_penalty": falldown_penalty,
                "collision_penalty": collision_penalty}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        obstacle_obs = np.clip(self._get_obstacles_observation(), -10, 10)
        obs =  np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        return np.concatenate([obs, obstacle_obs])

    def reset_model(self):
        self._obstacles = None
        self._num_curbs = 0
        self._num_ceils = 0
        self._put_obstacles()
        self.pass_state = [False] * (self._num_curbs + self._num_ceils)
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"])):
             self._step(self.unwrapped.action_space.sample())
        ob = self._get_obs()
        return ob

    def viewer_setup(self):
        geom_name_list = self.model.geom_names
        torso_idx = np.where([name == b'torso_geom' for name in geom_name_list])[0][0]
        geom_pos = self.model.data.geom_xpos[torso_idx]
        # distance, elevation, azimuth
        cam_pos = np.array([geom_pos[0], geom_pos[1]-3, geom_pos[2], 12, -10, 90])
        self.unwrapped._set_cam_position(self.viewer, cam_pos, -1)

    def _put_obstacles(self):
        geom_name_list = self.model.geom_names
        if self._obstacles is None:
            self._obstacles = {'curbs': None, 'ceils': None}
            curbs_x = np.array([3, 20, 35, 42])
            ceils_x = np.array([8, 25])
            if self._config["obs_randomness"] != 0:
                curbs_x, ceils_x = random_obstacles(curbs_x, ceils_x)

            # Curbs
            self._obstacles['curbs'] = {'pos': [], 'size': []}
            self._num_curbs = len(np.where([b'curb' in name for name in geom_name_list])[0])
            for i in range(self._num_curbs):
                idx = np.where([name.decode() == 'curb{}'.format(i+1) for name in geom_name_list])[0][0]
                pos = self.model.geom_pos.copy()
                pos[idx][0] = curbs_x[i]
                self.model.geom_pos = pos
                self._obstacles['curbs']['pos'].append(self.model.geom_pos[idx])
                self._obstacles['curbs']['size'].append(self.model.geom_size[idx])

            # Ceils
            self._obstacles['ceils'] = {'pos': [], 'size': []}
            self._num_ceils = len(np.where([b'ceiling' in name for name in geom_name_list])[0])
            for i in range(self._num_ceils):
                idx = np.where([name.decode() == 'ceiling{}'.format(i+1) for name in geom_name_list])[0][0]
                pos = self.model.geom_pos.copy()
                pos[idx][0] = ceils_x[i]
                self.model.geom_pos = pos
                self._obstacles['ceils']['pos'].append(self.model.geom_pos[idx])
                self._obstacles['ceils']['size'].append(self.model.geom_size[idx])
