import random

import numpy as np

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import ipdb


curbs_x = [3, 14, 23, 35, 42]

class MetaWalker2d_v1_Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "random_steps": 0,
                "curb_randomness": 0,
                "x_vel_weight": 1,
                "alive_reward": 0.1,
                "ctrl_weight": 1e-3,
                "time_penalty": 0,
                "pass_bonus": 100,
                "falldown_penalty": 5,
                "collision_penalty": 10,
                "applied_force": 100,
            }
        self._curbs = None
        self._stage = 0
        self._num_curbs = 0
        self.pass_state = [False] * self._num_curbs
        mujoco_env.MujocoEnv.__init__(self, "meta_walker2d_v1.xml", 4)
        utils.EzPickle.__init__(self)

        # get sub_policy observation space
        #sub_policy_obs = self.get_sub_policy_observation()
        sub_policy_obs_dim = self.observation_space.shape[0] - 2
        high = np.inf*np.ones(sub_policy_obs_dim)
        low = -high
        self.sub_policy_observation_space = spaces.Box(low, high)
        self.reward_type = ["alive_reward", "ctrl_reward", "x_vel_reward",
                            "pass_curb_bonus", "falldown_penalty", "collision_penalty"]

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs:
            # only consider the current observation
            if len(obs.shape) > 1:
                n = obs.shape[0]
                temp = np.reshape(obs, [n, self.stacked_obs, -1])
                sub_obs = temp[:, -1, :-2]
            else:
                temp = np.reshape(obs, [self.stacked_obs, -1])
                sub_obs = temp[-1, :-2]
            return sub_obs
        else:
            return obs[:-2]

    def pad_obs_tf(self, obs, pad):
        import tensorflow as tf
        if self.stacked_obs:
            # only consider the current observation
            temp = tf.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[:, :-2]
            padding = tf.ones_like(temp[:, -2:]) * pad
            sub_obs = tf.concat([sub_obs, padding], -1)
            sub_obs = tf.reshape(sub_obs, [-1])
            return sub_obs
        else:
            return obs[:-2]

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _apply_external_force(self, verbose=False):
        # randomly applied some force
        # if apply some torcs -> use self.model.data.xfrc_appied
        qfrc = self.model.data.qfrc_applied.copy()
        idx = np.random.randint(0, len(qfrc))

        if self._config["applied_force"] > 10:
            # if applied force is larger than 10, random force between 3 ~ 6
            force = np.random.rand(1)*3 + 3
        else:
            force = self._config["applied_force"]
        if verbose:
            print("Apply force: {}".format(force))

        qfrc[idx] = force
        self.model.data.qfrc_applied = qfrc

    def _get_curb_observation(self):
        if self._curbs is None:
            self._put_curbs()
        x_agent = self.get_body_com("torso")[0]
        self._stage = 0
        # where is the next curb
        for (pos, size) in zip(self._curbs['pos'], self._curbs['size']):
            if x_agent > pos[0] + size[0] + 1.5:
                self._stage += 1
            else:
                break
        if self._stage >= self._num_curbs:
            return (40, 40)
        else:
            curb_start = self._curbs['pos'][self._stage][0] - self._curbs['size'][self._stage][0]
            curb_end = self._curbs['pos'][self._stage][0] + self._curbs['size'][self._stage][0]
            return (curb_start - x_agent, curb_end - x_agent)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        stagebefore = self._stage
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        if random.random() < 0.1:
            self._apply_external_force()
        ob = self._get_obs()
        done = False
        stageafter = self._stage

        x_vel_reward = self._config["x_vel_weight"] * (posafter - posbefore) / self.dt
        alive_reward = self._config["alive_reward"]
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()
        falldown_penalty = 0
        if height <= 0.3:
            falldown_penalty = -1 * self._config["falldown_penalty"]

        pass_curb_bonus = 0
        if (stagebefore != stageafter) and (stagebefore < self._num_curbs) and (not self.pass_state[stagebefore]):
            pass_curb_bonus = self._config["pass_bonus"]
            self.pass_state[stagebefore] = True

        is_collision = self.collision_detection('curb')
        collision_penalty = 0
        if is_collision:
            collision_penalty = -1 * self._config["collision_penalty"]

        reward = alive_reward + ctrl_reward + x_vel_reward - \
                    self._config["time_penalty"] + pass_curb_bonus + falldown_penalty + collision_penalty

        if self.viewer is not None:
            self.viewer_setup()
        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "x_vel_reward": x_vel_reward,
                "pass_curb_bonus": pass_curb_bonus,
                "falldown_penalty": falldown_penalty,
                "collision_penalty": collision_penalty}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        curb_obs = self._get_curb_observation()
        obs =  np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        return np.concatenate([obs, curb_obs])

    def reset_model(self):
        self._curbs = None
        self._num_curbs = 0
        self._put_curbs()
        self.pass_state = [False] * self._num_curbs
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
        cam_pos = np.array([geom_pos[0], geom_pos[1]-3, geom_pos[2], 12, -20, 90])
        self.unwrapped._set_cam_position(self.viewer, cam_pos, -1)

    def _put_curbs(self):
        geom_name_list = self.model.geom_names
        if self._curbs is None:
            self._curbs = {'pos': [], 'size': []}
            self._num_curbs = len(np.where([b'curb' in name for name in geom_name_list])[0])
            for i in range(self._num_curbs):
                idx = np.where([name.decode() == 'curb{}'.format(i+1) for name in geom_name_list])[0][0]
                if self._config["curb_randomness"] != 0:
                    pos = self.model.geom_pos.copy()
                    pos[idx][0] = curbs_x[i] + np.random.rand(1) * 2
                    self.model.geom_pos = pos
                else:
                    pos = self.model.geom_pos.copy()
                    pos[idx][0] = curbs_x[i]
                    self.model.geom_pos = pos
                self._curbs['pos'].append(self.model.geom_pos[idx])
                self._curbs['size'].append(self.model.geom_size[idx])
