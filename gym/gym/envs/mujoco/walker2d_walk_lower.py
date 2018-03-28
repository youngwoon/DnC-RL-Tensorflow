import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb

def mass_center(model):
    mass = model.body_mass[:-1]
    xpos = model.data.xipos[:-1]
    return (np.sum(mass * xpos, 0) / np.sum(mass))[2]

class Walker2dLowerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "init_force": 0,
                "random_steps": 0,
                "applied_force": 0,
                "ceil_randomness": 0,
                "ceil_height": 0.8,
                "x_vel_weight": 1,
                "alive_reward": 0.1,
                "ctrl_weight": 1e-3,
                "collision_penalty": 10,
                "pass_bonus": 100,
            }
        self._ceils = None
        self._stage = 0
        self._num_ceils = 0
        self.pass_state = [False] * self._num_ceils
        self._vel = 0
        mujoco_env.MujocoEnv.__init__(self, "walker2d_lower.xml", 4)
        utils.EzPickle.__init__(self)
        # get sub_policy observation space
        self.sub_policy_observation_space = self.observation_space
        self.reward_type = ["x_vel_reward", "alive_reward", "ctrl_reward", "collision_penalty"]

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs is not None:
            # only consider the current observation
            temp = np.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[-1, :]
            return sub_obs
        else:
            return obs

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

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _get_ceil_observation(self):
        if self._ceils is None:
            self._put_ceils()
        x_agent = self.get_body_com("torso")[0]
        self._stage = 0
        # where is the next ceil
        for (pos, size) in zip(self._ceils['pos'], self._ceils['size']):
            if x_agent > pos[0] + size[0] + 1.5:
                self._stage += 1
            else:
                break
        if self._stage >= self._num_ceils:
            return (40, 40)
        else:
            ceil_start = self._ceils['pos'][self._stage][0] - self._ceils['size'][self._stage][0]
            ceil_end = self._ceils['pos'][self._stage][0] + self._ceils['size'][self._stage][0]
            return (ceil_start - x_agent, ceil_end - x_agent)

    def get_body_vel(self):
        return self._vel

    def _step(self, a):

        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        if random.random() < 0.1:
            self._apply_external_force()

        is_collision = self.collision_detection('ceiling')
        collision_penalty = 0
        if is_collision:
            collision_penalty = -1 * self._config["collision_penalty"]

        self._vel = (posafter - posbefore) / self.dt
        x_vel_reward = self._config["x_vel_weight"] * self._vel
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()
        alive_reward = self._config["alive_reward"]
        reward = alive_reward + ctrl_reward + collision_penalty + x_vel_reward

        ob = self._get_obs()

        done = False
        if self._ceils is not None:
            done = not (height > 0.3 and posafter < (self._ceils['pos'][0][0] + self._ceils['size'][0][0]))
        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "collision_penalty": collision_penalty,
                "x_vel_reward": x_vel_reward}

        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self._put_ceils()
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"]*self.unwrapped.frame_skip)):
            if random.random() < 0.1:
                self._apply_external_force()
            self._step(self.unwrapped.action_space.sample())
        return self._get_obs()

    def viewer_setup(self):
        geom_name_list = self.model.geom_names
        #torso_idx = np.where([name == b'torso_geom' for name in geom_name_list])[0][0]
        geom_pos = self.model.data.geom_xpos[2]
        # distance, elevation, azimuth
        cam_pos = np.array([geom_pos[0], geom_pos[1]-3, geom_pos[2], self.model.stat.extent * 0.5, -10, 90])
        self.unwrapped._set_cam_position(self.viewer, cam_pos, -1)


    def _put_ceils(self):
        geom_name_list = self.model.geom_names
        if self._ceils is None:
            self._ceils = {'pos': [], 'size': []}
            self._num_ceils = len(np.where([b'ceiling' in name for name in geom_name_list])[0])
            for i in range(self._num_ceils):
                idx = np.where([name.decode() == 'ceiling{}'.format(i+1) for name in geom_name_list])[0][0]
                if self._config["ceil_randomness"] != 0:
                    pos = self.model.geom_pos.copy()
                    pos[idx][0] = ceils_x[i] + np.random.rand(1)
                    pos[idx][2] = self._config["ceil_height"]
                    self.model.geom_pos = pos
                self._ceils['pos'].append(self.model.geom_pos[idx])
                self._ceils['size'].append(self.model.geom_size[idx])
