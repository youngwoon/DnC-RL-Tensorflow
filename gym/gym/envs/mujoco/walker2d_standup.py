import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb


def mass_center(model):
    mass = model.body_mass[:-1]
    xpos = model.data.xipos[:-1]
    # print(mass.shape)
    # print(xpos.shape)
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class Walker2dStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "standup_reward": 5,
                "init_force": 10,
                "random_steps": 200,
                "applied_force": 10,
                "ctrl_weight": 1e-3,
                "dont_move_weight": 0.5, 
                "required_height": 1,
            }
        self._vel = 0
        self._init_x = 0
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        # get sub_policy observation space
        self.sub_policy_observation_space = self.observation_space
        self.reward_type = ["standup_reward", "ctrl_reward", "dont_move_reward"]

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs is not None:
            # only consider the current observation
            temp = np.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[-1, :]
            return sub_obs
        else:
            return obs

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _apply_external_force(self, verbose=False):
        # randomly applied some force
        # if apply some torcs -> use self.model.data.xfrc_appied
        qfrc = self.model.data.qfrc_applied.copy()
        idx = np.random.randint(0, len(qfrc))
        ### initial stage: do large perturbation
        if self.timestep < 5 * self._config["random_steps"]:
            force = np.random.rand(1)*3 + self._config["init_force"]
        else:
            force = self._config["applied_force"]
        if verbose:
            print("Apply force: {}".format(force))

        qfrc[idx] = force
        self.model.data.qfrc_applied = qfrc

    def get_body_vel(self):
        return self._vel

    def _step(self, a):

        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        if random.random() < 0.1:
            self._apply_external_force()

        self._vel = (posafter - posbefore) / self.dt
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()
        standup_reward = 0
        if height > self._config["required_height"] and ang > -0.3 and ang < 0.3:
            standup_reward = self._config["standup_reward"]
        dont_move_reward = -1 * self._config["dont_move_weight"] * \
                    (np.max([np.abs(self._init_x - mass_center(self.model)[0]), 1]) - 1)

        reward = standup_reward + ctrl_reward + dont_move_reward
        done = False
        ob = self._get_obs()
        info = {"standup_reward": standup_reward,
                "ctrl_reward": ctrl_reward, 
                "dont_move_reward": dont_move_reward}

        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"])):
            if random.random() < 0.1:
                self._apply_external_force()
            self._step(self.unwrapped.action_space.sample())
        self._init_x = mass_center(self.model)[0]
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
