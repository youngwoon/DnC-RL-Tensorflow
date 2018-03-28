import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb

class Walker2dBalanceEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "tolerance": 0.2,
                "balance_penalty": 1.5,
                "alive_reward": 1,
                "ctrl_weight": 1e-3
            }
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]

        torso_idx = np.where([name == b'torso_geom' for name in self.model.geom_names])[0][0]
        torso_xyz = self.model.data.xpos[torso_idx]
        init_torso = self.model.body_ipos[torso_idx]
        if torso_xyz[2] < (init_torso[2] - self._config["tolerance"]) or \
            torso_xyz[2] > (init_torso[2] + self._config["tolerance"]):
            torso_balance_reward = -1 * self._config["balance_penalty"]
        else:
            torso_balance_reward = 0
        alive_reward = self._config["alive_reward"]
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()

        reward = torso_balance_reward + alive_reward - ctrl_reward
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
