import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb


def mass_center(model):
    return model.data.qpos[0]
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class Walker2dBackwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
            "random_steps": 0,
            "applied_force": 0,
            "alive_reward": 1,
            "x_vel_weight": 1,
            "angle_reward": 1,
            "height_reward": 1,
            "ctrl_weight": 1e-3,
            "x_vel_limit": 100,
        }
        self._vel = 0

        mujoco_env.MujocoEnv.__init__(self, "walker2d_new.xml", 4)
        utils.EzPickle.__init__(self)

        # get sub_policy observation space
        self.sub_policy_observation_space = self.observation_space

        self.reward_type = ["x_vel_reward", "ctrl_reward", "nz", "delta_h",
                            "angle_reward", "height_reward", "stand_reward"]

        self.ob_shape = {"joint": [17]}
        self.ob_type = self.ob_shape.keys()

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
        if self._config["applied_force"] > 10:
            # if applied force is larger than 10, random force between 3 ~ 6
            force = np.random.rand(1)*3 + 3
        else:
            force = self._config["applied_force"]
        if verbose:
            print("Apply force: {}".format(force))

        qfrc[idx] = force
        self.model.data.qfrc_applied = qfrc
        if self.action_space is not None:
            self.do_simulation(self.action_space.sample(), self.frame_skip)

        qfrc[idx] = 0
        self.model.data.qfrc_applied = qfrc

    def get_body_vel(self):
        return self._vel

    def _step(self, a):
        posbefore = mass_center(self.model)[0]
        self.do_simulation(a, self.frame_skip)
        posafter = mass_center(self.model)[0]

        if self._config["applied_force"] > 0 and random.random() < 0.1:
            self._apply_external_force()

        height = self.model.data.qpos[1, 0]
        delta_h = self.model.data.xpos[1, 2] - max(self.model.data.xpos[4, 2], self.model.data.xpos[7, 2])
        nz = np.cos(self.model.data.qpos[2])[0]
        self._vel = (posafter - posbefore) / self.dt

        vel = max(self._vel, -self._config["x_vel_limit"])
        x_vel_reward = -self._config["x_vel_weight"] * vel
        angle_reward = self._config["angle_reward"] * nz
        #height_reward = -self._config["height_reward"] * np.abs(delta_h - 1.2)
        height_reward = -self._config["height_reward"] * (1.0 - min(1.0, delta_h))
        ctrl_reward = -self._config["ctrl_weight"] * np.square(a).sum()
        stand_reward = self._config["alive_reward"]

        reward = x_vel_reward + angle_reward + height_reward + \
            ctrl_reward + stand_reward
        done = height < 0.5

        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward,
                "ctrl_reward": ctrl_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "stand_reward": stand_reward,
                "delta_h": delta_h,
                "nz": nz}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"])):
            self._step(self.action_space.sample())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 1.
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 60
