import random
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import ipdb

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class HumanoidSwimForwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self._config = {
                "random_steps": 0,
                "applied_force": 0,
                "x_vel_weight": 1,
                "height_penalty_weight": 0.5, 
                "ctrl_weight": 1e-3,
            }
        self._vel = 0
        self.reward_type = ["x_vel_reward", "quad_ctrl_cost", "quad_impact_cost", "height_penalty"]
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_pool.xml', 5)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])


    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs:
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

    def get_body_vel(self):
        return self._vel

    def _step(self, a):

        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)

        if random.random() < 0.1:
            self._apply_external_force()
        pos_after = mass_center(self.model)
        # qpos: relative position
        init_height = self.model.qpos0[2]
        after_height = self.model.data.qpos[2]

        self._vel = (pos_after[0] - pos_before[0]) / self.model.opt.timestep
        height_penalty = -1 * self._config["height_penalty_weight"] * \
                        (np.max([np.abs(after_height - init_height)[0], 1]) - 1)
        x_vel_reward = self._config["x_vel_weight"] * self._vel
        data = self.model.data
        quad_ctrl_cost = -1 *0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = -1 * min(quad_impact_cost, 10)

        reward = x_vel_reward + quad_ctrl_cost + quad_impact_cost + height_penalty

        done = bool(np.abs(after_height - init_height) > 2)
        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward, 
                "quad_ctrl_cost": quad_ctrl_cost, 
                "quad_impact_cost": quad_impact_cost,
                "height_penalty": height_penalty
                }

        return ob, reward, done, info

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        for _ in range(int(self._config["random_steps"])):
            self._step(self.unwrapped.action_space.sample())
 
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -10
