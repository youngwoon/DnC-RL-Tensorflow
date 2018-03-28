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


class Walker2dBackNForthEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
            "random_steps": 0,
            "applied_force": 0,
            "alive_reward": 1,
            "x_vel_weight": 3,
            "angle_reward": 0.1,
            "height_reward": 0.1,
            "ctrl_weight": 1e-3,
            "distant": 3,
            "x_vel_limit": 100,
        }
        self._vel = 0
        self._direction = 1

        #self.ob_shape = {"joint": [17], "pos_x": [1], "target_x": [1]}
        self.ob_shape = {"joint": [17], "target": [1]}
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "walker2d_new.xml", 4)
        utils.EzPickle.__init__(self)

        # get sub_policy observation space
        self.sub_policy_observation_space = self.observation_space

        self.reward_type = ["x_vel_reward", "ctrl_reward", "nz", "delta_h", "success",
                            "angle_reward", "height_reward", "stand_reward"]

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

    def get_body_vel(self):
        return self._vel

    def _step(self, a):
        #self.set_state(np.array([0, 1.25, 0, 0, 0, 0, 0, 0, 0]), self.init_qvel)
        posbefore = mass_center(self.model)[0]
        self.do_simulation(a, self.frame_skip)
        posafter = mass_center(self.model)[0]

        if self._config["applied_force"] > 0 and random.random() < 0.1:
            self._apply_external_force()

        height = self.model.data.qpos[1, 0]
        #delta_h = self.model.data.xpos[1, 2] - (self.model.data.xpos[4, 2] + self.model.data.xpos[7, 2])/2
        delta_h = self.model.data.xpos[1, 2] - max(self.model.data.xpos[4, 2], self.model.data.xpos[7, 2])
        nz = np.cos(self.model.data.qpos[2])[0]
        self._vel = (posafter - posbefore) / self.dt

        success = False
        if self.model.data.xpos[1, 0] > self._config["distant"] and self._direction > 0:
            print("change direction to backward")
            self._direction = -1
            success = True
        elif self.model.data.xpos[1, 0] < (-1*self._config["distant"]) and self._direction < 0:
            print("change direction to forward")
            self._direction = 1
            success = True

        # clip velocity for reward
        vel = min(self._vel, self._config["x_vel_limit"])
        vel = max(self._vel, -self._config["x_vel_limit"])
        x_vel_reward = self._direction * self._config["x_vel_weight"] * vel
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
                "nz": nz,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        target = self._config["distant"] * (-1 if self._direction < 0 else 1)
        target = np.array(target)
        #return np.concatenate([qpos[1:], np.clip(qvel, -10, 10),
        #    self.model.data.xpos[1, 0][np.newaxis, np.newaxis], target[np.newaxis, np.newaxis]]).ravel()
        rel_distance = target - qpos[0]
        rel_distance = np.clip(rel_distance, -5, 5)[0]
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10),
            rel_distance[np.newaxis, np.newaxis]]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :-1],
                'target': ob[:, -1:],
            }
        return {
            'joint': ob[:-1],
            'target': ob[-1:],
        }
        # if len(ob.shape) > 1:
        #     return {
        #         'joint': ob[:, :-2],
        #         'pos_x': ob[:, -2:-1],
        #         'target_x': ob[:, -1:],
        #     }
        # return {
        #     'joint': ob[:-2],
        #     'pos_x': ob[-2:-1],
        #     'target_x': ob[-1:],
        # }

    def reset_model(self):
        self._direction = 1
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qpos[0] = -self._config["distant"] + np.random.uniform(-0.1, 0.1)
        self.set_state(qpos, qvel)

        # set start/end position
        start_idx = np.where([name == b'hurdle0' for name in self.model.geom_names])[0][0]
        end_idx = np.where([name == b'hurdle1' for name in self.model.geom_names])[0][0]
        geom_pos = self.model.geom_pos.copy()
        geom_pos[start_idx][0] = self._config["distant"]
        geom_pos[start_idx][1] = -1.75
        geom_pos[end_idx][0] = -self._config["distant"]
        geom_pos[end_idx][1] = -1.75
        self.model.geom_pos = geom_pos
        for _ in range(int(self._config["random_steps"])):
            self._step(self.unwrapped.action_space.sample())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 1.
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 60
