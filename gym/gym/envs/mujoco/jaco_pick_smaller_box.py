import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoPickSmallerBoxEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "pick_reward": 10,
            "stable_reward": 10,
            "repeat": 0,
            "random_box": 1,
        })

        # state
        self._pick_count = 0
        self._init_box_pos = np.asarray([0.5, 0.2, 0.03])

        # env info
        self.reward_type += ["pick_reward", "success", "stable_reward"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick_smaller_box.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        stable_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_hand = self._get_distance_hand('box')
        box_z = self._get_box_pos()[2]
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist_hand < 0.06

        # pick
        if in_air and in_hand:
            pick_reward = self._config["pick_reward"] * min(0.2, box_z)
            self._pick_count += 1

        # fail
        if on_ground and self._pick_count > 0:
            done = True

        # success
        if self._pick_count == 50:
            success = True
            print('success')
            self._pick_count += 1  # to count success only once
            stable_reward = max(0, self._config["stable_reward"] - np.abs(self.model.data.qvel).mean())
            stable_reward += max(0, self._config["stable_reward"] * (
                1 - np.linalg.norm(self._init_box_pos_above - self._get_box_pos())))
            if self._config["repeat"] > 0:
                self.reset_box()
            else:
                done = True

        # print(self.model.data.qpos)
        # print(self.model.data.qvel)

        reward = ctrl_reward + pick_reward + stable_reward
        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "stable_reward": stable_reward,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos, np.clip(qvel, -30, 30)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_box(self):
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()

        # set box's initial position
        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.2 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.01])
        qpos[9:12] = self._init_box_pos

        # teh desired location after picking
        self._init_box_pos_above = self._init_box_pos
        self._init_box_pos_above[-1] = 0.2

        qpos[12:16] = self.init_qpos[12:16] + self.np_random.uniform(low=-.005, high=.005, size=4)
        qvel[9:15] = self.init_qvel[9:15] + self.np_random.uniform(low=-.005, high=.005, size=6)
        self.set_state(qpos, qvel)

        self._pick_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        return self._get_obs()
