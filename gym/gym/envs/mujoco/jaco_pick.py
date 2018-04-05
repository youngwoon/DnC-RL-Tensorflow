import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoPickEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "pick_reward": 100,
            "random_box": 1,
        })
        self._context = None

        # state
        self._pick_count = 0
        self._init_box_pos = np.asarray([0.5, 0.0, 0.03])

        # env info
        self.reward_type += ["pick_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def set_context(self, context):
        self._context = context

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_hand = self._get_distance_hand('box')
        box_z = self._get_box_pos()[2]
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist_hand < 0.08

        # pick
        if in_air and in_hand:
            pick_reward = self._config["pick_reward"] * box_z
            self._pick_count += 1

        # fail
        if on_ground and self._pick_count > 0:
            done = True

        # success
        if self._pick_count == 30:
            success = True
            done = True
            print('success')

        reward = ctrl_reward + pick_reward
        info = {"ctrl_reward_sum": ctrl_reward,
                "pick_reward_sum": pick_reward,
                "success_sum": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos, np.clip(qvel, -10, 10)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_box(self):
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()

        # set box's initial position
        sx, sy, ex, ey = -0.15, -0.15, 0.15, 0.15
        if self._context == 0:
            sx, sy = 0, 0
        elif self._context == 1:
            ex, sy = 0, 0
        elif self._context == 2:
            sx, ey = 0, 0
        elif self._context == 3:
            ex, ey = 0, 0

        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(sx, ex) * self._config["random_box"],
             0.15 + np.random.uniform(sy, ey) * self._config["random_box"],
             0.03])
        qpos[9:12] = self._init_box_pos

        self.set_state(qpos, qvel)

        self._pick_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        return self._get_obs()
