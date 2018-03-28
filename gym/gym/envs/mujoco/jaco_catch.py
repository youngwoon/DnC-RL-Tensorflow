import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoCatchEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "catch_reward": 1,
            "stable_reward": 10,
            "random_throw": 1,
        })

        # state
        self._catch_count = 0
        self._init_box_pos = np.asarray([0.5, 0.2, 0.1])

        # env info
        self.reward_type += ["catch_reward", "success", "stable_reward"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_catch.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        catch_reward = 0
        stable_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_hand = self._get_distance_hand('box')
        box_z = self._get_box_pos()[2]
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist_hand < 0.06

        # catch
        if in_air and in_hand:
            if self._catch_count == 0:
                self._init_box_pos = self._get_box_pos().copy()
                self._init_box_pos[2] = 0.2
            self._catch_count += 1
            catch_reward = self._config["catch_reward"]

        # success
        if self._catch_count == 150:
            done = True
            success = True
            print('success!')
            stable_reward = max(0, self._config["stable_reward"] - np.abs(self.model.data.qvel).mean())
            # stable_reward += max(0, self._config["stable_reward"] * (1 - np.linalg.norm(self._init_box_pos - self._get_box_pos())))
            stable_reward += max(0, self._config["stable_reward"] * (
                1 - np.linalg.norm(self._init_box_pos_above - self._get_box_pos())))

        # fail
        if on_ground:
            done = True

        reward = ctrl_reward + catch_reward + stable_reward
        info = {"ctrl_reward": ctrl_reward,
                "catch_reward": catch_reward,
                "stable_reward": stable_reward,
                "success": success}
        return ob, reward, done, info

    def _throw_box(self):
        # set initial force
        box_pos = self._get_box_pos()
        jaco_pos = self._get_pos('jaco_link_base')
        dx = 0.4 + np.random.uniform(0, 0.1) * self._config["random_throw"]
        dy = 0.3 + np.random.uniform(0, 0.1) * self._config["random_throw"]
        force = jaco_pos + [dx, dy, 1] - box_pos
        force = 110 * (force / np.linalg.norm(force))

        # apply force
        box_body_idx = np.where([name == b'box' for name in self.model.body_names])[0][0]
        xfrc = self.model.data.xfrc_applied.copy()
        xfrc[box_body_idx, :3] = force
        self.model.data.xfrc_applied = xfrc
        self.do_simulation(self.action_space.sample(), self.frame_skip)

        # reset force
        xfrc[box_body_idx] = 0
        self.model.data.xfrc_applied = xfrc

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        if self._with_rot == 0:
            qpos = qpos[:12]
            qvel = qvel[:12]
        return np.concatenate([qpos, np.clip(qvel, -30, 30)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        # set box's initial position
        qpos[9:12] = np.asarray([0, 2.0, 1.5])
        self.set_state(qpos, qvel)

        self._init_box_pos_above = [0.5, 0.2, 0.2]

        self._catch_count = 0

        self._throw_box()
        return self._get_obs()
