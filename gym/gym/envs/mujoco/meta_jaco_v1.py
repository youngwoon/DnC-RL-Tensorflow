import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


# Pick-and-catch, pick-and-pick, catch-and-catch
class MetaJaco_v1_Env(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "pick_reward": 10,
            "catch_reward": 10,
            "close_reward": 0,
            "random_box": 0,
            "random_throw": 0,
            "wait": 30,
            "pick_ratio": 0.5,
            "curriculum": 0,
        })

        # state
        self._t = 0
        self._air_count = 0
        self._is_catch = False
        self._ep_t = 0
        self._ep_count = 0

        # env info
        self.reward_type += ["shoot_reward", "catch_reward", "pick_reward", "close_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_catch.xml", 4)
        utils.EzPickle.__init__(self)

        self._ep_t = 0

    def _step(self, a):
        self._t += 1
        self._ep_t += 1
        if self._is_catch:
            if self._ep_t == 1 or (self._ep_t != self._t and self._t == self._config["wait"]):
                self._throw_box()
            elif self._ep_t != self._t and self._t < self._config["wait"]:
                self._set_box()

        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        reset = False
        pick_reward = 0
        catch_reward = 0
        close_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        box_z = self._get_box_pos()[2]
        dist = self._get_distance_hand('box')
        in_hand = dist < 0.06
        on_ground = box_z < 0.06
        in_air = box_z > 0.06

        # fail
        if self._is_catch and on_ground:
            done = True
            reset = True
            close_reward = -self._config["close_reward"] * dist

        # pick or catch
        if in_hand and in_air:
            self._air_count += 1
            if self._is_catch:
                catch_reward = self._config["catch_reward"]
            else:
                pick_reward = self._config["pick_reward"]

        # success or fail
        if self._t == 100 + self._config["wait"] or self._air_count >= 20:
            reset = True
            close_reward = -self._config["close_reward"] * dist
            if in_hand and self._air_count >= 20:
                success = True
                if self._is_catch:
                    print('success catch')
                else:
                    print('success pick')
            else:
                done = True

        if reset:
            self.reset_box()

        reward = ctrl_reward + pick_reward + catch_reward + close_reward
        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "catch_reward": catch_reward,
                "close_reward": close_reward,
                "success": success}
        return ob, reward, done, info

    def _set_box(self):
        # set box's initial position
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()
        qpos[9:16] = self.qpos_box
        qvel[9:15] = self.qvel_box
        self.set_state(qpos, qvel)

    def _throw_box(self):
        # set box's initial position
        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()
        qpos[9:16] = self.qpos_box
        qvel[9:15] = self.qvel_box
        self.set_state(qpos, qvel)

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
        return np.concatenate([qpos, np.clip(qvel, -30, 30)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_box(self):
        if self._ep_count < self._config['curriculum']:
            # pick-and-pick or catch-and-catch
            pass
        elif self._ep_count < self._config['curriculum'] * 5:
            # alternative
            self._is_catch = not self._is_catch
        else:
            # random
            self._is_catch = (np.random.rand(1) > self._config['pick_ratio'])[0]

        qpos = self.model.data.qpos.ravel().copy()
        qvel = self.model.data.qvel.ravel().copy()

        # set box's initial position
        if self._is_catch:
            qpos[9:12] = np.asarray([0, 2.0, 1.5])
        else:
            qpos[9:12] = [0.5 + np.random.uniform(0, 0.1) * self._config["random_box"],
                          0.2 + np.random.uniform(0, 0.1) * self._config["random_box"],
                          0.03]
        qpos[12:16] = self.init_qpos[12:16] + self.np_random.uniform(low=-.005, high=.005, size=4)
        qvel[9:15] = self.init_qvel[9:15] + self.np_random.uniform(low=-.005, high=.005, size=6)

        self.qpos_box = qpos[9:16]
        self.qvel_box = qvel[9:15]
        self.set_state(qpos, qvel)

        self._t = 0
        self._air_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self._ep_count < self._config['curriculum']:
            self._is_catch = (np.random.rand(1) > self._config['pick_ratio'])[0]
        self.reset_box()
        self._ep_t = 0
        self._ep_count += 1
        return self._get_obs()
