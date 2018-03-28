import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


# pick or catch -> shoot
class MetaJaco_v2_Env(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "air_reward": 1,
            "pick_reward": 2,
            "catch_reward": 2,
            "shoot_reward": 40,
            "dense_shoot": 0,
            "success_reward": 100,
            "random_box": 0,
            "random_target": 0,
            "random_throw": 0,
            "wait": 30,
            "pick_ratio": 0.5,
        })

        # state
        self._t = 0
        self._ep_t = 0
        self._is_catch = False
        self._catched = False
        self._picked = False
        self._air_count = 0
        self._hold_count = 0

        # env info
        self.reward_type += ["shoot_reward", "catch_reward", "air_reward", "pick_reward", "success"]
        self.ob_shape.update({"target": [3]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_shoot.xml", 4)
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
        shoot_reward = 0
        success_reward = 0
        air_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        box_z = self._get_box_pos()[2]
        dist = self._get_distance_hand('box')
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist < 0.06

        # shoot
        if in_air and not in_hand and (self._picked or self._catched): self._air_count += 1

        if self._is_catch:
            # catch
            if in_hand:
                if self._hold_count < 10:
                    catch_reward = self._config["catch_reward"]
                self._hold_count += 1
            # success
            if in_hand and not self._catched:
                self._init_dist = self._get_distance('box', 'target')
                self._catched = True
                print('success catch')
            # fail
            if on_ground:
                reset = True
        else:
            # pick
            if not on_ground and in_hand:
                if self._hold_count < 10:
                    pick_reward = self._config["pick_reward"]
                self._hold_count += 1
            # success
            if not on_ground and in_hand and not self._picked:
                self._init_dist = self._get_distance('box', 'target')
                self._picked = True
                print('success pick')
            # fail
            if on_ground and self._picked:
                reset = True

        # fail
        if self._t == 200 + self._config['wait']:
            reset = True

        if on_ground and (self._catched or self._picked):
            reset = True
            dist_target = self._get_distance('box', 'target')
            shoot_reward = self._config["shoot_reward"] * (1 - min(1., dist_target / self._init_dist))
            air_reward = self._config["air_reward"] * self._air_count
            air_reward = min(air_reward, 2 * shoot_reward)
            air_reward *= self._config["dense_shoot"]
            shoot_reward *= self._config["dense_shoot"]

            t_pos = self._get_target_pos()
            b_pos = self._get_box_pos()
            if abs(t_pos[0] - b_pos[0]) < 0.06 and abs(t_pos[1] - b_pos[1]) < 0.06:
                success = True
                success_reward = self._config["success_reward"]
                print('success shoot')

        if reset:
            #self.reset_box()
            #self.reset_model()
            done = True

        reward = ctrl_reward + shoot_reward + pick_reward + catch_reward + air_reward + success_reward
        info = {"ctrl_reward": ctrl_reward,
                "shoot_reward": shoot_reward,
                "pick_reward": pick_reward,
                "catch_reward": catch_reward,
                "air_reward": air_reward,
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
        if self._with_rot == 0:
            qpos = qpos[:12]
            qvel = qvel[:12]
        target = self._get_target_pos()[:, np.newaxis]
        return np.concatenate([qpos, np.clip(qvel, -30, 30), target]).ravel()

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :-3],
                'target': ob[:, -3:],
            }
        return {
            'joint': ob[:-3],
            'target': ob[-3:],
        }

    def reset_box(self):
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
        self._catched = False
        self._picked = False
        self._air_count = 0
        self._hold_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        # set target's initial position
        target_idx = np.where([name == b'target' for name in self.model.body_names])[0][0]
        body_pos = self.model.body_pos.copy()
        body_pos[target_idx][0:2] = [1.3, 0]
        body_pos[target_idx][0] += np.random.uniform(-1, 1) * self._config["random_target"]
        body_pos[target_idx][1] += np.random.uniform(-1, 1) * self._config["random_target"]
        self.model.body_pos = body_pos

        self.reset_box()
        self._ep_t = 0
        return self._get_obs()
