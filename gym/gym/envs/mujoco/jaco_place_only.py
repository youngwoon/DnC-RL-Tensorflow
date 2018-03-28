import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoPlaceOnlyEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "pick_reward": 100,
            "place_reward": 10,
            "success_reward": 100,
            "random_box": 0,
            "random_target": 0.1,
            "max_penalized_dist_ratio": 2.,
        })

        # state
        self._init_dist = np.inf
        self._t = 0
        self._picked = False
        self._pick_height = 0

        # env info
        self.reward_type += ["pick_reward", "place_reward", "success"]
        self.ob_shape.update({"target": [3]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_place.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        success_reward = 0
        place_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        box_z = self._get_box_pos()[2]
        dist_hand = self._get_distance_hand('box')
        dist_target = self._get_distance('box', 'target')
        on_ground = box_z < 0.035
        in_hand = dist_hand < 0.06

        # success
        if (on_ground and self._picked) or self._t == 100:
            done = True
            t_pos = self._get_target_pos()
            b_pos = self._get_box_pos()
            if self._picked and on_ground and abs(t_pos[0] - b_pos[0]) < 0.01 and abs(t_pos[1] - b_pos[1]) < 0.01 and abs(t_pos[2] - b_pos[2]) < 0.01:
                success = True
                success_reward = self._config["success_reward"]
                print('success!')
            if self._picked:
                place_reward = self._config["place_reward"] * (1 - min(
                    self._config["max_penalized_dist_ratio"], dist_target / self._init_dist))
                place_reward += self._config["place_reward"] * (1 - 10 * min(0.1, abs(t_pos[0] - b_pos[0]) + abs(t_pos[1] - b_pos[1])))

        # pick
        if in_hand and self._pick_height < min(0.2, box_z):
            pick_reward = self._config["pick_reward"] * (min(0.2, box_z) - self._pick_height)
            self._picked = True
            self._pick_height = box_z

        reward = ctrl_reward + pick_reward + place_reward + success_reward
        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "place_reward": place_reward,
                "success": success}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        target = self._get_target_pos()[:, np.newaxis]
        if self._with_rot == 0:
            qpos = qpos[:12]
            qvel = qvel[:12]
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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        # set box's initial position
        qpos[9] += np.random.uniform(-1, 1) * self._config["random_box"]
        qpos[10] += np.random.uniform(-1, 1) * self._config["random_box"]

        # initialize starting holding a box
        qpos_given = np.array([
            2.89388635e-01, 2.72388642e-01, -1.24199542e+00, -6.78454740e+00,
            1.14690654e+00, 9.01185527e+00, 6.29379506e-01, 6.99998244e-01,
            1.94140063e-04, 5.39575345e-01, 4.14449440e-01, 2.98347830e-01,
            -6.58797992e-01, -5.69180844e-01, 4.73323475e-01, -1.34101678e-01])
        qvel_given = np.array([
            2.56205361, -0.15591239, 2.52564981, -0.17614805,
            0.19007417, 7.4852305, -0.49073489, 0.01527384,
            0.1008432, -1.23376498, 1.41688187, -0.39876351,
            -1.24939007, 6.92172794, -0.46676828])
        self.set_state(qpos_given, qvel_given)

        # set target's initial position
        target_idx = np.where([name == b'target' for name in self.model.body_names])[0][0]
        body_pos = self.model.body_pos.copy()
        body_pos[target_idx][0:2] = [0.2, 0.5]
        body_pos[target_idx][0] += np.random.uniform(-1, 1) * self._config["random_target"]
        body_pos[target_idx][1] += np.random.uniform(-1, 1) * self._config["random_target"]
        self.model.body_pos = body_pos

        self._pick_height = 0
        self._t = 0
        self._picked = False
        self._init_dist = self._get_distance('box', 'target')
        return self._get_obs()
