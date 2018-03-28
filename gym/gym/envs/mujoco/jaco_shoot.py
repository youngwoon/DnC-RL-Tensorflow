import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.jaco import JacoEnv


class JacoShootEnv(JacoEnv):
    def __init__(self, with_rot=1):
        super().__init__(with_rot=with_rot)
        self._config.update({
            "air_reward": 1,
            "pick_reward": 100,
            "hold_reward": 0,
            "hold_reward_duration": 60,
            "shoot_reward": 40,
            "success_reward": 100,
            "random_box": 0,
            "random_target": 0,
        })

        # state
        self._init_dist = np.inf
        self._air_count = 0
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._pick_height = 0

        # env info
        self.reward_type += ["shoot_reward", "pick_reward", "hold_reward",
                             "success", "air_reward"]
        self.ob_shape.update({"target": [3]})
        self.ob_type = self.ob_shape.keys()

        mujoco_env.MujocoEnv.__init__(self, "jaco_shoot.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        air_reward = 0
        hold_reward = 0
        success_reward = 0
        shoot_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        box_z = self._get_box_pos()[2]
        dist_hand = self._get_distance_hand('box')
        dist_target = self._get_distance('box', 'target')
        in_air = box_z > 0.06
        on_ground = box_z < 0.06
        in_hand = dist_hand < 0.06

        # pick
        if in_air and not in_hand and self._picked: self._air_count += 1

        if in_hand and self._pick_height < min(0.2, box_z):
            pick_reward = self._config["pick_reward"] * (min(0.2, box_z) - self._pick_height)
            self._picked = True
            self._pick_height = box_z

        if in_hand and self._hold_duration < self._config['hold_reward_duration']:
            self._hold_duration += 1
            hold_reward = self._config["hold_reward"]

        # success or fail
        if self._t == 150 or (on_ground and self._air_count > 0):
            done = True
            t_pos = self._get_target_pos()
            b_pos = self._get_box_pos()
            if self._picked and on_ground and abs(t_pos[0] - b_pos[0]) < 0.06 and abs(t_pos[1] - b_pos[1]) < 0.06:
                success = True
                success_reward = self._config["success_reward"]
                print('success!')
            if self._picked and on_ground and self._air_count > 0:
                air_reward = self._config["air_reward"] * self._air_count
                shoot_reward = self._config["shoot_reward"] * (1 - min(1., dist_target / self._init_dist))
                air_reward = min(air_reward, 2 * shoot_reward)

        reward = ctrl_reward + pick_reward + hold_reward + \
            shoot_reward + air_reward + success_reward
        info = {"ctrl_reward": ctrl_reward,
                "pick_reward": pick_reward,
                "hold_reward": hold_reward,
                "shoot_reward": shoot_reward,
                "air_reward": air_reward,
                "success": success}
        return ob, reward, done, info

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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        # set box's initial position
        qpos[9] += np.random.uniform(-1, 1) * self._config["random_box"]
        qpos[10] += np.random.uniform(-1, 1) * self._config["random_box"]
        self.set_state(qpos, qvel)

        # set target's initial position
        target_idx = np.where([name == b'target' for name in self.model.body_names])[0][0]
        body_pos = self.model.body_pos.copy()
        body_pos[target_idx][0:2] = [1.3, 0]
        body_pos[target_idx][0] += np.random.uniform(-1, 1) * self._config["random_target"]
        body_pos[target_idx][1] += np.random.uniform(-1, 1) * self._config["random_target"]
        self.model.body_pos = body_pos

        self._air_count = 0
        self._pick_height = 0
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._init_dist = self._get_distance('box', 'target')
        return self._get_obs()
