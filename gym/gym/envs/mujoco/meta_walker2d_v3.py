import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb


class MetaWalker2d_v3_Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._config = {
                "x_vel_weight": 1,
                "alive_reward": 1,
                "ctrl_weight": 1e-3,
                "time_penalty": 0.1,
            }
        self._curbs = None
        self._stage = 0
        mujoco_env.MujocoEnv.__init__(self, "meta_walker2d_v3.xml", 4)
        utils.EzPickle.__init__(self)

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _get_curb_observation(self):
        x_agent = self.get_body_com("torso")[0]
        geom_name_list = self.model.geom_names
        if self._curbs is None:
            self._curbs = {'pos': [], 'size': []}
            num_curbs = len(np.where([b'curb' in name for name in geom_name_list])[0])
            for i in range(num_curbs):
                idx = np.where([name.decode() == 'curb{}'.format(i+1) for name in geom_name_list])[0][0]
                self._curbs['pos'].append(self.model.geom_pos[idx])
                self._curbs['size'].append(self.model.geom_size[idx])
        self._stage = 0
        # where is the next curb
        for (pos, size) in zip(self._curbs['pos'], self._curbs['size']):
            if x_agent > pos[0] + size[0]:
                self._stage += 1
            else:
                break
        return self._curbs['pos'][self._stage], self._curbs['size'][self._stage]

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]

        x_vel_reward = self._config["x_vel_weight"] * (posafter - posbefore) / self.dt
        alive_reward = self._config["alive_reward"]
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()

        reward = alive_reward - ctrl_reward + x_vel_reward - self._config["time_penalty"]
        done = not (height > 0.5)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        pos, size = self._get_curb_observation()
        ipdb.set_trace()
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
