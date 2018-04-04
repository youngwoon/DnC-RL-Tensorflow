import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class JacoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, with_rot=1):
        # config
        self._with_rot = with_rot
        self._config = {"ctrl_reward": 1e-4}

        # env info
        self.reward_type = ["ctrl_reward"]
        self.ob_shape = {"joint": [31]}
        if self._with_rot == 0:
            self.ob_shape["joint"] = [24]  # 4 for orientation, 3 for velocity

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _ctrl_reward(self, a):
        ctrl_reward = -self._config["ctrl_reward"] * np.square(a).sum()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.model.data.qvel).mean()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.model.data.qacc).mean()
        return ctrl_reward

    # get absolute coordinate
    def _get_pos(self, name):
        geom_idx = np.where([key.decode() == name for key in self.model.geom_names])
        if len(geom_idx[0]) > 0:
            return self.model.data.geom_xpos[geom_idx[0][0]]
        body_idx = np.where([key.decode() == name for key in self.model.body_names])
        if len(body_idx[0]) > 0:
            return self.model.body_pos[body_idx[0][0]]
        raise ValueError

    def _get_box_pos(self):
        return self._get_pos('box')

    def _get_target_pos(self):
        return self._get_pos('target')

    def _get_hand_pos(self):
        hand_pos = np.mean([self._get_pos(name) for name in [
            'jaco_link_hand', 'jaco_link_finger_1',
            'jaco_link_finger_2', 'jaco_link_finger_3']], 0)
        return hand_pos

    def _get_distance(self, name1, name2):
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_distance_hand(self, name):
        pos = self._get_pos(name)
        hand_pos = self._get_hand_pos()
        return np.linalg.norm(pos - hand_pos)

    def viewer_setup(self):
        #self.viewer.cam.trackbodyid = 1
        self.viewer.cam.trackbodyid = -1
        #self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.distance = 2
        #self.viewer.cam.azimuth = 260
        self.viewer.cam.azimuth = 170
        #self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[0] = 0.5
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20
