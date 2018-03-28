import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb

class AntBackForthEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        # only supprot two goals
        self._target = np.zeros([3])
        # common init
        mujoco_env.MujocoEnv.__init__(self, 'ant_target.xml', 5)
        utils.EzPickle.__init__(self)

    def _reset_target(self, t0=False):
        geom_name_list = self.model.geom_names
        target_idx = np.where([name == b'target' for name in geom_name_list])[0][0]
        geom_pos = self.model.geom_pos.copy()
        # Random generate distance (-3~3)
        rand_x = np.random.rand(1)*3
        if t0:
            rand_x *= np.random.choice([-1, 1])
        else:
            rand_x *= np.sign(geom_pos[target_idx, 0])
        geom_pos[target_idx, 0] = rand_x
        self.model.geom_pos = geom_pos
        self._target = geom_pos[target_idx]
         
    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        # Calculate the distance between the agent and the current target
        diff = np.sum(np.square(self.get_body_com("torso") - self._target))
        # Give target reward
        if diff < 0.5:
            target_reward = 50
            self._reset_target()
        else:
            target_reward = 0

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = target_reward - ctrl_cost - contact_cost + survive_reward

        # Use the z-axis of torso to determine `done`
        # Use the y-axis of torso to determine `done`
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0 \
            and np.abs(yposafter) < 3
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            target_reward=target_reward, 
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # Add current target coordination to the observation space
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self._target.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self._reset_target(t0=True)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.5
