import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb

class AntCurbEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._curbs = None
        self._stage = 0
        mujoco_env.MujocoEnv.__init__(self, 'ant_curb.xml', 5)
        utils.EzPickle.__init__(self)
         
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
        for (pos, size) in zip(self._curbs['pos'], self._curbs['size']):
            if x_agent > pos[0]:
                self._stage += 1
            else:
                break
        return self._curbs['pos'][self._stage]

    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        prev_stage = self._stage
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        forward_reward = (xposafter - xposbefore)/self.dt
        stage_change_reward = 0.
        if self._stage > prev_stage:
            stage_change_reward = 10
        elif self._stage < prev_stage:
            stage_change_reward = -10

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward + stage_change_reward

        # Use the z-axis of torso to determine `done`
        # Use the y-axis of torso to determine `done`
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0 \
            and np.abs(yposafter) < 3
        done = not notdone
        return ob, reward, done, dict(
            target_reward=forward_reward, 
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward, 
            stage_change_reward=stage_change_reward)

    def _get_obs(self):
        curb_obs = self._get_curb_observation()
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            curb_obs.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 70
        self.viewer.cam.distance = self.model.stat.extent * 0.6
