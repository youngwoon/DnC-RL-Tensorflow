import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import ipdb

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

def cosine_distance(x, y):
    a = np.sum(np.dot(x, y))
    def norm(vec):
        return np.sqrt(np.sum(np.square(vec)))
    b = norm(x) * norm(y)
    if b == 0.:
        return 0
    else:
        return a/b

class HumanoidCrawlEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
                "ceiling_reward": 2, 
                "side_weight": 0.5,
                "x_vel_weight": 0.25 
            }
        self._init_direction = np.array([0, 0, 0])
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_ceiling.xml', 5)
        utils.EzPickle.__init__(self)
        self._init_direction = self._get_direction()

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _get_direction(self):
        geom_name_list = self.model.geom_names
        head_idx = np.where([name == b'head' for name in geom_name_list])[0][0]
        nose_idx = np.where([name == b'nose' for name in geom_name_list])[0][0]
        head = self.model.geom_pos[head_idx]
        nose = self.model.geom_pos[nose_idx]
        direction = nose - head
        return direction
    
    # Done -> never terminate
    # Add ceiling reward: once the agent's hieght part is over the threshold -> penalty
    def _step(self, a):
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model)

        # ceiling reward
        geom_name_list = self.model.geom_names
        ceiling_idx = np.where([name == b'ceiling' for name in geom_name_list])[0][0]
        geom_pos = self.model.geom_pos.copy()
        ceiling_hieght = geom_pos[ceiling_idx][2]
        if pos_after[2] > (ceiling_hieght*0.7):
            ceiling_reward = -1*self._config["ceiling_reward"]
        else:
            ceiling_reward = 0

        side_reward = self._config["side_weight"] * np.abs(pos_after[1])
        data = self.model.data
        lin_vel_cost = self._config["x_vel_weight"] * (pos_after[0] - pos_before[0]) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost - side_reward + ceiling_reward
        qpos = self.model.data.qpos
        #done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        done = False
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, 
                     ceiling_reward=ceiling_reward, 
                     reward_quadctrl=-quad_ctrl_cost, 
                     reward_impact=-quad_impact_cost, 
                     side_reward=-side_reward)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -5
