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

class MetaHumanoid_v1_Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
                #"ceiling_reward": 2, 
                "side_weight": 0.5,
                "x_vel_weight": 0.25, 
                "time_penalty": 0.05, 
            }
        mujoco_env.MujocoEnv.__init__(self, 'meta_humanoid_v1.xml', 5)
        utils.EzPickle.__init__(self)

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _get_obs(self):
        data = self.model.data
        # add ceilign relative infomation to observation
        agent_pos_x = mass_center(self.model)[0]
        geom_name_list = self.model.geom_names
        ceiling_idx = np.where([name == b'ceiling' for name in geom_name_list])[0][0]
        ceiling_x = self.model.data.geom_xpos[ceiling_idx][0]
        ceiling_size_x = self.model.geom_size[ceiling_idx][0]
        ceiling_start = ceiling_x - ceiling_size_x
        ceiling_end = ceiling_x + ceiling_size_x
        ceiling_relative_start = ceiling_start - agent_pos_x
        ceiling_relative_end = ceiling_end - agent_pos_x
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat, 
                               ceiling_relative_start.flat, 
                               ceiling_relative_end.flat])

    # Done -> never terminate
    # reward -> the same as HumanoidFprward
    def _step(self, a):

        info = {}
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model)

        geom_name_list = self.model.geom_names
        ceiling_idx = np.where([name == b'ceiling' for name in geom_name_list])[0][0]
        ceiling_xyz = self.model.data.geom_xpos[ceiling_idx]
        ceiling_size = self.model.geom_size[ceiling_idx]
        '''
        if pos_after < (ceiling_xyz[0] - ceiling_size[0]):
            # first stage: forward
            pass    
        elif pos_after >= (ceiling_xyz[0] - ceiling_size[0]) and pos_after < (ceiling_xyz[0] + ceiling_size[0]) :
            # second state: crawling 
            geom_pos = self.model.geom_pos.copy()
            ceiling_hieght = geom_pos[ceiling_idx][2]
            if pos_after[2] > (ceiling_hieght*0.7):
                ceiling_reward = -1*self._config["ceiling_reward"]
            else:
                ceiling_reward = 0
            info.update({"ceiling_reward": ceiling_reward})
            reward += ceiling_reward
        else:
            # third stage: standup and forwarding
            pass
        '''

        # avoid deviate the route
        side_reward = self._config["side_weight"] * np.abs(pos_after[1])
        # going forward
        lin_vel_cost = self._config["x_vel_weight"] * (pos_after[0] - pos_before[0]) / self.model.opt.timestep
        # control
        quad_ctrl_cost = 0.1 * np.square(self.model.data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(self.model.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        # time penalty
        time_reward = -1 * self._config["time_penalty"]
        info.update({"side_reward": side_reward})
        info.update({"lin_vel_cost": lin_vel_cost})
        info.update({"quad_ctrl_cost": quad_ctrl_cost})
        info.update({"quad_impact_cost": quad_impact_cost})

        # total reward
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost - side_reward + time_reward
        done = pos_after[0] > (ceiling_xyz[0] + ceiling_size[0] + 5)
        return self._get_obs(), reward, done, info

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
