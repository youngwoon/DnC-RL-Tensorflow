import random
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
import ipdb

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class HumanoidHurdleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self._config = {
                "random_steps": 0,
                "applied_force": 0,
                "x_vel_weight": 5,
                "side_penalty_weight": 0.1, 
                "ctrl_weight": 0.02,
                "y_vel_weight": 0.1,
            }
        self._vel = 0
        self.reward_type = ["x_vel_reward", "ctrl_cost", "agent_height", "y_vel_reward", 
                            "side_reward", "delta_h", "upside_down_reward"]
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_hurdle.xml', 5)
        utils.EzPickle.__init__(self)
        sub_policy_obs_dim = self.observation_space.shape[0]
        high = np.inf*np.ones(sub_policy_obs_dim)
        low = -high
        self.sub_policy_observation_space = spaces.Box(low, high)

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat,
                               self.extra_obs.flat])


    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs:
            # only consider the current observation
            temp = np.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[-1, :-6]
            return sub_obs
        else:
            return obs
 
    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _apply_external_force(self, verbose=False):
        # randomly applied some force
        # if apply some torcs -> use self.model.data.xfrc_appied
        qfrc = self.model.data.qfrc_applied.copy()
        idx = np.random.randint(0, len(qfrc))

        if self._config["applied_force"] > 10:
            # if applied force is larger than 10, random force between 3 ~ 6
            force = np.random.rand(1)*3 + 3
        else:
            force = self._config["applied_force"]
        if verbose:
            print("Apply force: {}".format(force))

        qfrc[idx] = force
        self.model.data.qfrc_applied = qfrc


    def _put_hurdle(self, init=False):
        agent_pos = mass_center(self.model)
        geom_pos = self.model.geom_pos.copy()
        geom_size = self.model.geom_size.copy()
        if init or not hasattr(self, "extra_obs"):
            idx = np.where(['hurdle0' in name.decode() for name in self.model.geom_names])[0][0]
            pos_x = 3
            height_z = np.random.rand(1) * 0.5 + 0.1
            geom_pos[idx][0] = pos_x
            geom_size[idx][2] = height_z
            prev_pos_x = pos_x
            for i in range(1, 4):
                idx = np.where(['hurdle{}'.format(i) in name.decode() for name in self.model.geom_names])[0][0]
                height_z = np.random.rand(1) * 0.5 + 0.1
                gap = np.random.rand(1) + 4
                pos_x = prev_pos_x + gap
                geom_pos[idx][0] = pos_x 
                geom_size[idx][2] = height_z
                prev_pos_x = pos_x
        else:
            # check where is the agent
            dist = []
            for i in range(4):
                idx = np.where(['hurdle{}'.format(i) in name.decode() for name in self.model.geom_names])[0][0]
                dist.append((geom_pos[idx][0] + geom_size[idx][0] + 1) - agent_pos[0])

            dist = np.array(dist)
            # if need to change the hurdle (pass one hurdle)
            if np.min(dist) < -1:
                # change the extra observation
                change_idx = np.argmin(np.array(dist))
                change_idx = np.where(['hurdle{}'.format(change_idx) in name.decode() for name in self.model.geom_names])[0][0]
                last_idx = np.argmax(np.array(dist))
                last_idx = np.where(['hurdle{}'.format(last_idx) in name.decode() for name in self.model.geom_names])[0][0]
                height_z = np.random.rand(1) * 0.5 + 0.1
                gap = np.random.rand(1) + 4
                geom_pos[change_idx][0] = geom_pos[last_idx][0] + gap
                geom_size[change_idx][2] = height_z
            
        # set the hurdle position and size
        self.model.geom_pos = geom_pos
        self.model.geom_size = geom_size
        self.extra_obs = self._get_extra_obs()

    def _get_extra_obs(self):
        front_end = []; back_end = []; height = [];
        agent_pos = mass_center(self.model)
        geom_pos = self.model.geom_pos
        geom_size = self.model.geom_size
        for i in range(4):
            idx = np.where(['hurdle{}'.format(i) in name.decode() for name in self.model.geom_names])[0][0]
            front_end.append(geom_pos[idx][0] - geom_size[idx][0] - agent_pos[0])
            back_end.append(geom_pos[idx][0] + geom_size[idx][0] - agent_pos[0])
            height.append(geom_size[idx][2])

        front_end = np.array(front_end)
        back_end = np.array(back_end)
        height = np.array(height)
        sort_idx = np.argsort(front_end)
        front_end = front_end[sort_idx]
        back_end = back_end[sort_idx]
        height = height[sort_idx]
        
        extra_obs = np.concatenate([front_end[:2], back_end[:2], height[:2]], axis=0)
        return extra_obs

    def get_body_vel(self):
        return self._vel

    def _step(self, a):

        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        # check the hurdle
        self._put_hurdle()
        if random.random() < 0.1:
            self._apply_external_force()
        pos_after = mass_center(self.model)

        self._vel = (pos_after[0] - pos_before[0]) / self.model.opt.timestep
        x_vel_reward = -1 * self._config["x_vel_weight"] * min(self._vel, 20)
        y_vel = (pos_after[1] - pos_before[1]) / self.model.opt.timestep
        side_reward = -1 * self._config["side_penalty_weight"] * (self._vel ** 2 + y_vel ** 2)
        y_vel_reward = -1 * self._config["y_vel_weight"] * (y_vel ** 2)
        delta_h = self.model.data.xpos[1, 2] - (self.model.data.xpos[6, 2] + self.model.data.xpos[9, 2])/2

        agent_height = self.model.data.xpos[1][2]
        ctrl_cost = -1 * self._config["ctrl_weight"] * np.square(a).sum()

        upside_down_reward = -10 if delta_h < 0.5 else 0
        reward = x_vel_reward + ctrl_cost + y_vel_reward + side_reward + 0.02 + np.abs(delta_h - 1.2) + upside_down_reward

        height = pos_after[2]
        done = bool(height < 0.4)
        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward, 
                "ctrl_cost": ctrl_cost,
                "agent_height": agent_height,
                "y_vel_reward": y_vel_reward, 
                "side_reward": side_reward,
                "delta_h": delta_h,
                "upside_down_reward": upside_down_reward
                }

        return ob, reward, done, info

    def reset_model(self):
        self._put_hurdle(init=True)
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        for _ in range(int(self._config["random_steps"])):
            self._step(self.unwrapped.action_space.sample())
 
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.7
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 60
