import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb

curbs_x = [1]

class Walker2dJumpEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
            "random_steps": 0,
            "curb_randomness": 0,
            "x_vel_weight": 0.5,
            "applied_force": 100,
            "alive_reward": 0.5,
            "curb_height": 0.4, # initial height
            "ctrl_weight": 1e-3,
            "collision_penalty": 10,
            "pass_bonus": 100
        }
        self._curbs = None
        self._stage = 0
        self._num_curbs = 0
        self.pass_state = [False] * self._num_curbs

        mujoco_env.MujocoEnv.__init__(self, "walker2d_jump.xml", 4)
        utils.EzPickle.__init__(self)

        self.sub_policy_observation_space = self.observation_space

        self.reward_type = ["x_vel_reward", "alive_reward", "ctrl_reward",
                            "collision_penalty", "pass_curb_bonus"]

        self.ob_shape = {"joint": [17]}
        self.ob_type = self.ob_shape.keys()

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs:
            # only consider the current observation
            temp = np.reshape(obs, [self.stacked_obs, -1])
            sub_obs = temp[-1, :]
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

    def get_body_vel(self):
        return self._vel

    def _get_curb_observation(self):
        if self._curbs is None:
            self._put_curbs()
        x_agent = self.get_body_com("torso")[0]
        self._stage = 0
        # where is the next curb
        for (pos, size) in zip(self._curbs['pos'], self._curbs['size']):
            if x_agent > pos[0] + size[0] + 1.5:
                self._stage += 1
            else:
                break
        if self._stage >= self._num_curbs:
            return (40, 40)
        else:
            curb_start = self._curbs['pos'][self._stage][0] - self._curbs['size'][self._stage][0]
            curb_end = self._curbs['pos'][self._stage][0] + self._curbs['size'][self._stage][0]
            return (curb_start - x_agent, curb_end - x_agent)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        stagebefore = self._stage
        self.do_simulation(a, self.frame_skip)
        self._get_curb_observation()
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        stageafter = self._stage

        if random.random() < 0.1:
            self._apply_external_force()

        pass_curb_bonus = 0
        if (stagebefore != stageafter) and (stagebefore < self._num_curbs) and \
                (not self.pass_state[stagebefore]):
            pass_curb_bonus = self._config["pass_bonus"]
            self.pass_state[stagebefore] = True

        is_collision = self.collision_detection('curb')
        collision_penalty = 0
        if is_collision:
            collision_penalty = -1 * self._config["collision_penalty"]

        self._vel = (posafter - posbefore) / self.dt
        x_vel_reward = self._config["x_vel_weight"] * self._vel
        ctrl_reward = -1 * self._config["ctrl_weight"] * np.square(a).sum()
        alive_reward = self._config["alive_reward"]
        reward = alive_reward + ctrl_reward + collision_penalty + pass_curb_bonus + x_vel_reward

        x_agent = self.get_body_com("torso")[0]
        is_success = False
        if x_agent > self._curbs['pos'][-1][0] + self._curbs['size'][-1][0] + 2.5:
            is_success = True
        done = height < 0.6 or is_success
        ob = self._get_obs()
        info = {"alive_reward": alive_reward,
                "ctrl_reward": ctrl_reward,
                "collision_penalty": collision_penalty,
                "pass_curb_bonus": pass_curb_bonus,
                "x_vel_reward": x_vel_reward}

        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_model(self):
        self._curbs = None
        self._num_curbs = 0
        self._put_curbs()
        self.pass_state = [False] * self._num_curbs
        self.set_state(
            #self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            #self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"])):
            self._step(self.unwrapped.action_space.sample())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def _put_curbs(self):
        geom_name_list = self.model.geom_names
        if self._curbs is None:
            self._curbs = {'pos': [], 'size': []}
            self._num_curbs = len(np.where([b'curb' in name for name in geom_name_list])[0])
            for i in range(self._num_curbs):
                idx = np.where([name.decode() == 'curb{}'.format(i+1) for name in geom_name_list])[0][0]
                if self._config["curb_randomness"] != 0:
                    pos = self.model.geom_pos.copy()
                    pos[idx][0] = curbs_x[i] + np.random.rand(1)
                    self.model.geom_pos = pos
                self._curbs['pos'].append(self.model.geom_pos[idx])
                size = self.model.geom_size.copy()
                size[idx][2] = self._config["curb_height"]
                self.model.geom_size = size
                self._curbs['size'].append(self.model.geom_size[idx])
