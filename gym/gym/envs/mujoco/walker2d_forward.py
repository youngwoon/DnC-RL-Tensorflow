import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env
import ipdb


def mass_center(model):
    return model.data.qpos[0]
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                            'got {}.'.format(value_at_1))
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                                'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x*scale)**2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1/value_at_1)
        return 1 / np.cosh(x*scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1/value_at_1 - 1)
        return 1 / ((x*scale)**2 + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2*value_at_1 - 1) / np.pi
        scaled_x = x*scale
        return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi*scaled_x))/2, 0.0)

    elif sigmoid == 'linear':
        scale = 1-value_at_1
        scaled_x = x*scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1-value_at_1)
        scaled_x = x*scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1-value_at_1))
        return 1 - np.tanh(x*scale)**2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value


class Walker2dForwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._config = {
            "random_steps": 0,
            "applied_force": 0,
            "alive_reward": 1,
            "x_vel_weight": 1,
            "angle_reward": 1,
            "height_reward": 1,
            "ctrl_weight": 1e-3,
            "deepmind": 0,
            "x_vel_limit": 100,
        }
        self._vel = 0

        mujoco_env.MujocoEnv.__init__(self, "walker2d_new.xml", 4)
        utils.EzPickle.__init__(self)

        # get sub_policy observation space
        self.sub_policy_observation_space = self.observation_space

        self.reward_type = ["x_vel_reward", "ctrl_reward", "nz", "delta_h",
                            "angle_reward", "height_reward", "stand_reward"]

        self.ob_shape = {"joint": [17]}
        self.ob_type = self.ob_shape.keys()

    def convert_ob_to_sub_ob(self, obs):
        if self.stacked_obs is not None:
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
        if self.action_space is not None:
            self.do_simulation(self.action_space.sample(), self.frame_skip)

        qfrc[idx] = 0
        self.model.data.qfrc_applied = qfrc

    def get_body_vel(self):
        return self._vel

    def _step(self, a):
        posbefore = mass_center(self.model)[0]
        self.do_simulation(a, self.frame_skip)
        posafter = mass_center(self.model)[0]

        if self._config["applied_force"] > 0 and random.random() < 0.1:
            self._apply_external_force()

        height = self.model.data.qpos[1, 0]
        delta_h = self.model.data.xpos[1, 2] - max(self.model.data.xpos[4, 2], self.model.data.xpos[7, 2])
        nz = np.cos(self.model.data.qpos[2])[0]
        self._vel = (posafter - posbefore) / self.dt

        if self._config["deepmind"] > 0:
            height_reward = tolerance(height,
                                      bounds=(1.0, float('inf')),
                                      margin=0.5)
            angle_reward = (1 + nz) / 2
            stand_reward = (3 * height_reward + angle_reward) / 4
            x_vel_reward = tolerance(self._vel,
                                     bounds=(8, float('inf')),
                                     margin=4,
                                     value_at_margin=0.5,
                                     sigmoid='linear')
            reward = stand_reward * (5 * x_vel_reward + 1) / 6
            ctrl_reward = 0
            done = False
        else:
            vel = min(self._vel, self._config["x_vel_limit"])
            x_vel_reward = self._config["x_vel_weight"] * vel
            angle_reward = self._config["angle_reward"] * nz
            #height_reward = -self._config["height_reward"] * np.abs(delta_h - 1.2)
            height_reward = -self._config["height_reward"] * (1.0 - min(1.0, delta_h))
            ctrl_reward = -self._config["ctrl_weight"] * np.square(a).sum()
            stand_reward = self._config["alive_reward"]

            reward = x_vel_reward + angle_reward + height_reward + \
                ctrl_reward + stand_reward
            done = height < 0.5

        ob = self._get_obs()
        info = {"x_vel_reward": x_vel_reward,
                "ctrl_reward": ctrl_reward,
                "angle_reward": angle_reward,
                "height_reward": height_reward,
                "stand_reward": stand_reward,
                "delta_h": delta_h,
                "nz": nz}
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        for _ in range(int(self._config["random_steps"])):
            self._step(self.action_space.sample())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 1.
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 60
