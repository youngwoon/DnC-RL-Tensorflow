import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import ipdb

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.stacked_obs = None
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.timestep = 0
        self.ego_viewer = None
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        self._bounds = bounds
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def add_perturb_to_current_state(self, level=.005):
        qpos = self.model.data.qpos[:, 0]
        qvel = self.model.data.qvel[:, 0]
        self.set_state(qpos + self.np_random.uniform(low=-1*level, high=level, size=self.model.nq), 
                        qvel + self.np_random.uniform(low=-1*level, high=level, size=self.model.nv))

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        #ctrl = np.clip(ctrl, self.model.actuator_ctrlrange.copy()[:, 0], self.model.actuator_ctrlrange.copy()[:, 1])
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.timestep += 1
            self.model.step()

    def get_visual_observation(self):
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()
            #self._get_ego_viewer().loop_once()

    def _set_cam_position(self, viewer, cam_pos, trackid):
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = trackid

    def _get_ego_viewer(self):
        if self.ego_viewer is None:
            self.ego_viewer = mujoco_py.MjViewer(visible=True)
            self.ego_viewer.start()
            self.ego_viewer.set_model(self.model)
        geom_name_list = self.model.geom_names
        torso_idx = np.where([name == b'torso_geom' for name in geom_name_list])[0][0]
        geom_pos = self.model.data.geom_xpos[torso_idx]
        cam_pos = np.array([geom_pos[0]+1, geom_pos[1], geom_pos[2], 2.5, 0, 0])
        self._set_cam_position(self.ego_viewer, cam_pos, trackid=-1)
        return self.ego_viewer

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    def collision_detection(self, ref_name=None, body_name=None):
        assert ref_name is not None
        mjcontacts = self.model.data._wrapped.contents.contact
        ncon = self.model.data.ncon
        collision = False
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 , g2 = ct.geom1, ct.geom2
            g1 = self.model.geom_names[g1]
            g2 = self.model.geom_names[g2]
            if body_name is not None:
                if (g1.find(six.b(ref_name)) >= 0 or g2.find(six.b(ref_name)) >= 0) and \
                    (g1.find(six.b(body_name)) >= 0 or g2.find(six.b(body_name)) >= 0):
                    collision = True
                    break
            else:
                if (g1.find(six.b(ref_name)) >= 0 or g2.find(six.b(ref_name)) >= 0):
                    collision = True
                    break
        return collision
