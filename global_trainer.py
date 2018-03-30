import os.path as osp
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from mpi4py import MPI
import moviepy.editor as mpy
import tqdm

from baselines.common import zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam

import dataset


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


class GlobalTrainer(object):
    def __init__(self, name, env, runner, policy, config):
        self._name = name
        self._env = env.unwrapped
        self._runner = runner
        self._config = config
        self._policy = policy

        # global step
        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
        self._update_global_step = tf.assign(self.global_step, self.global_step + 1)

        # tensorboard summary
        self._time_str = time.strftime("%y-%m-%d_%H-%M-%S")
        self._is_chef = (MPI.COMM_WORLD.Get_rank() == 0)
        self.summary_name = []

        # build loss/optimizers
        self._build_distillation()

    def _build_distillation(self):
        config = self._config
        pi = self._policy

        # input placeholders
        ac = pi.pdtype.sample_placeholder([None])
        var_list = [v for v in pi.get_trainable_variables() if 'vf' not in v.name]
        self._adam = MpiAdam(var_list)
        fetch_dict = {'loss': tf.reduce_mean(pi.pd.neglogp(ac))}
        if self._is_chef:
            self.summary_name += ['global/' + key for key in fetch_dict.keys()]
            self.summary_name += ['global/grad_norm', 'global/global_norm']
        fetch_dict['g'] = U.flatgrad(fetch_dict['loss'], var_list, clip_norm=config.global_max_grad_norm)
        self._loss = U.function([ac] + pi.obs, fetch_dict)
        self._global_norm = U.function([], tf.global_norm([tf.cast(var, tf.float32) for var in pi.get_variables()]))

        # initialize and sync
        U.initialize()
        self._adam.sync()

    def update(self, step, ob, ac):
        info = defaultdict(list)
        config = self._config
        sess = U.get_session()
        global_step = sess.run(self.global_step)

        # backup checkpoint
        self.summary(step)

        sess.run(self._update_global_step)

        pi = self._policy
        ob_dict = self._env.get_ob_dict(ob)
        if self._config.obs_norm:
            for ob_name in pi.ob_type:
                pi.ob_rms[ob_name].update(ob_dict[ob_name])

        for (mb_ob, mb_ac) in dataset.iterbatches(
            (ob, ac), include_final_partial_batch=False, batch_size=64, shuffle=True):
            ob_list = pi.get_ob_list(mb_ob)
            fetched = self._loss(mb_ac, *ob_list)
            loss, g = fetched['loss'], fetched['g']
            self._adam.update(g, self._config.global_stepsize)

            info['global/loss'].append(np.mean(loss))
            info['global/grad_norm'].append(np.linalg.norm(g))

        for key, value in info.items():
            info[key] = np.mean(value)
        info['global/global_norm'] = self._global_norm()
        return info

    def summary(self, it):
        if not self._is_chef or it % self._config.ckpt_save_step > 0:
            return

        # save checkpoint
        fname = osp.join(self._config.log_dir, '%.5d' % it)
        U.save_state(fname)

        # record video
        if self._config.training_video_record:
            self.record(it)

    def record(self, ckpt_num=None):
        config = self._config

        ep_lens = []
        ep_rets = []

        record_dir = osp.join(config.log_dir, 'video')
        os.makedirs(record_dir, exist_ok=True)

        for _ in tqdm.trange(10):
            ep_traj = self._runner.rollout(True, True)
            ep_lens.append(ep_traj["ep_length"][0])
            ep_rets.append(ep_traj["ep_reward"][0])
            logger.log('Trial #{}: lengths {}, returns {}'.format(_, ep_traj["ep_length"][0], ep_traj["ep_reward"][0]))

            # Video recording
            visual_obs = ep_traj["visual_ob"]
            video_name = (config.video_prefix or '') + '{}{}.mp4'.format(
                '' if ckpt_num is None else 'ckpt_{}_'.format(ckpt_num), _)
            video_path = osp.join(record_dir, video_name)
            fps = 60.

            def f(t):
                frame_length = len(visual_obs)
                new_fps = 1./(1./fps + 1./frame_length)
                idx = min(int(t*new_fps), frame_length-1)
                return visual_obs[idx]
            video = mpy.VideoClip(f, duration=len(visual_obs)/fps+2)
            video.write_videofile(video_path, fps, verbose=False)

        logger.log('Episode Length: {}'.format(sum(ep_lens) / 10.))
        logger.log('Episode Rewards: {}'.format(sum(ep_rets) / 10.))
