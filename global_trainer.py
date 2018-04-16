import os.path as osp
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import moviepy.editor as mpy
import tqdm
from contextlib import contextmanager
from mpi4py import MPI
import imageio

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import colorize
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

        self._is_chef = (MPI.COMM_WORLD.Get_rank() == 0)

        # global step
        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
        self._update_global_step = tf.assign(self.global_step, self.global_step + 1)

        # tensorboard summary
        self.summary_name = ['global/length', 'global/reward', 'global/success']

        # build loss/optimizers
        self._build_distillation()

    def _build_distillation(self):
        config = self._config
        pi = self._policy

        self._global_norm = U.function(
            [], tf.global_norm([tf.cast(var, tf.float32) for var in pi.get_variables()]))

        # policy update
        ac = pi.pdtype.sample_placeholder([None])
        pol_var_list = [v for v in pi.get_trainable_variables() if 'pol' in v.name]
        self._pol_adam = MpiAdam(pol_var_list)
        pol_loss = tf.reduce_mean(pi.pd.neglogp(ac))
        #pol_loss = tf.reduce_mean(tf.square(pi.pd.sample() - ac))
        fetch_dict = {
            'loss': pol_loss,
            'g': U.flatgrad(pol_loss, pol_var_list,
                            clip_norm=config.global_max_grad_norm)
        }
        self._pol_loss = U.function([ac] + pi.ob, fetch_dict)
        self.summary_name += ['global/loss', 'global/grad_norm', 'global/global_norm']

        # value update
        if config.global_vf:
            ret = tf.placeholder(dtype=tf.float32, shape=[None], name='return')
            vf_var_list = [v for v in pi.get_trainable_variables() if 'vf' in v.name]
            self._vf_adam = MpiAdam(vf_var_list)
            vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
            fetch_dict = {
                'vf_loss': vf_loss,
                'vf_g': U.flatgrad(vf_loss, vf_var_list,
                                   clip_norm=config.global_max_grad_norm)
            }
            self._vf_loss = U.function([ret] + pi.ob, fetch_dict)
            self.summary_name += ['global/vf_loss', 'global/vf_grad_norm']

        # initialize and sync
        U.initialize()
        self._pol_adam.sync()
        if config.global_vf:
            self._vf_adam.sync()
        if config.debug:
            logger.log("[worker: {} global] Init param sum".format(MPI.COMM_WORLD.Get_rank()), self._adam.getflat().sum())

    @contextmanager
    def timed(self, msg):
        if self._is_chef:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def update(self, step, ob, ac, ret=None):
        info = defaultdict(list)
        config = self._config
        sess = U.get_session()
        global_step = sess.run(self.global_step)

        sess.run(self._update_global_step)

        pi = self._policy
        ob_dict = self._env.get_ob_dict(ob)
        if self._config.obs_norm == 'learn':
            for ob_name in pi.ob_type:
                pi.ob_rms[ob_name].update(ob_dict[ob_name])

        with self.timed("update global network"):
            for _ in range(self._config.global_iters):
                # policy network
                for (mb_ob, mb_ac) in dataset.iterbatches(
                        (ob, ac), include_final_partial_batch=False,
                        batch_size=self._config.global_batch_size):
                    ob_list = pi.get_ob_list(mb_ob)
                    fetched = self._pol_loss(mb_ac, *ob_list)
                    loss, g = fetched['loss'], fetched['g']
                    self._pol_adam.update(g, self._config.global_stepsize)
                    info['global/loss'].append(np.mean(loss))
                    info['global/grad_norm'].append(np.linalg.norm(g))

                if config.global_vf:
                    # value network
                    for (mb_ob, mb_ret) in dataset.iterbatches(
                            (ob, ret), include_final_partial_batch=False,
                            batch_size=self._config.global_batch_size):
                        ob_list = pi.get_ob_list(mb_ob)
                        fetched = self._vf_loss(mb_ret, *ob_list)
                        vf_loss, vf_g = fetched['vf_loss'], fetched['vf_g']
                        self._vf_adam.update(vf_g, self._config.global_stepsize)
                        info['global/vf_loss'].append(np.mean(vf_loss))
                        info['global/vf_grad_norm'].append(np.linalg.norm(vf_g))
        for key, value in info.items():
            info[key] = np.mean(value)
        info['global/global_norm'] = self._global_norm()
        return info

    def summary(self, it):
        info = self.evaluate(it, record=self._config.training_video_record)

        # save checkpoint
        if it % self._config.ckpt_save_step == 0:
            fname = osp.join(self._config.log_dir, '%.5d' % it)
            U.save_state(fname)

        return info

    def evaluate(self, ckpt_num=None, record=False):
        config = self._config

        ep_lens = []
        ep_rets = []
        ep_success = []

        if record:
            record_dir = osp.join(config.log_dir, 'video')
            os.makedirs(record_dir, exist_ok=True)

        for _ in tqdm.trange(10):
            ep_traj = self._runner.rollout(True, True)
            ep_lens.append(ep_traj["ep_length"][0])
            ep_rets.append(ep_traj["ep_reward"][0])
            ep_success.append(ep_traj["ep_success"][0])
            logger.log('[{}] Trial #{}: lengths {}, returns {}'.format(
                self._name, _, ep_traj["ep_length"][0], ep_traj["ep_reward"][0]))

            # Video recording
            if record:
                visual_obs = ep_traj["visual_ob"]
                video_name = '{}{}_{}{}.{}'.format(config.video_prefix or '', self._name,
                    '' if ckpt_num is None else 'ckpt_{}_'.format(ckpt_num), _, config.video_format)
                video_path = osp.join(record_dir, video_name)

                if config.video_format == 'mp4':
                    fps = 60.

                    def f(t):
                        frame_length = len(visual_obs)
                        new_fps = 1./(1./fps + 1./frame_length)
                        idx = min(int(t*new_fps), frame_length-1)
                        return visual_obs[idx]
                    video = mpy.VideoClip(f, duration=len(visual_obs)/fps+2)
                    video.write_videofile(video_path, fps, verbose=False)
                elif config.video_format == 'gif':
                    imageio.mimsave(video_path, visual_obs, fps=100)

        logger.log('[{}] Episode Length: {}'.format(self._name, np.mean(ep_lens)))
        logger.log('[{}] Episode Rewards: {}'.format(self._name, np.mean(ep_rets)))
        return {'global/length': np.mean(ep_lens),
                'global/reward': np.mean(ep_rets),
                'global/success': np.mean(ep_success)}
