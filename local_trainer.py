import os.path as osp
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from mpi4py import MPI
import tqdm
import moviepy.editor as mpy
from contextlib import contextmanager

from baselines.common import zipsame
from baselines import logger
from baselines.common import colorize
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.statistics import stats
import dataset


class LocalTrainer(object):
    def __init__(self, name, env, runner, policy, old_policy, global_policy, config):
        self._name = name
        self._env = env.unwrapped
        self._runner = runner
        self._config = config
        self._policy = policy
        self._old_policy = old_policy

        self._entcoeff = config.entcoeff
        self._optim_epochs = config.optim_epochs
        self._optim_stepsize = config.optim_stepsize
        self._optim_batchsize = config.optim_batchsize

        # global step
        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
        self.update_global_step = tf.assign(self.global_step, self.global_step + 1)

        # set to the global network
        self._init_network = U.function([], tf.group(
            *[v1.assign(v2) for v1, v2 in zip(global_policy.var_list, policy.var_list)]))

        # tensorboard summary
        self._time_str = time.strftime("%y-%m-%d_%H-%M-%S")
        self._is_chef = (MPI.COMM_WORLD.Get_rank() == 0)
        self._num_workers = MPI.COMM_WORLD.Get_size()
        if self._is_chef:
            self.summary_name = ["reward", "length"]
            self.summary_name += env.unwrapped.reward_type

        # build loss/optimizers
        self._build_trpo()

        if self._is_chef and self._config.is_train:
            self.summary_name = ['{}/{}'.format(self._name, key) for key in self.summary_name]
            #self.ep_stats = stats(self.summary_name)
            self.writer = U.file_writer(config.log_dir)

    def init_network(self):
        self._init_network()

    @contextmanager
    def timed(self, msg):
        if self._is_chef:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def _all_mean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= self._num_workers
        return out

    def _build_trpo(self):
        pi = self._policy
        oldpi = self._old_policy

        # input placeholders
        obs = pi.obs
        ac = pi.pdtype.sample_placeholder([None], name='action')
        atarg = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name='return')  # Empirical return

        # policy
        all_var_list = pi.get_trainable_variables()
        self.pol_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        self.vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        self._vf_adam = MpiAdam(self.vf_var_list)

        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = tf.reduce_mean(kl_oldnew)
        mean_ent = tf.reduce_mean(ent)
        pol_entpen = -self._config.entcoeff * mean_ent

        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
        pol_surr = tf.reduce_mean(ratio * atarg)
        pol_loss = pol_surr + pol_entpen

        pol_losses = {'pol_loss': pol_loss,
                      'pol_surr': pol_surr,
                      'pol_entpen': pol_entpen,
                      'kl': mean_kl,
                      'entropy': mean_ent}
        if self._is_chef:
            self.summary_name += ['vf_loss']
            self.summary_name += pol_losses.keys()

        self._get_flat = U.GetFlat(self.pol_var_list)
        self._set_from_flat = U.SetFromFlat(self.pol_var_list)
        klgrads = tf.gradients(mean_kl, self.pol_var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in self.pol_var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
        fvp = U.flatgrad(gvp, self.pol_var_list)

        self._update_oldpi = U.function([],[], updates=[
            tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self._compute_losses = U.function(obs + [ac, atarg], pol_losses)
        pol_losses = dict(pol_losses)
        pol_losses.update({'g': U.flatgrad(pol_loss, self.pol_var_list)})
        self._compute_lossandgrad = U.function(obs + [ac, atarg], pol_losses)
        self._compute_fvp = U.function([flat_tangent] + obs + [ac, atarg], fvp)
        self._compute_vflossandgrad = U.function(obs + [ret], U.flatgrad(vf_loss, self.vf_var_list))
        self._compute_vfloss = U.function(obs + [ret], vf_loss)

        # initialize and sync
        U.initialize()
        th_init = self._get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self._set_from_flat(th_init)
        self._vf_adam.sync()
        print("Init param sum", th_init.sum())

    def generate_rollout(self):
        sess = U.get_session()

        # rollout
        with self.timed("sampling"):
            rollout = self._runner.rollout(stochastic=True)
            self._runner.add_advantage(rollout, 0.99, 0.98)
        self.rollout = rollout

    def update(self, sess):
        config = self._config

        with sess.as_default(), sess.graph.as_default():
            global_step = sess.run(self.global_step)
            # rollout
            self.generate_rollout()

            # train policy
            info = self._update_policy(self.rollout, global_step)

            for key, value in self.rollout.items():
                if key.startswith('ep_'):
                    info[key.split('ep_')[1]] = np.mean(value)
            info = {'{}/{}'.format(self._name, key):value for key, value in info.items()}
            #self.ep_stats.add_all_summary_dict(self.writer, info, global_step)
            global_step = sess.run(self.update_global_step)

    def evaluate(self, rollout, ckpt_num=None):
        config = self._config

        ep_lens = []
        ep_rets = []
        if config.record:
            record_dir = osp.join(config.log_dir, 'video')
            os.makedirs(record_dir, exist_ok=True)

        for _ in tqdm.trange(10):
            ep_traj = self._runner.rollout(True, True)
            ep_lens.append(ep_traj["ep_length"][0])
            ep_rets.append(ep_traj["ep_reward"][0])
            logger.log('Trial #{}: lengths {}, returns {}'.format(_, ep_traj["ep_length"][0], ep_traj["ep_reward"][0]))

            # Video recording
            if config.record:
                visual_obs = ep_traj["visual_obs"]
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

    def _update_policy(self, seg, it):
        pi = self._policy
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        atarg = (atarg - atarg.mean()) / atarg.std()

        if self._is_chef:
            info = defaultdict(list)

        ob_dict = self._env.get_ob_dict(ob)
        if self._config.obs_norm:
            for ob_name in pi.ob_type:
                pi.ob_rms[ob_name].update(ob_dict[ob_name])

        ob_list = pi.get_ob_list(ob_dict)
        args = ob_list + [ac, atarg]
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return self._all_mean(self._compute_fvp(p, *fvpargs)) + self._config.cg_damping * p

        self._update_oldpi()

        with self.timed("computegrad"):
            lossbefore = self._compute_lossandgrad(*args)
            lossbefore = {k: self._all_mean(np.array(lossbefore[k])) for k in sorted(lossbefore.keys())}
        g = lossbefore['g']

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with self.timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self._config.cg_iters, verbose=self._is_chef)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self._config.max_kl)
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore['pol_loss']
            stepsize = 1.0
            thbefore = self._get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self._set_from_flat(thnew)
                meanlosses = self._compute_losses(*args)
                meanlosses = {k: self._all_mean(np.array(meanlosses[k])) for k in sorted(meanlosses.keys())}
                #print('mean', [float(meanlosses[k]) for k in ['pol_loss', 'kl', 'pol_entpen', 'pol_surr', 'entropy']])
                if self._is_chef:
                    for key, value in meanlosses.items():
                        if key != 'g':
                            info['trpo/' + key].append(value)
                surr = meanlosses['pol_loss']
                kl = meanlosses['kl']
                meanlosses = np.array(list(meanlosses.values()))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > self._config.max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self._set_from_flat(thbefore)
            if self._num_workers > 1 and it % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self._vf_adam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        with self.timed("vf"):
            for _ in range(self._config.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret),
                include_final_partial_batch=False, batch_size=64):
                    ob_list = pi.get_ob_list(mbob)
                    g = self._all_mean(self._compute_vflossandgrad(*ob_list, mbret))
                    self._vf_adam.update(g, self._config.vf_stepsize)
                    vf_loss = self._all_mean(np.array(self._compute_vfloss(*ob_list, mbret)))
                    if self._is_chef:
                        info['trpo/vf_loss'].append(vf_loss)

        if self._is_chef:
            for key, value in info.items():
                info[key] = np.mean(value)
            return info
        return None


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
