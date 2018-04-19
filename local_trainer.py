import os.path as osp
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from mpi4py import MPI
import tqdm
import moviepy.editor as mpy
import imageio
from contextlib import contextmanager

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import zipsame, colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg

import dataset


class LocalTrainer(object):
    def __init__(self, id, env, runner, policy, old_policy, pis, global_policy, config):
        self.id = id
        self._name = 'local_%d' % id
        self._env = env.unwrapped
        self._runner = runner
        self._config = config
        self._policy = policy
        self._old_policy = old_policy
        self._pis = pis

        self._ent_coeff = config.ent_coeff

        # set to the global network
        self._init_network = U.function([], tf.group(
            *[tf.assign(v2, v1) for v1, v2 in zip(global_policy.var_list, policy.var_list)]))

        # copy weights to the global network
        self._copy_network = U.function([], tf.group(
            *[tf.assign(v1, v2) for v1, v2 in zip(global_policy.var_list, policy.var_list)]))

        # tensorboard summary
        self._is_chef = (MPI.COMM_WORLD.Get_rank() == 0)
        self._num_workers = MPI.COMM_WORLD.Get_size()
        self.summary_name = ["reward", "length", "adv"]
        self.summary_name += env.unwrapped.reward_type

        # build loss/optimizers
        self._build_trpo()

        if self._config.is_train:
            self.summary_name = ['{}/{}'.format(self._name, key) for key in self.summary_name]

    def init_network(self):
        self._init_network()

    def copy_network(self):
        self._copy_network()

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
        other_pis = self._pis

        # input placeholders
        ob = pi.ob
        ac = pi.pdtype.sample_placeholder([None], name='action')
        atarg = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name='return')  # Empirical return

        # policy
        all_var_list = pi.get_trainable_variables()
        pol_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        self._vf_adam = MpiAdam(vf_var_list)

        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = tf.reduce_mean(kl_oldnew)
        mean_ent = tf.reduce_mean(ent)
        pol_entpen = -self._config.ent_coeff * mean_ent

        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
        pol_surr = tf.reduce_mean(ratio * atarg)

        # divergence
        other_obs = [] # put id-th data
        for other_pi in other_pis:
            other_obs.extend(other_pi.obs[self.id])
        my_obs_for_other = flatten_lists(pi.obs) # put i-th data
        other_obs_for_other = [] # put i-th data
        for i, other_pi in enumerate(other_pis):
            other_obs_for_other.extend(other_pi.obs[i])

        pairwise_divergence = [tf.constant(0.)]
        for i, other_pi in enumerate(other_pis):
            if i != self.id:
                pairwise_divergence.append(tf.reduce_mean(pi.pds[self.id].kl(other_pi.pds[self.id])))
                pairwise_divergence.append(tf.reduce_mean(other_pi.pds[i].kl(pi.pds[i])))
        pol_divergence = self._config.divergence_coeff * tf.reduce_mean(pairwise_divergence)

        pol_loss = pol_surr + pol_entpen + pol_divergence
        pol_losses = {'pol_loss': pol_loss,
                      'pol_surr': pol_surr,
                      'pol_entpen': pol_entpen,
                      'pol_divergence': pol_divergence,
                      'kl': mean_kl,
                      'entropy': mean_ent}
        self.summary_name += ['vf_loss']
        self.summary_name += pol_losses.keys()

        self._get_flat = U.GetFlat(pol_var_list)
        self._set_from_flat = U.SetFromFlat(pol_var_list)
        klgrads = tf.gradients(mean_kl, pol_var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in pol_var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
        fvp = U.flatgrad(gvp, pol_var_list)

        self._update_oldpi = U.function([],[], updates=[
            tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        obs_pairwise = other_obs + my_obs_for_other + other_obs_for_other + ob
        self._compute_losses = U.function(obs_pairwise + [ac, atarg], pol_losses)
        pol_losses = dict(pol_losses)
        pol_losses.update({'g': U.flatgrad(pol_loss, pol_var_list)})
        self._compute_lossandgrad = U.function(obs_pairwise + [ac, atarg], pol_losses)
        self._compute_fvp = U.function([flat_tangent] + obs_pairwise + [ac, atarg], fvp)
        self._compute_vflossandgrad = U.function(ob + [ret], U.flatgrad(vf_loss, vf_var_list))
        self._compute_vfloss = U.function(ob + [ret], vf_loss)

        # initialize and sync
        U.initialize()
        th_init = self._get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self._set_from_flat(th_init)
        self._vf_adam.sync()
        rank = MPI.COMM_WORLD.Get_rank()

        if self._config.debug:
            logger.log("[worker: {} local net: {}] Init pol param sum: {}".format(rank, self.id, th_init.sum()))
            logger.log("[worker: {} local net: {}] Init vf param sum: {}".format(rank, self.id, self._vf_adam.getflat().sum()))

    def generate_rollout(self, sess, context=None):
        with sess.as_default(), sess.graph.as_default():
            with self.timed("sampling"):
                rollout = self._runner.rollout(stochastic=True, context=context)
                self._runner.add_advantage(rollout, 0.99, 0.98)
            self.rollout = rollout

    def update(self, sess, rollouts, global_step):
        config = self._config

        with sess.as_default(), sess.graph.as_default():
            # train policy
            info = self._update_policy(rollouts, global_step)

            for key, value in rollouts[self.id].items():
                if key.startswith('ep_'):
                    info[key.split('ep_')[1]] = np.mean(value)

            if self._is_chef:
                logger.log('[worker {}] iter: {}, rewards: {}, length: {}'.format(
                    self.id, global_step, np.mean(info["reward"]), np.mean(info["length"])))
            info = {'{}/{}'.format(self._name, key):np.mean(value) for key, value in info.items()}
            return info

    def evaluate(self, ckpt_num=None, record=False, context=None):
        config = self._config

        ep_lens = []
        ep_rets = []

        if record:
            record_dir = osp.join(config.log_dir, 'video')
            os.makedirs(record_dir, exist_ok=True)

        for _ in tqdm.trange(5):
            ep_traj = self._runner.rollout(True, True, context)
            ep_lens.append(ep_traj["ep_length"][0])
            ep_rets.append(ep_traj["ep_reward"][0])
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

    def update_ob_rms(self, rollouts):
        assert self._config.obs_norm == 'learn'
        ob = np.concatenate([rollout['ob'] for rollout in rollouts])
        ob_dict = self._env.get_ob_dict(ob)
        for ob_name in self._policy.ob_type:
            self._policy.ob_rms[ob_name].update(ob_dict[ob_name])

    def _update_policy(self, rollouts, it):
        pi = self._policy
        seg = rollouts[self.id]
        info = defaultdict(list)

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        atarg = (atarg - atarg.mean()) / atarg.std()
        info['adv'] = np.mean(atarg)

        other_ob_list = []
        for i, other_pi in enumerate(self._pis):
            other_ob_list.extend(other_pi.get_ob_list(rollouts[i]["ob"]))

        ob_list = pi.get_ob_list(ob)
        args = ob_list * self._config.num_contexts + \
            other_ob_list * 2 + ob_list + [ac, atarg]
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return self._all_mean(self._compute_fvp(p, *fvpargs)) + self._config.cg_damping * p

        self._update_oldpi()

        with self.timed("compute gradient"):
            lossbefore = self._compute_lossandgrad(*args)
            lossbefore = {k: self._all_mean(np.array(lossbefore[k])) for k in sorted(lossbefore.keys())}
        g = lossbefore['g']

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with self.timed("compute conjugate gradient"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self._config.cg_iters, verbose=False)
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
                for key, value in meanlosses.items():
                    if key != 'g':
                        info[key].append(value)
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
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:]), paramsums

        with self.timed("updating value function"):
            for _ in range(self._config.vf_iters):
                for (mbob, mbret) in dataset.iterbatches(
                        (ob, tdlamret), include_final_partial_batch=False,
                        batch_size=self._config.vf_batch_size):
                    ob_list = pi.get_ob_list(mbob)
                    g = self._all_mean(self._compute_vflossandgrad(*ob_list, mbret))
                    self._vf_adam.update(g, self._config.vf_stepsize)
                    vf_loss = self._all_mean(np.array(self._compute_vfloss(*ob_list, mbret)))
                    info['vf_loss'].append(vf_loss)

        for key, value in info.items():
            info[key] = np.mean(value)
        return info


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
