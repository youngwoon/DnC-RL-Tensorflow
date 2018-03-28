import time
import numpy as np
import os
import ipdb

import tensorflow as tf
from collections import deque
from mpi4py import MPI
from contextlib import contextmanager
from tqdm import tqdm
import moviepy.editor as mpy

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.statistics import stats


# Sample one trajectory (until trajectory end)
def traj_episode_generator(pi, env, horizon, stochastic, info_list, record=False, render=False):

    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    vpreds = []
    visual_obs = []
    info = {}
    for i in info_list:
        info.update({i: []})

    while True:
        if record:
            visual_ob = env.unwrapped.get_visual_observation()
            visual_obs.append(visual_ob)
        if render:
            env.render()
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)
        vpreds.append(vpred)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        for k, v in info.items():
            if k == 'vel':
                vel = env.unwrapped.get_body_vel()
                v.append(vel)

        cur_ep_ret += rew
        cur_ep_len += 1
        if t > 0 and (new or t % horizon == 0):
            # convert list into numpy array
            obs = np.array(obs)
            rews = np.array(rews)
            news = np.array(news)
            acs = np.array(acs)
            if record:
                yield {"ob":obs, "rew":rews, "new":news, "ac":acs, "vpred": vpreds, "nextvpred": vpred * (1 - new),
                    "ep_ret":cur_ep_ret, "ep_len":cur_ep_len, "visual_obs":visual_obs, "info":info}
            else:
                yield {"ob":obs, "rew":rews, "new":news, "ac":acs, "vpred": vpreds, "nextvpred": vpred * (1 - new),
                    "ep_ret":cur_ep_ret, "ep_len":cur_ep_len, "info":info}
            ob = env.reset()

            for k, v in info.items():
                info[k] = []

            cur_ep_ret = 0; cur_ep_len = 0; t = 0

            # Initialize history arrays
            obs = []
            rews = []
            news = []
            acs = []
            vpreds = []
            visual_obs = []
        t += 1


def traj_segment_generator(pi, env, horizon, stochastic, render=False):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    reward_details = {}
    for name in env.unwrapped.reward_type:
        reward_details.update({name: np.zeros(horizon, 'float32')  })

    while True:
        if render:
            env.render()
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            data = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            for name in env.unwrapped.reward_type:
                data.update({name: np.mean(reward_details[name])})
            yield data
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)
        for name in env.unwrapped.reward_type:
            reward_details[name][i] = info[name]

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, checkpoint_dir, log_dir, *,
          render,
          timesteps_per_batch, # what to train on
          max_kl,
          cg_iters,
          gamma, lam, # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("{}/pi".format(env.spec.id), ob_space, ac_space)
    oldpi = policy_func("{}/oldpi".format(env.spec.id), ob_space, ac_space)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[2] == "pol"]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[2] == "vf"]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
                                                   for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    if rank == 0:
        writer = U.file_writer(log_dir)
    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, render=render)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    reward_details_buffer = {}
    for name in env.unwrapped.reward_type:
        reward_details_buffer.update({name: deque(maxlen=40)})
    if rank == 0:
        ep_stats = stats(["Episode_rewards", "Episode_length"] + env.unwrapped.reward_type)
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        if iters_so_far % 100 == 0:
            U.save_state('{}/trpo-{}'.format(checkpoint_dir, iters_so_far))
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        for name in env.unwrapped.reward_type:
            lrlocal += ([seg[name]], )
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        log_data  = map(flatten_lists, zip(*listoflrpairs))
        log_data = [i for i in log_data]
        lens, rews = log_data[0], log_data[1]

        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        for i, name in enumerate(env.unwrapped.reward_type):
            reward_details_buffer[name].extend(log_data[i+2])

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
            to_write = [np.mean(rewbuffer), np.mean(lenbuffer)]
            for name in env.unwrapped.reward_type:
                to_write += [np.mean(reward_details_buffer[name]),]
            ep_stats.add_all_summary(writer, to_write, iters_so_far)
        iters_so_far += 1

def evaluate(env,
             policy_func,
             timesteps_per_batch,
             load_model_path,
             video_prefix,
             record,
             render,
             info_list,
             gamma,
             lam):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("{}/pi".format(env.spec.id), ob_space, ac_space)
    ob = U.get_placeholder_cached(name="ob")
    ret = tf.placeholder(dtype=tf.float32, shape=[None])
    vferr = U.mean(tf.square(pi.vpred - ret))
    compute_vfloss = U.function([ob, ret], vferr)
    U.initialize()
    U.load_state(load_model_path)

    ep_gen = traj_episode_generator(pi, env, 1024, stochastic=False, record=record, render=render, info_list=info_list)
    ep_lens = []
    ep_rets = []
    visual_obs = []
    info = {}
    log_value = True
    if log_value:
        info.update({"value": []})
    for i in info_list:
        info.update({i: []})

    if record:
        record_dir = os.path.join(os.path.dirname(load_model_path), 'video')
        os.makedirs(record_dir, exist_ok=True)
    for _ in tqdm(range(10)):
        ep_traj = ep_gen.__next__()
        add_vtarg_and_adv(ep_traj, gamma, lam)
        ep_lens.append(ep_traj["ep_len"])
        ep_rets.append(ep_traj["ep_ret"])
        if log_value:
            info['value'].append(ep_traj["vpred"])
        for k in info_list:
            info[k].append(ep_traj['info'][k])

        # Video recording
        if _ % 2 == 0 and record:
            visual_obs = ep_traj["visual_obs"]
            if video_prefix is None:
                video_path = os.path.join(record_dir, '{}.mp4'.format(_))
            else:
                video_path = os.path.join(record_dir, '{}-{}.mp4'.format(video_prefix, _))
            fps = 15.

            def f(t):
                frame_length = len(visual_obs)
                new_fps = 1./(1./fps + 1./frame_length)
                idx = min(int(t*new_fps), frame_length-1)
                return visual_obs[idx]
            video = mpy.VideoClip(f, duration=len(visual_obs)/fps+2)
            video.write_videofile(video_path, fps, verbose=False)

    print('Episode Length: {}'.format(sum(ep_lens)/10.))
    print('Episode Rewards: {}'.format(sum(ep_rets)/10.))
    #if len(info_list) != 0:
    #    for k, v in info.items():
    #        v_flatten = flatten_lists(v)
    #        avg = sum(v_flatten) / len(v_flatten)
    #        print("Average {}: {}".format(k, avg))


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
